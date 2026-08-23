from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from datetime import UTC
from gzip import compress
from typing import Any

import httpx2
import pytest
from ccflow_etl import ExecutionPolicy

from ccflow_http import HTTPAuth, HTTPConfig, HTTPModel, HTTPRequest, HTTPRequestContext, HTTPResponseResult, HTTPRetryPolicy, safe_request_dump


@dataclass
class FakeResponse:
    value: Any
    status_code: int = 200
    headers: dict[str, str] | None = None
    url: str = "https://api.example.test/v1/tickers/AAA"

    @property
    def content(self) -> bytes:
        return b"payload"

    @property
    def text(self) -> str:
        return "payload"

    def json(self) -> Any:
        return self.value

    def raise_for_status(self) -> None:
        return None


def test_http_model_renders_request_and_returns_json(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append({"client": self.kwargs, "request": kwargs})
            return FakeResponse(value={"status": "OK", "results": [{"ticker": "AAA"}]}, headers={"x-limit-remaining": "9"})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers/{{ ticker }}",
        query={"date": "{{ date }}", "apiKey": "{{ api_key }}"},
        headers={"Authorization": "Bearer {{ token }}"},
        response_format="json",
        timeout=12.5,
    )

    result = model(
        HTTPRequestContext(
            template_values={"ticker": "AAA", "date": "2024-01-03", "api_key": "secret", "token": "abc"},
            query={"adjusted": True},
        )
    )

    assert result.value == {"status": "OK", "results": [{"ticker": "AAA"}]}
    assert result.status_code == 200
    assert result.headers == {"x-limit-remaining": "9"}
    assert result.url == "https://api.example.test/v1/tickers/AAA"
    assert calls == [
        {
            "client": {"base_url": "https://api.example.test", "timeout": 12.5, "follow_redirects": True},
            "request": {
                "method": "GET",
                "url": "/v1/tickers/AAA",
                "params": {"date": "2024-01-03", "apiKey": "secret", "adjusted": True},
                "headers": {"Authorization": "Bearer abc"},
                "json": None,
                "content": None,
            },
        }
    ]


def test_http_model_can_explain_request_without_network():
    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers/{{ ticker }}",
        query={"date": "{{ date }}"},
    )

    request = model.build_request(HTTPRequestContext(template_values={"ticker": "AAA", "date": "2024-01-03"}))

    assert request.method == "GET"
    assert request.url == "/v1/tickers/AAA"
    assert request.params == {"date": "2024-01-03"}


def test_safe_request_dump_redacts_secret_params_and_headers():
    request = HTTPRequest(
        method="GET",
        url="/v1/tickers/AAA",
        params={"date": "2024-01-03", "apiKey": "query-secret", "page_token": "cursor-secret"},
        headers={"Authorization": "Bearer header-secret", "X-Request-ID": "abc"},
    )

    assert safe_request_dump(request) == {
        "method": "GET",
        "url": "/v1/tickers/AAA",
        "params": {"date": "2024-01-03", "apiKey": "***", "page_token": "***"},
        "headers": {"Authorization": "***", "X-Request-ID": "abc"},
        "json_data": None,
        "content": None,
    }


def test_http_model_error_message_omits_secret_query_values(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            request = httpx2.Request("GET", "https://api.example.test/v1/tickers/AAA?apiKey=secret")
            response = httpx2.Response(429, request=request)
            raise httpx2.HTTPStatusError("rate limited with apiKey=secret", request=request, response=response)

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    model = HTTPModel(base_url="https://api.example.test", path="/v1/tickers/{{ ticker }}", query={"apiKey": "{{ api_key }}"})

    with pytest.raises(RuntimeError, match="HTTP GET /v1/tickers/AAA failed with status 429") as error:
        model(HTTPRequestContext(template_values={"ticker": "AAA", "api_key": "secret"}))

    assert "secret" not in str(error.value)
    assert "apiKey" not in str(error.value)


def test_http_model_retries_retryable_status_and_captures_rate_limit(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                request = httpx2.Request("GET", "https://api.example.test/v1/tickers")
                response = httpx2.Response(429, request=request, headers={"retry-after": "1"})
                raise httpx2.HTTPStatusError("rate limited", request=request, response=response)
            return FakeResponse(value={"status": "OK"}, headers={"x-ratelimit-remaining": "8"})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    result = HTTPModel(base_url="https://api.example.test", path="/v1/tickers", max_attempts=2)(HTTPRequestContext())

    assert len(calls) == 2
    assert result.value == {"status": "OK"}
    assert result.attempts == 2
    assert result.rate_limit == {"x-ratelimit-remaining": "8"}


def test_http_model_consumes_shared_retry_policy(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                request = httpx2.Request("GET", "https://api.example.test/v1/tickers")
                response = httpx2.Response(503, request=request)
                raise httpx2.HTTPStatusError("unavailable", request=request, response=response)
            return FakeResponse(value={"status": "OK"})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    model = HTTPModel(base_url="https://api.example.test", path="/v1/tickers", retry_policy=HTTPRetryPolicy(max_attempts=2, retry_status_codes=[503]))

    assert model(HTTPRequestContext()).attempts == 2


def test_http_model_consumes_shared_retry_delay_and_execution_policy(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                request = httpx2.Request("GET", "https://api.example.test/v1/tickers")
                response = httpx2.Response(429, request=request)
                raise httpx2.HTTPStatusError("rate limited", request=request, response=response)
            return FakeResponse(value={"status": "OK"})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers",
        retry_policy=HTTPRetryPolicy(max_attempts=2, retry_status_codes=[429], wait_initial=1.25),
        execution_policy=ExecutionPolicy(requests_per_interval=1, interval_seconds=2.0),
    )
    sleeps = []
    request_times = iter([100.0, 101.25])
    monkeypatch.setattr(model, "_sleep", sleeps.append)
    monkeypatch.setattr(model, "_now", lambda: next(request_times))

    result = model(HTTPRequestContext())

    assert len(calls) == 2
    assert sleeps == [1.25, 0.75]
    assert result.retry_events == [
        {
            "attempt": 1,
            "outcome": "retry",
            "delay_seconds": 1.25,
            "status_code": 429,
            "category": "rate_limit",
            "message": "retryable status code 429",
        }
    ]
    assert result.retry_summary == {"attempts": 2, "retried": 1, "failed": 0, "succeeded": 1}


def test_http_model_retries_timeout_exception(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise httpx2.TimeoutException("timed out")
            return FakeResponse(value={"status": "OK"})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    result = HTTPModel(base_url="https://api.example.test", path="/v1/tickers", max_attempts=2)(HTTPRequestContext())

    assert len(calls) == 2
    assert result.value == {"status": "OK"}
    assert result.attempts == 2


def test_http_model_retries_5xx_until_attempts_are_exhausted(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            request = httpx2.Request("GET", "https://api.example.test/v1/tickers?apiKey=secret")
            response = httpx2.Response(500, request=request)
            raise httpx2.HTTPStatusError("server error with apiKey=secret", request=request, response=response)

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    model = HTTPModel(base_url="https://api.example.test", path="/v1/tickers", query={"apiKey": "secret"}, max_attempts=2)

    with pytest.raises(RuntimeError, match="HTTP GET /v1/tickers failed with status 500") as error:
        model(HTTPRequestContext())

    assert len(calls) == 2
    assert "secret" not in str(error.value)
    assert "apiKey" not in str(error.value)


def test_http_model_paginates_massive_style_next_url(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return FakeResponse(value={"results": [{"ticker": "AAA"}], "next_url": "/v3/reference/tickers?cursor=2"})
            return FakeResponse(value={"results": [{"ticker": "BBB"}]})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    result = HTTPModel(base_url="https://api.example.test", path="/v3/reference/tickers", paginate=True)(HTTPRequestContext())

    assert [call["url"] for call in calls] == ["/v3/reference/tickers", "/v3/reference/tickers"]
    assert calls[1]["params"] == {"cursor": "2"}
    assert result.value["results"] == [{"ticker": "AAA"}, {"ticker": "BBB"}]
    assert result.pages == 2


def test_http_model_preserves_query_auth_for_next_url_pagination(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return FakeResponse(value={"results": [{"ticker": "AAA"}], "next_url": "/v3/reference/tickers?cursor=2"})
            return FakeResponse(value={"results": [{"ticker": "BBB"}]})

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", FakeClient)

    HTTPModel(
        base_url="https://api.example.test",
        path="/v3/reference/tickers",
        query={"apiKey": "secret"},
        paginate=True,
    )(HTTPRequestContext())

    assert calls[1]["url"] == "/v3/reference/tickers"
    assert calls[1]["params"] == {"cursor": "2", "apiKey": "secret"}


def test_http_model_sends_next_url_query_and_preserved_auth_with_httpx2_transport():
    seen_urls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen_urls.append(str(request.url))
        if len(seen_urls) == 1:
            return httpx2.Response(200, json={"results": [{"ticker": "AAA"}], "next_url": "/v3/reference/tickers?cursor=2"})
        return httpx2.Response(200, json={"results": [{"ticker": "BBB"}]})

    model = HTTPModel(
        config=HTTPConfig(base_url="https://api.example.test", transport=httpx2.MockTransport(handler)),
        path="/v3/reference/tickers",
        query={"apiKey": "secret"},
        paginate=True,
    )

    result = model(HTTPRequestContext())

    assert seen_urls == [
        "https://api.example.test/v3/reference/tickers?apiKey=secret",
        "https://api.example.test/v3/reference/tickers?cursor=2&apiKey=secret",
    ]
    assert result.value["results"] == [{"ticker": "AAA"}, {"ticker": "BBB"}]


def test_http_model_applies_config_and_all_auth_strategies_with_mock_transport():
    seen_requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen_requests.append(request)
        return httpx2.Response(200, json={"status": "OK"}, headers={"x-ratelimit-remaining": "7"})

    transport = httpx2.MockTransport(handler)
    config = HTTPConfig(base_url="https://api.example.test", timeout=4.0, headers={"Accept": "application/json"}, transport=transport)

    bearer = HTTPModel(config=config, path="/v1/{{ resource }}", auth=HTTPAuth(strategy="bearer", token="{{ token }}"))
    bearer_result = bearer(HTTPRequestContext(template_values={"resource": "tickers", "token": "bearer-token"}))

    api_header = HTTPModel(config=config, path="/v1/tickers", auth=HTTPAuth(strategy="api_key_header", name="X-API-Key", value="{{ api_key }}"))
    api_header(HTTPRequestContext(template_values={"api_key": "header-key"}))

    api_query = HTTPModel(config=config, path="/v1/tickers", auth=HTTPAuth(strategy="api_key_query", name="apiKey", value="{{ api_key }}"))
    api_query(HTTPRequestContext(template_values={"api_key": "query-key"}))

    basic = HTTPModel(config=config, path="/v1/tickers", auth=HTTPAuth(strategy="basic", username="{{ user }}", password="{{ password }}"))
    basic(HTTPRequestContext(template_values={"user": "svc", "password": "secret"}))

    no_auth = HTTPModel(config=config, path="/v1/tickers", auth=HTTPAuth(strategy="none"))
    no_auth(HTTPRequestContext())

    assert isinstance(bearer_result, HTTPResponseResult)
    assert str(seen_requests[0].url) == "https://api.example.test/v1/tickers"
    assert seen_requests[0].headers["accept"] == "application/json"
    assert seen_requests[0].headers["authorization"] == "Bearer bearer-token"
    assert seen_requests[1].headers["x-api-key"] == "header-key"
    assert dict(seen_requests[2].url.params) == {"apiKey": "query-key"}
    assert seen_requests[3].headers["authorization"].startswith("Basic ")
    assert b64decode(seen_requests[3].headers["authorization"].removeprefix("Basic ")).decode("utf-8") == "svc:secret"
    assert "authorization" not in seen_requests[4].headers


def test_http_model_parses_csv_and_gzip_responses_with_mock_transport():
    def csv_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="ticker,volume\nAAA,10\nBBB,20\n")

    csv_result = HTTPModel(
        config=HTTPConfig(base_url="https://api.example.test", transport=httpx2.MockTransport(csv_handler)),
        path="/csv",
        response_format="csv",
    )(HTTPRequestContext())

    def gzip_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=compress(b'{"status":"OK"}'))

    gzip_result = HTTPModel(
        config=HTTPConfig(base_url="https://api.example.test", transport=httpx2.MockTransport(gzip_handler)),
        path="/gzip",
        response_format="gzip",
    )(HTTPRequestContext())

    assert csv_result.value == [{"ticker": "AAA", "volume": "10"}, {"ticker": "BBB", "volume": "20"}]
    assert gzip_result.value == b'{"status":"OK"}'


def test_http_model_supports_cursor_page_and_offset_pagination_with_mock_transport():
    cursor_requests = []

    def cursor_handler(request: httpx2.Request) -> httpx2.Response:
        cursor_requests.append(dict(request.url.params))
        if "cursor" not in request.url.params:
            return httpx2.Response(200, json={"results": [{"id": 1}], "next_cursor": "abc"})
        return httpx2.Response(200, json={"results": [{"id": 2}]})

    cursor_result = HTTPModel(
        config=HTTPConfig(base_url="https://api.example.test", transport=httpx2.MockTransport(cursor_handler)),
        path="/items",
        paginate=True,
        pagination_mode="cursor",
        next_cursor_field="next_cursor",
        cursor_param="cursor",
    )(HTTPRequestContext())

    page_requests = []

    def page_handler(request: httpx2.Request) -> httpx2.Response:
        page_requests.append(dict(request.url.params))
        page = int(request.url.params["page"])
        payload = {1: [{"id": 1}], 2: [{"id": 2}]}.get(page, [])
        return httpx2.Response(200, json={"results": payload})

    page_result = HTTPModel(
        config=HTTPConfig(base_url="https://api.example.test", transport=httpx2.MockTransport(page_handler)),
        path="/items",
        paginate=True,
        pagination_mode="page",
        page_param="page",
        page_start=1,
        max_pages=5,
    )(HTTPRequestContext())

    offset_requests = []

    def offset_handler(request: httpx2.Request) -> httpx2.Response:
        offset_requests.append(dict(request.url.params))
        offset = int(request.url.params["offset"])
        payload = {0: [{"id": 1}], 2: [{"id": 2}]}.get(offset, [])
        return httpx2.Response(200, json={"results": payload})

    offset_result = HTTPModel(
        config=HTTPConfig(base_url="https://api.example.test", transport=httpx2.MockTransport(offset_handler)),
        path="/items",
        paginate=True,
        pagination_mode="offset",
        offset_param="offset",
        limit_param="limit",
        limit=2,
        max_pages=5,
    )(HTTPRequestContext())

    assert cursor_requests == [{}, {"cursor": "abc"}]
    assert cursor_result.value["results"] == [{"id": 1}, {"id": 2}]
    assert page_requests == [{"page": "1"}, {"page": "2"}, {"page": "3"}]
    assert page_result.value["results"] == [{"id": 1}, {"id": 2}]
    assert offset_requests == [{"offset": "0", "limit": "2"}, {"offset": "2", "limit": "2"}, {"offset": "4", "limit": "2"}]
    assert offset_result.value["results"] == [{"id": 1}, {"id": 2}]


def _retrying_client(calls, response_factory):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                response_factory()
            return FakeResponse(value={"status": "OK"})

    return FakeClient


def _raise_status(status_code, headers=None):
    request = httpx2.Request("GET", "https://api.example.test/v1/tickers")
    response = httpx2.Response(status_code, request=request, headers=headers or {})
    raise httpx2.HTTPStatusError("error", request=request, response=response)


@pytest.mark.parametrize(("exception_type", "category"), [(httpx2.ConnectError, "connection"), (httpx2.RemoteProtocolError, "exception")])
def test_http_model_retries_transport_error(monkeypatch, exception_type, category):
    calls = []

    def fail_once():
        raise exception_type("transport failed")

    monkeypatch.setattr("ccflow_http.base.httpx2.Client", _retrying_client(calls, fail_once))

    result = HTTPModel(base_url="https://api.example.test", path="/v1/tickers", max_attempts=2)(HTTPRequestContext())

    assert len(calls) == 2
    assert result.attempts == 2
    assert result.retry_events[0]["category"] == category


def test_http_model_honors_retry_after_seconds_over_backoff(monkeypatch):
    calls = []
    monkeypatch.setattr("ccflow_http.base.httpx2.Client", _retrying_client(calls, lambda: _raise_status(429, {"retry-after": "45"})))

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers",
        retry_policy=HTTPRetryPolicy(max_attempts=2, wait_initial=1.0),
    )
    sleeps = []
    monkeypatch.setattr(model, "_sleep", sleeps.append)

    result = model(HTTPRequestContext())

    assert len(calls) == 2
    assert sleeps == [45.0]
    assert result.retry_events[0]["delay_seconds"] == 45.0


def test_http_model_uses_backoff_when_it_exceeds_retry_after(monkeypatch):
    calls = []
    monkeypatch.setattr("ccflow_http.base.httpx2.Client", _retrying_client(calls, lambda: _raise_status(429, {"retry-after": "1"})))

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers",
        retry_policy=HTTPRetryPolicy(max_attempts=2, wait_initial=5.0),
    )
    sleeps = []
    monkeypatch.setattr(model, "_sleep", sleeps.append)

    model(HTTPRequestContext())

    assert sleeps == [5.0]


def test_http_model_caps_retry_after_at_retry_after_max(monkeypatch):
    calls = []
    monkeypatch.setattr("ccflow_http.base.httpx2.Client", _retrying_client(calls, lambda: _raise_status(429, {"retry-after": "9999"})))

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers",
        retry_policy=HTTPRetryPolicy(max_attempts=2, wait_initial=1.0, retry_after_max=300.0),
    )
    sleeps = []
    monkeypatch.setattr(model, "_sleep", sleeps.append)

    model(HTTPRequestContext())

    assert sleeps == [300.0]


def test_http_model_honors_retry_after_http_date(monkeypatch):
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    calls = []
    retry_at = format_datetime(datetime.now(UTC) + timedelta(seconds=60))
    monkeypatch.setattr("ccflow_http.base.httpx2.Client", _retrying_client(calls, lambda: _raise_status(429, {"retry-after": retry_at})))

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers",
        retry_policy=HTTPRetryPolicy(max_attempts=2, wait_initial=1.0),
    )
    sleeps = []
    monkeypatch.setattr(model, "_sleep", sleeps.append)

    model(HTTPRequestContext())

    assert len(sleeps) == 1
    assert 50.0 <= sleeps[0] <= 60.0


def test_http_model_ignores_invalid_retry_after(monkeypatch):
    calls = []
    monkeypatch.setattr("ccflow_http.base.httpx2.Client", _retrying_client(calls, lambda: _raise_status(429, {"retry-after": "soon"})))

    model = HTTPModel(
        base_url="https://api.example.test",
        path="/v1/tickers",
        retry_policy=HTTPRetryPolicy(max_attempts=2, wait_initial=1.0),
    )
    sleeps = []
    monkeypatch.setattr(model, "_sleep", sleeps.append)

    model(HTTPRequestContext())

    assert sleeps == [1.0]
