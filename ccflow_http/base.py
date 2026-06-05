from base64 import b64encode
from csv import DictReader
from gzip import decompress
from io import StringIO
from time import monotonic, sleep
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import httpx
from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult, PyObjectPath
from ccflow.utils.retry import RetryPolicy
from ccflow_etl import ExecutionPolicy
from jinja2 import Environment
from pydantic import Field

__all__ = (
    "HTTPConfig",
    "HTTPAuth",
    "HTTPContext",
    "HTTPRequestContext",
    "HTTPRequest",
    "HTTPRetryPolicy",
    "HTTPResponseResult",
    "HTTPResult",
    "HTTPModel",
    "redact_mapping",
    "safe_request_dump",
)

ResponseFormat = Literal["json", "text", "bytes", "csv", "gzip"]
HTTPMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
HTTPAuthStrategy = Literal["none", "bearer", "api_key_header", "api_key_query", "basic"]
HTTPPaginationMode = Literal["next_url", "cursor", "page", "offset"]
HTTPRetryOutcome = Literal["retry", "failed"]


class HTTPConfig(BaseModel):
    base_url: str = ""
    timeout: float = 30.0
    follow_redirects: bool = True
    headers: Dict[str, str] = Field(default_factory=dict)
    transport: Optional[Any] = None


class HTTPAuth(BaseModel):
    strategy: HTTPAuthStrategy = "none"
    token: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    scheme: str = "Bearer"


class HTTPContext(ContextBase):
    path: Optional[str] = None
    query: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    template_values: Dict[str, Any] = Field(default_factory=dict)
    json_body: Optional[Any] = None
    content: Optional[Union[bytes, str]] = None


class HTTPRequestContext(HTTPContext): ...


class HTTPRequest(BaseModel):
    method: HTTPMethod
    url: str
    params: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    json_data: Optional[Any] = None
    content: Optional[Union[bytes, str]] = None


def redact_mapping(values: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in values.items():
        normalized_key = key.lower().replace("_", "").replace("-", "")
        if normalized_key in {"apikey", "authorization", "password"} or "token" in normalized_key or "secret" in normalized_key:
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def safe_request_dump(request: HTTPRequest) -> Dict[str, Any]:
    request_data = request.model_dump(exclude={"type_"})
    request_data["params"] = redact_mapping(request_data.get("params", {}))
    request_data["headers"] = redact_mapping(request_data.get("headers", {}))
    return request_data


class HTTPRetryEvent(BaseModel):
    attempt: int
    outcome: HTTPRetryOutcome
    delay_seconds: float = 0.0
    status_code: Optional[int] = None
    exception_type: Optional[str] = None
    category: Optional[str] = None
    message: Optional[str] = None


class HTTPRetryPolicy(RetryPolicy):
    retry_status_codes: List[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_exceptions: List[PyObjectPath] = Field(
        default_factory=lambda: [PyObjectPath.validate(httpx.TimeoutException), PyObjectPath.validate(httpx.ConnectError)]
    )
    timeout_exception_types: List[str] = Field(
        default_factory=lambda: ["TimeoutError", "TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"]
    )

    def should_retry_status(self, status_code: Optional[int], attempt: int) -> bool:
        return status_code in self.retry_status_codes and attempt < self.max_attempts

    def should_retry_exception(self, exception: BaseException, attempt: int) -> bool:
        return self._should_retry(exception) and attempt < self.max_attempts

    def delay_seconds(self, attempt: int, jitter_value: Optional[float] = None) -> float:
        if attempt < 1:
            raise ValueError("attempt must be greater than or equal to 1")
        return self._compute_delay(attempt, jitter_value=jitter_value)

    def retry_delay_seconds(self, attempt: int, total_wait_seconds: float) -> Optional[float]:
        delay_seconds = self.delay_seconds(attempt)
        if self.max_delay is not None and total_wait_seconds + delay_seconds > self.max_delay:
            return None
        return delay_seconds

    def exception_category(self, exception: BaseException) -> str:
        exception_names = {type(exception).__name__}
        exception_names.update(base.__name__ for base in type(exception).__mro__)
        if exception_names.intersection(self.timeout_exception_types) or any("Timeout" in name for name in exception_names):
            return "timeout"
        if any("Connect" in name or "Connection" in name for name in exception_names):
            return "connection"
        return "exception"

    def status_category(self, status_code: Optional[int]) -> str:
        if status_code == 429:
            return "rate_limit"
        if status_code == 408:
            return "timeout"
        if status_code is not None and 500 <= status_code <= 599:
            return "server_error"
        return "status"


class HTTPResponseResult(GenericResult[Any]):
    status_code: int
    headers: Dict[str, str] = Field(default_factory=dict)
    url: str = ""
    attempts: int = 1
    pages: int = 1
    rate_limit: Dict[str, str] = Field(default_factory=dict)
    retry_events: List[Dict[str, Any]] = Field(default_factory=list)
    retry_summary: Dict[str, int] = Field(default_factory=dict)


class HTTPResult(HTTPResponseResult): ...


class HTTPModel(CallableModel):
    config: Optional[HTTPConfig] = None
    auth: HTTPAuth = Field(default_factory=HTTPAuth)
    method: HTTPMethod = "GET"
    base_url: str = ""
    path: str = ""
    query: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0
    follow_redirects: bool = True
    response_format: ResponseFormat = "json"
    json_body: Optional[Any] = None
    content: Optional[Union[bytes, str]] = None
    max_attempts: int = 1
    retry_status_codes: List[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_policy: Optional[HTTPRetryPolicy] = None
    execution_policy: Optional[ExecutionPolicy] = None
    paginate: bool = False
    max_pages: int = 100
    pagination_mode: HTTPPaginationMode = "next_url"
    next_url_field: str = "next_url"
    next_cursor_field: str = "next_cursor"
    cursor_param: str = "cursor"
    cursor_start: Optional[str] = None
    page_param: str = "page"
    page_start: int = 1
    offset_param: str = "offset"
    offset_start: int = 0
    limit_param: str = "limit"
    limit: Optional[int] = None
    results_field: str = "results"

    @property
    def context_type(self):
        return HTTPRequestContext

    @property
    def result_type(self):
        return HTTPResult

    def _template_data(self, context: HTTPContext) -> Dict[str, Any]:
        data = context.model_dump(exclude_none=True)
        data.update(context.template_values)
        return data

    def _render(self, value: Any, data: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            return Environment().from_string(value).render(**data)
        return value

    def _render_mapping(self, values: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._render(value, data) for key, value in values.items() if value is not None}

    def _render_required_auth_value(self, value: Optional[str], field_name: str, data: Dict[str, Any]) -> str:
        if value is None:
            raise ValueError(f"HTTP auth strategy {self.auth.strategy!r} requires {field_name}.")
        return str(self._render(value, data))

    def _apply_auth(self, headers: Dict[str, str], query: Dict[str, Any], data: Dict[str, Any]) -> None:
        match self.auth.strategy:
            case "none":
                return
            case "bearer":
                token = self._render_required_auth_value(self.auth.token, "token", data)
                headers["Authorization"] = f"{self.auth.scheme} {token}"
            case "api_key_header":
                name = self._render_required_auth_value(self.auth.name, "name", data)
                value = self._render_required_auth_value(self.auth.value, "value", data)
                headers[name] = value
            case "api_key_query":
                name = self._render_required_auth_value(self.auth.name, "name", data)
                value = self._render_required_auth_value(self.auth.value, "value", data)
                query[name] = value
            case "basic":
                username = self._render_required_auth_value(self.auth.username, "username", data)
                password = self._render_required_auth_value(self.auth.password, "password", data)
                encoded = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
                headers["Authorization"] = f"Basic {encoded}"
            case _:
                raise ValueError(f"Unsupported HTTP auth strategy: {self.auth.strategy}")

    def _base_url(self) -> str:
        return self.base_url or (self.config.base_url if self.config else "")

    def _timeout(self) -> float:
        if self.config and self.timeout == 30.0:
            return self.config.timeout
        return self.timeout

    def _follow_redirects(self) -> bool:
        if self.config and self.follow_redirects is True:
            return self.config.follow_redirects
        return self.follow_redirects

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "base_url": self._base_url(),
            "timeout": self._timeout(),
            "follow_redirects": self._follow_redirects(),
        }
        if self.config and self.config.transport is not None:
            kwargs["transport"] = self.config.transport
        return kwargs

    def build_request(self, context: Optional[HTTPContext] = None) -> HTTPRequest:
        context = context or HTTPRequestContext()
        data = self._template_data(context)

        path = context.path or self.path
        query = {**self.query, **context.query}
        config_headers = self.config.headers if self.config else {}
        headers = {**config_headers, **self.headers, **context.headers}

        rendered_query = self._render_mapping(query, data)
        rendered_headers = self._render_mapping(headers, data)
        self._apply_auth(rendered_headers, rendered_query, data)

        return HTTPRequest(
            method=self.method,
            url=self._render(path, data),
            params=rendered_query,
            headers=rendered_headers,
            json_data=context.json_body if context.json_body is not None else self.json_body,
            content=context.content if context.content is not None else self.content,
        )

    def _response_value(self, response: httpx.Response) -> Any:
        match self.response_format:
            case "json":
                return response.json()
            case "text":
                return response.text
            case "bytes":
                return response.content
            case "csv":
                return list(DictReader(StringIO(response.text)))
            case "gzip":
                return decompress(response.content)
            case _:
                raise ValueError(f"Unsupported response format: {self.response_format}")

    def _safe_url(self, request: HTTPRequest) -> str:
        return request.url.split("?", 1)[0]

    def _rate_limit_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        rate_limit = {}
        for key, value in headers.items():
            normalized_key = key.lower()
            if "ratelimit" in normalized_key or normalized_key == "retry-after" or "rate-limit" in normalized_key:
                rate_limit[normalized_key] = value
        return rate_limit

    def _retry_policy(self) -> HTTPRetryPolicy:
        return self.retry_policy or HTTPRetryPolicy(max_attempts=self.max_attempts, retry_status_codes=self.retry_status_codes)

    def _sleep(self, delay_seconds: float) -> None:
        if delay_seconds > 0:
            sleep(delay_seconds)

    def _now(self) -> float:
        return monotonic()

    def _throttle(self, previous_started_at: Optional[float]) -> float:
        now = self._now()
        if self.execution_policy is None:
            return now
        delay_seconds = self.execution_policy.rate_delay_seconds(previous_started_at=previous_started_at, now=now)
        if delay_seconds > 0:
            self._sleep(delay_seconds)
        return now + delay_seconds

    def _retry_event(self, **values: Any) -> Dict[str, Any]:
        return HTTPRetryEvent(**values).model_dump(exclude={"type_"}, exclude_none=True)

    def _retry_summary(self, events: List[Dict[str, Any]], attempts: int, succeeded: bool) -> Dict[str, int]:
        return {
            "attempts": attempts,
            "retried": sum(1 for event in events if event["outcome"] == "retry"),
            "failed": sum(1 for event in events if event["outcome"] == "failed"),
            "succeeded": 1 if succeeded else 0,
        }

    def _extract_field(self, value: Any, field: str) -> Any:
        current = value
        for part in field.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _request_once(
        self, client: httpx.Client, request: HTTPRequest, previous_started_at: Optional[float]
    ) -> Tuple[httpx.Response, int, List[Dict[str, Any]], float]:
        attempts = 0
        events: List[Dict[str, Any]] = []
        retry_policy = self._retry_policy()
        total_retry_wait_seconds = 0.0
        while True:
            attempts += 1
            previous_started_at = self._throttle(previous_started_at)
            try:
                response = client.request(
                    method=request.method,
                    url=request.url,
                    params=request.params,
                    headers=request.headers,
                    json=request.json_data,
                    content=request.content,
                )
                response.raise_for_status()
                return response, attempts, events, previous_started_at
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if retry_policy.should_retry_status(status_code, attempts):
                    delay_seconds = retry_policy.retry_delay_seconds(attempts, total_wait_seconds=total_retry_wait_seconds)
                    if delay_seconds is not None:
                        events.append(
                            self._retry_event(
                                attempt=attempts,
                                outcome="retry",
                                delay_seconds=delay_seconds,
                                status_code=status_code,
                                category=retry_policy.status_category(status_code),
                                message=f"retryable status code {status_code}",
                            )
                        )
                        self._sleep(delay_seconds)
                        total_retry_wait_seconds += delay_seconds
                        continue
                events.append(
                    self._retry_event(
                        attempt=attempts,
                        outcome="failed",
                        status_code=status_code,
                        category=retry_policy.status_category(status_code),
                        message=f"retryable status code {status_code}",
                    )
                )
                status_label = status_code if status_code is not None else "unknown"
                raise RuntimeError(f"HTTP {request.method} {self._safe_url(request)} failed with status {status_label}") from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if retry_policy.should_retry_exception(exc, attempts):
                    delay_seconds = retry_policy.retry_delay_seconds(attempts, total_wait_seconds=total_retry_wait_seconds)
                    if delay_seconds is not None:
                        events.append(
                            self._retry_event(
                                attempt=attempts,
                                outcome="retry",
                                delay_seconds=delay_seconds,
                                exception_type=type(exc).__name__,
                                category=retry_policy.exception_category(exc),
                                message=str(exc),
                            )
                        )
                        self._sleep(delay_seconds)
                        total_retry_wait_seconds += delay_seconds
                        continue
                events.append(
                    self._retry_event(
                        attempt=attempts,
                        outcome="failed",
                        exception_type=type(exc).__name__,
                        category=retry_policy.exception_category(exc),
                        message=str(exc),
                    )
                )
                raise RuntimeError(f"HTTP {request.method} {self._safe_url(request)} failed with {type(exc).__name__}") from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"HTTP {request.method} {self._safe_url(request)} failed with {type(exc).__name__}") from exc

    def _merge_page_values(self, values: List[Any]) -> Any:
        if not values:
            return []
        if all(isinstance(value, dict) and isinstance(value.get(self.results_field), list) for value in values):
            merged = dict(values[-1])
            merged[self.results_field] = [item for value in values for item in value[self.results_field]]
            merged.pop(self.next_url_field, None)
            merged.pop(self.next_cursor_field, None)
            return merged
        return values

    def _page_items(self, value: Any) -> Optional[List[Any]]:
        items = self._extract_field(value, self.results_field)
        return items if isinstance(items, list) else None

    def _request_with_params(self, request: HTTPRequest, params: Dict[str, Any]) -> HTTPRequest:
        return request.model_copy(update={"params": {**request.params, **params}})

    def _is_sensitive_query_param(self, key: str) -> bool:
        normalized_key = key.lower().replace("_", "").replace("-", "")
        return normalized_key in {"apikey", "authorization", "password"} or "token" in normalized_key or "secret" in normalized_key

    def _next_url_request(self, request: HTTPRequest, next_url: str) -> HTTPRequest:
        next_url_parts = urlsplit(next_url)
        next_url_params = dict(parse_qsl(next_url_parts.query, keep_blank_values=True))
        for key, value in request.params.items():
            if key not in next_url_params and self._is_sensitive_query_param(key):
                next_url_params[key] = value
        next_url_without_query = urlunsplit((next_url_parts.scheme, next_url_parts.netloc, next_url_parts.path, "", next_url_parts.fragment))
        return request.model_copy(update={"url": next_url_without_query, "params": next_url_params})

    def _initial_paginated_request(self, request: HTTPRequest) -> HTTPRequest:
        if not self.paginate:
            return request
        match self.pagination_mode:
            case "next_url":
                return request
            case "cursor":
                return self._request_with_params(request, {self.cursor_param: self.cursor_start}) if self.cursor_start is not None else request
            case "page":
                if self.page_param in request.params:
                    return request
                return self._request_with_params(request, {self.page_param: self.page_start})
            case "offset":
                params = dict(request.params)
                params.setdefault(self.offset_param, self.offset_start)
                if self.limit is not None:
                    params.setdefault(self.limit_param, self.limit)
                if self.limit_param not in params:
                    raise ValueError("Offset pagination requires a limit or existing limit parameter.")
                return request.model_copy(update={"params": params})
            case _:
                raise ValueError(f"Unsupported pagination mode: {self.pagination_mode}")

    def _next_paginated_request(self, request: HTTPRequest, value: Any) -> Optional[HTTPRequest]:
        if self._page_items(value) == []:
            return None
        match self.pagination_mode:
            case "next_url":
                next_url = self._extract_field(value, self.next_url_field) if isinstance(value, dict) else None
                return self._next_url_request(request, next_url) if next_url else None
            case "cursor":
                next_cursor = self._extract_field(value, self.next_cursor_field)
                return self._request_with_params(request, {self.cursor_param: next_cursor}) if next_cursor else None
            case "page":
                next_page = int(request.params.get(self.page_param, self.page_start)) + 1
                return self._request_with_params(request, {self.page_param: next_page})
            case "offset":
                limit = int(request.params[self.limit_param])
                next_offset = int(request.params.get(self.offset_param, self.offset_start)) + limit
                return self._request_with_params(request, {self.offset_param: next_offset, self.limit_param: limit})
            case _:
                raise ValueError(f"Unsupported pagination mode: {self.pagination_mode}")

    @Flow.call
    def __call__(self, context: HTTPRequestContext) -> HTTPResult:
        request = self._initial_paginated_request(self.build_request(context))

        with httpx.Client(**self._client_kwargs()) as client:
            values = []
            retry_events = []
            total_attempts = 0
            pages = 0
            previous_started_at = None
            while True:
                response, attempts, events, previous_started_at = self._request_once(client, request, previous_started_at)
                total_attempts += attempts
                retry_events.extend(events)
                pages += 1
                value = self._response_value(response)
                values.append(value)

                if not self.paginate or pages >= self.max_pages:
                    break
                next_request = self._next_paginated_request(request, value)
                if next_request is None:
                    break
                request = next_request

            return HTTPResult(
                value=self._merge_page_values(values) if self.paginate else values[-1],
                status_code=response.status_code,
                headers=dict(response.headers or {}),
                url=str(response.url),
                attempts=total_attempts,
                pages=pages,
                rate_limit=self._rate_limit_headers(dict(response.headers or {})),
                retry_events=retry_events,
                retry_summary=self._retry_summary(retry_events, attempts=total_attempts, succeeded=True),
            )
