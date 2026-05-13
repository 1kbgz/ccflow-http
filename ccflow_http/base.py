from base64 import b64encode
from csv import DictReader
from gzip import decompress
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult
from ccflow_etl import RetryPolicy
from jinja2 import Environment
from pydantic import Field

__all__ = (
    "HTTPConfig",
    "HTTPAuth",
    "HTTPContext",
    "HTTPRequestContext",
    "HTTPRequest",
    "HTTPResponseResult",
    "HTTPResult",
    "HTTPModel",
)

ResponseFormat = Literal["json", "text", "bytes", "csv", "gzip"]
HTTPMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
HTTPAuthStrategy = Literal["none", "bearer", "api_key_header", "api_key_query", "basic"]
HTTPPaginationMode = Literal["next_url", "cursor", "page", "offset"]


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


class HTTPResponseResult(GenericResult[Any]):
    status_code: int
    headers: Dict[str, str] = Field(default_factory=dict)
    url: str = ""
    attempts: int = 1
    pages: int = 1
    rate_limit: Dict[str, str] = Field(default_factory=dict)


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
    retry_policy: Optional[RetryPolicy] = None
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

    def _retry_policy(self) -> RetryPolicy:
        return self.retry_policy or RetryPolicy(max_attempts=self.max_attempts, retry_status_codes=self.retry_status_codes)

    def _extract_field(self, value: Any, field: str) -> Any:
        current = value
        for part in field.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _request_once(self, client: httpx.Client, request: HTTPRequest) -> Tuple[httpx.Response, int]:
        attempts = 0
        retry_policy = self._retry_policy()
        while True:
            attempts += 1
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
                return response, attempts
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if retry_policy.should_retry_status(status_code, attempts):
                    continue
                status_label = status_code if status_code is not None else "unknown"
                raise RuntimeError(f"HTTP {request.method} {self._safe_url(request)} failed with status {status_label}") from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if retry_policy.should_retry_exception(exc, attempts):
                    continue
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
                return request.model_copy(update={"url": next_url, "params": {}}) if next_url else None
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
            total_attempts = 0
            pages = 0
            while True:
                response, attempts = self._request_once(client, request)
                total_attempts += attempts
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
            )
