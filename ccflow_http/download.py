import json
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx2
from ccflow import CallableModel, DateContext, Flow, ResultBase
from ccflow_etl import ArtifactWriteFileContext, ArtifactWriteFileModel, ArtifactWriteFileResult
from pydantic import Field

from .base import _REQUEST_SENSITIVE_PARAMS, HTTPModel, HTTPRequest, HTTPRequestContext, redact_url

__all__ = (
    "HTTPArtifactCaptureModel",
    "HTTPArtifactCaptureResult",
    "HTTPDownloadContext",
    "HTTPDownloadModel",
    "HTTPDownloadResult",
)


class HTTPDownloadContext(HTTPRequestContext):
    output_path: Path


class HTTPDownloadResult(ResultBase):
    path: str
    size: int
    sha256: str
    status_code: int
    url: str
    etag: str | None = None
    last_modified: str | None = None
    content_type: str | None = None
    attempts: int = 1
    retry_events: list[dict[str, Any]] = Field(default_factory=list)


class HTTPDownloadModel(HTTPModel):
    """Stream a response body to a local file without holding it in memory.

    The file is written to a temporary path beside the target and renamed into place only after the full body
    has been received, so a failed download never leaves a partial file at the target path.
    """

    chunk_size: int = Field(default=1 << 20, ge=1)

    @property
    def context_type(self) -> type[HTTPDownloadContext]:
        return HTTPDownloadContext

    @property
    def result_type(self) -> type[HTTPDownloadResult]:
        return HTTPDownloadResult

    def _send(self, client: httpx2.Client, request: HTTPRequest) -> httpx2.Response:
        built = client.build_request(
            method=request.method,
            url=request.url,
            params=request.params,
            headers=request.headers,
            json=request.json_data,
            content=request.content,
        )
        return client.send(built, stream=True)

    @Flow.call
    def __call__(self, context: HTTPDownloadContext) -> HTTPDownloadResult:
        request = self.build_request(context)
        sensitive_params = request._sensitive_params
        path = context.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        digest = sha256()
        size = 0
        token = _REQUEST_SENSITIVE_PARAMS.set(sensitive_params)
        try:
            with httpx2.Client(**self._client_kwargs()) as client:
                response, attempts, events, _ = self._request_once(client, request, None)
                try:
                    with temp_path.open("wb") as file:
                        for chunk in response.iter_bytes(self.chunk_size):
                            file.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                finally:
                    response.close()
            temp_path.replace(path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            _REQUEST_SENSITIVE_PARAMS.reset(token)
        return HTTPDownloadResult(
            path=str(path),
            size=size,
            sha256=digest.hexdigest(),
            status_code=response.status_code,
            url=redact_url(str(response.url), sensitive_params),
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            content_type=response.headers.get("content-type"),
            attempts=attempts,
            retry_events=events,
        )


class HTTPArtifactCaptureResult(ResultBase):
    date: date
    key: str
    manifest_key: str
    status: str
    download: HTTPDownloadResult | None = None
    write: ArtifactWriteFileResult | None = None
    manifest_write: ArtifactWriteFileResult | None = None


class HTTPArtifactCaptureModel(CallableModel):
    """Capture one HTTP resource for a date: download it, store it under a dated key, and store a manifest beside it.

    The manifest records the redacted URL, capture time, status, response headers, size, and SHA-256 of the stored
    payload. An existing payload is skipped before downloading unless ``overwrite`` is set.
    """

    download: HTTPDownloadModel
    writer: ArtifactWriteFileModel
    key_template: str = Field(description="Payload key with {date}, {year}, {month}, and {day} placeholders.")
    workspace: Path = Path("cc-etl-workspace")
    dataset: str | None = None
    media_type: str | None = None
    overwrite: bool = False
    keep_local: bool = Field(default=False, description="Keep the downloaded file in the workspace after it is stored.")
    explain: bool = False

    @property
    def context_type(self) -> type[DateContext]:
        return DateContext

    @property
    def result_type(self) -> type[HTTPArtifactCaptureResult]:
        return HTTPArtifactCaptureResult

    def key(self, context: DateContext) -> str:
        value = context.date
        return self.key_template.format(date=value.isoformat(), year=f"{value.year:04d}", month=f"{value.month:02d}", day=f"{value.day:02d}")

    def _write(self, key: str, path: Path, media_type: str | None, dry_run: bool, metadata: dict[str, Any]) -> ArtifactWriteFileResult:
        return self.writer(
            ArtifactWriteFileContext(
                key=key,
                path=path,
                media_type=media_type,
                dataset=self.dataset,
                stage="extract",
                overwrite=self.overwrite,
                dry_run=dry_run,
                metadata=metadata,
            )
        )

    def _exists(self, key: str) -> bool:
        exists = getattr(self.writer.store, "exists", None)
        return bool(exists and exists(key))

    @Flow.call
    def __call__(self, context: DateContext) -> HTTPArtifactCaptureResult:
        key = self.key(context)
        manifest_key = f"{key}.manifest.json"
        local_path = self.workspace / key
        manifest_path = self.workspace / manifest_key
        metadata = {"capture_date": context.date.isoformat()}
        if self.explain:
            return HTTPArtifactCaptureResult(
                date=context.date,
                key=key,
                manifest_key=manifest_key,
                status="planned",
                write=self._write(key, local_path, self.media_type, True, metadata),
                manifest_write=self._write(manifest_key, manifest_path, "application/json", True, metadata),
            )
        if not self.overwrite and self._exists(key):
            return HTTPArtifactCaptureResult(date=context.date, key=key, manifest_key=manifest_key, status="exists")

        download = self.download(HTTPDownloadContext(output_path=local_path))
        manifest = {
            "key": key,
            "capture_date": context.date.isoformat(),
            "captured_at": datetime.now(UTC).isoformat(),
            **download.model_dump(mode="json", exclude={"type_", "path", "retry_events"}),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            write = self._write(key, local_path, self.media_type or download.content_type, False, {**metadata, "sha256": download.sha256})
            manifest_write = self._write(manifest_key, manifest_path, "application/json", False, metadata)
        finally:
            if not self.keep_local:
                local_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
        status = next((value for value in (write.status, manifest_write.status) if value != "exists"), "exists")
        return HTTPArtifactCaptureResult(
            date=context.date, key=key, manifest_key=manifest_key, status=status, download=download, write=write, manifest_write=manifest_write
        )
