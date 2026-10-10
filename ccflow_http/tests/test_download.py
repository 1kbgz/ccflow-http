import json
from datetime import date
from hashlib import sha256

import httpx2
import pytest
from ccflow_etl import ArtifactWriteFileModel, LocalFileOutput

from ccflow_http import HTTPArtifactCaptureModel, HTTPAuth, HTTPConfig, HTTPDownloadContext, HTTPDownloadModel, HTTPRetryPolicy

PAYLOAD = b"x" * 5000


def _download_model(handler, **kwargs):
    return HTTPDownloadModel(
        config=HTTPConfig(base_url="https://files.example.test", transport=httpx2.MockTransport(handler)),
        path="/bulk/data.zip",
        chunk_size=1024,
        **kwargs,
    )


def test_http_download_streams_body_to_file_with_checksum_and_headers(tmp_path):
    def handler(request):
        return httpx2.Response(
            200, content=PAYLOAD, headers={"ETag": '"abc"', "Last-Modified": "Fri, 09 Oct 2026 04:35:00 GMT", "Content-Type": "application/zip"}
        )

    target = tmp_path / "out" / "data.zip"
    result = _download_model(handler, auth=HTTPAuth(strategy="api_key_query", name="key", value="secret"))(HTTPDownloadContext(output_path=target))

    assert target.read_bytes() == PAYLOAD
    assert result.size == len(PAYLOAD)
    assert result.sha256 == sha256(PAYLOAD).hexdigest()
    assert (result.etag, result.last_modified, result.content_type) == ('"abc"', "Fri, 09 Oct 2026 04:35:00 GMT", "application/zip")
    assert result.url == "https://files.example.test/bulk/data.zip?key=***"
    assert not list(target.parent.glob(".*.tmp"))


def test_http_download_retries_status_and_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("ccflow_http.base.sleep", lambda seconds: None)
    calls = []

    def flaky(request):
        calls.append(request)
        return httpx2.Response(503) if len(calls) == 1 else httpx2.Response(200, content=PAYLOAD)

    policy = HTTPRetryPolicy(max_attempts=2, wait_initial=0.0, wait_max=0.0, wait_jitter=0.0)
    result = _download_model(flaky, retry_policy=policy)(HTTPDownloadContext(output_path=tmp_path / "data.zip"))
    assert result.attempts == 2

    target = tmp_path / "failed.zip"
    with pytest.raises(RuntimeError, match="failed with status 404"):
        _download_model(lambda request: httpx2.Response(404))(HTTPDownloadContext(output_path=target))
    assert not target.exists()
    assert not list(tmp_path.glob(".failed.zip.*.tmp"))


def _capture(tmp_path, handler, **kwargs):
    store = LocalFileOutput(path=tmp_path / "store")
    model = HTTPArtifactCaptureModel(
        download=_download_model(handler),
        writer=ArtifactWriteFileModel(store=store),
        key_template="bulk/data/{year}/{month}/{date}.zip",
        workspace=tmp_path / "workspace",
        dataset="bulk-data",
        **kwargs,
    )
    return model, store


def test_http_capture_stores_payload_and_manifest_and_cleans_workspace(tmp_path):
    model, store = _capture(tmp_path, lambda request: httpx2.Response(200, content=PAYLOAD, headers={"ETag": '"abc"'}))

    result = model(date(2026, 10, 11))
    manifest = json.loads(store.file_path("bulk/data/2026/10/2026-10-11.zip.manifest.json").read_text())

    assert result.status == "written"
    assert store.file_path("bulk/data/2026/10/2026-10-11.zip").read_bytes() == PAYLOAD
    assert manifest["sha256"] == sha256(PAYLOAD).hexdigest()
    assert manifest["size"] == len(PAYLOAD)
    assert manifest["etag"] == '"abc"'
    assert manifest["capture_date"] == "2026-10-11"
    assert manifest["url"] == "https://files.example.test/bulk/data.zip"
    assert not [path for path in (tmp_path / "workspace").rglob("*") if path.is_file()]


def test_http_capture_skips_existing_payload_without_downloading(tmp_path):
    calls = []
    model, _ = _capture(tmp_path, lambda request: calls.append(request) or httpx2.Response(200, content=PAYLOAD))
    model(date(2026, 10, 11))

    result = model(date(2026, 10, 11))

    assert result.status == "exists"
    assert len(calls) == 1


def test_http_capture_explain_does_no_network_or_storage_io(tmp_path):
    def handler(request):
        raise AssertionError("explain made a request")

    model, _ = _capture(tmp_path, handler, explain=True)

    result = model(date(2026, 10, 11))

    assert result.status == "planned"
    assert result.write.status == "planned"
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "workspace").exists()
