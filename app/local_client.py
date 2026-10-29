"""Small, local-only HTTP client shared by the CLI and MCP server."""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "anthill"}
_DURATIONS = {30, 45, 60}


class LocalApiError(RuntimeError):
    """An API error that is safe to display without exposing request data."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _base_url() -> str:
    value = os.getenv("ANTHILL_URL", "http://127.0.0.1:8090").rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise LocalApiError("ANTHILL_URL must point to the local AntHill service.") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _ALLOWED_HOSTS
        or (port is not None and not 1 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise LocalApiError("ANTHILL_URL must use HTTP and a local AntHill host (localhost, 127.0.0.1, ::1, or anthill).")
    return value


def _request(path: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 15):
    if not path.startswith("/") or ".." in path.split("/"):
        raise LocalApiError("Invalid local API path.")
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(local_service_url(path), data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get_content_type()
            return body, content_type
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode("utf-8"))
            message = str(error.get("error", "The local AntHill service returned an error."))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            message = "The local AntHill service returned an error."
        raise LocalApiError(f"HTTP {exc.code}: {message}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LocalApiError("Could not reach the local AntHill service. Start it with docker compose up -d.") from None


def request_json(path: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 15) -> dict:
    body, content_type = _request(path, method=method, payload=payload, timeout=timeout)
    if content_type != "application/json":
        raise LocalApiError("The local AntHill service returned an unexpected response.")
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LocalApiError("The local AntHill service returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise LocalApiError("The local AntHill service returned an unexpected response.")
    return result


def check_readiness() -> dict:
    return request_json("/api/readiness", timeout=5)


def local_service_url(path: str) -> str:
    if not path.startswith("/") or ".." in path.split("/"):
        raise LocalApiError("Invalid local API path.")
    return _base_url() + path


def generate_video(story: str, duration_seconds: int = 30) -> dict:
    if not isinstance(story, str) or not 25 <= len(story.strip()) <= 6000:
        raise LocalApiError("Story must contain between 25 and 6,000 characters.")
    if isinstance(duration_seconds, bool) or duration_seconds not in _DURATIONS:
        raise LocalApiError("Duration must be 30, 45, or 60 seconds.")
    return request_json(
        "/api/jobs", method="POST", payload={"story": story.strip(), "duration_seconds": duration_seconds}
    )


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise LocalApiError("Job ID must be 12 lowercase hexadecimal characters.")
    return job_id


def get_video_job(job_id: str) -> dict:
    return request_json(f"/api/jobs/{_validate_job_id(job_id)}")


def list_video_jobs() -> dict:
    return request_json("/api/jobs")


def retry_video_job(job_id: str) -> dict:
    return request_json(f"/api/jobs/{_validate_job_id(job_id)}/retry", method="POST", payload={})


def download_video(job: dict, destination: Path) -> Path:
    """Download a completed local MP4 with path and same-origin checks."""
    video = job.get("video")
    if not isinstance(video, str):
        raise LocalApiError("This job does not have a finished video yet.")
    parsed = urllib.parse.urlsplit(video)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/videos/AntHill_"):
        raise LocalApiError("The video link was not a local AntHill download.")
    expected = f"/videos/AntHill_{_validate_job_id(str(job.get('id', '')))}.mp4"
    if parsed.path != expected or parsed.query or parsed.fragment:
        raise LocalApiError("The video link did not match this local job.")
    target = destination.expanduser().resolve()
    if target.exists() and target.is_dir():
        raise LocalApiError("Choose a file path for the video download.")
    if target.exists():
        raise LocalApiError("The selected output file already exists; choose a new path to preserve it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(local_service_url(parsed.path), headers={"Accept": "video/mp4"})
    try:
        with opener.open(request, timeout=300) as response:
            if response.headers.get_content_type() != "video/mp4":
                raise LocalApiError("The local service did not return an MP4 video.")
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=target.parent,
                prefix=f".{target.name}.", suffix=".partial",
            ) as output:
                temporary = Path(output.name)
                header = response.read(12)
                if len(header) < 12 or header[4:8] != b"ftyp":
                    raise LocalApiError("The local service did not return a valid MP4 video.")
                output.write(header)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if temporary.stat().st_size < 12:
            raise LocalApiError("The local service returned an empty video.")
        try:
            # Same-directory hard-link creation is atomic and fails when a target
            # already exists, avoiding a check-then-replace overwrite race.
            os.link(temporary, target)
        except FileExistsError:
            raise LocalApiError("The selected output file already exists; choose a new path to preserve it.") from None
    except urllib.error.HTTPError as exc:
        raise LocalApiError(f"Could not download the local video (HTTP {exc.code}).") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise LocalApiError("Could not download the video from the local AntHill service.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
