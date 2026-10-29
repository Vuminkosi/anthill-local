"""MCP stdio tools for dev tools that connect to the local AntHill API."""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from local_client import (
    check_readiness as _check_readiness,
    generate_video as _generate_video,
    get_video_job as _get_video_job,
    list_video_jobs as _list_video_jobs,
    local_service_url,
    retry_video_job as _retry_video_job,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
server = FastMCP("AntHill Local Video Factory", json_response=True)


@server.tool()
def generate_video(story: str, duration_seconds: int = 30) -> dict:
    """Queue a 30, 45, or 60 second local motion-comic video. Poll with get_video_job."""
    return {"job": _generate_video(story, duration_seconds)}


@server.tool()
def get_video_job(job_id: str) -> dict:
    """Read a job's queue state, progress, failure, or completed local video URL."""
    job = _get_video_job(job_id)
    if job.get("video"):
        job["video_url"] = local_service_url(job["video"])
    return {"job": job}


@server.tool()
def list_video_jobs() -> dict:
    """List the most recent local AntHill video jobs."""
    return _list_video_jobs()


@server.tool()
def retry_video_job(job_id: str) -> dict:
    """Requeue a failed local video job, preserving its story and duration."""
    return {"job": _retry_video_job(job_id)}


@server.tool()
def check_readiness() -> dict:
    """Check that the approved local image, speech, and storyboard models are installed."""
    return {"readiness": _check_readiness()}


if __name__ == "__main__":
    server.run(transport="stdio")
