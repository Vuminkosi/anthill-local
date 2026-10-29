"""HTTP interface and durable, single-worker queue for AntHill jobs."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from model_setup import IMAGE_FILES, IMAGE_PATH, VOICE_FILES, VOICE_PATH
from pipeline import ROOT, local_ollama_base, run_job

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100_000
app.config["JOB_DATABASE"] = Path(os.getenv("JOB_DATABASE", str(ROOT / "state" / "jobs.sqlite3")))
app.config["MAX_PENDING_JOBS"] = int(os.getenv("MAX_PENDING_JOBS", "10"))

logger = logging.getLogger("anthill.server")
_worker_lock = threading.Lock()
_worker_started = False
_valid_job_id = re.compile(r"^[a-f0-9]{12}$")
_valid_video_name = re.compile(r"^AntHill_[a-f0-9]{12}\.mp4$")
_durations = {30, 45, 60}
_public_error = "Video generation failed. Check the application logs, then retry this story."


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not allow a local readiness request to follow an unexpected redirect."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _connect() -> sqlite3.Connection:
    database = Path(app.config["JOB_DATABASE"])
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _initialize_database() -> None:
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                story TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                status TEXT NOT NULL,
                progress TEXT NOT NULL,
                percent INTEGER NOT NULL,
                video TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        # Jobs interrupted by an application restart are safe to run again:
        # every artifact is scoped to its job ID and the final file is atomic.
        connection.execute(
            "UPDATE jobs SET status='queued', progress='Resuming after application restart', "
            "percent=0, updated_at=? WHERE status='working'",
            (time.time(),),
        )


def _public_job(row: sqlite3.Row) -> dict:
    result = {
        "id": row["id"],
        "status": row["status"],
        "progress": row["progress"],
        "percent": row["percent"],
        "duration_seconds": row["duration_seconds"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if row["status"] == "done" and row["video"]:
        result["video"] = f"/videos/{row['video']}"
    if row["status"] == "error":
        result["error"] = _public_error
    return result


def _local_model_status() -> tuple[bool, dict]:
    """Inspect selected model files and the configured local Ollama service only."""
    image_model = os.getenv("IMAGE_MODEL", "Lykon/dreamshaper-8")
    image_path = IMAGE_PATH
    image_ready = (
        image_model == "Lykon/dreamshaper-8"
        and all((image_path / filename).is_file() for filename in IMAGE_FILES)
    )
    voice_name = os.getenv("KOKORO_VOICE", "af_heart")
    voice_ready = all((VOICE_PATH / filename).is_file() for filename in VOICE_FILES)
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    storyboard_ready = False
    try:
        ollama_url = local_ollama_base()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )
        with opener.open(ollama_url + "/api/tags", timeout=2) as response:
            import json

            available = json.loads(response.read().decode("utf-8")).get("models", [])
            storyboard_ready = any(
                model.get("name") == ollama_model or model.get("model") == ollama_model
                for model in available
            )
    except (OSError, RuntimeError, ValueError, urllib.error.URLError):
        logger.info("Local Ollama model readiness check is unavailable")
    return image_ready and voice_ready and storyboard_ready, {
        "image": {"name": image_model, "installed": image_ready},
        "narration": {"name": f"Kokoro 82M ({voice_name})", "installed": voice_ready},
        "storyboard": {"name": ollama_model, "installed": storyboard_ready},
    }


def _get_job(job_id: str):
    with _connect() as connection:
        return connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def _claim_next_job():
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at, id LIMIT 1"
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        now = time.time()
        updated = connection.execute(
            "UPDATE jobs SET status='working', progress='Starting local generation', percent=1, updated_at=? "
            "WHERE id=? AND status='queued'",
            (now, row["id"]),
        )
        connection.commit()
        return row["id"] if updated.rowcount == 1 else None


def _update_progress(job_id: str, message: str, percent: int | None = None) -> None:
    safe_message = str(message).strip()[:240] or "Working"
    bounded_percent = max(0, min(99, int(percent))) if percent is not None else None
    with _connect() as connection:
        if bounded_percent is None:
            connection.execute(
                "UPDATE jobs SET progress=?, updated_at=? WHERE id=? AND status='working'",
                (safe_message, time.time(), job_id),
            )
        else:
            connection.execute(
                "UPDATE jobs SET progress=?, percent=?, updated_at=? WHERE id=? AND status='working'",
                (safe_message, bounded_percent, time.time(), job_id),
            )


def _run_claimed_job(job_id: str) -> None:
    row = _get_job(job_id)
    if row is None:
        return
    try:
        output = run_job(
            job_id,
            row["story"],
            lambda message, percent=None: _update_progress(job_id, message, percent),
            duration_seconds=row["duration_seconds"],
        )
        expected_name = f"AntHill_{job_id}.mp4"
        output_path = Path(output).resolve()
        output_root = (ROOT / "output").resolve()
        if output_path.parent != output_root or output_path.name != expected_name or not output_path.is_file():
            raise RuntimeError("Video pipeline returned an unexpected output path")
        with _connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='done', progress='Video ready', percent=100, video=?, error=NULL, updated_at=? "
                "WHERE id=? AND status='working'",
                (expected_name, time.time(), job_id),
            )
        logger.info("Video job %s completed", job_id)
    except Exception as exc:
        # Third-party failures can echo prompts; record the failure category without exception text.
        logger.error("Video job %s failed (%s)", job_id, type(exc).__name__)
        with _connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='error', progress='Generation failed', error=?, updated_at=? WHERE id=?",
                (_public_error, time.time(), job_id),
            )


def _worker_loop() -> None:
    while True:
        try:
            job_id = _claim_next_job()
            if job_id is None:
                time.sleep(1)
                continue
            _run_claimed_job(job_id)
        except Exception:
            logger.exception("Background job worker encountered a database error")
            time.sleep(2)


def start_worker() -> None:
    """Start one GPU queue worker; called once by the single-process app entry point."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _initialize_database()
        thread = threading.Thread(target=_worker_loop, name="anthill-job-worker", daemon=True)
        thread.start()
        _worker_started = True


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    try:
        with _connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return jsonify(status="ok")
    except sqlite3.Error:
        logger.exception("Health check could not reach the job store")
        return jsonify(status="unavailable"), 503


@app.get("/api/readiness")
def readiness():
    """Report approved local model availability without downloading or initializing models."""
    ready, models = _local_model_status()
    return jsonify(
        status="ready" if ready else "models-needed",
        models=models,
    )


@app.post("/api/jobs")
def start_job():
    if not request.is_json:
        return jsonify(error="Send the story as JSON."), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="The request must contain a story."), 400
    story = payload.get("story")
    if not isinstance(story, str):
        return jsonify(error="Enter a story as text."), 400
    story = story.strip()
    if len(story) < 25:
        return jsonify(error="Please enter at least 25 characters."), 400
    if len(story) > 6000:
        return jsonify(error="Stories must be 6,000 characters or fewer."), 400
    requested_duration = payload.get("duration_seconds", 30)
    if isinstance(requested_duration, bool) or not isinstance(requested_duration, int) or requested_duration not in _durations:
        return jsonify(error="Choose a 30, 45, or 60-second video."), 400
    models_ready, models = _local_model_status()
    if not models_ready:
        return jsonify(error="Local models are still being installed or are unavailable. Check readiness and retry.", models=models), 503

    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        pending = connection.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','working')").fetchone()[0]
        if pending >= app.config["MAX_PENDING_JOBS"]:
            connection.rollback()
            return jsonify(error="The video queue is full. Please try again shortly."), 503
        connection.execute(
            "INSERT INTO jobs(id,story,duration_seconds,status,progress,percent,created_at,updated_at) "
            "VALUES(?,?,?,'queued','Waiting for the local generator',0,?,?)",
            (job_id, story, requested_duration, now, now),
        )
        connection.commit()
    logger.info("Queued video job %s (%ss)", job_id, requested_duration)
    return jsonify(id=job_id, status="queued"), 202


@app.get("/api/jobs")
def recent_jobs():
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 20").fetchall()
    return jsonify(jobs=[_public_job(row) for row in rows])


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    if not _valid_job_id.fullmatch(job_id):
        return jsonify(error="Video job not found."), 404
    row = _get_job(job_id)
    return jsonify(_public_job(row)) if row else (jsonify(error="Video job not found."), 404)


@app.post("/api/jobs/<job_id>/retry")
def retry_job(job_id: str):
    if not _valid_job_id.fullmatch(job_id):
        return jsonify(error="Video job not found."), 404
    existing = _get_job(job_id)
    if existing is None:
        return jsonify(error="Video job not found."), 404
    if existing["status"] != "error":
        return jsonify(error="Only failed videos can be retried."), 409
    models_ready, models = _local_model_status()
    if not models_ready:
        return jsonify(error="Local models are still being installed or are unavailable. Check readiness and retry.", models=models), 503
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            connection.rollback()
            return jsonify(error="Video job not found."), 404
        if row["status"] != "error":
            connection.rollback()
            return jsonify(error="Only failed videos can be retried."), 409
        pending = connection.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','working')").fetchone()[0]
        if pending >= app.config["MAX_PENDING_JOBS"]:
            connection.rollback()
            return jsonify(error="The video queue is full. Please try again shortly."), 503
        connection.execute(
            "UPDATE jobs SET status='queued', progress='Waiting for retry', percent=0, video=NULL, error=NULL, updated_at=? WHERE id=?",
            (time.time(), job_id),
        )
        connection.commit()
    logger.info("Retry queued for video job %s", job_id)
    return jsonify(id=job_id, status="queued"), 202


@app.get("/videos/<path:name>")
def video(name: str):
    if not _valid_video_name.fullmatch(name):
        return jsonify(error="Video not found."), 404
    row = _get_job(name.removeprefix("AntHill_").removesuffix(".mp4"))
    if row is None or row["status"] != "done" or row["video"] != name:
        return jsonify(error="Video not found."), 404
    path = (ROOT / "output" / name).resolve()
    if path.parent != (ROOT / "output").resolve() or not path.is_file():
        return jsonify(error="Video not found."), 404
    return send_file(path, mimetype="video/mp4", as_attachment=True, download_name=name, conditional=True)


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify(error="Stories must be 6,000 characters or fewer."), 413


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    start_worker()
    app.run(
        host=os.getenv("ANTHILL_BIND_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        threaded=True,
        use_reloader=False,
    )
