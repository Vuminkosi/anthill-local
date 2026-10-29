"""Contract tests for the durable local video job API; no model inference is run."""

from __future__ import annotations

import gc
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

server = importlib.import_module("server")


class VideoJobsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="anthill-api-test-")
        self.root = Path(self.temp_dir.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.original_root = server.ROOT
        server.ROOT = self.root
        server.app.config.update(
            TESTING=True,
            JOB_DATABASE=self.root / "state" / "jobs.sqlite3",
            MAX_PENDING_JOBS=2,
            JOB_RETENTION=20,
        )
        server._initialize_database()
        self.client = server.app.test_client()
        self.model_readiness = patch.object(
            server,
            "_local_model_status",
            return_value=(True, {"image": {"installed": True}, "narration": {"installed": True}, "storyboard": {"installed": True}}),
        )
        self.model_readiness.start()
        self.addCleanup(self.model_readiness.stop)

    def tearDown(self) -> None:
        server.ROOT = self.original_root
        gc.collect()
        self.temp_dir.cleanup()

    def submit(self, **overrides):
        payload = {"story": "A kind waitress helped a nervous man during breakfast."}
        payload.update(overrides)
        return self.client.post("/api/jobs", json=payload)

    def test_story_and_duration_validation_returns_helpful_errors(self) -> None:
        self.assertEqual(self.client.post("/api/jobs", data="plain text").status_code, 400)
        self.assertEqual(self.client.post("/api/jobs", json=["story"]).status_code, 400)
        self.assertEqual(self.submit(story=17).status_code, 400)
        self.assertEqual(self.submit(story="short").status_code, 400)
        self.assertEqual(self.submit(story="x" * 6001).status_code, 400)
        self.assertEqual(self.submit(duration_seconds=True).status_code, 400)
        self.assertEqual(self.submit(duration_seconds=31).status_code, 400)
        self.assertEqual(self.submit(duration_seconds="30").status_code, 400)
        too_large = self.client.post("/api/jobs", data=("x" * 110_000), content_type="application/json")
        self.assertIn(too_large.status_code, (400, 413))

    def test_accepted_job_is_persisted_and_lists_only_public_fields(self) -> None:
        response = self.submit(duration_seconds=45)
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["id"]
        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["status"], "queued")
        self.assertEqual(status["duration_seconds"], 45)
        self.assertEqual(status["percent"], 0)
        self.assertNotIn("story", status)
        self.assertNotIn("error", status)
        self.assertEqual(self.client.get("/api/jobs").get_json()["jobs"][0]["id"], job_id)

    def test_queue_capacity_is_bounded(self) -> None:
        self.assertEqual(self.submit().status_code, 202)
        self.assertEqual(self.submit().status_code, 202)
        response = self.submit()
        self.assertEqual(response.status_code, 503)

    def test_job_worker_persists_progress_and_final_output(self) -> None:
        job_id = self.submit().get_json()["id"]

        def fake_run_job(actual_id, _story, progress, duration_seconds=30):
            progress("Generating scene 3", 45)
            self.assertEqual(duration_seconds, 30)
            path = self.output / f"AntHill_{actual_id}.mp4"
            path.write_bytes(b"test video bytes")
            return path

        with patch.object(server, "run_job", side_effect=fake_run_job):
            server._claim_next_job()
            server._run_claimed_job(job_id)

        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["status"], "done")
        self.assertEqual(status["percent"], 100)
        self.assertEqual(status["progress"], "Video ready")
        download = self.client.get(status["video"])
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.data, b"test video bytes")
        self.assertEqual(download.mimetype, "video/mp4")
        download.close()

    def test_failure_is_sanitized_and_retry_preserves_job(self) -> None:
        secret_story = "Private detail: do not expose this story to API users."
        job_id = self.submit(story=secret_story).get_json()["id"]
        with patch.object(server, "run_job", side_effect=RuntimeError(secret_story)):
            server._claim_next_job()
            server._run_claimed_job(job_id)

        failed = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(failed["status"], "error")
        self.assertNotIn(secret_story, str(failed))
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/retry").status_code, 202)
        retried = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["percent"], 0)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/retry").status_code, 409)

    def test_restart_recovers_interrupted_work_without_exposing_story(self) -> None:
        job_id = self.submit().get_json()["id"]
        server._claim_next_job()
        server._initialize_database()
        recovered = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["progress"], "Resuming after application restart")

    def test_video_route_rejects_arbitrary_paths_and_incomplete_jobs(self) -> None:
        self.assertEqual(self.client.get("/videos/../state/jobs.sqlite3").status_code, 404)
        self.assertEqual(self.client.get("/videos/AntHill_000000000000.mp4").status_code, 404)

    def test_health_and_readiness_do_not_trigger_downloads(self) -> None:
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")
        with patch.object(
            server,
            "_local_model_status",
            return_value=(False, {"image": {"installed": False}, "narration": {"installed": True}, "storyboard": {"installed": True}}),
        ):
            response = self.client.get("/api/readiness")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "models-needed")

    def test_readiness_rejects_partial_dreamshaper_snapshot(self) -> None:
        image_path = self.root / "models" / "huggingface" / "dreamshaper"
        for relative in (
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ):
            target = image_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")

        voice_path = self.root / "models" / "voices" / "kokoro"
        for relative in ("kokoro-v1_0.pth", "config.json", "voices/af_heart.pt"):
            target = voice_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")

        ollama_reply = io.BytesIO(json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode())
        opener = unittest.mock.MagicMock()
        opener.open.return_value = ollama_reply
        self.model_readiness.stop()
        with patch.dict(
            os.environ,
            {
                "IMAGE_MODEL": "Lykon/dreamshaper-8",
                "OLLAMA_URL": "http://ollama:11434",
                "OLLAMA_MODEL": "llama3.2:3b",
                "KOKORO_VOICE": "af_heart",
            },
        ), patch.object(server, "IMAGE_PATH", image_path), patch.object(server, "VOICE_PATH", voice_path), patch.object(
            server.urllib.request, "build_opener", return_value=opener
        ):
            ready, models = server._local_model_status()

        self.assertFalse(ready)
        self.assertFalse(models["image"]["installed"])
        self.assertTrue(models["narration"]["installed"])
        self.assertTrue(models["storyboard"]["installed"])

    def test_readiness_ollama_redirect_is_not_followed(self) -> None:
        self.model_readiness.stop()
        redirect = urllib.error.HTTPError(
            "http://ollama:11434/api/tags", 302, "Found", {"Location": "https://example.com/api/tags"}, None
        )
        opener = unittest.mock.MagicMock()
        opener.open.side_effect = redirect
        with patch.dict(os.environ, {"OLLAMA_URL": "http://ollama:11434"}), patch.object(
            server.urllib.request, "build_opener", return_value=opener
        ) as build_opener:
            ready, models = server._local_model_status()

        self.assertFalse(ready)
        self.assertFalse(models["storyboard"]["installed"])
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args.args[0], "http://ollama:11434/api/tags")
        self.assertTrue(any(isinstance(handler, server._NoRedirectHandler) for handler in build_opener.call_args.args))


if __name__ == "__main__":
    unittest.main()
