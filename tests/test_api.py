"""Contract tests for the durable local video job API; no model inference is run."""

from __future__ import annotations

import gc
import importlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

server = importlib.import_module("server")
pipeline = importlib.import_module("pipeline")


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
            return_value=(True, {"video": {"installed": True}, "narration": {"installed": True}, "storyboard": {"installed": True}}),
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
        self.assertEqual(status["phase"], "queued")
        self.assertTrue(status["title"].startswith("Story video - "))
        self.assertNotIn("story", status)
        self.assertNotIn("error", status)
        self.assertEqual(self.client.get("/api/jobs").get_json()["jobs"][0]["id"], job_id)

    def test_story_tone_and_character_guidance_are_persisted_with_defaults(self) -> None:
        defaults = self.submit().get_json()
        default_status = self.client.get(f"/api/jobs/{defaults['id']}").get_json()
        self.assertEqual(default_status["tone"], "heartfelt")
        self.assertEqual(default_status["character"], "")
        self.assertEqual(default_status["adjustments"], "")

        response = self.submit(
            tone="suspenseful",
            character="A tall, silver-haired botanist with round glasses.",
            adjustments="Make the ending hopeful and add a quiet visual callback.",
        )
        self.assertEqual(response.status_code, 202)
        status = self.client.get(f"/api/jobs/{response.get_json()['id']}").get_json()
        self.assertEqual(status["tone"], "suspenseful")
        self.assertEqual(status["character"], "A tall, silver-haired botanist with round glasses.")
        self.assertEqual(status["adjustments"], "Make the ending hopeful and add a quiet visual callback.")

    def test_unsupported_tones_and_oversized_guidance_are_rejected(self) -> None:
        for tone in ("romantic", "", None, 5):
            with self.subTest(tone=tone):
                self.assertEqual(self.submit(tone=tone).status_code, 400)
        self.assertEqual(self.submit(character="x" * 501).status_code, 400)
        self.assertEqual(self.submit(adjustments="x" * 1001).status_code, 400)

    def test_cancel_queued_job_prevents_it_from_being_claimed(self) -> None:
        job_id = self.submit().get_json()["id"]

        response = self.client.post(f"/api/jobs/{job_id}/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "cancelled")
        cancelled = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["phase"], "cancelled")
        self.assertEqual(server._claim_next_job(), None)

    def test_cancel_rejects_invalid_and_unknown_job_ids(self) -> None:
        self.assertEqual(self.client.post("/api/jobs/not-a-job/cancel").status_code, 404)
        self.assertEqual(self.client.post("/api/jobs/000000000000/cancel").status_code, 404)

    def test_cancel_active_job_is_idempotent_and_finishes_as_cancelled(self) -> None:
        job_id = self.submit().get_json()["id"]
        self.assertEqual(server._claim_next_job(), job_id)
        started = threading.Event()
        cancellation_observed = threading.Event()
        release_worker = threading.Event()

        def wait_for_cancellation(_actual_id, _story, _progress, duration_seconds=30, cancelled=None, **_settings):
            self.assertEqual(duration_seconds, 30)
            self.assertIsNotNone(cancelled)
            started.set()
            while not cancelled():
                threading.Event().wait(0.01)
            cancellation_observed.set()
            self.assertTrue(release_worker.wait(5))
            raise pipeline.GenerationCancelled()

        with patch.object(server, "run_job", side_effect=wait_for_cancellation):
            worker = threading.Thread(target=server._run_claimed_job, args=(job_id,))
            worker.start()
            self.assertTrue(started.wait(5))

            response = self.client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.get_json()["status"], "cancelling")
            self.assertTrue(cancellation_observed.wait(5))

            duplicate = self.client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(duplicate.status_code, 202)
            self.assertEqual(duplicate.get_json()["status"], "cancelling")
            release_worker.set()
            worker.join(5)

        self.assertFalse(worker.is_alive())
        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["status"], "cancelled")
        self.assertEqual(status["phase"], "cancelled")
        self.assertNotIn("video", status)
        self.assertFalse((self.output / f"AntHill_{job_id}.mp4").exists())

    def test_cancellation_wins_race_against_pipeline_success(self) -> None:
        job_id = self.submit().get_json()["id"]
        self.assertEqual(server._claim_next_job(), job_id)
        started = threading.Event()
        release_worker = threading.Event()

        def finish_after_cancel(actual_id, _story, _progress, duration_seconds=30, cancelled=None, **_settings):
            self.assertEqual(duration_seconds, 30)
            self.assertIsNotNone(cancelled)
            started.set()
            self.assertTrue(release_worker.wait(5))
            path = self.output / f"AntHill_{actual_id}.mp4"
            path.write_bytes(b"must not be published after cancellation")
            return path

        with patch.object(server, "run_job", side_effect=finish_after_cancel):
            worker = threading.Thread(target=server._run_claimed_job, args=(job_id,))
            worker.start()
            self.assertTrue(started.wait(5))

            response = self.client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(response.status_code, 202)
            release_worker.set()
            worker.join(5)

        self.assertFalse(worker.is_alive())
        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["status"], "cancelled")
        self.assertNotIn("video", status)
        self.assertFalse((self.output / f"AntHill_{job_id}.mp4").exists())

    def test_cancel_terminal_job_returns_conflict_with_current_status(self) -> None:
        job_id = self.submit().get_json()["id"]
        with self.output.joinpath(f"AntHill_{job_id}.mp4").open("wb") as video:
            video.write(b"finished")
        with server._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='done', phase='complete', video=?, percent=100 WHERE id=?",
                (f"AntHill_{job_id}.mp4", job_id),
            )

        response = self.client.post(f"/api/jobs/{job_id}/cancel")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["status"], "done")
        self.assertTrue((self.output / f"AntHill_{job_id}.mp4").is_file())

    def test_progress_phase_is_persisted_and_exposed(self) -> None:
        job_id = self.submit().get_json()["id"]
        self.assertEqual(server._claim_next_job(), job_id)

        server._update_progress(job_id, "Generating cartoon scene 2 of 6", 29, "scenes")

        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["phase"], "scenes")
        self.assertEqual(status["progress"], "Generating cartoon scene 2 of 6")
        self.assertEqual(status["percent"], 29)

    def test_database_migration_adds_phase_and_recovers_interrupted_jobs(self) -> None:
        legacy_database = self.root / "state" / "legacy.sqlite3"
        legacy_database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(legacy_database) as connection:
            connection.execute(
                """CREATE TABLE jobs (
                    id TEXT PRIMARY KEY, story TEXT NOT NULL, duration_seconds INTEGER NOT NULL,
                    status TEXT NOT NULL, progress TEXT NOT NULL, percent INTEGER NOT NULL,
                    video TEXT, error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("aaaaaaaaaaaa", "private story", 30, "working", "Generating", 22, None, None, 1, 1),
            )
        original_database = server.app.config["JOB_DATABASE"]
        server.app.config["JOB_DATABASE"] = legacy_database
        try:
            server._initialize_database()
            recovered = server._get_job("aaaaaaaaaaaa")
            with server._connect() as connection:
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
        finally:
            server.app.config["JOB_DATABASE"] = original_database

        self.assertIn("phase", columns)
        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["phase"], "queued")
        self.assertEqual(recovered["progress"], "Resuming after application restart")
        self.assertEqual(recovered["tone"], "heartfelt")
        self.assertEqual(recovered["character"], "")
        self.assertEqual(recovered["adjustments"], "")

    def test_restart_finishes_a_pending_cancellation_instead_of_requeueing(self) -> None:
        job_id = self.submit().get_json()["id"]
        self.assertEqual(server._claim_next_job(), job_id)
        with server._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status='cancelling', phase='cancelling', progress='Stopping' WHERE id=?",
                (job_id,),
            )

        server._initialize_database()

        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["status"], "cancelled")
        self.assertEqual(status["phase"], "cancelled")

    def test_queue_capacity_is_bounded(self) -> None:
        self.assertEqual(self.submit().status_code, 202)
        self.assertEqual(self.submit().status_code, 202)
        response = self.submit()
        self.assertEqual(response.status_code, 503)

    def test_job_worker_persists_progress_and_final_output(self) -> None:
        job_id = self.submit().get_json()["id"]

        def fake_run_job(actual_id, _story, progress, duration_seconds=30, **_settings):
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
        job_id = self.submit(
            story=secret_story,
            tone="dramatic",
            character="A tired paramedic with a blue scarf.",
            adjustments="Keep the rescue non-graphic and end with relief.",
        ).get_json()["id"]
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
        self.assertEqual(retried["tone"], "dramatic")
        self.assertEqual(retried["character"], "A tired paramedic with a blue scarf.")
        self.assertEqual(retried["adjustments"], "Keep the rescue non-graphic and end with relief.")
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

    def test_index_uses_character_feature_dropdown(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('<select id="character" name="character"', html)
        self.assertIn('<option value="" selected>Let the story decide</option>', html)
        self.assertIn('<option value="A dog">A dog</option>', html)
        self.assertIn('<option value="A stick person">Stick person</option>', html)
        self.assertNotIn('<input id="character"', html)
        self.assertIn('<option value="did_you_know">Did You Know?</option>', html)
        self.assertIn('<option value="explainer">Explainer</option>', html)
        self.assertIn('id="research-category"', html)
        self.assertIn('id="research-subcategory"', html)
        self.assertIn("Human Relations", html)
        self.assertIn("The Book of Enoch", html)
        self.assertIn("Angels and heavenly help available to us", html)
        self.assertIn("Marriage and Relationship Dynamics", html)
        self.assertIn("Business", html)

    def test_researched_video_types_require_valid_category_and_subcategory(self) -> None:
        self.assertEqual(self.submit(video_format="did_you_know", category="science").status_code, 400)
        self.assertEqual(self.submit(video_format="explainer", category="science", subcategory="bogus").status_code, 400)
        response = self.submit(video_format="did_you_know", category="science", subcategory="space", story="")
        self.assertEqual(response.status_code, 202)
        job = self.client.get(f"/api/jobs/{response.get_json()['id']}").get_json()
        self.assertEqual(job["video_format"], "did_you_know")
        self.assertEqual(job["category"], "science")
        self.assertEqual(job["subcategory"], "space")
        self.assertEqual(job["sources"], [])
        self.assertEqual(job["title"], "Did You Know: Space and astronomy")

    def test_readiness_exposes_ltx_video_state_without_starting_model_setup(self) -> None:
        models = {
            "narration": {"installed": True},
            "storyboard": {"installed": True},
            "video": {"name": "LTX-Video 2B Distilled", "installed": False, "active": True},
        }
        with patch.object(server, "_local_model_status", return_value=(False, models)) as status:
            response = self.client.get("/api/readiness")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["models"]["video"], models["video"])
        status.assert_called_once_with()

    def test_readiness_blocks_when_ltx_assets_are_incomplete(self) -> None:
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
                "OLLAMA_URL": "http://ollama:11434",
                "OLLAMA_MODEL": "llama3.2:3b",
                "KOKORO_VOICE": "af_heart",
            },
        ), patch.object(server, "VOICE_PATH", voice_path), patch.object(
            server, "ltx_video_assets_ready", return_value=False
        ), patch.object(server.urllib.request, "build_opener", return_value=opener
        ):
            ready, models = server._local_model_status()

        self.assertFalse(ready)
        self.assertTrue(models["narration"]["installed"])
        self.assertTrue(models["storyboard"]["installed"])
        self.assertFalse(models["video"]["installed"])

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
