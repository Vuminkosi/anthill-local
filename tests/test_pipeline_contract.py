"""Local-only inference safeguards for the story pipeline."""

from __future__ import annotations

import importlib
import io
import json
import tempfile
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

try:
    import pipeline
    import model_setup
    import server
except ImportError as exc:
    pipeline = None
    model_setup = None
    server = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


@unittest.skipIf(pipeline is None, f"pipeline dependencies unavailable: {_IMPORT_ERROR}")
class LocalInferenceContractTests(unittest.TestCase):
    def test_progress_callback_keeps_legacy_one_and_two_argument_signatures(self) -> None:
        calls = []

        pipeline._progress(lambda message: calls.append((message,)), "Narrating", 12, "narration")
        pipeline._progress(lambda message, percent: calls.append((message, percent)), "Scenes", 40, "scenes")

        self.assertEqual(calls, [("Narrating",), ("Scenes", 40)])

    def test_progress_callback_receives_optional_generation_phase(self) -> None:
        calls = []

        pipeline._progress(lambda message, percent, phase: calls.append((message, percent, phase)), "Scenes", 40, "scenes")

        self.assertEqual(calls, [("Scenes", 40, "scenes")])

    def test_storyboard_prompt_receives_tone_character_and_adjustments(self) -> None:
        scenes = [
            {"image": f"scene {index}", "caption": f"beat {index}", "narration": "spoken words " * 6}
            for index in range(6)
        ]
        plan = {"character_bible": "A consistent character", "narration": "spoken words " * 18, "scenes": scenes}
        response = unittest.mock.Mock(status_code=200)
        response.iter_lines.return_value = [json.dumps({"response": json.dumps(plan), "done": True})]
        session = unittest.mock.MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.post.return_value = response
        story = "A botanist finds a lantern and helps a stranded village before dawn."
        character = "Mara: silver braids, round glasses, green coat, brass leaf pin."
        adjustments = "Make the revelation gently funny, then finish with a hopeful sunrise."

        with patch.dict("os.environ", {"OLLAMA_URL": "http://ollama:11434"}), patch.object(
            pipeline.requests, "Session", return_value=session
        ), patch.object(pipeline, "_retry", side_effect=lambda operation, *_args, **_kwargs: operation()):
            result = pipeline.storyboard(
                story, tone="comedic", character=character, adjustments=adjustments
            )

        self.assertEqual(result, plan)
        prompt = session.post.call_args.kwargs["json"]["prompt"]
        self.assertIn('"comedic"', prompt)
        self.assertIn(character, prompt)
        self.assertIn(adjustments, prompt)
        self.assertIn(story, prompt)

    def test_legacy_storyboard_and_run_job_call_signatures_remain_supported(self) -> None:
        story = "A traveler finds a lost dog and returns it before the storm."
        plan = {
            "character_bible": "A traveler and a small dog",
            "narration": "The traveler returns the dog before the storm.",
            "scenes": [
                {"image": "A traveler", "caption": "A kind discovery", "narration": "spoken words " * 6}
                for _ in range(6)
            ],
        }
        with tempfile.TemporaryDirectory(prefix="anthill-legacy-call-") as temporary:
            output = Path(temporary) / "output"
            progress_messages = []
            with patch.object(pipeline, "OUT", output), patch.object(
                pipeline, "storyboard", return_value=plan
            ) as storyboard, patch.object(pipeline, "narrate"), patch.object(
                pipeline, "make_images", return_value=[]
            ), patch.object(pipeline, "render"):
                result = pipeline.run_job("legacy-signature", story, progress_messages.append)

        self.assertEqual(result, output / "AntHill_legacy-signature.mp4")
        self.assertEqual(storyboard.call_args.args, (story, 30, None, "heartfelt", "", ""))
        self.assertTrue(progress_messages)

    def test_run_job_checks_cancellation_before_storyboard_or_model_work(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anthill-cancel-test-") as temporary:
            output = Path(temporary) / "output"
            with patch.object(pipeline, "OUT", output), patch.object(pipeline, "storyboard") as storyboard:
                with self.assertRaises(pipeline.GenerationCancelled):
                    pipeline.run_job("cancelled-test", "A sufficiently long story to pass validation.", lambda *_args: None,
                                     cancelled=lambda: True)

        storyboard.assert_not_called()

    def test_storyboard_rejects_remote_ollama_before_sending_story(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_URL": "https://example.com"}), patch.object(
            pipeline.requests, "Session"
        ) as session:
            with self.assertRaisesRegex(RuntimeError, "only use a local Ollama"):
                pipeline.storyboard("A private marker story that must remain local.")
        session.assert_not_called()

    def test_storyboard_allows_only_loopback_or_compose_ollama_hosts(self) -> None:
        for endpoint in ("http://ollama:11434", "http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"):
            with self.subTest(endpoint=endpoint), patch.dict("os.environ", {"OLLAMA_URL": endpoint}):
                self.assertEqual(pipeline.local_ollama_base(), endpoint)

    def test_local_ollama_request_disables_proxies_and_redirects(self) -> None:
        response = unittest.mock.Mock(status_code=302)
        session = unittest.mock.MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.post.return_value = response
        with patch.dict("os.environ", {"OLLAMA_URL": "http://ollama:11434"}), patch.object(
            pipeline.requests, "Session", return_value=session
        ), patch.object(pipeline, "_retry", side_effect=lambda operation, *_args, **_kwargs: operation()):
            with self.assertRaisesRegex(RuntimeError, "Local Ollama storyboard failed"):
                pipeline.storyboard("A local story that is long enough to test.")
        self.assertFalse(session.trust_env)
        self.assertFalse(session.post.call_args.kwargs["allow_redirects"])

    def test_custom_kokoro_model_path_derives_the_voice_from_its_model_directory(self) -> None:
        import model_setup

        try:
            with patch.dict(os.environ, {"KOKORO_MODEL_PATH": "custom/kokoro.pth"}, clear=False):
                os.environ.pop("KOKORO_VOICE_PATH", None)
                importlib.reload(model_setup)
                expected_root = model_setup.MODELS_ROOT / "custom"
                self.assertEqual(model_setup.KOKORO_MODEL_FILE, expected_root / "kokoro.pth")
                self.assertEqual(model_setup.KOKORO_VOICE_FILE, expected_root / "voices" / "af_heart.pt")

                custom_voice = expected_root / "special_voice.pt"
                os.environ["KOKORO_VOICE_PATH"] = str(custom_voice)
                importlib.reload(model_setup)
                self.assertEqual(model_setup.KOKORO_VOICE_FILE, custom_voice)
        finally:
            importlib.reload(model_setup)

    def test_separate_kokoro_model_and_voice_paths_match_manifest_and_readiness(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anthill-kokoro-paths-") as temporary:
            root = Path(temporary)
            model_file = root / "weights" / "kokoro.pth"
            voice_file = root / "voices" / "af_heart.pt"
            environment = {
                "ANTHILL_DATA": str(root / "models"),
                "KOKORO_MODEL_PATH": str(model_file),
                "KOKORO_VOICE_PATH": str(voice_file),
                "KOKORO_VOICE": "af_heart",
                "IMAGE_MODEL": "Lykon/dreamshaper-8",
                "IMAGE_MODEL_PATH": str(root / "custom-images" / "dreamshaper"),
                "OLLAMA_URL": "http://ollama:11434",
                "OLLAMA_MODEL": "llama3.2:3b",
            }
            with patch.dict(os.environ, environment):
                importlib.reload(model_setup)
                model_file.parent.mkdir(parents=True)
                model_file.write_bytes(b"weights")
                model_setup.KOKORO_CONFIG_FILE.write_text("{}", encoding="utf-8")
                voice_file.parent.mkdir(parents=True)
                voice_file.write_bytes(b"voice")
                image_path = Path(environment["IMAGE_MODEL_PATH"])
                for relative in model_setup.IMAGE_FILES:
                    target = image_path / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"image")

                manifest = model_setup._model_manifest()["models"]
                self.assertTrue(manifest["narration"]["ready"])
                self.assertEqual(Path(manifest["narration"]["local_path"]), model_setup.VOICE_PATH)
                self.assertTrue(manifest["image"]["ready"])
                with patch.object(pipeline, "IMAGE_PATH", model_setup.IMAGE_PATH):
                    self.assertEqual(pipeline._local_model_path(), model_setup.IMAGE_PATH)
                tags = io.BytesIO(json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode("utf-8"))
                opener = unittest.mock.MagicMock()
                opener.open.return_value = tags
                with patch.object(server, "IMAGE_PATH", image_path), patch.object(
                    server, "IMAGE_FILES", model_setup.IMAGE_FILES
                ), patch.object(server, "VOICE_PATH", model_setup.VOICE_PATH), patch.object(
                    server, "VOICE_FILES", model_setup.VOICE_FILES
                ), patch.object(server.urllib.request, "build_opener", return_value=opener):
                    ready, status = server._local_model_status()
                self.assertTrue(ready)
                self.assertTrue(status["narration"]["installed"])
                self.assertTrue(status["image"]["installed"])
            importlib.reload(model_setup)


if __name__ == "__main__":
    unittest.main()
