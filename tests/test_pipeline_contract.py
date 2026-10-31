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
            {
                "image": f"scene {index}",
                "caption": f"beat {index}",
                "narration": "spoken words " * 6,
                "action": f"Mara moves through scene {index}",
            }
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
        request = session.post.call_args
        self.assertEqual(request.args, ("http://ollama:11434/api/generate",))
        self.assertFalse(session.trust_env)
        payload = request.kwargs["json"]
        self.assertEqual(payload["format"], "json")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["options"]["num_gpu"], 0)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertLessEqual(payload["options"]["num_predict"], 1400)
        prompt = payload["prompt"]
        self.assertIn('"comedic"', prompt)
        self.assertIn(character, prompt)
        self.assertIn(adjustments, prompt)
        self.assertIn(story, prompt)
        self.assertIn("concise action string", prompt)
        self.assertIn("naming visible movement", prompt)
        self.assertIn("each action at most 8 words", prompt)
        self.assertIn("each caption 3-7 words", prompt)
        self.assertIn("Scene 1 must open on the story's central hook", prompt)
        self.assertIn("original chronological order across scenes 2-5", prompt)
        self.assertIn("Scene 6 must show the story's stated ending", prompt)
        self.assertIn("specific enough to recognize across scenes", prompt)
        self.assertIn("location or environment", prompt)

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
                pipeline, "make_ltx_clips", return_value=[]
            ), patch.object(pipeline, "render"):
                result = pipeline.run_job("legacy-signature", story, progress_messages.append)

        self.assertEqual(result, output / "AntHill_legacy-signature.mp4")
        self.assertEqual(storyboard.call_args.args, (story, 30, None, "heartfelt", "", ""))
        self.assertTrue(progress_messages)

    def test_run_job_uses_ltx_clips_and_passes_them_to_render(self) -> None:
        story = "A traveler finds a lost dog and returns it before the storm."
        plan = {
            "character_bible": "A traveler and a small dog",
            "narration": "The traveler returns the dog before the storm.",
            "scenes": [{"image": "A traveler", "caption": "A kind discovery", "narration": "spoken words " * 6}],
        }
        with tempfile.TemporaryDirectory(prefix="anthill-motion-order-") as temporary:
            root = Path(temporary)
            clips = [root / "motion.mp4"]
            order = []

            def narrate(*_args):
                order.append("narration")

            def make_ltx(*args):
                order.append("ltx")
                self.assertEqual(args[1], plan["character_bible"])
                return clips

            def render(*args, **kwargs):
                order.append("render")
                self.assertIsNone(args[1])
                self.assertEqual(kwargs["motion_clips"], clips)

            with patch.object(pipeline, "OUT", root / "output"), patch.object(
                pipeline, "storyboard", return_value=plan
            ), patch.object(pipeline, "narrate", side_effect=narrate) as narrate_mock, patch.object(
                pipeline, "make_ltx_clips", side_effect=make_ltx
            ), patch.object(pipeline, "render", side_effect=render):
                pipeline.run_job("motion-order", story, lambda *_args: None)

        self.assertEqual(narrate_mock.call_count, 1)
        self.assertEqual(order, ["narration", "ltx", "render"])

    def test_run_job_honors_cancellation_raised_during_motion_generation(self) -> None:
        story = "A traveler finds a lost dog and returns it before the storm."
        plan = {
            "character_bible": "A traveler and a small dog",
            "narration": "The traveler returns the dog before the storm.",
            "scenes": [{"image": "A traveler", "caption": "A kind discovery", "narration": "spoken words " * 6}],
        }
        with tempfile.TemporaryDirectory(prefix="anthill-motion-cancel-") as temporary:
            output = Path(temporary) / "output"
            cancellation_checks = []

            def should_cancel():
                cancellation_checks.append(True)
                return len(cancellation_checks) >= 4

            def cancel_during_animation(*args):
                self.assertTrue(callable(args[-1]))
                if args[-1]():
                    raise pipeline.GenerationCancelled("cancelled in animation")

            with patch.object(pipeline, "OUT", output), patch.object(
                pipeline, "storyboard", return_value=plan
            ), patch.object(pipeline, "narrate"), patch.object(
                pipeline, "make_ltx_clips", side_effect=cancel_during_animation
            ) as animation, patch.object(pipeline, "render") as render:
                with self.assertRaises(pipeline.GenerationCancelled):
                    pipeline.run_job("motion-cancel", story, lambda *_args: None, cancelled=should_cancel)

        animation.assert_called_once()
        self.assertEqual(animation.call_args.args[1], plan["character_bible"])
        render.assert_not_called()
        self.assertGreaterEqual(len(cancellation_checks), 4)

    def test_motion_render_zooms_each_scene_before_caption_and_uses_its_frame_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anthill-motion-render-zoom-") as temporary:
            root = Path(temporary)
            work_dir = root / "work"
            work_dir.mkdir()
            scenes = [
                {"caption": f"Beat {index}", "narration": f"Narration for beat {index}."}
                for index in range(4)
            ]
            images = [root / f"still_{index}.png" for index in range(4)]
            voices = [root / f"voice_{index}.wav" for index in range(4)]
            motion_clips = [root / f"motion_{index}.mp4" for index in range(4)]
            for path in (*images, *voices, *motion_clips):
                path.write_bytes(b"fixture")

            recorded_commands = []

            def record_command(args, **_kwargs):
                recorded_commands.append(args)
                Path(args[-1]).write_bytes(b"fixture output")
                return unittest.mock.Mock(stdout="", returncode=0)

            with patch.object(pipeline, "command", side_effect=record_command), patch.object(
                pipeline, "duration", side_effect=[8.0, 1.0, 8.0, 1.0, 8.0, 1.0, 8.0, 1.0, 31.0]
            ), patch.object(pipeline, "music"), patch.object(pipeline, "_caption_overlay"):
                pipeline.render(
                    scenes, images, voices, work_dir, root / "final.mp4", lambda *_args: None,
                    seconds=31, motion_clips=motion_clips,
                )

        scene_commands = [
            args for args in recorded_commands
            if "-filter_complex" in args and "-frames:v" in args
        ]
        self.assertEqual(len(scene_commands), len(scenes))
        expected_counts = [233, 233, 232, 232]
        for args, expected_count in zip(scene_commands, expected_counts):
            filter_graph = args[args.index("-filter_complex") + 1]
            scene_layer = filter_graph.split(";", 1)[0]
            self.assertLess(scene_layer.index("[0:v]"), scene_layer.index("zoompan"))
            self.assertLess(scene_layer.index("zoompan"), scene_layer.index("[base]"))
            self.assertIn(f"2*PI*on/{expected_count - 1}", scene_layer)
            self.assertIn("[1:v]", filter_graph)
            self.assertIn("overlay=", filter_graph)
            frame_count_index = args.index("-frames:v")
            self.assertEqual(int(args[frame_count_index + 1]), expected_count)

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

    def test_model_manifest_contains_only_active_video_and_narration_models(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anthill-ltx-manifest-") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            encoder = root / "text-encoder"
            config = root / "config"
            with patch.object(model_setup, "MODELS_ROOT", root / "models"), patch.object(
                model_setup, "LTX_CHECKPOINT_PATH", checkpoint
            ), patch.object(model_setup, "LTX_TEXT_ENCODER_PATH", encoder), patch.object(
                model_setup, "LTX_CONFIG_PATH", config
            ):
                for relative in model_setup.LTX_CHECKPOINT_FILES:
                    path = checkpoint / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
                for relative in (
                    "text_encoder/config.json",
                    "text_encoder/model-00001-of-00002.safetensors",
                    "text_encoder/model-00002-of-00002.safetensors",
                    "text_encoder/model.safetensors.index.json",
                    "tokenizer/spiece.model",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/added_tokens.json",
                ):
                    path = encoder / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
                config_path = config / model_setup.LTX_CONFIG_FILE
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text("pipeline_type: multi-scale\\n", encoding="utf-8")
                manifest = model_setup._model_manifest()["models"]
                self.assertEqual(set(manifest), {"video", "narration", "storyboard"})
                self.assertTrue(manifest["video"]["ready"])
                self.assertEqual(manifest["video"]["variant"], "2B distilled BF16 0.9.8")
                self.assertNotIn("image", manifest)
                self.assertNotIn("animation", manifest)

    def test_ltx_verify_only_checks_active_video_and_narration_models(self) -> None:
        with patch.object(model_setup, "_snapshot") as snapshot, patch.object(
            model_setup,
            "_write_manifest",
            return_value={"models": {"video": {"ready": True}, "narration": {"ready": True}}},
        ) as write_manifest, patch("sys.argv", ["model_setup", "--verify-only"]):
            self.assertEqual(model_setup.main(), 0)
        snapshot.assert_not_called()
        write_manifest.assert_called_once_with()

    def test_ltx_download_uses_only_the_pinned_active_assets(self) -> None:
        with patch.object(model_setup, "_snapshot") as snapshot, patch.object(
            model_setup, "_write_manifest", return_value={"models": {}}
        ), patch("sys.argv", ["model_setup", "--download-ltx-video"]):
            self.assertEqual(model_setup.main(), 0)
        self.assertEqual(snapshot.call_count, 3)
        self.assertEqual(snapshot.call_args_list[0].args[:2], (model_setup.LTX_VIDEO_REPO, model_setup.LTX_VIDEO_REVISION))
        self.assertEqual(snapshot.call_args_list[1].args[:2], (model_setup.LTX_TEXT_ENCODER_REPO, model_setup.LTX_TEXT_ENCODER_REVISION))
        self.assertEqual(snapshot.call_args_list[2].kwargs["repo_type"], "space")


if __name__ == "__main__":
    unittest.main()
