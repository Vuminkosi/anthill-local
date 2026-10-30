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
                pipeline, "make_motion_clips", return_value=[]
            ), patch.object(pipeline, "render"):
                result = pipeline.run_job("legacy-signature", story, progress_messages.append)

        self.assertEqual(result, output / "AntHill_legacy-signature.mp4")
        self.assertEqual(storyboard.call_args.args, (story, 30, None, "heartfelt", "", ""))
        self.assertTrue(progress_messages)

    def test_run_job_generates_motion_from_text_without_stills_and_passes_clips_to_render(self) -> None:
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

            def make_motion(*args):
                order.append("motion")
                self.assertIsNone(args[1])
                return clips

            def render(*args, **kwargs):
                order.append("render")
                self.assertIsNone(args[1])
                self.assertEqual(kwargs["motion_clips"], clips)

            with patch.object(pipeline, "OUT", root / "output"), patch.object(
                pipeline, "storyboard", return_value=plan
            ), patch.object(pipeline, "narrate", side_effect=narrate) as narrate_mock, patch.object(
                pipeline, "make_images"
            ) as make_images, patch.object(
                pipeline, "make_motion_clips", side_effect=make_motion
            ), patch.object(pipeline, "render", side_effect=render):
                pipeline.run_job("motion-order", story, lambda *_args: None)

        make_images.assert_not_called()
        self.assertEqual(narrate_mock.call_count, 1)
        self.assertEqual(order, ["narration", "motion", "render"])

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
                pipeline, "make_images"
            ) as make_images, patch.object(
                pipeline, "make_motion_clips", side_effect=cancel_during_animation
            ) as animation, patch.object(pipeline, "render") as render:
                with self.assertRaises(pipeline.GenerationCancelled):
                    pipeline.run_job("motion-cancel", story, lambda *_args: None, cancelled=should_cancel)

        animation.assert_called_once()
        self.assertIsNone(animation.call_args.args[1])
        make_images.assert_not_called()
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
                pipeline, "duration", side_effect=[1.0, 1.0, 1.0, 1.0, 31.0]
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

    def test_missing_local_motion_adapter_fails_without_loading_remote_models(self) -> None:
        import types

        with tempfile.TemporaryDirectory(prefix="anthill-no-motion-adapter-") as temporary:
            adapter = Path(temporary) / "adapter"
            fake_torch = types.ModuleType("torch")
            fake_diffusers = types.ModuleType("diffusers")
            fake_adapter_loader = unittest.mock.Mock()
            fake_pipeline_loader = unittest.mock.Mock()
            fake_diffusers.MotionAdapter = unittest.mock.Mock(from_pretrained=fake_adapter_loader)
            fake_diffusers.AnimateDiffPipeline = unittest.mock.Mock(from_pretrained=fake_pipeline_loader)
            fake_diffusers.DDIMScheduler = unittest.mock.Mock()
            with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}), patch.object(
                pipeline, "_motion_model_path", return_value=adapter
            ):
                with self.assertRaisesRegex(RuntimeError, "inside the app container.*never downloads"):
                    pipeline.make_motion_clips([{"image": "scene"}], [Path("still.png")], "character",
                                               Path(temporary), lambda *_args: None)

        fake_adapter_loader.assert_not_called()
        fake_pipeline_loader.assert_not_called()

    def test_text_to_video_motion_pipeline_is_local_stylized_and_cancellable(self) -> None:
        import types

        with tempfile.TemporaryDirectory(prefix="anthill-motion-callback-") as temporary:
            root = Path(temporary)
            adapter_path = root / "adapter"
            base_path = root / "dreamshaper"
            for relative in pipeline.ANIMATION_FILES:
                target = adapter_path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"adapter")
            base_weight = base_path / "unet" / "diffusion_pytorch_model.fp16.safetensors"
            base_weight.parent.mkdir(parents=True)
            base_weight.write_bytes(b"base")
            callback_results = []
            pipeline_calls = []
            memory_calls = []
            initial_scheduler_config = object()
            source_scheduler_config = object()
            scheduler_result = types.SimpleNamespace(config=source_scheduler_config)

            class FakeMotionPipe:
                scheduler = types.SimpleNamespace(config=initial_scheduler_config)

                def enable_vae_slicing(self):
                    memory_calls.append(("vae_slicing",))

                def enable_vae_tiling(self):
                    memory_calls.append(("vae_tiling",))

                def enable_attention_slicing(self, value):
                    memory_calls.append(("attention_slicing", value))

                def enable_model_cpu_offload(self):
                    memory_calls.append(("cpu_offload",))

                def __call__(self, prompt, negative_prompt, height, width, num_frames,
                             num_inference_steps, guidance_scale, generator,
                             callback_on_step_end=None):
                    if self.scheduler is not scheduler_result:
                        raise AssertionError("DDIM scheduler must be assigned before generation")
                    call = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "generator": generator,
                    }
                    pipeline_calls.append(call)
                    kwargs = {"latents": object()}
                    returned = callback_on_step_end(self, 0, 1, kwargs)
                    callback_results.append((kwargs, returned))
                    frame = pipeline.Image.new("RGB", (width, height), "green")
                    return types.SimpleNamespace(frames=[[frame]])

            fake_pipe = FakeMotionPipe()
            fake_torch = types.ModuleType("torch")
            fake_torch.float16 = "float16"
            fake_torch.Generator = lambda **_kwargs: types.SimpleNamespace(
                manual_seed=lambda _seed: object()
            )
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            fake_diffusers = types.ModuleType("diffusers")
            adapter_loader = unittest.mock.Mock(return_value=object())
            pipeline_loader = unittest.mock.Mock(return_value=fake_pipe)
            scheduler_loader = unittest.mock.Mock(return_value=scheduler_result)
            fake_diffusers.MotionAdapter = types.SimpleNamespace(from_pretrained=adapter_loader)
            fake_diffusers.AnimateDiffPipeline = types.SimpleNamespace(from_pretrained=pipeline_loader)
            fake_diffusers.DDIMScheduler = types.SimpleNamespace(from_config=scheduler_loader)

            cancellation_checks = []

            def cancel_during_step():
                cancellation_checks.append(True)
                return len(cancellation_checks) >= 3

            with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}), patch.dict(
                os.environ,
                {"ANIMATEDIFF_WIDTH": "256", "ANIMATEDIFF_HEIGHT": "448",
                 "ANIMATEDIFF_FRAMES": "8", "ANIMATEDIFF_STEPS": "4"},
            ), patch.object(pipeline, "_motion_model_path", return_value=adapter_path), patch.object(
                pipeline, "_local_model_path", return_value=base_path
            ), patch.object(pipeline, "command") as command:
                clips = pipeline.make_motion_clips(
                    [{"image": "A rabbit runs through a moonlit garden"}], None,
                    "Mara, silver braids and a green coat", root / "success",
                    lambda *_args: None, cancelled=lambda: False,
                )
                self.assertEqual(len(clips), 1)
                self.assertIs(callback_results[0][0], callback_results[0][1])
                self.assertEqual(callback_results[0][1].keys(), {"latents"})
                self.assertEqual(pipeline_calls[0]["num_frames"], 8)
                self.assertEqual(pipeline_calls[0]["num_inference_steps"], 4)
                self.assertEqual(pipeline_calls[0]["guidance_scale"], 7.5)
                self.assertNotIn("video", pipeline_calls[0])
                self.assertNotIn("strength", pipeline_calls[0])
                self.assertIn("3D cartoon animation", pipeline_calls[0]["prompt"])
                self.assertIn("recognizable expressive characters", pipeline_calls[0]["prompt"])
                self.assertIn("full-body action", pipeline_calls[0]["prompt"])
                self.assertIn("Mara, silver braids and a green coat", pipeline_calls[0]["prompt"])
                self.assertIn("rabbit runs through a moonlit garden", pipeline_calls[0]["prompt"])
                self.assertIn("static pose", pipeline_calls[0]["negative_prompt"])

                adapter_loader.assert_called_once_with(
                    str(adapter_path), torch_dtype=fake_torch.float16, variant="fp16",
                    use_safetensors=True, local_files_only=True,
                )
                pipeline_loader.assert_called_once()
                base_model_args, base_model_kwargs = pipeline_loader.call_args
                self.assertEqual(base_model_args, (str(base_path),))
                self.assertIs(base_model_kwargs["motion_adapter"], adapter_loader.return_value)
                self.assertIs(base_model_kwargs["torch_dtype"], fake_torch.float16)
                self.assertEqual(base_model_kwargs["variant"], "fp16")
                self.assertTrue(base_model_kwargs["use_safetensors"])
                self.assertTrue(base_model_kwargs["local_files_only"])
                self.assertTrue(base_model_kwargs["low_cpu_mem_usage"])
                scheduler_loader.assert_called_once_with(
                    initial_scheduler_config,
                    beta_schedule="linear",
                    timestep_spacing="linspace",
                    steps_offset=1,
                    clip_sample=False,
                )
                self.assertEqual(
                    memory_calls,
                    [("vae_slicing",), ("vae_tiling",), ("attention_slicing", "max"), ("cpu_offload",)],
                )

                with patch.dict(os.environ, {"ANIMATEDIFF_STEPS": "30"}):
                    pipeline.make_motion_clips(
                        [{"image": "A rabbit runs through a moonlit garden"}], None,
                        "Mara, silver braids and a green coat", root / "maximum-steps",
                        lambda *_args: None, cancelled=lambda: False,
                    )
                self.assertEqual(pipeline_calls[-1]["num_inference_steps"], 30)
                self.assertEqual(pipeline_calls[-1]["guidance_scale"], 7.5)

                with patch.dict(
                    os.environ,
                    {"ANIMATEDIFF_WIDTH": "256", "ANIMATEDIFF_HEIGHT": "448", "ANIMATEDIFF_FRAMES": "8"},
                    clear=True,
                ):
                    pipeline.make_motion_clips(
                        [{"image": "A rabbit runs through a moonlit garden"}], None,
                        "Mara, silver braids and a green coat", root / "default-steps",
                        lambda *_args: None, cancelled=lambda: False,
                    )
                self.assertEqual(pipeline_calls[-1]["num_inference_steps"], 25)
                self.assertEqual(pipeline_calls[-1]["guidance_scale"], 7.5)

                for invalid_steps in (3, 31):
                    with self.subTest(invalid_steps=invalid_steps), patch.dict(
                        os.environ, {"ANIMATEDIFF_STEPS": str(invalid_steps)}
                    ), self.assertRaisesRegex(ValueError, "between 4 and 30"):
                        pipeline.make_motion_clips(
                            [{"image": "A rabbit runs through a moonlit garden"}], None,
                            "Mara, silver braids and a green coat", root / f"invalid-steps-{invalid_steps}",
                            lambda *_args: None, cancelled=lambda: False,
                        )

                callback_results.clear()
                pipeline_calls.clear()
                cancellation_checks.clear()
                with self.assertRaises(pipeline.GenerationCancelled):
                    pipeline.make_motion_clips(
                        [{"image": "A rabbit runs through a moonlit garden"}], None,
                        "Mara, silver braids and a green coat", root / "cancelled",
                        lambda *_args: None, cancelled=cancel_during_step,
                    )

            self.assertEqual(command.call_count, 3)

    def test_motion_prompt_prioritizes_explicit_action_over_long_descriptions(self) -> None:
        action = "Emma reaches for Rob’s hand"
        character = "Mara has silver braids, round glasses, a green coat, a brass leaf pin, " + "distinctive details " * 12
        scene = {
            "action": action,
            "caption": "A quiet reunion",
            "image": "Rain-soaked station with a blue train and silver lights. "
            + "A very long image description should not displace the visual cue or action. " * 12,
        }

        prompt = pipeline._motion_prompt(scene, character)

        self.assertTrue(prompt.startswith("3D cartoon animation, recognizable expressive characters, full-body action."))
        self.assertIn(f"Action: {action}.", prompt)
        self.assertLess(prompt.index(action), prompt.index("Scene:"))
        self.assertLess(prompt.index("Scene:"), prompt.index("Character design:"))
        self.assertIn("Rain-soaked station with a blue train", prompt)
        self.assertIn("Mara has silver braids", prompt)
        self.assertIn("Vertical 9:16", prompt)
        self.assertLessEqual(len(prompt.split()), 55)

    def test_motion_prompt_uses_caption_when_action_is_missing(self) -> None:
        scene = {
            "caption": "Mara waves as the train pulls away",
            "image": "A crowded station glows in the pale morning light.",
        }

        prompt = pipeline._motion_prompt(scene, "Mara in a green coat")

        self.assertIn("Action: Mara waves as the train pulls away.", prompt)

    def test_motion_prompt_prefers_image_movement_over_static_caption(self) -> None:
        scene = {
            "caption": "A quiet reunion",
            "image": "The platform glows beneath station lights. Mara reaches for Rob’s hand beside the train.",
        }

        prompt = pipeline._motion_prompt(scene, "Mara and Rob at the station")

        self.assertIn("Action: Mara reaches for Rob’s hand beside the train.", prompt)

    def test_motion_prompt_extracts_movement_sentence_from_image_fallback(self) -> None:
        scene = {
            "image": "A quiet courtyard glows beneath the evening sky. Emma reaches for Rob’s hand near the fountain.",
        }

        prompt = pipeline._motion_prompt(scene, "Emma and Rob at dusk")

        self.assertIn("Action: Emma reaches for Rob’s hand near the fountain.", prompt)

    def test_motion_prompt_caps_explicit_action_at_nine_words(self) -> None:
        action = "reaches " * 25
        prompt = pipeline._motion_prompt({"action": action}, "Mara in a green coat")
        selected_action = prompt.split("Action: ", 1)[1].split(". Scene:", 1)[0]

        self.assertEqual(selected_action, " ".join(action.split()[:9]))
        self.assertLessEqual(len(selected_action.split()), 9)

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
                animation_path = root / "animatediff"
                for relative in model_setup.ANIMATION_FILES:
                    target = animation_path / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"animation")

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
                ), patch.object(server, "ANIMATION_PATH", animation_path), patch.object(
                    server, "ANIMATION_FILES", model_setup.ANIMATION_FILES
                ), patch.object(server.urllib.request, "build_opener", return_value=opener):
                    ready, status = server._local_model_status()
                self.assertTrue(ready)
                self.assertTrue(status["narration"]["installed"])
                self.assertTrue(status["image"]["installed"])
                self.assertTrue(status["animation"]["installed"])
            importlib.reload(model_setup)

    def test_animatediff_manifest_is_pinned_and_ready_only_with_both_exact_files(self) -> None:
        expected_files = (
            "config.json",
            "diffusion_pytorch_model.fp16.safetensors",
        )
        with tempfile.TemporaryDirectory(prefix="anthill-animatediff-manifest-") as temporary:
            animation_path = Path(temporary) / "adapter"
            with patch.object(model_setup, "ANIMATION_PATH", animation_path), patch.object(
                model_setup, "ANIMATION_FILES", expected_files
            ):
                model = model_setup._model_manifest()["models"]["animation"]
                self.assertEqual(model["repository"], "guoyww/animatediff-motion-adapter-v1-5-2")
                self.assertEqual(model["revision"], "504374388860e52e206813f44f1ab60a3814152a")
                self.assertEqual(model["files"], list(expected_files))
                self.assertEqual(model_setup.ANIMATION_FILES, expected_files)
                self.assertEqual(pipeline.ANIMATION_FILES, expected_files)
                self.assertEqual(Path(model["local_path"]), animation_path)
                self.assertFalse(model["ready"])

                (animation_path / expected_files[0]).parent.mkdir(parents=True, exist_ok=True)
                (animation_path / expected_files[0]).write_bytes(b"config")
                self.assertFalse(model_setup._model_manifest()["models"]["animation"]["ready"])

                (animation_path / expected_files[1]).write_bytes(b"weights")
                self.assertTrue(model_setup._model_manifest()["models"]["animation"]["ready"])

    def test_animatediff_verify_only_does_not_download_files(self) -> None:
        with patch.object(model_setup, "_snapshot") as snapshot, patch.object(
            model_setup,
            "_write_manifest",
            return_value={
                "models": {
                    name: {"ready": True}
                    for name in ("image", "narration", "animation")
                }
            },
        ) as write_manifest, patch("sys.argv", ["model_setup", "--verify-only"]):
            self.assertEqual(model_setup.main(), 0)

        snapshot.assert_not_called()
        write_manifest.assert_called_once_with()

    def test_animatediff_download_command_uses_only_the_pinned_file_allow_list(self) -> None:
        with patch.object(model_setup, "_snapshot") as snapshot, patch.object(
            model_setup, "_write_manifest", return_value={"models": {}}
        ), patch("sys.argv", ["model_setup", "--download-animation"]):
            self.assertEqual(model_setup.main(), 0)

        snapshot.assert_called_once_with(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            "504374388860e52e206813f44f1ab60a3814152a",
            model_setup.ANIMATION_PATH,
            ("config.json", "diffusion_pytorch_model.fp16.safetensors"),
        )


if __name__ == "__main__":
    unittest.main()
