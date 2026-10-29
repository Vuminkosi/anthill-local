"""Opt-in real FFmpeg motion-comic export smoke test using generated fixtures.

Set ANTHILL_RUN_FFMPEG_TESTS=1 in the app container to run. This does not
invoke or claim to verify any AI model.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

try:
    import pipeline
    from PIL import Image, ImageDraw
except ImportError as exc:
    pipeline = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


@unittest.skipUnless(os.getenv("ANTHILL_RUN_FFMPEG_TESTS") == "1", "opt-in CPU FFmpeg smoke test")
@unittest.skipIf(pipeline is None, f"pipeline dependencies unavailable: {_IMPORT_ERROR}")
class FfmpegMotionComicSmokeTests(unittest.TestCase):
    def test_fixture_scenes_export_portrait_mp4_with_audio_and_subtitles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anthill-ffmpeg-test-") as temporary:
            work = Path(temporary)
            images = []
            voices = []
            scenes = []
            for index in range(6):
                image_path = work / f"scene_{index}.png"
                image = Image.new("RGB", (360, 640), (30 + index * 20, 70, 120))
                ImageDraw.Draw(image).ellipse((80, 120, 280, 320), fill=(240, 180, 90))
                image.save(image_path)
                images.append(image_path)

                voice_path = work / f"voice_{index}.wav"
                with wave.open(str(voice_path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(24000)
                    audio.writeframes(b"\0\0" * (24000 // 2))
                voices.append(voice_path)
                scenes.append({
                    "caption": f"Story beat {index + 1}",
                    "narration": f"The kindness gave him hope, scene {index + 1}.",
                })

            output = work / "fixture.mp4"
            pipeline.render(scenes, images, voices, work, output, lambda *_args: None, seconds=30)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)
            self.assertIn("The kindness gave him hope", (work / "subtitles.srt").read_text(encoding="utf-8"))
            probe = pipeline.command([
                "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,codec_name",
                "-of", "json", str(output),
            ], timeout=30)
            import json
            streams = json.loads(probe.stdout)["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")
            self.assertEqual((video["width"], video["height"]), (1080, 1920))
            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(audio["codec_name"], "aac")
            self.assertAlmostEqual(pipeline.duration(output), 30, delta=0.3)


if __name__ == "__main__":
    unittest.main()
