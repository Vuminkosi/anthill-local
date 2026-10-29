"""CLI and local-client contract tests; calls are stubbed and make no inference requests."""

from __future__ import annotations

import io
from email.message import Message
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import cli
import local_client


class LocalClientTests(unittest.TestCase):
    def test_remote_endpoint_is_rejected_before_network_access(self) -> None:
        with patch.dict("os.environ", {"ANTHILL_URL": "https://example.com"}), patch.object(
            local_client.urllib.request, "build_opener"
        ) as build_opener:
            with self.assertRaisesRegex(local_client.LocalApiError, "local AntHill host"):
                local_client.check_readiness()
        build_opener.assert_not_called()

    def test_userinfo_and_remote_port_are_rejected(self) -> None:
        for value in ("http://user:pass@localhost:8080", "http://localhost:0", "http://192.168.1.20:8080"):
            with self.subTest(value=value), patch.dict("os.environ", {"ANTHILL_URL": value}):
                with self.assertRaises(local_client.LocalApiError):
                    local_client.check_readiness()

    def test_arbitrary_loopback_port_is_allowed_without_a_cloud_host(self) -> None:
        with patch.dict("os.environ", {"ANTHILL_URL": "http://127.0.0.1:8090"}):
            self.assertEqual(local_client._base_url(), "http://127.0.0.1:8090")

    def test_default_endpoint_uses_the_selected_local_port(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(local_client._base_url(), "http://127.0.0.1:8090")

    def test_video_download_requires_matching_local_job_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "video.mp4"
            job = {"id": "0123456789ab", "video": "https://example.com/video.mp4"}
            with self.assertRaisesRegex(local_client.LocalApiError, "not a local AntHill"):
                local_client.download_video(job, destination)
            self.assertFalse(destination.exists())

    def test_streaming_download_preserves_existing_partial_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "story.mp4"
            sentinel = target.with_name(target.name + ".partial")
            sentinel.write_bytes(b"user-owned partial sentinel")
            data = b"\x00\x00\x00\x18ftypmp42" + b"fixture payload"
            headers = Message()
            headers["Content-Type"] = "video/mp4"
            response = unittest.mock.MagicMock()
            response.headers = headers
            response.__enter__.return_value = response
            stream = io.BytesIO(data)
            response.read.side_effect = stream.read
            opener = unittest.mock.MagicMock()
            opener.open.return_value = response
            job = {"id": "0123456789ab", "video": "/videos/AntHill_0123456789ab.mp4"}

            with patch.dict("os.environ", {"ANTHILL_URL": "http://127.0.0.1:8080"}), patch.object(
                local_client.urllib.request, "build_opener", return_value=opener
            ):
                local_client.download_video(job, target)

            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(sentinel.read_bytes(), b"user-owned partial sentinel")
            self.assertEqual(list(Path(temporary).glob(".story.mp4.*.partial")), [])

    def test_malformed_stream_preserves_partial_and_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "story.mp4"
            sentinel = target.with_name(target.name + ".partial")
            sentinel.write_bytes(b"user-owned partial sentinel")
            headers = Message()
            headers["Content-Type"] = "video/mp4"
            response = unittest.mock.MagicMock()
            response.headers = headers
            response.__enter__.return_value = response
            response.read.side_effect = io.BytesIO(b"not-an-mp4").read
            opener = unittest.mock.MagicMock()
            opener.open.return_value = response
            job = {"id": "0123456789ab", "video": "/videos/AntHill_0123456789ab.mp4"}

            with patch.dict("os.environ", {"ANTHILL_URL": "http://127.0.0.1:8080"}), patch.object(
                local_client.urllib.request, "build_opener", return_value=opener
            ):
                with self.assertRaises(local_client.LocalApiError):
                    local_client.download_video(job, target)

            self.assertFalse(target.exists())
            self.assertEqual(sentinel.read_bytes(), b"user-owned partial sentinel")
            self.assertEqual(list(Path(temporary).glob(".story.mp4.*.partial")), [])

    def test_download_does_not_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "story.mp4"
            target.write_bytes(b"user-owned video")
            job = {"id": "0123456789ab", "video": "/videos/AntHill_0123456789ab.mp4"}
            with patch.object(local_client.urllib.request, "build_opener") as build_opener:
                with self.assertRaisesRegex(local_client.LocalApiError, "already exists"):
                    local_client.download_video(job, target)
            self.assertEqual(target.read_bytes(), b"user-owned video")
            build_opener.assert_not_called()


class CliTests(unittest.TestCase):
    def test_generate_submits_selected_duration_and_prints_job(self) -> None:
        story = "A kind waitress paid for a nervous man's breakfast and changed his morning."
        expected = {"id": "0123456789ab", "status": "queued", "duration_seconds": 45}
        stdout = io.StringIO()
        with patch.object(cli, "generate_video", return_value=expected) as generate, redirect_stdout(stdout):
            result = cli.main(["generate", "--story", story, "--duration", "45"])
        self.assertEqual(result, 0)
        generate.assert_called_once_with(story, 45)
        self.assertEqual(__import__("json").loads(stdout.getvalue()), expected)

    def test_generate_can_read_story_from_file(self) -> None:
        story = "A small act of kindness helped a lonely visitor feel welcome."
        with tempfile.TemporaryDirectory() as temporary:
            story_file = Path(temporary) / "story.txt"
            story_file.write_text(story, encoding="utf-8")
            with patch.object(cli, "generate_video", return_value={"status": "queued"}) as generate, redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["generate", "--story-file", str(story_file)]), 0)
            generate.assert_called_once_with(story, 30)

    def test_invalid_output_without_wait_fails_before_submission(self) -> None:
        stderr = io.StringIO()
        with patch.object(cli, "generate_video") as generate, redirect_stderr(stderr):
            result = cli.main([
                "generate", "--story", "A sufficiently long story for validation.", "--output", "video.mp4"
            ])
        self.assertEqual(result, 1)
        self.assertIn("together with --wait", stderr.getvalue())
        generate.assert_not_called()

    def test_status_reads_job_and_does_not_generate(self) -> None:
        expected = {"id": "0123456789ab", "status": "working", "percent": 37}
        stdout = io.StringIO()
        with patch.object(cli, "get_video_job", return_value=expected) as get_job, redirect_stdout(stdout):
            result = cli.main(["status", "0123456789ab"])
        self.assertEqual(result, 0)
        get_job.assert_called_once_with("0123456789ab")
        self.assertEqual(__import__("json").loads(stdout.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
