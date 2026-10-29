"""Command-line client for the local AntHill video factory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from local_client import (
    LocalApiError,
    check_readiness,
    download_video,
    generate_video,
    get_video_job,
    list_video_jobs,
    retry_video_job,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and manage videos with your local AntHill service.")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Queue a story for video generation")
    story = generate.add_mutually_exclusive_group(required=True)
    story.add_argument("--story", help="Story text; use - to read the story from standard input")
    story.add_argument("--story-file", type=Path, help="UTF-8 text file containing the story")
    generate.add_argument("--duration", type=int, choices=(30, 45, 60), default=30)
    _add_wait_options(generate)

    status = commands.add_parser("status", help="Show one job")
    status.add_argument("job_id")

    commands.add_parser("list", help="List recent jobs")

    retry = commands.add_parser("retry", help="Retry a failed job")
    retry.add_argument("job_id")
    _add_wait_options(retry)

    commands.add_parser("readiness", help="Check whether all selected local models are installed")
    return parser


def _add_wait_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true", help="Wait for completion and print final status")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum wait time in seconds (1-3600)")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Seconds between status checks (0.5-30)")
    parser.add_argument("--output", type=Path, help="Download the finished MP4 to this file path")


def _read_story(args: argparse.Namespace) -> str:
    if args.story_file is not None:
        try:
            return args.story_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise LocalApiError(f"Could not read the story file: {exc}") from None
    if args.story == "-":
        return sys.stdin.read(20_001)
    return args.story


def _validate_wait_options(args: argparse.Namespace) -> None:
    if not 1 <= args.timeout <= 3600:
        raise LocalApiError("Wait timeout must be from 1 to 3,600 seconds.")
    if not 0.5 <= args.poll_interval <= 30:
        raise LocalApiError("Poll interval must be from 0.5 to 30 seconds.")
    if args.output is not None and not args.wait:
        raise LocalApiError("Use --output together with --wait to download a finished video.")


def _wait_for_job(job: dict, args: argparse.Namespace) -> dict:
    deadline = time.monotonic() + args.timeout
    while job.get("status") in {"queued", "working"}:
        if time.monotonic() >= deadline:
            raise LocalApiError(f"Wait timed out; job {job.get('id')} is still {job.get('status')}.")
        time.sleep(args.poll_interval)
        job = get_video_job(job["id"])
    if job.get("status") == "error":
        raise LocalApiError(job.get("error", "Video generation failed."))
    if job.get("status") != "done":
        raise LocalApiError("The local service returned an unknown job status.")
    if args.output is not None:
        job["downloaded_to"] = str(download_video(job, args.output))
    return job


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            _validate_wait_options(args)
            job = generate_video(_read_story(args), args.duration)
            if args.wait:
                job = _wait_for_job(job, args)
        elif args.command == "status":
            job = get_video_job(args.job_id)
        elif args.command == "list":
            job = list_video_jobs()
        elif args.command == "readiness":
            job = check_readiness()
        elif args.command == "retry":
            _validate_wait_options(args)
            job = retry_video_job(args.job_id)
            if args.wait:
                job = _wait_for_job(job, args)
        else:
            parser.error("Unknown command")
            return 2
    except LocalApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(job, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
