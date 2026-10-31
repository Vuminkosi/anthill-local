"""Explicit, allow-listed setup for the locally approved AntHill models.

This command downloads only model files explicitly approved by the user. Runtime
inference is local-only and never calls this module automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(os.getenv("ANTHILL_DATA", "/data"))
MODELS_ROOT = DATA_ROOT / "models"
LTX_VIDEO_REPO = "Lightricks/LTX-Video"
LTX_VIDEO_REVISION = "19560f8b59a58a0431baf39c78cd8a60a86b9c33"
LTX_TEXT_ENCODER_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"
LTX_TEXT_ENCODER_REVISION = "dcc28e445c2cc7310d53d438e50a0edd7bec3407"
LTX_CONFIG_REPO = "Lightricks/ltx-video-distilled"
LTX_CONFIG_REVISION = "9c24b4737c3c177bba269fc57039a02247f7f2fe"
LTX_PATH = MODELS_ROOT / "ltx-video"
LTX_CHECKPOINT_PATH = LTX_PATH / "checkpoint"
LTX_TEXT_ENCODER_PATH = LTX_PATH / "text-encoder"
LTX_CONFIG_PATH = LTX_PATH / "config"
LTX_CHECKPOINT_FILES = (
    "ltxv-2b-0.9.8-distilled.safetensors",
    "ltxv-spatial-upscaler-0.9.8.safetensors",
    "LTX-Video-Open-Weights-License-0.X.txt",
)
LTX_TEXT_ENCODER_FILES = (
    "text_encoder/config.json",
    "text_encoder/model-*.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/added_tokens.json",
)
LTX_CONFIG_FILE = "configs/ltxv-2b-0.9.8-distilled.yaml"

def ltx_video_assets_ready() -> bool:
    return (
        all((LTX_CHECKPOINT_PATH / name).is_file() for name in LTX_CHECKPOINT_FILES)
        and (LTX_TEXT_ENCODER_PATH / "text_encoder/config.json").is_file()
        and len(list((LTX_TEXT_ENCODER_PATH / "text_encoder").glob("model-*.safetensors"))) >= 2
        and all(
            (LTX_TEXT_ENCODER_PATH / name).is_file()
            for name in (
                "text_encoder/model.safetensors.index.json",
                "tokenizer/spiece.model",
                "tokenizer/tokenizer_config.json",
                "tokenizer/special_tokens_map.json",
                "tokenizer/added_tokens.json",
            )
        )
        and (LTX_CONFIG_PATH / LTX_CONFIG_FILE).is_file()
    )


def _configured_model_path(variable: str, default_path: Path) -> Path:
    value = os.getenv(variable)
    if not value:
        return default_path
    configured = Path(value)
    return configured if configured.is_absolute() else MODELS_ROOT / configured


VOICE_REPO = "hexgrad/Kokoro-82M"
VOICE_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
VOICE_NAME = os.getenv("KOKORO_VOICE", "af_heart")


KOKORO_MODEL_FILE = _configured_model_path(
    "KOKORO_MODEL_PATH", MODELS_ROOT / "voices" / "kokoro" / "kokoro-v1_0.pth"
)
KOKORO_MODEL_DIR = KOKORO_MODEL_FILE.parent
KOKORO_CONFIG_FILE = KOKORO_MODEL_DIR / "config.json"
KOKORO_VOICE_FILE = _configured_model_path(
    "KOKORO_VOICE_PATH", KOKORO_MODEL_DIR / "voices" / f"{VOICE_NAME}.pt"
)
try:
    VOICE_PATH = Path(os.path.commonpath((KOKORO_MODEL_DIR, KOKORO_VOICE_FILE.parent)))
except ValueError:
    # On Windows, unrelated drive letters have no shared path. The absolute
    # voice entry below remains valid when joined to this model directory.
    VOICE_PATH = KOKORO_MODEL_DIR
logger = logging.getLogger("anthill.model_setup")

VOICE_DOWNLOAD_FILES = ("config.json", "kokoro-v1_0.pth", f"voices/{VOICE_NAME}.pt")
try:
    _voice_manifest_path = str(KOKORO_VOICE_FILE.relative_to(VOICE_PATH))
except ValueError:
    _voice_manifest_path = str(KOKORO_VOICE_FILE)
VOICE_FILES = (
    str(KOKORO_CONFIG_FILE.relative_to(VOICE_PATH)),
    str(KOKORO_MODEL_FILE.relative_to(VOICE_PATH)),
    _voice_manifest_path,
)
def _snapshot(
    repo_id: str,
    revision: str,
    destination: Path,
    files: tuple[str, ...],
    *,
    repo_type: str = "model",
) -> None:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading approved model %s at revision %s", repo_id, revision)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=list(files),
        max_workers=2,
    )
    missing = []
    for name in files:
        candidates = list(destination.glob(name)) if any(character in name for character in "*?[") else [destination / name]
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(name)
    if missing:
        raise RuntimeError(f"Model download is incomplete; missing files: {', '.join(missing)}")


def _model_manifest() -> dict:
    voice_ready = all((VOICE_PATH / name).is_file() for name in VOICE_FILES)
    ltx_video_ready = ltx_video_assets_ready()
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "narration": {
                "repository": VOICE_REPO,
                "revision": VOICE_REVISION,
                "voice": VOICE_NAME,
                "local_path": str(VOICE_PATH),
                "ready": voice_ready,
                "files": list(VOICE_FILES),
            },
            "video": {
                "repository": LTX_VIDEO_REPO,
                "revision": LTX_VIDEO_REVISION,
                "variant": "2B distilled BF16 0.9.8",
                "local_path": str(LTX_PATH),
                "ready": ltx_video_ready,
                "checkpoint_files": list(LTX_CHECKPOINT_FILES),
                "text_encoder_repository": LTX_TEXT_ENCODER_REPO,
                "text_encoder_revision": LTX_TEXT_ENCODER_REVISION,
                "config_file": LTX_CONFIG_FILE,
            },
            "storyboard": {
                "provider": "local Ollama",
                "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                "ready": None,
                "note": "Ollama model availability is checked by the Ollama service, not this downloader.",
            },
        },
    }


def _write_manifest() -> dict:
    manifest = _model_manifest()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = MODELS_ROOT / "model_manifest.json"
    temporary = destination.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download only user-approved AntHill local model files.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--download-voice", action="store_true", help="Download Kokoro 82M and the af_heart voice")
    selection.add_argument("--download-ltx-video", action="store_true", help="Download LTX-Video 2B distilled BF16 weights and text encoder")
    selection.add_argument("--all-approved", action="store_true", help="Download LTX-Video and Kokoro files")
    parser.add_argument("--verify-only", action="store_true", help="Write readiness data without downloading model files")
    args = parser.parse_args()

    if args.verify_only and (args.download_voice or args.download_ltx_video or args.all_approved):
        parser.error("--verify-only cannot be combined with download options")
    if not args.verify_only and not (args.download_voice or args.download_ltx_video or args.all_approved):
        parser.error("Select --download-voice, --download-ltx-video, --all-approved, or --verify-only")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if not args.verify_only:
            # Compose sets offline mode for inference; downloads happen only after
            # an explicit setup flag, so temporarily permit this one CLI process.
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["HUGGINGFACE_HUB_OFFLINE"] = "0"
        if args.download_voice or args.all_approved:
            _snapshot(VOICE_REPO, VOICE_REVISION, KOKORO_MODEL_DIR, VOICE_DOWNLOAD_FILES)
            default_voice = KOKORO_MODEL_DIR / f"voices/{VOICE_NAME}.pt"
            if KOKORO_VOICE_FILE.resolve() != default_voice.resolve():
                KOKORO_VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
                temporary_voice = KOKORO_VOICE_FILE.with_suffix(KOKORO_VOICE_FILE.suffix + ".partial")
                shutil.copyfile(default_voice, temporary_voice)
                temporary_voice.replace(KOKORO_VOICE_FILE)
            default_weight = KOKORO_MODEL_DIR / "kokoro-v1_0.pth"
            if KOKORO_MODEL_FILE.resolve() != default_weight.resolve():
                temporary_weight = KOKORO_MODEL_FILE.with_suffix(KOKORO_MODEL_FILE.suffix + ".partial")
                shutil.copyfile(default_weight, temporary_weight)
                temporary_weight.replace(KOKORO_MODEL_FILE)
        if args.download_ltx_video or args.all_approved:
            _snapshot(LTX_VIDEO_REPO, LTX_VIDEO_REVISION, LTX_CHECKPOINT_PATH, LTX_CHECKPOINT_FILES)
            _snapshot(LTX_TEXT_ENCODER_REPO, LTX_TEXT_ENCODER_REVISION, LTX_TEXT_ENCODER_PATH, LTX_TEXT_ENCODER_FILES)
            _snapshot(
                LTX_CONFIG_REPO,
                LTX_CONFIG_REVISION,
                LTX_CONFIG_PATH,
                (LTX_CONFIG_FILE,),
                repo_type="space",
            )
        manifest = _write_manifest()
    except Exception as exc:
        logger.exception("Approved model setup failed")
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2))
    if args.verify_only and not all(
        model["ready"] is True
        for name, model in manifest["models"].items()
            if name in ("video", "narration")
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
