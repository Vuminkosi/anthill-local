"""Local story-to-motion-comic video pipeline.

Image generation deliberately uses short-lived, CPU-offloaded diffusion pipelines
to fit consumer GPUs. It creates animated stills, not full video diffusion.
"""

from __future__ import annotations

import json
import gc
import inspect
import logging
import os
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from model_setup import (
    IMAGE_PATH,
    KOKORO_CONFIG_FILE,
    KOKORO_MODEL_FILE,
    KOKORO_VOICE_FILE,
    VOICE_NAME as KOKORO_VOICE,
)

ROOT = Path(os.getenv("ANTHILL_DATA", "/data"))
OUT = ROOT / "output"
MODELS = ROOT / "models"
KOKORO_REPO = "hexgrad/Kokoro-82M"
LINUX_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
WINDOWS_FONT = Path("C:/Windows/Fonts/arialbd.ttf")
APPROVED_IMAGE_MODEL = "Lykon/dreamshaper-8"
APPROVED_IMAGE_REVISION = "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917"
SCENE_COUNT = 6
logger = logging.getLogger("anthill.pipeline")
Progress = Callable[..., None]


def _progress(callback: Progress, message: str, percent: int) -> None:
    """Keep compatibility with the original one-argument Flask callback."""
    try:
        parameters = inspect.signature(callback).parameters.values()
        accepts_two = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        accepts_two = accepts_two or len(inspect.signature(callback).parameters) >= 2
    except (TypeError, ValueError):
        accepts_two = False
    if accepts_two:
        callback(message, percent)
    else:
        callback(message)


def command(args: list[str], stdin: str | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            args,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout} seconds: {args[0]}") from exc
    if result.returncode:
        logger.error("Command failed (%s): %s", result.returncode, args[0])
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-1200:]}")
    return result


def _retry(operation: Callable[[], object], description: str, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt == attempts - 1:
                raise
            wait = min(2 ** attempt, 8)
            logger.warning("%s failed; retry %s/%s in %ss", description, attempt + 1, attempts - 1, wait)
            time.sleep(wait)


def local_ollama_base() -> str:
    base = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
    parsed = urlsplit(base)
    local_hosts = {"ollama", "localhost", "127.0.0.1", "::1"}
    try:
        valid_port = parsed.port in (None, 11434)
    except ValueError:
        valid_port = False
    if (
        parsed.scheme != "http"
        or parsed.hostname not in local_hosts
        or not valid_port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Storyboards can only use a local Ollama service; external inference URLs are blocked")
    return base


def storyboard(story: str, duration_seconds: int = 30) -> dict:
    target_words = round(duration_seconds * 2.15)
    prompt = (
        "Return ONLY valid JSON with exactly these keys: character_bible (string), narration (string), scenes (array). "
        f"Create exactly {SCENE_COUNT} scenes for a {duration_seconds}-second emotional story. "
        f"The narration must be about {target_words} English words, split naturally across six scenes. "
        "Each scene must contain image, caption, and narration strings. Each narration must be a contiguous spoken part "
        "of the complete story narration, with no extra claims. Captions are short story beats, 3-9 words. "
        "character_bible must describe each recurring character's face, age, hair, clothing and distinguishing details. "
        "Every image description must repeat the character details from character_bible and describe a single coherent "
        "3D animated cartoon film still, expressive emotion, cinematic lighting, vertical 9:16 framing, no text. "
        "Follow the supplied story faithfully and do not invent facts. Story:\n" + story[:6000]
    )
    ollama_base = local_ollama_base()
    url = ollama_base + "/api/generate"

    def request_plan() -> dict:
        # Ignore ambient HTTP proxy settings and reject redirects so story text
        # cannot leave the host through a proxy or an Ollama redirect response.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                url,
                json={
                    "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.35, "num_predict": 2200, "num_ctx": 4096, "num_gpu": 0},
                },
                timeout=(10, 360),
                allow_redirects=False,
            )
        if 300 <= response.status_code < 400:
            raise RuntimeError("Local Ollama returned a redirect; external redirects are blocked")
        response.raise_for_status()
        obj = json.loads(response.json()["response"])
        scenes = obj.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != SCENE_COUNT:
            raise ValueError("Storyboard must contain exactly six scenes")
        for field in ("character_bible", "narration"):
            if not isinstance(obj.get(field), str) or not obj[field].strip():
                raise ValueError(f"Storyboard is missing {field}")
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"Scene {index} must be an object")
            for field in ("image", "caption", "narration"):
                if not isinstance(scene.get(field), str) or not scene[field].strip():
                    raise ValueError(f"Scene {index} is missing {field}")
        # Keep transcript and scene speech aligned; reject obviously malformed plans.
        scene_text = " ".join(scene["narration"].strip() for scene in scenes)
        if len(scene_text.split()) < duration_seconds:
            raise ValueError("Storyboard narration is too short for the selected duration")
        return obj

    try:
        return _retry(request_plan, "Ollama storyboard", attempts=4)
    except Exception as exc:
        raise RuntimeError(
            "Local Ollama storyboard failed. Confirm the Ollama service is running and the configured model is installed."
        ) from exc


_kokoro_pipeline = None


def _get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    if not all(path.is_file() for path in (KOKORO_MODEL_FILE, KOKORO_VOICE_FILE, KOKORO_CONFIG_FILE)):
        raise RuntimeError(
            "Kokoro speech files are missing. Run `python -m model_setup --download-voice` inside the app container. "
            "Runtime narration never downloads models or voices."
        )

    try:
        import torch
        from kokoro import KModel, KPipeline

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
        model = KModel(
            repo_id=KOKORO_REPO,
            config=str(KOKORO_CONFIG_FILE),
            model=str(KOKORO_MODEL_FILE),
        ).to("cpu").eval()
        _kokoro_pipeline = KPipeline(
            lang_code=KOKORO_VOICE[0],
            repo_id=KOKORO_REPO,
            model=model,
            trf=False,
            device="cpu",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize local Kokoro narration. Check the installed `kokoro` and `misaki[en]` packages "
            "and the cached model/voice files."
        ) from exc
    return _kokoro_pipeline


def narrate(text: str, path: Path) -> None:
    try:
        pipeline = _get_kokoro_pipeline()
        audio_chunks = [
            result.audio.detach().cpu().numpy()
            for result in pipeline(
                text,
                voice=str(KOKORO_VOICE_FILE),
            )
            if result.audio is not None
        ]
        if not audio_chunks:
            raise RuntimeError("Kokoro returned no audio")
        audio = np.concatenate(audio_chunks).astype(np.float32)
        temporary = path.with_suffix(".partial.wav")
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        temporary.replace(path)
        if path.stat().st_size < 44:
            raise RuntimeError("Kokoro produced an invalid narration WAV")
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("Kokoro speech files are missing"):
            raise
        raise RuntimeError("Local Kokoro narration failed; inspect the container log for details.") from exc


def _local_model_path() -> Path:
    if os.getenv("IMAGE_MODEL", APPROVED_IMAGE_MODEL) != APPROVED_IMAGE_MODEL:
        raise RuntimeError("Only the locally approved DreamShaper 8 FP16 model is enabled")
    return IMAGE_PATH


def _diffusion_pipeline(model_path: Path, image_to_image: bool, torch):
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    pipeline_type = AutoPipelineForImage2Image if image_to_image else AutoPipelineForText2Image
    pipe = pipeline_type.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        low_cpu_mem_usage=True,
    )
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    return pipe


def make_images(scenes: list[dict], character_bible: str, work_dir: Path, progress: Progress) -> list[Path]:
    import torch

    width = int(os.getenv("IMAGE_WIDTH", "512"))
    height = int(os.getenv("IMAGE_HEIGHT", "896"))
    if width > 640 or height > 1152 or width % 8 or height % 8:
        raise ValueError("Image dimensions must be divisible by 8 and no larger than 640x1152 for the low-VRAM profile")
    model_path = _local_model_path()
    required_weight = model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    if not required_weight.is_file():
        raise RuntimeError(
            "The approved DreamShaper 8 FP16 model is missing. Run `python -m model_setup --model dreamshaper-8` "
            "inside the app container, then retry. Runtime generation never downloads model files."
        )
    logger.info("Loading local image model %s (%sx%s, CPU offload)", model_path, width, height)
    _progress(progress, "Loading the installed local image model...", 20)
    paths: list[Path] = []
    reference: Image.Image | None = None
    # DreamShaper is a standard Stable Diffusion 1.5 checkpoint. Load text-to-image
    # once for the reference shot, then load image-to-image once for all later shots.
    # Each pipeline is released before the next is constructed to bound VRAM use.
    try:
        for image_to_image in (False, True):
            first_index = 1 if image_to_image else 0
            last_index = len(scenes) if image_to_image else min(1, len(scenes))
            if first_index >= last_index:
                continue
            pipe = _diffusion_pipeline(model_path, image_to_image, torch)
            try:
                for index in range(first_index, last_index):
                    scene = scenes[index]
                    _progress(progress, f"Generating cartoon scene {index + 1}/{len(scenes)}...", 22 + index * 7)
                    # Scene content leads, so CLIP's 77-token input budget sees the
                    # important composition before compact identity and style cues.
                    prompt = (
                        scene["image"][:420]
                        + ", consistent character: " + character_bible[:180]
                        + ", expressive 3D cartoon film still, warm cinematic light, portrait 9:16, no text"
                    )
                    output = work_dir / f"image_{index:02d}.png"
                    reference_image = reference

                    def generate(current_width: int, current_height: int) -> Image.Image:
                        generator = torch.Generator(device="cpu").manual_seed(
                            int(os.getenv("IMAGE_SEED", "4242")) + index
                        )
                        if image_to_image:
                            result = pipe(
                                prompt=prompt,
                                image=reference_image.resize(
                                    (current_width, current_height), Image.Resampling.LANCZOS
                                ),
                                strength=float(os.getenv("IMAGE_REFERENCE_STRENGTH", "0.62")),
                                guidance_scale=float(os.getenv("IMAGE_GUIDANCE", "7.0")),
                                num_inference_steps=int(os.getenv("IMAGE_STEPS", "24")),
                                generator=generator,
                            )
                        else:
                            result = pipe(
                                prompt=prompt,
                                height=current_height,
                                width=current_width,
                                num_inference_steps=int(os.getenv("IMAGE_STEPS", "24")),
                                guidance_scale=float(os.getenv("IMAGE_GUIDANCE", "7.0")),
                                generator=generator,
                            )
                        return result.images[0]

                    try:
                        try:
                            image = generate(width, height)
                        except RuntimeError as exc:
                            if "out of memory" not in str(exc).lower() or width <= 384 or height <= 672:
                                raise
                            logger.warning("Image generation ran out of memory; retrying once at 384x672")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            image = generate(384, 672)
                        image.save(output)
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if reference is None:
                        reference = image.copy()
                    paths.append(output)
            finally:
                del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return paths
    except OSError as exc:
        raise RuntimeError(
            "The local DreamShaper 8 FP16 model could not be loaded. Re-run the approved model setup command and "
            "check that this container's PyTorch build supports the installed NVIDIA GPU."
        ) from exc


def card(image: Image.Image, caption: str, path: Path) -> None:
    canvas = image.convert("RGB").resize((1080, 1920), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.truetype(str(_caption_font_path()), 52)
    words = str(caption).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] > 960 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    lines = lines[:3]
    line_height = 72
    height = line_height * len(lines)
    # Keep story-beat cards near the top so they do not collide with lower
    # third narration subtitles and remain clear of the common social overlays.
    y = 360
    draw.rounded_rectangle((48, y - 28, 1032, y + height + 22), radius=26, fill=(0, 0, 0, 178))
    for line_index, line in enumerate(lines):
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(((1080 - box[2]) / 2, y + line_index * line_height), line, font=font,
                  fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _caption_font_path() -> Path:
    font_path = os.getenv("ANTHILL_FONT")
    if font_path:
        selected_font = Path(font_path)
        if not selected_font.is_file():
            raise RuntimeError(f"Configured caption font was not found: {selected_font}")
    else:
        selected_font = next(
            (candidate for candidate in (LINUX_FONT, WINDOWS_FONT) if candidate.is_file()),
            None,
        )
        if selected_font is None:
            raise RuntimeError(
                "No caption font was found. Set ANTHILL_FONT to a readable .ttf font file "
                "(for example C:/Windows/Fonts/arialbd.ttf)."
            )
    try:
        ImageFont.truetype(str(selected_font), 52)
    except OSError as exc:
        raise RuntimeError(f"Caption font could not be loaded: {selected_font}") from exc
    return selected_font


def duration(path: Path) -> float:
    result = command([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], timeout=30)
    return float(result.stdout.strip())


def music(path: Path, seconds: int, sample_rate: int = 24000) -> None:
    # Original synthesized ambient bed; no third-party recording or API.
    sample_count = seconds * sample_rate
    samples = np.arange(sample_count, dtype=np.float64) / sample_rate
    base = np.zeros_like(samples)
    progression = ((220, 261.63, 329.63), (196, 246.94, 293.66),
                   (174.61, 220, 261.63), (196, 246.94, 329.63))
    for index, notes in enumerate(progression):
        start = index * seconds / len(progression)
        end = (index + 1) * seconds / len(progression)
        selected = (samples >= start) & (samples < end)
        relative = samples[selected] - start
        segment_length = end - start
        envelope = np.minimum(1, relative / 1.5) * np.minimum(1, (segment_length - relative) / 2)
        base[selected] = sum(np.sin(2 * np.pi * frequency * relative) for frequency in notes) * envelope / len(notes)
    pcm = (base * 1200).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_subtitles(scenes: list[dict], seconds: int, path: Path) -> None:
    scene_seconds = seconds / len(scenes)
    rows = []
    for index, scene in enumerate(scenes):
        start = index * scene_seconds
        end = (index + 1) * scene_seconds
        text = re.sub(r"\s+", " ", scene["narration"]).strip()
        rows.append(f"{index + 1}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(rows), encoding="utf-8")


def _escape_subtitle_filter(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return (
        f"subtitles='{value}':force_style='PlayResX=1080,PlayResY=1920,FontName=DejaVu Sans,"
        "FontSize=54,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,"
        "Shadow=1,Alignment=2,MarginL=80,MarginR=80,MarginV=300'"
    )


def render(scenes: list[dict], images: list[Path], voices: list[Path], work_dir: Path,
           final_path: Path, progress: Progress, seconds: int = 30) -> None:
    clips = []
    scene_frames = [seconds * 30 // len(scenes)] * len(scenes)
    for index in range(seconds * 30 % len(scenes)):
        scene_frames[index] += 1
    for index, (scene, image_path, frame_count) in enumerate(zip(scenes, images, scene_frames)):
        _progress(progress, f"Animating scene {index + 1}/{len(scenes)}...", 70 + index * 3)
        captioned = work_dir / f"caption_{index}.png"
        card(Image.open(image_path), scene["caption"], captioned)
        clip = work_dir / f"part_{index:02d}.mp4"
        fps = 30
        zoom = f"zoompan=z='min(1+on*0.00035,1.07)':d=1:s=1080x1920:fps={fps}"
        command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(captioned),
            "-vf", f"{zoom},format=yuv420p", "-frames:v", str(frame_count), "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "22", str(clip),
        ], timeout=300)
        clips.append(clip)

        # Keep scene narration within its scene; gently time-stretch and pad as needed.
        voice_duration = duration(voices[index])
        scene_duration = frame_count / fps
        tempo = voice_duration / scene_duration
        tempo_filters = []
        while tempo > 2.0:
            tempo_filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            tempo_filters.append("atempo=0.5")
            tempo /= 0.5
        tempo_filters.append(f"atempo={tempo:.5f}")
        audio_part = work_dir / f"voice_{index:02d}.wav"
        command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(voices[index]),
            "-af", f"{','.join(tempo_filters)},apad,atrim=duration={scene_duration:.5f}",
            "-ar", "24000", "-ac", "1", str(audio_part),
        ], timeout=120)

    video_list = work_dir / "video_segments.txt"
    video_list.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips), encoding="utf-8")
    audio_list = work_dir / "audio_segments.txt"
    audio_list.write_text("".join(f"file '{(work_dir / f'voice_{i:02d}.wav').as_posix()}'\n" for i in range(len(scenes))), encoding="utf-8")
    silent = work_dir / "silent.mp4"
    narration = work_dir / "narration.wav"
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(video_list), "-c", "copy", str(silent)])
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list), "-c", "copy", str(narration)])
    bed = work_dir / "ambient.wav"
    music(bed, seconds)
    subtitles = work_dir / "subtitles.srt"
    write_subtitles(scenes, seconds, subtitles)
    temporary = final_path.with_suffix(".partial.mp4")
    _progress(progress, "Mixing narration and music, then exporting a portrait MP4...", 92)
    filters = f"[0:v]{_escape_subtitle_filter(subtitles)}[video];[2:a]volume=0.18[music];[1:a][music]amix=inputs=2:duration=first:normalize=0[audio]"
    command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(narration),
        "-i", str(bed), "-filter_complex", filters, "-map", "[video]", "-map", "[audio]", "-t", str(seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(temporary),
    ], timeout=900)
    if duration(temporary) < seconds - 0.25:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("The exported video was shorter than requested")
    temporary.replace(final_path)


def run_job(job_id: str, story: str, progress: Progress, duration_seconds: int = 30) -> Path:
    if duration_seconds not in (30, 45, 60):
        raise ValueError("Video duration must be 30, 45, or 60 seconds")
    if not isinstance(story, str) or len(story.strip()) < 25:
        raise ValueError("Please provide a story with at least 25 characters")
    OUT.mkdir(parents=True, exist_ok=True)
    work_dir = OUT / ("working_" + re.sub(r"[^A-Za-z0-9_-]", "", job_id)[:40])
    work_dir.mkdir(exist_ok=True)
    logger.info("Starting local motion-comic job %s (%ss)", job_id, duration_seconds)
    _progress(progress, "Writing a storyboard with local Ollama...", 5)
    plan = storyboard(story, duration_seconds)
    (work_dir / "storyboard.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    _progress(progress, "Generating expressive local narration...", 12)
    voices = []
    for index, scene in enumerate(plan["scenes"]):
        voice_path = work_dir / f"narration_{index:02d}.wav"
        narrate(scene["narration"], voice_path)
        voices.append(voice_path)
        _progress(progress, f"Recording narration {index + 1}/{len(plan['scenes'])}...", 12 + index)
    images = make_images(plan["scenes"], plan["character_bible"], work_dir, progress)
    final = OUT / f"AntHill_{re.sub(r'[^A-Za-z0-9_-]', '', job_id)[:40]}.mp4"
    render(plan["scenes"], images, voices, work_dir, final, progress, duration_seconds)
    logger.info("Completed local motion-comic job %s: %s", job_id, final.name)
    return final
