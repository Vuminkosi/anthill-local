# AntHill Local Video Factory

AntHill turns a short story into a locally generated vertical video. Paste a story, choose **30, 45, or 60 seconds**, and select **Generate**. The output is an H.264/AAC MP4 at **1080 × 1920** with continuous LTX-generated live-action-style scene footage, spoken narration, burned-in subtitles, and locally synthesized ambient music.

The pipeline creates a six-scene storyboard with local Ollama, generates a separate continuous video clip for each scene with LTX-Video 2B distilled, and assembles those clips with Kokoro narration, subtitles, and music. Generated footage is enlarged for the 1080 × 1920 export; character identity and visual quality can vary between scenes.

All story planning, image generation, narration, and video assembly run locally in Docker. No host Python or Ollama installation is required. Ollama cloud features are disabled. Model downloads are needed for initial installation; inference does not require paid APIs or send prompts to a hosted model service. Docker images and Python packages also need internet access when first installed.

**Validation status:** LTX source integration is in progress. A complete LTX GPU video and end-to-end story-to-MP4 are **NOT VERIFIED** until the app finishes a generation on the target machine.

## Requirements

- Windows 11, a current NVIDIA driver, and Docker Desktop using its WSL2 engine.
- An NVIDIA RTX GPU with 8 GB VRAM is the target profile. LTX uses sequential CPU offload and a CPU-resident text encoder; Ollama and Kokoro run on CPU.
- 32 GB system RAM is recommended, and the Docker Desktop WSL memory limit must be high enough for LTX model initialization. A 16 GB WSL cap has not yet completed initialization.
- Plan for about 28 GB of model files, plus Docker images, Python dependencies, caches, and temporary render files. Keep more space available during initial downloads.

The setup downloads the pinned LTX-Video 2B distilled checkpoint, PixArt T5 text encoder, spatial upscaler, and configuration to the Docker volume on D:, plus Kokoro voice assets. Ollama retains its own model volume. Runtime generation uses installed files only and does not fetch model weights.

The active video model is the pinned LTX-Video 0.9.8 2B distilled multi-scale pipeline. Checkpoint assets persist in the `ltx_video_models` Docker volume on D:. DreamShaper and AnimateDiff are no longer part of model setup or the active generation path.

## Install and start

Install the current NVIDIA driver and Docker Desktop, enable **Use the WSL 2 based engine**, then start Docker Desktop. In PowerShell, check that Docker can see the GPU before building the app:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

From the folder containing `compose.yaml`, verify the Compose file:

```powershell
docker compose config --quiet
```

From the folder containing `compose.yaml`, run this command. Compose builds the app, starts Ollama, downloads the configured Llama 3.2 model, and installs LTX and Kokoro assets. Initial model downloads need internet access; later starts reuse the named LTX and Ollama volumes and the voice files under `models`:

```powershell
docker compose up --build -d
```

The app binds to **http://localhost:8090** on Windows and port 8080 inside Docker. Wait for the application health check, then open **http://localhost:8090**. The first video-generation job can take longer while LTX initializes the checkpoint and text encoder.

### Optional local configuration

The defaults in `compose.yaml` work without credentials. No inference credential or hosted provider account is required. `.env` and `.env.example` are intentionally excluded from Git. Create `.env` beside `compose.yaml` only if you need a host port override or Hugging Face requires a download token:

```dotenv
ANTHILL_PORT=8090
# Optional Hugging Face token for model downloads; public model files normally need no token.
HF_TOKEN=
```

Do not put paid inference API keys in this file. `HF_TOKEN` is only for downloading model files and is not an inference credential. Keep any real token private.

## Generate a story

Paste a story or choose **Use example story** to load the sample about a diner waitress who quietly pays for a man's breakfast while he is recovering from surgery. Choose a duration and generate a video. The dashboard reports job progress and provides playback and download when the MP4 is ready. Recent job status is stored locally so it can be recovered after an app restart. The render queue runs one job at a time because the GPU is shared.

The service accepts stories between 25 and 6,000 characters and durations of 30, 45, or 60 seconds. Videos are saved under `output`. The local `state` database retains story text and job metadata; `output/working_<job-id>` retains storyboards, generated images, narration, subtitles, music, and other render intermediates. These user files stay on disk after app restarts and are not automatically removed. Keep the project folder private and copy any files you need before removing them manually.

## Run, stop, and inspect logs

```powershell
# Start services after the first build

docker compose up -d

# Follow app logs
docker compose logs -f anthill

# Check service health and installed model status
Invoke-RestMethod http://localhost:8090/health
Invoke-RestMethod http://localhost:8090/api/readiness

# Stop the services; named Ollama model storage, local models, jobs, and videos persist.
docker compose down
```

`docker compose down` stops and removes the Compose containers and network. It does not remove the named Ollama model volume or the bind-mounted `models`, `state`, and `output` folders. Do not add `--volumes` unless you intend to remove the Ollama model volume.

## Use the command line

The same local job queue is available through `app/cli.py`. Run it inside the running container so it uses the app's local service configuration and installed dependencies. In PowerShell, pipe a UTF-8 story file from Windows into the container; a Windows path such as `C:\AntHill\story.txt` is not a path inside the Linux container:

```powershell
Get-Content -Raw -Encoding utf8 .\story.txt | docker compose exec -T anthill python cli.py generate --story - --duration 30 --wait --output /data/output/story-video.mp4

# Read a job by ID, list recent jobs, retry a failed job, or check local model readiness.
docker compose exec -T anthill python cli.py status 0123456789ab
docker compose exec -T anthill python cli.py list
docker compose exec -T anthill python cli.py retry 0123456789ab --wait
docker compose exec -T anthill python cli.py readiness
```

The generated MP4 saved to `/data/output` also appears in the Windows `output` folder. If you generate without `--wait`, the command prints the job ID immediately; use `status` to check progress.

## Connect an MCP-capable development tool

AntHill provides a local stdio MCP server. A client that supports stdio MCP can launch it through Docker Compose, so it uses the same local queue and model setup. The repository includes a sample configuration at [`examples/mcp-config.json`](examples/mcp-config.json). Register an entry like this in that client's MCP configuration, replacing `C:/AntHill` with the absolute path to this checkout:

```json
{
  "mcpServers": {
    "anthill-local": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "C:/AntHill/compose.yaml",
        "exec",
        "-T",
        "anthill",
        "python",
        "mcp_server.py"
      ]
    }
  }
}
```

The AntHill app container must be running before the client launches this server. Each development tool has its own configuration location and restart procedure; follow that tool's instructions to register the entry. The server exposes tools to generate a video, inspect one job, list recent jobs, retry a failed job, and check model readiness. Calls use the configured local AntHill service and do not invoke hosted inference.

## Output and limitations

- MP4 video: 1080 × 1920, H.264, 30 frames per second, AAC audio, `yuv420p`, and fast-start metadata.
- LTX-Video generates a continuous clip for each scene; short clips are rejected instead of looped to fill scene timing. Complete 30–60-second story-to-MP4 generation remains **NOT VERIFIED** until a full app job succeeds.
- Scene subtitles are burned into the video; a matching SRT subtitle file is retained with the job's working files.
- Narration is generated scene by scene and fitted to the scene timing. Background music is synthesized locally; it is not a licensed song or a music-generation model.
- Image consistency is best-effort. Inspect the exported video before publishing.
- The same six-scene structure is used at each duration; longer durations give each scene and its narration more time and can take longer to generate.

## Troubleshooting

**Docker says no compatible GPU/device driver is available.** Confirm Docker Desktop is running with its WSL2 engine, update the Windows NVIDIA driver, and rerun the `nvidia/cuda` smoke command above. Restart Docker Desktop after a driver update.

**Model setup fails.** Check `docker compose logs ollama ollama-model-setup model-setup`, confirm internet access for the initial downloads, then rerun `docker compose up --build -d`. LTX and Ollama use named Docker volumes; Kokoro voice files remain under `models`.

**The app says an approved model is missing.** Check `docker compose logs model-setup anthill`, then rerun `docker compose up --build -d` and retry the job. The app uses offline Hugging Face access after model setup; it will not silently download models during inference.

**LTX initialization or generation runs out of memory.** Check Docker Desktop's WSL memory cap and available GPU memory. The 16 GB WSL cap configured on the current machine has not completed the LTX initialization probe; model setup itself does not prove video inference works.

**A job fails.** Read the application logs for the underlying error. Use the dashboard's retry action after fixing the cause. Job errors in the browser are intentionally general and do not expose internal stack traces.

**A job is slow.** LTX generates two distilled passes for each scene. Model initialization and the first GPU generation take longer than subsequent scene clips.

## Model and software licenses

- [LTX-Video](https://github.com/Lightricks/LTX-Video) model weights are distributed under the included Lightricks Open Weights license. Review its terms before redistribution or commercial use.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is Apache-2.0. Check the selected voice file's upstream terms as well.
- [Llama 3.2](https://ollama.com/library/llama3.2) is governed by Meta's Llama 3.2 Community License and acceptable-use terms; review them for your intended use.
- Ollama, Docker, PyTorch, Diffusers, Piper's retained package dependency, and the other project dependencies have their own licenses. Review the applicable notices before redistribution.

## Video generation status

The active app path uses local LTX-Video 2B distilled weights. A GPU smoke test produced and played a 5.125-second H.264 clip at 448x800 and 8 fps. The app readiness endpoint reports LTX-Video 2B Distilled as active. A full story-to-MP4 job was submitted but cancelled during local storyboard generation, so full-job integration remains unverified. DreamShaper and AnimateDiff are no longer used.
