# AntHill Local Video Factory

AntHill turns a short story into a locally generated vertical motion comic. Paste a story, choose **30, 45, or 60 seconds**, and select **Generate**. The output is an H.264/AAC MP4 at **1080 × 1920** with scene illustrations, gentle camera movement, spoken narration, burned-in subtitles, and an original synthesized ambient music bed.

This is a motion comic: generated still images move through camera effects. It does not make continuous, full-body character animation. Images are generated at a lower resolution to fit an 8 GB GPU, then enlarged for export. DreamShaper reference-image prompting helps keep the recurring character similar between scenes, but image generation cannot guarantee identical faces or details. Kokoro produces natural English speech; it has limited direct control over emotional acting.

All story planning, image generation, narration, and video assembly run locally in Docker. No host Python or Ollama installation is required. Ollama cloud features are disabled. Model downloads are needed for initial installation; inference does not require paid APIs or send prompts to a hosted model service. Docker images and Python packages also need internet access when first installed.

**Runtime validation is incomplete in the current environment.** Docker Compose parses and NVIDIA GPU passthrough to an `nvidia-smi` container check passed, but Docker Engine HTTP 500 errors and timeouts have blocked the app image build and model setup. No AI weights are installed. GPU tensor inference, MP4 generation, and browser validation remain **NOT VERIFIED**.

## Requirements

- Windows 11, a current NVIDIA driver, and Docker Desktop using its WSL2 engine.
- An NVIDIA RTX GPU with 8 GB VRAM is recommended for the selected image model. The image service uses CUDA with model CPU offload; Ollama and Kokoro run on CPU to preserve GPU memory.
- 32 GB system RAM is recommended.
- At least 20 GB of free disk space is a practical starting point for the model files, Docker images, Python dependencies, caches, and outputs. Keep additional space available for videos and temporary render files.

The setup pins selected model revisions and downloads only the FP16 DreamShaper components required for inference. Download size is about 2.2 GB for DreamShaper, roughly 360 MB for Kokoro and its selected voice, and about 2 GB for the Llama 3.2 3B Ollama model. Total model downloads are approximately 4.6 GB; supporting container and package downloads add more.

## Install and start

Install the current NVIDIA driver and Docker Desktop, enable **Use the WSL 2 based engine**, then start Docker Desktop. In PowerShell, check that Docker can see the GPU before building the app:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

From the folder containing `compose.yaml`, verify the Compose file:

```powershell
docker compose config --quiet
```

From the folder containing `compose.yaml`, run this single command. Compose builds the app, starts Ollama, downloads the approved Llama 3.2 model, then runs the DreamShaper and Kokoro model setup service before starting the web app. The first start downloads about 4.6 GB of model files and needs internet access. Later starts reuse the models stored in `models` and the Ollama volume:

```powershell
docker compose up --build -d
```

The app binds to **http://localhost:8090** on Windows and port 8080 inside Docker. Wait for the application health check, then open **http://localhost:8090**. The first image-generation job can take considerably longer while model weights load. The selected models stay in the `models` folder and Ollama's named Docker volume across app restarts.

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
- Six illustrated story scenes use gentle zoom effects. This is not true AI video animation.
- Scene subtitles are burned into the video; a matching SRT subtitle file is retained with the job's working files.
- Narration is generated scene by scene and fitted to the scene timing. Background music is synthesized locally; it is not a licensed song or a music-generation model.
- Image consistency is best-effort. Inspect the exported video before publishing.
- The same six-scene structure is used at each duration; longer durations give each scene and its narration more time and can take longer to generate.

## Troubleshooting

**Docker says no compatible GPU/device driver is available.** Confirm Docker Desktop is running with its WSL2 engine, update the Windows NVIDIA driver, and rerun the `nvidia/cuda` smoke command above. Restart Docker Desktop after a driver update.

**Model setup fails.** Check `docker compose logs ollama ollama-model-setup model-setup`, confirm internet access for the initial downloads, then rerun `docker compose up --build -d`. Ollama stores its model in the persistent `ollama_models` volume; DreamShaper and Kokoro files are stored in `models`.

**The app says an approved model is missing.** Check `docker compose logs model-setup anthill`, then rerun `docker compose up --build -d` and retry the job. The app uses offline Hugging Face access after model setup; it will not silently download models during inference.

**Image generation runs out of VRAM.** Close other GPU applications and retry. The renderer releases the diffusion model between text-to-image and image-to-image stages, uses CPU offload and attention/VAE memory options, and has one lower-resolution retry for a CUDA out-of-memory error. The scene images will still be enlarged for the final export.

**A job fails.** Read the application logs for the underlying error. Use the dashboard's retry action after fixing the cause. Job errors in the browser are intentionally general and do not expose internal stack traces.

**A job is slow.** The model files run locally and image generation is the slowest step. CPU narration and storyboarding avoid competing for the limited GPU memory. The first job also loads the image model into memory.

## Model and software licenses

- [DreamShaper 8](https://huggingface.co/Lykon/dreamshaper-8) uses the CreativeML OpenRAIL-M license. Review its usage restrictions before commercial distribution.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is Apache-2.0. Check the selected voice file's upstream terms as well.
- [Llama 3.2](https://ollama.com/library/llama3.2) is governed by Meta's Llama 3.2 Community License and acceptable-use terms; review them for your intended use.
- Ollama, Docker, PyTorch, Diffusers, Piper's retained package dependency, and the other project dependencies have their own licenses. Review the applicable notices before redistribution.

## True AI animation research

True character motion is a separate follow-up. [CogVideoX-2B](https://huggingface.co/zai-org/CogVideoX-2b) is a candidate to benchmark first: its official repository is about 13.8 GB, and the model card reports Diffusers FP16 inference starting at 4 GB VRAM, with model offload and VAE slicing/tiling optimizations. The vendor says those memory figures were tested on A100/H100 GPUs, so performance and compatibility on this RTX 5050 remain unverified. CogVideoX-2B targets short 720 × 480, 8 fps clips, so it would provide scene inserts rather than replace the full 1080 × 1920 narration pipeline. Its model license is Apache-2.0. [Wan 2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) is another option, but its published 8.19 GB VRAM requirement for a five-second 480p clip leaves no practical headroom on this 8 GB laptop. Neither model is installed or part of the current pipeline.
