# AntHill Local Video Factory

AntHill turns a short story into a locally generated vertical motion comic. Paste a story, choose **30, 45, or 60 seconds**, and select **Generate**. The output is an H.264/AAC MP4 at **1080 × 1920** with scene illustrations, gentle camera movement, spoken narration, burned-in subtitles, and an original synthesized ambient music bed.

The pipeline animates each illustrated scene into a short generated motion clip, then loops clips as needed while assembling the full narration-led video. This creates scene motion, but not continuous full-body animation throughout the finished video. Images and motion clips are generated at reduced resolution for 8 GB GPUs, then enlarged for export. DreamShaper reference-image prompting helps keep recurring characters similar, but exact identity is not guaranteed. Kokoro produces natural English speech with limited direct control over emotional acting.

All story planning, image generation, narration, and video assembly run locally in Docker. No host Python or Ollama installation is required. Ollama cloud features are disabled. Model downloads are needed for initial installation; inference does not require paid APIs or send prompts to a hosted model service. Docker images and Python packages also need internet access when first installed.

**Validation status:** The Docker app image has built, DreamShaper, Kokoro, and AnimateDiff model setup completed, and NVIDIA GPU passthrough was verified on an RTX 5050 with 8 GB VRAM. A 16-frame 384 × 672 AnimateDiff GPU clip was generated, and its first and last frames were nonidentical. The full automated test suite ran in Docker: 52 passed and 1 was skipped. Complete 30–60-second story-to-MP4 generation and browser playback remain **NOT VERIFIED**.

## Requirements

- Windows 11, a current NVIDIA driver, and Docker Desktop using its WSL2 engine.
- An NVIDIA RTX GPU with 8 GB VRAM is recommended for the selected image model. The image service uses CUDA with model CPU offload; Ollama and Kokoro run on CPU to preserve GPU memory.
- 32 GB system RAM is recommended.
- At least 20 GB of free disk space is a practical starting point for the model files, Docker images, Python dependencies, caches, and outputs. Keep additional space available for videos and temporary render files.

The setup pins selected model revisions and downloads the FP16 model files for local inference. Current model downloads are about 2.2 GB for DreamShaper, roughly 360 MB for Kokoro and its selected voice, and about 2 GB for Llama 3.2 3B. The approved AnimateDiff motion adapter adds approximately 1.82 GB, bringing the expected total to about 6.4 GB; supporting container and package downloads add more. Actual sizes can vary slightly by file selection and download metadata.

The pipeline now includes an AnimateDiff animation phase. It uses the pinned FP16 adapter, [`guoyww/animatediff-motion-adapter-v1-5-2`](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2), with the existing DreamShaper model components. Adapter files are stored under `models/animatediff/guoyww--animatediff-motion-adapter-v1-5-2` and persist across container restarts. Model downloads run in the Docker model-setup service; inference uses local files and does not fetch models at runtime. A short 16-frame 384 × 672 GPU smoke clip was verified on an RTX 5050 with 8 GB VRAM. Complete 30–60-second real-story video generation has not yet been verified.

## Install and start

Install the current NVIDIA driver and Docker Desktop, enable **Use the WSL 2 based engine**, then start Docker Desktop. In PowerShell, check that Docker can see the GPU before building the app:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

From the folder containing `compose.yaml`, verify the Compose file:

```powershell
docker compose config --quiet
```

From the folder containing `compose.yaml`, run this single command. Compose builds the app, starts Ollama, downloads the approved Llama 3.2 model, then runs Docker model setup for DreamShaper, Kokoro, and the AnimateDiff adapter before starting the web app. The initial model download is approximately 6.4 GB and needs internet access. Later starts reuse files stored in `models` and the Ollama volume. To download only the AnimateDiff adapter separately, run `docker compose run --rm --no-deps model-setup python model_setup.py --download-animation`. This runs the existing model-setup image without starting its dependencies and stores the adapter under `models`:

```powershell
docker compose up --build -d
```

The app binds to **http://localhost:8090** on Windows and port 8080 inside Docker. Wait for the application health check, then open **http://localhost:8090**. The first video-generation job can take considerably longer while image and motion model weights load. The selected models stay in the `models` folder and Ollama's named Docker volume across app restarts.

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
- The implemented animation phase uses DreamShaper scene stills with the AnimateDiff motion adapter to generate a short clip per scene (16 frames at 8 fps by default). Each clip is about two seconds at the default 16 frames and 8 fps, then looped as needed during assembly into the narrated 30–60-second video. This produces generated scene motion, but not continuous full-body animation across the whole video, and it does not guarantee consistent character identity. A short 16-frame 384 × 672 GPU smoke clip was verified on an RTX 5050 with 8 GB VRAM; complete 30–60-second real-story video generation remains unverified.
- Scene subtitles are burned into the video; a matching SRT subtitle file is retained with the job's working files.
- Narration is generated scene by scene and fitted to the scene timing. Background music is synthesized locally; it is not a licensed song or a music-generation model.
- Image consistency is best-effort. Inspect the exported video before publishing.
- The same six-scene structure is used at each duration; longer durations give each scene and its narration more time and can take longer to generate.

## Troubleshooting

**Docker says no compatible GPU/device driver is available.** Confirm Docker Desktop is running with its WSL2 engine, update the Windows NVIDIA driver, and rerun the `nvidia/cuda` smoke command above. Restart Docker Desktop after a driver update.

**Model setup fails.** Check `docker compose logs ollama ollama-model-setup model-setup`, confirm internet access for the initial downloads, then rerun `docker compose up --build -d`. Ollama stores its model in the persistent `ollama_models` volume; DreamShaper, Kokoro, and AnimateDiff files are stored in `models`.

**The app says an approved model is missing.** Check `docker compose logs model-setup anthill`, then rerun `docker compose up --build -d` and retry the job. The app uses offline Hugging Face access after model setup; it will not silently download models during inference.

**Image generation runs out of VRAM.** Close other GPU applications and retry. The renderer releases the diffusion model between text-to-image and image-to-image stages, uses CPU offload and attention/VAE memory options, and has one lower-resolution retry for a CUDA out-of-memory error. The scene images will still be enlarged for the final export.

**A job fails.** Read the application logs for the underlying error. Use the dashboard's retry action after fixing the cause. Job errors in the browser are intentionally general and do not expose internal stack traces.

**A job is slow.** The model files run locally and image generation is the slowest step. CPU narration and storyboarding avoid competing for the limited GPU memory. The first job also loads the image model into memory.

## Model and software licenses

- [DreamShaper 8](https://huggingface.co/Lykon/dreamshaper-8) uses the CreativeML OpenRAIL-M license. Review its usage restrictions before commercial distribution.
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is Apache-2.0. Check the selected voice file's upstream terms as well.
- [Llama 3.2](https://ollama.com/library/llama3.2) is governed by Meta's Llama 3.2 Community License and acceptable-use terms; review them for your intended use.
- Ollama, Docker, PyTorch, Diffusers, Piper's retained package dependency, and the other project dependencies have their own licenses. Review the applicable notices before redistribution.

## Animation status and limitations

AnimateDiff is implemented as a generation phase that turns each DreamShaper scene image into a short motion clip, then assembles those clips with narration, subtitles, and music into a 1080 × 1920 MP4. The clips repeat to fill their scene durations. This is real generated scene motion, but it does not provide continuous full-body animation through the entire video or guarantee exact character identity across scenes. A short 16-frame 384 × 672 GPU smoke clip was verified on an RTX 5050 with 8 GB VRAM. Complete 30–60-second real-story output is not yet verified.

The adapter is distributed under its upstream model license; review the [model card](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2) before redistribution or commercial use. The app uses locally stored weights for inference and does not download models at runtime.
