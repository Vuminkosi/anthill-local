[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not (Test-Path -LiteralPath '.env.example')) {
        @(
            '# Local inference settings. HF_TOKEN is optional and only used for model downloads.'
            '# No cloud inference or paid API credentials are used by default.'
            'ANTHILL_PORT=8090'
            'OLLAMA_URL=http://ollama:11434'
            'OLLAMA_MODEL=llama3.2:3b'
            'IMAGE_MODEL=Lykon/dreamshaper-8'
            'IMAGE_MODEL_VARIANT=fp16'
            'IMAGE_MODEL_PATH=/data/models/huggingface/Lykon--dreamshaper-8'
            'ANIMATEDIFF_MODEL_PATH=/data/models/animatediff/guoyww--animatediff-motion-adapter-v1-5-2'
            'KOKORO_MODEL_PATH=/data/models/voices/kokoro/kokoro-v1_0.pth'
            'KOKORO_VOICE_PATH=/data/models/voices/kokoro/voices/af_heart.pt'
            'HF_TOKEN='
        ) | Set-Content -LiteralPath '.env.example' -Encoding utf8
    }

    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'

        Write-Host 'Created local .env from the example. HF_TOKEN is optional for public models.'
    }

    Write-Host 'Building the app and starting Ollama; the approved Llama 3.2 model will be installed locally.'
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed with exit code $LASTEXITCODE."
    }

    Write-Host 'Downloading the approved DreamShaper 8 FP16 image model, Kokoro narration model, and AnimateDiff FP16 motion adapter.'
    docker compose --profile model-setup run --rm model-setup
    if ($LASTEXITCODE -ne 0) {
        throw "Local model setup failed with exit code $LASTEXITCODE. Review the output above and retry after resolving the download error."
    }

    Write-Host 'All approved model files are installed. Open http://localhost:8090 to generate a video.'
}
finally {
    Pop-Location
}
