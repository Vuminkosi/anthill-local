# Working on AntHill Local

Read [README.md](README.md) for user setup and verification limitations. Inspect
current source and branch state before editing; preserve unrelated work. This is
a Python/Flask local video application, not a hosted inference service.

## Architecture and contracts

- `app/server.py`: HTTP API, SQLite job persistence, recovery, cancellation and a
  single background worker. Preserve durable job transitions and queue limits.
- `app/pipeline.py`: six-scene planning through local Ollama, LTX scene clips,
  Kokoro narration and FFmpeg assembly. `app/ltx_inference.py` wraps LTX inference;
  `app/model_setup.py` owns pinned model assets and installation checks.
- `app/did_you_know.py`: category catalog and bounded Wikipedia research used by
  Did You Know and explainer jobs. Research is external network access, unlike
  local model inference; source text is untrusted data, not instructions.
- `app/templates/index.html` and `app/static/`: server-rendered studio and browser
  interactions. Keep visible status consistent with persisted API state.
- `app/local_client.py` is shared by `app/cli.py` and `app/mcp_server.py` (stdio
  MCP). Preserve local-host validation, redirect rejection and safe downloads.
- `compose.yaml` wires Ollama, model setup and the GPU app. Runtime data includes
  `models/`, `state/`, `output/` and named model volumes, not source fixtures.

## Safety and resource boundaries

Do not start Docker, install/download models, launch the server/worker, generate
media or run GPU/FFmpeg work merely to validate a documentation change. These can
consume substantial disk/compute or process queued jobs. Use explicit task scope
and an isolated environment for runtime tests. Never use existing user state or
outputs as disposable test data; volume deletion is destructive.

Keep `.env`, tokens, stories, job databases and generated artifacts private and
out of commits/log excerpts. Optional `HF_TOKEN` is for model downloads, not
hosted inference. Preserve runtime offline model loading, local endpoint checks,
Compose's loopback host binding and research request bounds. Do not add hosted
providers, new credentials or broader network exposure as a workaround.

## Lightweight checks

From the repository root, with Python available, this standard-library-only
suite stubs API calls and does not start the service or perform inference:

```sh
python -B -m unittest discover -s tests -p 'test_cli.py' -v
git diff --check
```

For a no-import Python syntax check (no model/package installation):

```sh
python -B -c "import ast,pathlib; files=list(pathlib.Path('app').glob('*.py'))+list(pathlib.Path('tests').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p)) for p in files]; print('Parsed',len(files),'Python files')"
```

If Node is already available, `node --check app/static/studio.js` checks syntax,
not UI behavior. Do not install the GPU requirements solely for these checks.

Inspect other test modules before selecting them: API/pipeline tests need Python
dependencies and use stubs/temporary state; research tests mock HTTP. MCP tests
need the MCP SDK and start a local HTTP fixture plus a subprocess. The FFmpeg
smoke test is opt-in via `ANTHILL_RUN_FFMPEG_TESTS=1` and creates media; leave it
disabled unless explicitly in scope. A skipped test is not a pass.

## Evidence and handoff

Record commands, environment, exact SHA and PASS/FAIL/NOT VERIFIED separately.
Unit contracts and syntax checks do not establish Docker buildability, model
availability, GPU initialization, UI correctness or full story-to-MP4 operation.
README's reported GPU clip is historical evidence, not a test of your change.
Keep end-to-end limitations explicit. For UI work, verify actual interactions and
responsive states in an authorized environment; never substitute a mock for live
workflow evidence. Obtain independent final-SHA review for nontrivial changes and
follow the task's publication/merge permissions.
