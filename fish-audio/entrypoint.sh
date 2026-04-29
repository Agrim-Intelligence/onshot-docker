#!/bin/bash
# OnShot fish-audio pod entrypoint.
#
# Boots a single FastAPI server on :8000 that exposes:
#   POST /v1/bgm  — MusicGen Medium for per-scene background music
#   POST /v1/sfx  — AudioGen Medium for per-shot sound effects
#   GET  /v1/health
#
# Diagnostic discipline (CI-O-58w pattern from lipsync v1.6.7):
#   * preflight import check writes /workspace/preflight.log
#   * sleep infinity if the server dies, so SSH + log inspection stay up
#   * NO `set -e` — a single command failing shouldn't bypass the debug tail
#
# Env vars consumed:
#   AUDIO_API_KEY            required (the API key the orchestrator sends)
#   MUSICGEN_MODEL           medium | large (default: medium)
#   FISH_SPEECH_AUTOSTART    true | false   (default: false; not baked here)
#   PUBLIC_KEY               injected by RunPod for SSH access
#
# Logs land in /workspace/{preflight,server,sshd}.log

mkdir -p /workspace
cd /workspace

# Audiocraft GPU kernels need cuDNN libs from the torch pip wheel on
# LD_LIBRARY_PATH (otherwise SIGBUS on libcudnn_ops_infer.so.8 — same gotcha
# deploy_fish.sh handles in its STEP 4 prologue).
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12")
CUDNN_LIB="/usr/local/lib/python${PY_VER}/dist-packages/nvidia/cudnn/lib"
[ -d "$CUDNN_LIB" ] && export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

# HF cache where models were baked during build.
export HF_HOME=/workspace/models
export TORCH_HOME=/workspace/models
export AUDIO_TTS_PROVIDER=fish
# AUDIO_API_KEY must be supplied by the launcher; api_server.py rejects requests
# without a matching X-API-Key header. Provide a default so the server boots
# (the real key is injected via env when the pod is launched).
export AUDIO_API_KEY="${AUDIO_API_KEY:-sk-audio-2024-change-me}"

# --- Start sshd if available (runpod base ships sshd but its own entrypoint
# starts it — we overrode that, so do it ourselves). Inject the RunPod-supplied
# PUBLIC_KEY into root's authorized_keys so the operator can SSH in for log
# inspection. Without this, the "sleep infinity" debug fallback below is
# unreachable. Same pattern as lipsync v1.6.7.
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
if command -v sshd >/dev/null 2>&1; then
    mkdir -p /run/sshd
    [ -f /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -A 2>/dev/null || true
    /usr/sbin/sshd -D > /workspace/sshd.log 2>&1 &
    echo "[entrypoint] sshd started (pid=$!)"
fi

# --- Pre-flight import checks ---
# Validate the runtime stack BEFORE we kick off the API server. Any
# ImportError here lands in /workspace/preflight.log with a clean traceback
# instead of being buried in the server's startup spam after a crash.
{
  echo "=== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python3 - <<'PYEOF' 2>&1
import sys, importlib
print("python:", sys.version)
mods = [
    "torch", "torchvision", "torchaudio",
    "numpy", "scipy", "soundfile",
    "audiocraft", "audiocraft.models",
    "transformers", "accelerate", "spacy",
    "fastapi", "uvicorn", "pydantic",
    "ormsgpack", "httpx", "librosa",
    "torchmetrics", "einops", "encodec",
]
for name in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {name}=={v}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

# Verify CUDA + cuDNN actually load (audiocraft GPU path).
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cudnn version: {torch.backends.cudnn.version()}")
        print(f"  device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  cuda probe failed: {e}")

# Verify baked HF cache: MusicGen Medium + AudioGen Medium must be present.
import os
hub = "/workspace/models/hub"
if os.path.isdir(hub):
    snaps = [d for d in os.listdir(hub) if d.startswith("models--")]
    print(f"  hf cache snapshots: {len(snaps)}")
    for s in sorted(snaps):
        print(f"    - {s}")
else:
    print(f"  hf cache MISSING at {hub}")
PYEOF
} > /workspace/preflight.log 2>&1
echo "[entrypoint] preflight done; see /workspace/preflight.log"

# --- Start FastAPI audio server on :8000 ---
if [ -f /opt/api_server.py ]; then
    cd /workspace
    nohup python3 /opt/api_server.py > /workspace/server.log 2>&1 &
    SERVER_PID=$!
    echo "[entrypoint] api_server started (pid=$SERVER_PID)"
else
    echo "[entrypoint] FATAL: /opt/api_server.py missing — sleep-only mode"
    SERVER_PID=""
fi

# Stream logs to container stdout for `docker logs` / RunPod web UI.
tail -F /workspace/server.log /workspace/preflight.log 2>/dev/null &
TAIL_PID=$!

# Wait for the server to die (or never started).
if [ -n "$SERVER_PID" ]; then
    wait -n "$SERVER_PID"
    EXIT_CODE=$?
else
    EXIT_CODE=255
fi

echo "[entrypoint] api_server exited (code=$EXIT_CODE); container will sleep" \
     "indefinitely so SSH + log inspection stay available. Inspect" \
     "/workspace/preflight.log + /workspace/server.log."
kill -TERM $TAIL_PID 2>/dev/null || true

# Keep PID 1 alive forever so the container doesn't crash-loop.
sleep infinity
