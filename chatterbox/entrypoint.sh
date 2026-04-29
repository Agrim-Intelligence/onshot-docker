#!/bin/bash
# OnShot chatterbox-multilingual TTS pod entrypoint.
#
# Boots a single FastAPI server on :8000 that exposes:
#   POST /api/v1/voice/chatterbox_basic   — Hindi/multilingual TTS
#   POST /api/v1/voice/chatterbox_clone   — Zero-shot voice cloning
#   GET  /api/v1/download/{id}            — Download audio
#   GET  /health
#
# Same diagnostic discipline as fish-audio v1.0.x and lipsync v1.6.7:
#   * preflight import check writes /workspace/preflight.log
#   * sleep infinity if the server dies, so SSH + log inspection stay up
#   * NO `set -e` — a single command failing shouldn't bypass the debug tail
#
# Env vars consumed:
#   AUDIO_API_KEY    required (the API key the orchestrator sends)
#   PUBLIC_KEY       injected by RunPod for SSH access
#
# Logs land in /workspace/{preflight,server,sshd}.log

mkdir -p /workspace
cd /workspace

export AUDIO_TTS_PROVIDER=chatterbox
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# HF cache where weights were baked during build
export HF_HOME=/workspace/models
export TORCH_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models

# AUDIO_API_KEY must be supplied by the launcher; chatterbox_api_server.py
# rejects requests without a matching X-API-Key header.
export AUDIO_API_KEY="${AUDIO_API_KEY:-sk-audio-2024-change-me}"

# --- Start sshd if available (runpod base ships sshd but its own entrypoint
# starts it — we overrode that, so do it ourselves). Inject the RunPod-supplied
# PUBLIC_KEY into root's authorized_keys so the operator can SSH in for log
# inspection. Without this, the "sleep infinity" debug fallback below is
# unreachable. Same pattern as lipsync v1.6.7 + fish-audio v1.0.x.
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
# Validate the runtime stack BEFORE the API server boots. Catches
# torch/chatterbox/transformers ABI mismatches early with a clean traceback
# in /workspace/preflight.log.
{
  echo "=== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python3 - <<'PYEOF' 2>&1
import sys, importlib
print("python:", sys.version)
mods = [
    "torch", "torchvision", "torchaudio",
    "numpy", "scipy", "soundfile",
    "chatterbox", "chatterbox.mtl_tts",
    "transformers", "fastapi", "uvicorn",
]
for name in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {name}=={v}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

# Verify CUDA + cuDNN actually load.
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cudnn version: {torch.backends.cudnn.version()}")
        print(f"  device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  cuda probe failed: {e}")

# Verify baked HF cache: ResembleAI/chatterbox snapshot must be present.
import os
hub = "/workspace/models/hub"
expected = "models--ResembleAI--chatterbox"
if os.path.isdir(hub):
    snaps = [d for d in os.listdir(hub) if d.startswith("models--")]
    print(f"  hf cache snapshots: {len(snaps)}")
    for s in sorted(snaps):
        print(f"    - {s}")
    chatter_dir = os.path.join(hub, expected)
    if os.path.isdir(chatter_dir):
        # walk into snapshots/ and report file sizes
        snap_root = os.path.join(chatter_dir, "snapshots")
        if os.path.isdir(snap_root):
            for s in os.listdir(snap_root):
                snap = os.path.join(snap_root, s)
                for f in sorted(os.listdir(snap)):
                    full = os.path.join(snap, f)
                    if os.path.islink(full):
                        full = os.path.realpath(full)
                    if os.path.isfile(full):
                        size_mb = os.path.getsize(full) / 1024 / 1024
                        print(f"      {f}: {size_mb:.1f} MB")
    else:
        print(f"  CHATTERBOX WEIGHTS MISSING — expected {chatter_dir}")
else:
    print(f"  hf cache MISSING at {hub}")
PYEOF
} > /workspace/preflight.log 2>&1
echo "[entrypoint] preflight done; see /workspace/preflight.log"

# --- Start FastAPI chatterbox server on :8000 ---
if [ -f /opt/chatterbox_api_server.py ]; then
    cd /workspace
    nohup python3 /opt/chatterbox_api_server.py > /workspace/server.log 2>&1 &
    SERVER_PID=$!
    echo "[entrypoint] chatterbox_api_server started (pid=$SERVER_PID)"
else
    echo "[entrypoint] FATAL: /opt/chatterbox_api_server.py missing — sleep-only mode"
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

echo "[entrypoint] chatterbox_api_server exited (code=$EXIT_CODE); container will" \
     "sleep indefinitely so SSH + log inspection stay available." \
     "Inspect /workspace/preflight.log + /workspace/server.log."
kill -TERM $TAIL_PID 2>/dev/null || true

# Keep PID 1 alive forever so the container doesn't crash-loop.
sleep infinity
