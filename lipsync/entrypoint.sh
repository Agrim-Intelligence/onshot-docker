#!/bin/bash
# OnShot lipsync pod entrypoint.
#
# Boots two cooperating servers in the same container:
#   :8188 — ComfyUI (LatentSync inference engine)
#   :8000 — api_server_postprocess.py sidecar (GFPGAN/CodeFormer/feathered-blend)
#
# Both write to /workspace/*.log; the trailing `tail -F` pipes logs into the
# container's stdout so `docker logs` (and RunPod's web UI) sees them.
#
# Boot resilience (CI-O-58w): if either server exits, we DO NOT exit 1 — we
# `sleep infinity` so the container stays up and SSH/proxy traffic keeps
# working, allowing live debugging of the failure (logs in /workspace, env
# inspection, etc). Without this, a runtime import error (e.g. cv2/numpy ABI
# mismatch) crash-loops the container forever — no SSH, no port proxy,
# nothing to grab logs from.
set -e

mkdir -p /workspace
cd /workspace

# CI-O-21 safety net (also baked into Dockerfile ENV, kept here for clarity).
export TORCHAUDIO_USE_BACKEND_DISPATCHER=1
export TORCHAUDIO_BACKEND=soundfile

# --- Pre-flight import checks (CI-O-58w) ---
# Validate the runtime stack BEFORE we kick off the heavy services. Any
# ImportError here lands in /workspace/preflight.log with a clean traceback,
# instead of being buried in ComfyUI's startup spam after a crash.
{
  echo "=== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python3 - <<'PYEOF' 2>&1
import sys; print("python:", sys.version)
import importlib
mods = [
    "torch", "torchvision", "torchaudio", "torchcodec",
    "numpy", "scipy", "cv2", "PIL",
    "diffusers", "transformers", "accelerate", "safetensors",
    "mediapipe", "librosa", "imageio", "soundfile",
    "basicsr", "facexlib", "gfpgan", "lpips", "realesrgan",
    "fastapi", "uvicorn", "onnxruntime",
]
for name in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {name}=={v}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
PYEOF
} > /workspace/preflight.log 2>&1
echo "[entrypoint] preflight done; see /workspace/preflight.log"

# --- Start ComfyUI on :8188 ---
cd /opt/ComfyUI
python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --disable-auto-launch \
    > /workspace/comfyui.log 2>&1 &
COMFY_PID=$!

# --- Start postprocess sidecar on :8000 ---
nohup python3 /opt/api_server_postprocess.py \
    > /workspace/postprocess.log 2>&1 &
POSTPROC_PID=$!

echo "[entrypoint] ComfyUI pid=$COMFY_PID, Postprocess pid=$POSTPROC_PID"

# Stream both logs to container stdout so `docker logs` shows them live.
tail -F /workspace/comfyui.log /workspace/postprocess.log /workspace/preflight.log &
TAIL_PID=$!

# Wait on either real server dying (ignore the tail — it lives forever).
# `wait -n` returns when the first job exits; we DO NOT propagate that as a
# container exit, so SSH stays up for debugging.
wait -n $COMFY_PID $POSTPROC_PID
EXIT_CODE=$?

echo "[entrypoint] one of the servers exited (code=$EXIT_CODE); container will sleep" \
     "indefinitely so SSH + log inspection stay available. Inspect" \
     "/workspace/comfyui.log /workspace/postprocess.log /workspace/preflight.log."
kill -TERM $TAIL_PID 2>/dev/null || true

# Keep PID 1 alive forever so the container doesn't crash-loop. Operator can
# `docker logs <id>` or SSH in to see what went wrong.
sleep infinity
