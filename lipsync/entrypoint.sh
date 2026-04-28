#!/bin/bash
# OnShot lipsync pod entrypoint.
#
# Boots two cooperating servers in the same container:
#   :8188 — ComfyUI (LatentSync inference engine)
#   :8000 — api_server_postprocess.py sidecar (GFPGAN/CodeFormer/feathered-blend)
#
# Both write to /workspace/*.log; the trailing `tail -F` pipes logs into the
# container's stdout so `docker logs` (and RunPod's web UI) sees them. If
# either process exits, the container exits 1 and the pod orchestrator can
# decide whether to restart.
set -e

mkdir -p /workspace
cd /workspace

# CI-O-21 safety net (also baked into Dockerfile ENV, kept here for clarity).
export TORCHAUDIO_USE_BACKEND_DISPATCHER=1
export TORCHAUDIO_BACKEND=soundfile

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
tail -F /workspace/comfyui.log /workspace/postprocess.log &
TAIL_PID=$!

# Wait on either real server dying (ignore the tail — it lives forever).
# `wait -n` returns when the first job exits; we propagate non-zero.
wait -n $COMFY_PID $POSTPROC_PID
EXIT_CODE=$?

echo "[entrypoint] one of the servers exited (code=$EXIT_CODE); bringing down the container"
kill -TERM $TAIL_PID 2>/dev/null || true
exit 1
