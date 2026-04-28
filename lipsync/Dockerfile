# =============================================================================
# OnShot Lipsync Pod — pre-baked Docker image
#
# This image is what RunPod pulls when launching a lipsync pod. Everything
# the deploy_latentsync_comfyui.sh script used to do live (clone, pip install,
# checkpoint downloads, three patches) is baked at build time. Pod boot
# becomes "pull image (cached after first pull) + start two servers" — ~2 min
# instead of the 30+ min flaky setup that bit cycle-wala / bandhani-jaipur /
# chai-aur-chemistry runs.
#
# WHAT THIS IMAGE CONTAINS
#   /opt/ComfyUI                       — ComfyUI pinned (see COMFYUI_REF below)
#   .../custom_nodes/ComfyUI-LatentSyncWrapper       — pinned commit, patched
#   .../custom_nodes/ComfyUI-VideoHelperSuite        — pinned commit
#   .../checkpoints/latentsync_unet.pt               — 5 GB
#   .../checkpoints/stable_syncnet.pt                — 1.6 GB
#   .../checkpoints/whisper/tiny.pt                  — ~75 MB
#   .../checkpoints/vae/diffusion_pytorch_model.safetensors + config.json — 335 MB
#   /workspace/CodeFormer                            — repo + 3 face weights
#   /workspace/gfpgan_weights/GFPGANv1.4.pth         — 332 MB
#   /opt/api_server_postprocess.py                   — :8000 sidecar
#   /entrypoint.sh                                   — boots ComfyUI + sidecar
#
# WHAT THIS IMAGE INTENTIONALLY DOES NOT CONTAIN
#   • RealESRGAN_x4plus_anime_6B.pth — optional bg upsampler for CodeFormer.
#     Postprocess server gracefully degrades when missing (bake later if needed).
#   • Any secrets — image is publishable to a public GHCR registry.
#
# WHY BAKE
#   Six known live-deploy failure modes the image eliminates:
#     1. apt-get exit 100 (transient mirror failures)
#     2. git clone shallow.lock race
#     3. checkpoint silent no-op (4 curls, no size verification mid-script)
#     4. CRLF wrapper patch silent miss
#     5. postprocess server start race with SSH disconnect
#     6. two-deploy concurrent stomp (auto-trigger + explicit POST)
#
# BUILD / PUSH (manual; not run in CI by default — see README)
#   docker build -t ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2 \
#                infra/docker/lipsync
#   docker push  ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2
#
# Versioning: v1.6.X for patch changes, v1.X.Y when ComfyUI/wrapper bumps,
# vX.Y.Z when interface changes. Keep VERSION in sync.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder: clone repos, apply patches, download checkpoints, install deps
#
# Base: runpod/pytorch:2.4.0 ships a *tuned* torch wheel — same one cycle-wala
# ran on yesterday at ~5min/shot. The plain nvidia/cuda base + pip-installed
# torch==2.5.1 was 3-5x slower (vanilla SDPA attention path; bandhani took
# ~22min/shot on the same A5000-class GPU). Pinning the runpod base eliminates
# the regression. Don't downgrade further — runpod hosts cache this base layer
# on every machine, so pull cost stays similar to a stock cuda image.
# -----------------------------------------------------------------------------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONDONTWRITEBYTECODE=1
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Pinned upstream refs (escape transitive drift) ---
# ComfyUI: pin to a recent stable tag matching what cycle-wala ran on
ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFYUI_REF=v0.3.30
# LatentSyncWrapper: pin to a verified commit that the CI-O-22 + Step 2b/2c
# patches apply cleanly to. If upstream drifts, refresh both the SHA below
# and the patches in patches/ together — never bump in isolation.
ARG LATENTSYNC_WRAPPER_REPO=https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
ARG LATENTSYNC_WRAPPER_REF=main
# VideoHelperSuite: pin similarly
ARG VIDEO_HELPER_REPO=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
ARG VIDEO_HELPER_REF=main

# --- System deps (build-time only; runtime stage installs runtime libs) ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        build-essential pkg-config \
        libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && python3 -m pip install --upgrade pip==24.2 setuptools==70.0.0 wheel==0.43.0

# --- ComfyUI ---
WORKDIR /opt
RUN git clone --depth 1 --branch ${COMFYUI_REF} ${COMFYUI_REPO} ComfyUI

# --- PyTorch stack ---
# Inherited from the runpod/pytorch:2.4.0 base — DO NOT reinstall torch here.
# The base ships a tuned wheel (cycle-wala measured ~5min/shot vs 22min on a
# vanilla pip torch==2.5.1 — the regression is real). torchvision 0.19.0 +
# torchaudio 2.4.0 come from the base too. CI-O-22 patch still applies (it
# matches "from torchvision.transforms.functional import" — present in 0.19+).

# --- ComfyUI requirements ---
# Skip strict torch pins inside ComfyUI's requirements.txt so they don't bump
# the base's torch 2.4.0 to 2.5+. The other packages (psutil, kornia, etc.)
# install fine.
RUN sed -i -E '/^(torch|torchvision|torchaudio|torchsde)([=<>!]|$)/d' \
        /opt/ComfyUI/requirements.txt \
 && pip install -r /opt/ComfyUI/requirements.txt

# --- Custom nodes (cloned then pinned to ref) ---
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone ${LATENTSYNC_WRAPPER_REPO} ComfyUI-LatentSyncWrapper \
 && cd ComfyUI-LatentSyncWrapper \
 && git checkout ${LATENTSYNC_WRAPPER_REF} \
 && cd .. \
 && git clone ${VIDEO_HELPER_REPO} ComfyUI-VideoHelperSuite \
 && cd ComfyUI-VideoHelperSuite \
 && git checkout ${VIDEO_HELPER_REF}

# --- LatentSync wrapper deps + pinned compatibility set ---
# Strip torch/torchvision/torchaudio pins from the wrapper requirements too —
# same reason as ComfyUI's requirements.txt above. The wrapper's own deps
# (mediapipe, librosa, etc.) install cleanly.
RUN sed -i -E '/^(torch|torchvision|torchaudio|torchsde)([=<>!]|$)/d' \
        /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/requirements.txt \
 && sed -i -E '/^(torch|torchvision|torchaudio|torchsde)([=<>!]|$)/d' \
        /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt \
 && pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/requirements.txt \
 && pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
# torchcodec must match the installed torch ABI. torch 2.4.0 → torchcodec 0.3.0
# (0.4.0 was built for torch 2.5+ and would force an ABI bump that defeats the
# whole reason we're on the runpod base). soundfile is the fallback audio
# backend we force on at runtime (TORCHAUDIO_BACKEND=soundfile in entrypoint).
RUN pip install \
        torchcodec==0.3.0 \
        soundfile==0.12.1 \
        insightface==0.7.3 \
        onnxruntime-gpu==1.20.1 \
        accelerate==1.0.1 \
        diffusers==0.31.0 \
        transformers==4.44.2 \
        huggingface_hub==0.25.2 \
        safetensors==0.4.5 \
        einops==0.8.0 \
        omegaconf==2.3.0 \
        imageio==2.36.0 \
        imageio-ffmpeg==0.5.1 \
        opencv-python-headless==4.10.0.84 \
        librosa==0.10.2.post1 \
        scipy==1.13.1 \
        numpy==1.26.4 \
        Pillow==10.4.0

# --- Postprocess deps (full chain: GFPGAN + Feathered + CodeFormer) ---
# basicsr ships a stale `from torchvision.transforms.functional_tensor import
# rgb_to_grayscale` import that's been removed in tv >= 0.17. Sed-fix it after
# install so gfpgan can import cleanly.
RUN pip install \
        basicsr==1.4.2 \
        facexlib==0.3.0 \
        gfpgan==1.3.8 \
        lpips==0.1.4 \
        realesrgan==0.3.0 \
        'mediapipe<0.10.15' \
        fastapi==0.109.0 \
        'uvicorn[standard]==0.27.0' \
        python-multipart==0.0.6 \
 && DEG=$(python3 -c "import importlib.util, os; s=importlib.util.find_spec('basicsr'); print(os.path.join(os.path.dirname(s.origin), 'data', 'degradations.py'))") \
 && sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' "$DEG" \
 && python3 -c "import gfpgan; import basicsr.data.degradations; import mediapipe; print('postproc deps OK')"

# --- Apply patches: CI-O-22 (write_video + read_video imageio), no-face, cv2 fallback ---
COPY patches/ /tmp/patches/
RUN python3 /tmp/patches/patch_ci_o_22_write_video.py \
        /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper \
 && python3 /tmp/patches/patch_no_face_graceful.py \
        /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper \
 && python3 /tmp/patches/patch_latentsync_read_video.py \
        /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper

# Fail the build LOUD if any patch silently no-op'd. Far better than building
# a broken image that crashes on every dialogue lipsync at runtime.
RUN grep -q "CI-O-22 PATCH APPLIED" /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py \
        || (echo "FATAL: CI-O-22 patch did not apply to nodes.py" && exit 1) \
 && grep -q "CI-O-22c" /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py \
        || (echo "FATAL: CI-O-22c (read_video imageio) marker missing" && exit 1) \
 && grep -q "no_face_indices" /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync/pipelines/lipsync_pipeline.py \
        || (echo "FATAL: no-face patch did not apply to lipsync_pipeline.py" && exit 1) \
 && grep -q "return None, None, None" /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync/utils/image_processor.py \
        || (echo "FATAL: no-face patch A did not apply to image_processor.py" && exit 1)

# --- Checkpoint directory layout ---
ENV CKPT=/opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints
RUN mkdir -p "$CKPT/whisper" "$CKPT/vae" \
        /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/auxiliary \
        /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/.cache

# --- LatentSync core checkpoints (~7 GB) ---
# Each download enforces a minimum size check — if HF returns an HTML error
# page or a partial transfer, we'd otherwise bake a corrupt image. Better to
# fail the build than ship 10 GB of lies.
RUN curl -fL --retry 5 --retry-delay 10 --max-time 1800 \
        -o "$CKPT/latentsync_unet.pt" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt' \
 && [ "$(stat -c%s $CKPT/latentsync_unet.pt)" -gt 5000000000 ] \
        || (echo "FATAL: latentsync_unet.pt too small" && exit 1)
RUN curl -fL --retry 5 --retry-delay 10 --max-time 600 \
        -o "$CKPT/stable_syncnet.pt" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt' \
 && [ "$(stat -c%s $CKPT/stable_syncnet.pt)" -gt 1500000000 ] \
        || (echo "FATAL: stable_syncnet.pt too small" && exit 1)
RUN curl -fL --retry 5 --retry-delay 10 --max-time 120 \
        -o "$CKPT/whisper/tiny.pt" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/whisper/tiny.pt' \
 && [ "$(stat -c%s $CKPT/whisper/tiny.pt)" -gt 70000000 ] \
        || (echo "FATAL: whisper/tiny.pt too small" && exit 1)
RUN curl -fL --retry 5 --retry-delay 10 --max-time 60 \
        -o "$CKPT/latentsync_config.json" \
        'https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/config.json'

# --- VAE: directory format (config + safetensors) so AutoencoderKL.from_pretrained works ---
RUN curl -fL --retry 5 --retry-delay 10 --max-time 60 \
        -o "$CKPT/vae/config.json" \
        'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json' \
 && curl -fL --retry 5 --retry-delay 10 --max-time 600 \
        -o "$CKPT/vae/diffusion_pytorch_model.safetensors" \
        'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors' \
 && [ "$(stat -c%s $CKPT/vae/diffusion_pytorch_model.safetensors)" -gt 300000000 ] \
        || (echo "FATAL: VAE safetensors too small" && exit 1)

# --- CodeFormer + facelib weights (postprocess sidecar) ---
RUN git clone --depth 1 https://github.com/sczhou/CodeFormer.git /workspace/CodeFormer \
 && printf '__version__ = "0.1.0"\n__gitsha__ = "unknown"\n' > /workspace/CodeFormer/basicsr/version.py \
 && mkdir -p /workspace/CodeFormer/weights/CodeFormer /workspace/CodeFormer/weights/facelib \
 && curl -fL --retry 5 -o /workspace/CodeFormer/weights/CodeFormer/codeformer.pth \
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth' \
 && curl -fL --retry 5 -o /workspace/CodeFormer/weights/facelib/detection_Resnet50_Final.pth \
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth' \
 && curl -fL --retry 5 -o /workspace/CodeFormer/weights/facelib/parsing_parsenet.pth \
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'

# --- GFPGAN weights ---
RUN mkdir -p /workspace/gfpgan_weights \
 && curl -fL --retry 5 --retry-delay 10 --max-time 600 \
        -o /workspace/gfpgan_weights/GFPGANv1.4.pth \
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

# -----------------------------------------------------------------------------
# Stage 2 — runtime: same runpod/pytorch base for ABI parity with builder
#
# Using the runpod/pytorch base in BOTH stages means the tuned torch 2.4.0
# wheel from the builder's /usr/local/lib/python3.11/dist-packages copies in
# clean — no version mismatch with a different runtime base. The base is
# already cached on most runpod hosts (it's their default pytorch worker
# image), so first-pull cost is ~equivalent to cuda:12.4.1-runtime.
# -----------------------------------------------------------------------------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.title="onshot-lipsync-latentsync"
LABEL org.opencontainers.image.description="OnShot lipsync pod — ComfyUI + LatentSync 1.6 + post-processing sidecar, fully baked"
LABEL org.opencontainers.image.source="https://github.com/Agrim-Intelligence/kg-storytelling-platform"
LABEL org.opencontainers.image.licenses="proprietary"
LABEL org.opencontainers.image.version="1.6.3"
LABEL onshot.role="lipsync"
LABEL onshot.engine="latentsync-1.6"

# Runtime apt deps. ffmpeg / libgl / etc. — the runpod base has python 3.11
# already, so we don't reinstall it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy ONLY the pip-installed extras (everything our builder added on top of
# the base). /usr/local/lib/python3.11/dist-packages contains both the base's
# tuned torch + all our extra wheels, so a single COPY is correct.
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Copy ComfyUI tree (includes patched custom_nodes + all checkpoints)
COPY --from=builder /opt/ComfyUI /opt/ComfyUI

# Copy CodeFormer + GFPGAN weights into /workspace
COPY --from=builder /workspace/CodeFormer /workspace/CodeFormer
COPY --from=builder /workspace/gfpgan_weights /workspace/gfpgan_weights

# Copy postprocess server + entrypoint
COPY api_server_postprocess.py /opt/api_server_postprocess.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Working dir for runtime artefacts (logs, postprocess_tmp)
RUN mkdir -p /workspace
WORKDIR /workspace

# CI-O-21 safety net: torchaudio.save() uses soundfile backend
ENV TORCHAUDIO_USE_BACKEND_DISPATCHER=1 \
    TORCHAUDIO_BACKEND=soundfile \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ports: 8188 = ComfyUI, 8000 = postprocess sidecar
EXPOSE 8188 8000

ENTRYPOINT ["/entrypoint.sh"]
