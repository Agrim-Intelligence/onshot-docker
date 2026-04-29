# =============================================================================
# OnShot Fish-Audio Pod — pre-baked Docker image (v1.0.0)
#
# Bakes everything deploy_fish.sh did live (system deps, dedicated venv,
# torch 2.8 pin, audiocraft --no-deps + manual deps, MusicGen Medium +
# AudioGen Medium model downloads). Pod boot becomes "pull image (cached
# after first pull) + start FastAPI server" — ~2 min instead of the 30+ min
# flaky live deploy that bit cycle-wala / bandhani-jaipur runs (4 deploy
# failures in a single bandhani-jaipur attempt → BGM + SFX skipped entirely).
#
# DESIGN PRINCIPLE
# ----------------
# Mirror deploy_fish.sh EXACTLY. That script worked on cycle-wala. The
# same lesson from lipsync v1.6.2 → v1.6.7 applies: don't deviate from
# the working contract. Only deviation here is using the system Python
# (no venv) since this image is single-purpose — the venv in the deploy
# script existed to isolate fish from xtts/chatterbox on a shared pod,
# not because of any package incompatibility with the system stack.
#
# WHAT THIS IMAGE CONTAINS
#   /opt/api_server.py                         — FastAPI server (1300 lines)
#   /workspace/models/hub/                     — HF cache, pre-populated
#       facebook/musicgen-medium               — ~3 GB BGM model
#       facebook/audiogen-medium               — ~3 GB SFX model
#       google-t5/t5-large                     — text encoder, pulled by audiocraft
#       facebook/encodec_32khz                 — neural audio codec
#   pip-installed: audiocraft + manual deps    — see Step 5 below
#
# WHAT THIS IMAGE INTENTIONALLY DOES NOT CONTAIN
#   * fish-speech v2 S2-Pro: chatterbox handles TTS in our stack;
#     fish-audio pod is BGM/SFX only. Set START_FISH_SPEECH=true in env
#     and add the relevant deps if a future episode genuinely needs S2-Pro.
#   * MusicGen Large (only Medium baked): fits 20-24GB GPUs and matches
#     what the deploy script defaults to.
#   * Any secrets — image is publishable to a public GHCR registry.
#
# WHY BAKE
#   Six known live-deploy failure modes the image eliminates:
#     1. apt-get exit 100 (transient mirror failures)
#     2. audiocraft pulling stale torch==2.1.0 → torchvision::nms missing
#     3. HF model download silent partial (2 GB safetensors → 0-byte file
#        + script marches on)
#     4. cuDNN libs missing from LD_LIBRARY_PATH → SIGBUS in libcudnn_*
#     5. venv vs system-pip drift across torch reinstalls
#     6. SSH disconnect mid-deploy leaving "setup_done" with no server
#
# BUILD / PUSH (manual; CI handles main pushes)
#   docker build -t ghcr.io/agrim-intelligence/fish-audio:v1.0.0 fish-audio
#   docker push  ghcr.io/agrim-intelligence/fish-audio:v1.0.0
# =============================================================================

# -----------------------------------------------------------------------------
# Single stage on runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404. The base
# image ships torch 2.8.0 + cuDNN configured for CUDA 12.8.1 — exactly what
# audiocraft + MusicGen need to run without the silent-downgrade trap.
# Multi-stage is overkill: most of the image is the 6 GB of pre-baked HF
# weights, which can't be shrunk via a runtime-only stage.
# -----------------------------------------------------------------------------
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONDONTWRITEBYTECODE=1
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Step 2: System deps (mirrors deploy_fish.sh:151-160) ---
# Most of these (ffmpeg, build tools) are already in the base; idempotent.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        ffmpeg libsndfile1 git wget curl build-essential pkg-config \
        libavformat-dev libavcodec-dev libavutil-dev libavfilter-dev \
        libswscale-dev libswresample-dev openssh-server \
 && rm -rf /var/lib/apt/lists/*

# --- Step 4: Pin torch / torchaudio / torchvision (mirrors deploy_fish.sh:223-256) ---
# The base ships torch 2.8.0+cu128 already. We force-reinstall torchaudio +
# torchvision to the matching cu128 build because audiocraft will pull stale
# pins otherwise. --no-deps so pip never sees audiocraft's torch==2.1.0
# constraint. Index URL is the cu128 wheel index, same as the deploy script.
RUN pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        --no-deps \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        torchvision==0.23.0+cu128

# --- Step 5: Core numeric + server deps (mirrors deploy_fish.sh:271-285) ---
# Same package list, same order, same pins as the script. NO version pins
# beyond what the script applies (numpy 1.26.4, accelerate 0.25.0, etc).
RUN pip install --break-system-packages --no-cache-dir \
        numpy==1.26.4 \
        soundfile==0.12.1 \
 && pip install --break-system-packages --no-cache-dir \
        fastapi uvicorn python-multipart pydantic \
 && pip install --break-system-packages --no-cache-dir \
        accelerate==0.25.0 \
 && pip install --break-system-packages --no-cache-dir \
        "transformers>=4.40,<5" \
 && pip install --break-system-packages --no-cache-dir \
        ormsgpack httpx \
 && pip install --break-system-packages --no-cache-dir \
        spacy --prefer-binary

# --- Step 5 (cont): audiocraft + manual deps (mirrors deploy_fish.sh:288-309) ---
# audiocraft pins torch==2.1.0 — install with --no-deps so pip never sees
# the stale constraint. Then install audiocraft's runtime deps manually
# (the ones NOT in audiocraft's own pin set, but required to import).
# demucs needs --no-deps for the same reason (duplicate torch otherwise).
RUN pip install --break-system-packages --no-cache-dir \
        audiocraft --no-deps \
 && pip install --break-system-packages --no-cache-dir \
        librosa torchmetrics einops sentencepiece encodec lameenc flashy \
        num2words av --prefer-binary \
 && pip install --break-system-packages --no-cache-dir \
        demucs --no-deps \
 && pip install --break-system-packages --no-cache-dir \
        diffq julius openunmix dora-search treetable --prefer-binary

# --- Step 5 (final): Re-pin numpy 1.26.4 ---
# audiocraft + a few transitive deps may have nudged numpy back to 2.x;
# numpy 2 breaks audiocraft's encodec wrapper at runtime. Pin again, last.
RUN pip install --break-system-packages --no-cache-dir --force-reinstall \
        numpy==1.26.4

# --- Step 5b: Patch audiocraft for xformers-free operation ---
# audiocraft 1.3.0 hard-imports `from xformers import ops` at module load.
# There is no stable xformers wheel for torch 2.8 + Python 3.12, so we
# make that import optional and fall back to torch attention. Without
# this, the very next step (`from audiocraft.models import MusicGen`)
# raises ModuleNotFoundError and the whole build fails.
# Verbatim port of deploy_fish.sh:309-380 — same patches, same files,
# idempotent. Build fails LOUDLY if either patch silently no-ops.
COPY patches/patch_audiocraft_xformers.py /tmp/patches/patch_audiocraft_xformers.py
RUN python3 /tmp/patches/patch_audiocraft_xformers.py \
 && python3 -c "import audiocraft.modules.transformer; print('audiocraft import OK after xformers patch')"

# --- Step 5c: Pre-download T5 tokenizers (CI-O-FISH-1) ---
# audiocraft's MusicGen + AudioGen call `T5Tokenizer.from_pretrained("t5-base")`
# at construction time (audiocraft/modules/conditioners.py:415). The legacy
# "t5-base" / "t5-large" repo names were renamed on HF to "google-t5/t5-base"
# and "google-t5/t5-large"; the redirect can be flaky on GHA runners (clean
# HF cache + rate limits → 404). The deploy script doesn't hit this because
# the runpod base image has these cached from prior pulls.
#
# Fix: explicit snapshot_download with retries, then symlink the legacy
# cache name to the canonical one so transformers' `from_pretrained` finds
# the local copy without hitting Hub. This makes the t5 fetch a one-time
# build cost instead of a runtime risk.
ENV HF_HOME=/workspace/models \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ENABLE_HF_TRANSFER=0
# CI-O-FISH-2: runpod/pytorch:1.0.2 base sets HF_HUB_ENABLE_HF_TRANSFER=1
# but doesn't install hf_transfer, so snapshot_download fails immediately
# with "Fast download using 'hf_transfer' is enabled but package missing".
# Override the env var (above) AND install hf_transfer (below) so both
# the slow + fast paths work, regardless of whether the env wins.
RUN pip install --break-system-packages --no-cache-dir hf_transfer
RUN mkdir -p /workspace/models /opt /workspace/audio-server /tmp/audio-outputs
ENV HF_HOME=/workspace/models \
    TORCH_HOME=/workspace/models

# CI-O-FISH-5 (v1.0.5): inline the slim into the SAME RUN as each download.
# v1.0.4 ran download + slim in separate RUNs and the image grew to 30 GB
# (vs 20 GB on v1.0.3) because Docker overlay cannot delete files from a
# previous layer — the slim's `shutil.rmtree(blobs)` only writes whiteout
# markers; the bytes still live in the prior download layer. To actually
# slim, the download + symlink-resolve + blobs/ delete must all happen in
# the SAME RUN so the layer commits the FINAL state directly.
#
# Per-model RUN (one layer each, naturally parallel-pull-friendly):
#   t5-base:        ~250 MB  (tokenizer + model)
#   t5-large:       ~3 GB    (the bigger T5)
#   MusicGen Med:   ~1.7 GB  (UNet + encodec, dedup'd of t5)
#   AudioGen Med:   ~1.7 GB  (UNet + encodec, dedup'd of t5)
# Total expected: ~7-8 GB of model weights baked, vs 30 GB on v1.0.4.

# --- Step 5c-1: t5-base download + inline slim ---
RUN python3 - <<'PYEOF'
import pathlib, shutil, time
from huggingface_hub import snapshot_download

hub = pathlib.Path("/workspace/models/hub")
hub.mkdir(parents=True, exist_ok=True)

# Download canonical name
last_err = None
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id="google-t5/t5-base", cache_dir=str(hub),
            allow_patterns=["*.json","*.txt","*.model","spiece.model","tokenizer*","special_tokens_map.json","added_tokens.json","*.safetensors","*.bin"])
        break
    except Exception as e:
        last_err = e
        print(f"  attempt {attempt} failed: {e}", flush=True)
        time.sleep(5 * attempt)
else:
    raise RuntimeError(f"Failed to download t5-base: {last_err}")

# Symlink legacy alias
legacy = hub / "models--t5-base"
canonical = hub / "models--google-t5--t5-base"
if canonical.exists() and not legacy.exists():
    legacy.symlink_to(canonical)
    print(f"  symlink: {legacy.name} → {canonical.name}", flush=True)

# Inline slim: resolve symlinks → real files in snapshots/, delete blobs/
snap_root = canonical / "snapshots"
if snap_root.is_dir():
    for snap in snap_root.iterdir():
        for f in snap.iterdir():
            if f.is_symlink():
                target = f.resolve()
                if target.is_file():
                    f.unlink()
                    shutil.copy2(str(target), str(f))
blobs = canonical / "blobs"
if blobs.is_dir():
    sz = sum(p.stat().st_size for p in blobs.rglob("*") if p.is_file())
    shutil.rmtree(str(blobs))
    print(f"  removed t5-base blobs/ ({sz/1024/1024:.0f} MB)", flush=True)
sz_final = sum(p.stat().st_size for p in canonical.rglob("*") if p.is_file())
print(f"  t5-base final: {sz_final/1024/1024:.0f} MB", flush=True)
PYEOF

# --- Step 5c-2: t5-large download + inline slim ---
RUN python3 - <<'PYEOF'
import pathlib, shutil, time
from huggingface_hub import snapshot_download

hub = pathlib.Path("/workspace/models/hub")

last_err = None
for attempt in range(1, 4):
    try:
        snapshot_download(repo_id="google-t5/t5-large", cache_dir=str(hub),
            allow_patterns=["*.json","*.txt","*.model","spiece.model","tokenizer*","special_tokens_map.json","added_tokens.json","*.safetensors","*.bin"])
        break
    except Exception as e:
        last_err = e
        print(f"  attempt {attempt} failed: {e}", flush=True)
        time.sleep(5 * attempt)
else:
    raise RuntimeError(f"Failed to download t5-large: {last_err}")

legacy = hub / "models--t5-large"
canonical = hub / "models--google-t5--t5-large"
if canonical.exists() and not legacy.exists():
    legacy.symlink_to(canonical)
    print(f"  symlink: {legacy.name} → {canonical.name}", flush=True)

snap_root = canonical / "snapshots"
if snap_root.is_dir():
    for snap in snap_root.iterdir():
        for f in snap.iterdir():
            if f.is_symlink():
                target = f.resolve()
                if target.is_file():
                    f.unlink()
                    shutil.copy2(str(target), str(f))
blobs = canonical / "blobs"
if blobs.is_dir():
    sz = sum(p.stat().st_size for p in blobs.rglob("*") if p.is_file())
    shutil.rmtree(str(blobs))
    print(f"  removed t5-large blobs/ ({sz/1024/1024:.0f} MB)", flush=True)
sz_final = sum(p.stat().st_size for p in canonical.rglob("*") if p.is_file())
print(f"  t5-large final: {sz_final/1024/1024:.0f} MB", flush=True)

# Sanity: T5Tokenizer must work via legacy alias
from transformers import T5Tokenizer
tok = T5Tokenizer.from_pretrained("t5-base")
print(f"  T5Tokenizer.from_pretrained('t5-base') OK", flush=True)
PYEOF

# --- Step 6a: MusicGen Medium download + inline slim ---
RUN python3 - <<'PYEOF'
import pathlib, shutil
from audiocraft.models import MusicGen
print("Downloading MusicGen Medium...", flush=True)
m = MusicGen.get_pretrained("facebook/musicgen-medium")
del m
print("MusicGen Medium downloaded", flush=True)

# Inline slim: resolve symlinks across ALL freshly-touched HF cache dirs
# (audiocraft.get_pretrained also pulls encodec_32khz + sometimes sd-vae;
# slim them all so blobs/ never lands in this layer).
hub = pathlib.Path("/workspace/models/hub")
for model_dir in hub.glob("models--*"):
    if model_dir.is_symlink():
        continue
    snap_root = model_dir / "snapshots"
    if snap_root.is_dir():
        for snap in snap_root.iterdir():
            for f in snap.iterdir():
                if f.is_symlink():
                    target = f.resolve()
                    if target.is_file():
                        f.unlink()
                        shutil.copy2(str(target), str(f))
    blobs = model_dir / "blobs"
    if blobs.is_dir():
        sz = sum(p.stat().st_size for p in blobs.rglob("*") if p.is_file())
        shutil.rmtree(str(blobs))
        print(f"  removed {model_dir.name}/blobs/ ({sz/1024/1024:.0f} MB)", flush=True)
PYEOF

# --- Step 6b: AudioGen Medium download + inline slim ---
RUN python3 - <<'PYEOF'
import pathlib, shutil
from audiocraft.models import AudioGen
print("Downloading AudioGen Medium...", flush=True)
a = AudioGen.get_pretrained("facebook/audiogen-medium")
del a
print("AudioGen Medium downloaded", flush=True)

hub = pathlib.Path("/workspace/models/hub")
for model_dir in hub.glob("models--*"):
    if model_dir.is_symlink():
        continue
    snap_root = model_dir / "snapshots"
    if snap_root.is_dir():
        for snap in snap_root.iterdir():
            for f in snap.iterdir():
                if f.is_symlink():
                    target = f.resolve()
                    if target.is_file():
                        f.unlink()
                        shutil.copy2(str(target), str(f))
    blobs = model_dir / "blobs"
    if blobs.is_dir():
        sz = sum(p.stat().st_size for p in blobs.rglob("*") if p.is_file())
        shutil.rmtree(str(blobs))
        print(f"  removed {model_dir.name}/blobs/ ({sz/1024/1024:.0f} MB)", flush=True)

# Final cleanup + verify
hf_meta = hub / ".huggingface"
if hf_meta.is_dir():
    shutil.rmtree(str(hf_meta))
for p in hub.rglob("*.lock"):
    p.unlink()
for p in hub.rglob("*.partial"):
    p.unlink()

import sys
def model_size(pattern):
    for d in hub.iterdir():
        if d.is_symlink():
            continue
        if pattern in d.name:
            return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    return 0
bgm = model_size("facebook--musicgen-medium")
sfx = model_size("facebook--audiogen-medium")
print(f"MusicGen Medium: {bgm/1024/1024:.0f} MB")
print(f"AudioGen Medium: {sfx/1024/1024:.0f} MB")
if bgm < 1_500_000_000 or sfx < 1_500_000_000:
    print("FATAL: model size below threshold", file=sys.stderr)
    sys.exit(1)
total = sum(p.stat().st_size for p in hub.rglob("*") if p.is_file())
print(f"final hub size: {total/1024/1024:.0f} MB")
PYEOF

# --- Bake the api_server + entrypoint ---
COPY api_server.py /opt/api_server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Working dir for runtime artefacts (logs, output WAVs)
WORKDIR /workspace

# Runtime env (also exported in entrypoint for clarity)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/models \
    TORCH_HOME=/workspace/models

LABEL org.opencontainers.image.title="onshot-fish-audio"
LABEL org.opencontainers.image.description="OnShot fish-audio pod — MusicGen Medium (BGM) + AudioGen Medium (SFX), fully baked"
LABEL org.opencontainers.image.source="https://github.com/Agrim-Intelligence/onshot-docker"
LABEL org.opencontainers.image.licenses="proprietary"
LABEL org.opencontainers.image.version="1.0.5"
LABEL onshot.role="audio"
LABEL onshot.engine="fish-audiocraft"

# Port: 8000 = FastAPI server (BGM + SFX endpoints)
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
