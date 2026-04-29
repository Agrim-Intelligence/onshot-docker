# =============================================================================
# OnShot Chatterbox-Multilingual TTS Pod — pre-baked Docker image (v1.0.0)
#
# Bakes everything deploy_chatterbox.sh did live (system deps, venv,
# chatterbox install from GitHub, FastAPI server) PLUS the ~2.7 GB of
# ResembleAI/chatterbox model weights that the runtime install would have
# pulled on first boot. Pod boot becomes "pull image (cached after first
# pull) + start FastAPI server" — ~2 min instead of the 8-15 min on-first-
# boot HF download that bit cycle-wala / bandhani-jaipur (HF rate-limit
# variability triggered watchdog timeouts mid-download).
#
# DESIGN PRINCIPLE
# ----------------
# Mirror deploy_chatterbox.sh exactly (no version pins beyond what the
# script applies — i.e. none) and bake the HF weights so first request
# doesn't pay the multi-GB HF download.
#
# Lessons from fish-audio v1.0.x iterations applied preemptively:
#   * Pre-download HF weights via huggingface_hub.snapshot_download with
#     retries, rather than letting chatterbox lazy-load them at runtime.
#   * Sanity-check `from chatterbox.mtl_tts import ChatterboxMultilingualTTS`
#     after install so the build fails LOUDLY if the GitHub HEAD shifts and
#     breaks the API surface.
#
# WHAT THIS IMAGE CONTAINS
#   /opt/chatterbox_api_server.py              — FastAPI server (272 lines)
#   /workspace/models/hub/                     — HF cache, pre-populated
#       ResembleAI/chatterbox/                 — ~2.7 GB total
#         t3_mtl23ls_v2.safetensors            — ~2.4 GB (multilingual T3)
#         s3gen.pt                             — ~250 MB (speaker generator)
#         ve.pt                                — voice encoder
#         conds.pt                             — conditioning vectors
#         Cangjie5_TC.json                     — Chinese segmentation
#         grapheme_mtl_merged_expanded_v1.json — multilingual grapheme map
#   pip-installed: chatterbox + soundfile + fastapi + uvicorn
#
# WHAT THIS IMAGE INTENTIONALLY DOES NOT CONTAIN
#   * Any secrets — image is publishable to a public GHCR registry.
#
# WHY BAKE
#   Eliminates the on-first-boot HF download, which is the dominant
#   failure mode of the live deploy:
#     1. Unauthenticated HF rate-limit → partial downloads / 429 retries
#     2. 360s → 1200s watchdog bumps in deploy_chatterbox.sh (CI-O-58d) only
#        masked the symptom; baked weights eliminate the problem.
#     3. SSH disconnect mid-download leaves "setup_done" with no model.
# =============================================================================

# -----------------------------------------------------------------------------
# Single stage on runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2404. The tag
# in deploy_chatterbox.sh:8 ("2.6.1-py3.11-cuda12.4.1-devel-ubuntu22.04")
# refers to the OLD runpod tag scheme and was unpublished by runpod when
# they switched to the "<runpod-ver>-cu<cuda>-torch<torch>-ubuntu<distro>"
# format. This tag matches the same intent (torch 2.6 + CUDA 12.8) under
# the new naming and is part of the same family fish-audio v1.0.x uses.
# -----------------------------------------------------------------------------
FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2404

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONDONTWRITEBYTECODE=1
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Step 1: System packages (mirrors deploy_chatterbox.sh:42-46) ---
# ffmpeg + libsndfile1 for audio I/O. openssh-server for entrypoint sshd
# (the runpod base ships sshd but our ENTRYPOINT replaces theirs, so we
# explicitly start sshd in entrypoint.sh — same fix as fish-audio v1.0.x).
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        ffmpeg libsndfile1-dev libsndfile1 \
        ca-certificates curl git wget \
        openssh-server \
 && rm -rf /var/lib/apt/lists/*

# --- Step 2: Install chatterbox + server deps (mirrors deploy_chatterbox.sh:55-72) ---
# IMPORTANT: chatterbox-multilingual is only available from GitHub HEAD —
# pip doesn't ship the multilingual TTS class. The script pins to no specific
# commit; we follow the same convention so the bake matches what's deployed
# today. If a future ResembleAI commit breaks ChatterboxMultilingualTTS,
# pin a commit hash here in v1.x.x.
RUN pip install --break-system-packages --no-cache-dir --upgrade pip
RUN pip install --break-system-packages --no-cache-dir \
        git+https://github.com/resemble-ai/chatterbox.git
RUN pip install --break-system-packages --no-cache-dir \
        soundfile python-multipart fastapi uvicorn \
        huggingface_hub

# Sanity check: ChatterboxMultilingualTTS must be importable. Fail LOUDLY
# at build time if the GitHub HEAD has drifted past our compatibility window.
RUN python3 -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('ChatterboxMultilingualTTS import OK')"

# --- Step 3: Pre-download ResembleAI/chatterbox weights (~2.7 GB) ---
# The runtime install pulls these from HF on first model construction.
# CI-O-58d in deploy_chatterbox.sh bumped the watchdog to 1200s because
# unauthenticated HF rate limits make this download wildly variable
# (8-15 min observed, occasional partial-then-fail). Baking eliminates
# the variability completely.
ENV HF_HOME=/workspace/models \
    TORCH_HOME=/workspace/models \
    TRANSFORMERS_CACHE=/workspace/models \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ENABLE_HF_TRANSFER=0
# CI-O-CHATTER-1: runpod base sets HF_HUB_ENABLE_HF_TRANSFER=1 but does
# not install hf_transfer, so snapshot_download fails immediately with
# "Fast download using 'hf_transfer' is enabled but package missing".
# Override the env var (above) AND install hf_transfer (below) so both
# the slow + fast paths work, regardless of whether the env wins.
RUN pip install --break-system-packages --no-cache-dir hf_transfer
RUN mkdir -p /workspace/models /opt /tmp/audio-outputs /root/audio-logs

# CI-O-CHATTER-2 (v1.0.3): the v1.0.2 layer for this step was 9.7 GB
# compressed for a ~2.7 GB model. Two issues:
#   1. allow_patterns=["*.safetensors","*.pt","*.pth","*.bin",...] pulled
#      every weight format ResembleAI publishes — including duplicates of
#      the same weights in different formats.
#   2. snapshot_download leaves both blob files AND symlinks (and a
#      .huggingface/ metadata dir). Docker's overlay layer captures all
#      of it, doubling the apparent layer size.
#
# v1.0.3 fix:
#   1. Tighten allow_patterns to the exact 6 files chatterbox needs
#      (per deploy_chatterbox.sh CI-O-58d note).
#   2. After download, resolve symlinks → real files in snapshots/, then
#      delete the blobs/ dir (Docker layer keeps only one copy of each).
#   3. Remove .huggingface/ metadata + any .lock / .partial files.
# CI-O-CHATTER-3 (v1.0.4): inline slim into SAME RUN as download.
# v1.0.3 ran download (L38=2.76GB) + slim (L39=2.76GB) in separate RUNs;
# Docker overlay can't actually delete L38's blobs (only mask them with
# whiteouts in L39), so total grew rather than shrank. v1.0.4 fixes this
# by doing snapshot_download + symlink resolve + blobs/ delete + verify
# all in ONE RUN — the layer commits the FINAL state directly with no
# leftover blobs/.
RUN python3 - <<'PYEOF'
import os, pathlib, sys, time, shutil
from huggingface_hub import snapshot_download

repo_id = "ResembleAI/chatterbox"
cache_root = pathlib.Path("/workspace/models/hub")
cache_root.mkdir(parents=True, exist_ok=True)

# Tight allow_patterns: only the 6 files chatterbox actually loads.
ALLOW = [
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "ve.pt",
    "conds.pt",
    "Cangjie5_TC.json",
    "grapheme_mtl_merged_expanded_v1.json",
]

last_err = None
for attempt in range(1, 4):
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root),
            allow_patterns=ALLOW,
        )
        print(f"OK on attempt {attempt}: {path}", flush=True)
        break
    except Exception as e:
        last_err = e
        print(f"attempt {attempt} failed: {e}", flush=True)
        time.sleep(10 * attempt)
else:
    raise RuntimeError(f"Failed to download {repo_id}: {last_err}")

# --- Inline slim: resolve symlinks → real files, then delete blobs/ ---
chatter = cache_root / "models--ResembleAI--chatterbox"
snap_root = chatter / "snapshots"
if not snap_root.is_dir():
    print(f"FATAL: snapshot dir missing at {snap_root}", file=sys.stderr)
    sys.exit(1)

# Resolve symlinks in-place
for snap in snap_root.iterdir():
    for f in snap.iterdir():
        if f.is_symlink():
            target = f.resolve()
            if target.is_file():
                f.unlink()
                shutil.copy2(str(target), str(f))

# Delete blobs/ (now redundant with resolved files in snapshots/)
blobs = chatter / "blobs"
if blobs.is_dir():
    sz = sum(p.stat().st_size for p in blobs.rglob("*") if p.is_file())
    shutil.rmtree(str(blobs))
    print(f"  removed blobs/ ({sz/1024/1024:.0f} MB freed)", flush=True)

# Cleanup metadata + leftovers
hf_meta = cache_root / ".huggingface"
if hf_meta.is_dir():
    shutil.rmtree(str(hf_meta))
for p in cache_root.rglob("*.lock"):
    p.unlink()
for p in cache_root.rglob("*.partial"):
    p.unlink()

# Verify expected files all present + total > 2GB
total_bytes = 0
found = set()
for snap in snap_root.iterdir():
    for f in snap.iterdir():
        if f.is_file():
            sz = f.stat().st_size
            total_bytes += sz
            found.add(f.name)
missing = set(ALLOW) - found
if missing:
    print(f"FATAL: missing expected files: {missing}", file=sys.stderr)
    sys.exit(1)
if total_bytes < 2_000_000_000:
    print(f"FATAL: total weights only {total_bytes/1024/1024:.0f} MB", file=sys.stderr)
    sys.exit(1)
print(f"All chatterbox weights baked OK ({total_bytes/1024/1024:.0f} MB)")
PYEOF

# Step 3b removed (v1.0.4): slim is now inline in Step 3 above.

# --- Step 4: Bake api_server + entrypoint ---
COPY chatterbox_api_server.py /opt/chatterbox_api_server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="onshot-chatterbox-multilingual"
LABEL org.opencontainers.image.description="OnShot chatterbox-multilingual TTS pod — Hindi/multilingual TTS + zero-shot voice cloning, fully baked"
LABEL org.opencontainers.image.source="https://github.com/Agrim-Intelligence/onshot-docker"
LABEL org.opencontainers.image.licenses="proprietary"
LABEL org.opencontainers.image.version="1.0.4"
LABEL onshot.role="audio"
LABEL onshot.engine="chatterbox-multilingual"

# Port: 8000 = FastAPI server (chatterbox endpoints)
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
