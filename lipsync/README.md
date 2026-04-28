# OnShot Lipsync Pod — Pre-baked Docker Image

This directory builds a single Docker image that contains everything a
LatentSync lipsync pod needs at runtime. RunPod pulls the image and the pod
boots in ~2 min instead of running a 30+ min flaky setup script per launch.

## Why this exists

Live-deploy of `deploy_latentsync_comfyui.sh` repeatedly fails in production
runs (cycle-wala, bandhani-jaipur, chai-aur-chemistry). The six failure modes
the baked image eliminates:

1. **`apt-get` exit 100** on fresh `runpod-slim` images (transient mirror
   failures during pip's underlying apt resolution).
2. **`git clone` shallow.lock race** when `.git/shallow.lock` is left from an
   interrupted prior attempt.
3. **Checkpoint silent no-op** — Step 5 curls 4 files (~7.7 GB total). The
   `[ -f X ] && skip || curl X` idiom can leave a 0-byte / partial file and
   march on; the pod ends up "setup_done" with missing checkpoints.
4. **CRLF wrapper patch silent miss** — `ComfyUI-LatentSyncWrapper` ships
   with CRLF endings in `nodes.py`; our patches matched LF and silently
   exited rc=1 four+ separate times in the field.
5. **Postprocess server start race** — Step 8 starts the sidecar via
   `nohup ... & disown` which races SSH disconnect.
6. **Two-deploy concurrent stomp** — launch auto-trigger + explicit deploy
   POST sometimes both run, overwriting each other's files.

The Dockerfile bakes pinned ComfyUI + LatentSyncWrapper + VideoHelperSuite,
applies all three patches with build-time `grep` verification (build fails
loud if any patch silently no-ops), pre-downloads + size-verifies all 7 GB
of checkpoints, and ships an entrypoint that boots both servers cleanly.

## Image contents

| Path | Size | Purpose |
|------|-----:|---------|
| `/opt/ComfyUI` | ~2 GB | Pinned to `v0.3.30` |
| `…/custom_nodes/ComfyUI-LatentSyncWrapper` | ~50 MB | Patched (CI-O-22, no-face, cv2 fallback) |
| `…/custom_nodes/ComfyUI-VideoHelperSuite` | ~10 MB | Required for `VHS_LoadVideo`, `VHS_VideoCombine` |
| `…/checkpoints/latentsync_unet.pt` | 5.0 GB | LatentSync 1.6 UNet |
| `…/checkpoints/stable_syncnet.pt` | 1.6 GB | SyncNet discriminator |
| `…/checkpoints/whisper/tiny.pt` | 75 MB | Audio encoder |
| `…/checkpoints/vae/*` | 335 MB | sd-vae-ft-mse (config + safetensors) |
| `/workspace/CodeFormer + facelib weights` | ~700 MB | Postprocess face restore |
| `/workspace/gfpgan_weights/GFPGANv1.4.pth` | 332 MB | Pre-blend face restore |
| `/opt/api_server_postprocess.py` | — | :8000 sidecar |

Total uncompressed: **~10 GB**, mostly checkpoints.

What is **not** baked: `RealESRGAN_x4plus_anime_6B.pth`. The postprocess
server gracefully degrades when it's missing (CodeFormer simply runs without
the bg upsampler). Bake later if anime-style background upscaling becomes
mandatory.

## Build (manual — first build only)

```bash
# From repo root
docker build \
    -t ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2 \
    -t ghcr.io/agrim-intelligence/lipsync-latentsync:latest \
    lipsync
```

Build time on a fast Linux box with a fast pipe: **~30 min** (most of which
is the `latentsync_unet.pt` + `stable_syncnet.pt` downloads from HuggingFace).
On a 500 Mbps connection, expect 25-35 min end-to-end.

After the first push, subsequent builds reuse cached layers — only the layers
above whatever changed re-run, so a patch-only change rebuilds in <2 min.

## Push to GitHub Container Registry

```bash
# One-time: log in to ghcr.io with a PAT that has write:packages scope
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

# Then push
docker push ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2
docker push ghcr.io/agrim-intelligence/lipsync-latentsync:latest
```

The org segment **must be lowercase** (`agrim-intelligence`, not
`Agrim-Intelligence`). GHCR is case-insensitive on lookup but Docker pulls
expect the lowercase form. **Confirm with @swapnil before the first push** —
if the org slug differs from `agrim-intelligence` the rest of the wiring
needs to match.

By default the package is **public** on GHCR (zero egress cost). If you
want it private the first time you push, do it via the GitHub UI before any
`docker pull` runs against it.

## Test locally before pushing

```bash
docker run --rm --gpus all \
    -p 8188:8188 -p 8000:8000 \
    ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2

# In another terminal:
curl -s localhost:8188/object_info | jq 'has("LatentSyncNode")'
# Must print: true

curl -s localhost:8000/   # postprocess sidecar root
# Should respond with FastAPI's default OpenAPI redirect
```

If `LatentSyncNode` is not present, the wrapper failed to load — check
`docker logs <container>` for `comfyui.log` output. The most common cause
is a missing checkpoint (build should have caught this; if it didn't, the
`stat` check thresholds need raising).

## RunPod consumption

RunPod accepts the raw image name in the GraphQL `podFindAndDeployOnDemand`
mutation:

```graphql
podFindAndDeployOnDemand(input: {
    imageName: "ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2"
    ports: "8188/http,8000/http,22/tcp"
    containerDiskInGb: 30
    volumeInGb: 0
    dockerEntrypoint: ""    # use the image's ENTRYPOINT
})
```

Per-machine pull is cached, so only the first pod on a given physical host
pays the ~10 GB download. Subsequent pods on the same machine boot in ~2 min
(image pull is local).

**Wiring follow-up (NOT in this PR):** `services/media-orchestration-service/
src/media_orchestration_service/api/v1_runpod_instance_handler.py` has a
`RUNPOD_TEMPLATE_MAP`. For the new image no template is needed — RunPod can
boot the raw image as long as `dockerEntrypoint` is empty and ports are
declared. That handler change is its own PR.

## Versioning

| Bump | When | Examples |
|------|------|----------|
| **patch** `v1.6.X` | Patch script update, dep bump within compat range | CRLF normaliser change, mediapipe minor |
| **minor** `v1.X.Y` | ComfyUI tag bump, wrapper SHA bump, new postprocess feature | ComfyUI v0.3.30 → v0.4.x |
| **major** `vX.Y.Z` | Breaking interface change | New port, new entrypoint protocol, image renamed |

Always update `VERSION` in this dir + the `LABEL ... version` in the
Dockerfile in the same commit. The CI workflow reads `VERSION` to tag.

## CI build (automatic)

`.github/workflows/build-lipsync.yml` triggers on:

* changes under `lipsync/**`
* changes to `services/media-orchestration-service/lipsync-hosting-setup/api_server_postprocess.py`

It builds for `linux/amd64` and pushes to `ghcr.io/agrim-intelligence/
lipsync-latentsync:<VERSION>` plus `:latest`. CI uses `GITHUB_TOKEN` (no PAT
needed) — the token has `packages: write` scope by default for the repo's
own org packages.

To force a rebuild without a code change, bump `VERSION` and push.

## Local dev tip

When debugging on a real workload, `docker run` mounts a host dir under
`/workspace/postprocess_tmp` to avoid filling the container's writable
layer:

```bash
docker run --rm --gpus all \
    -p 8188:8188 -p 8000:8000 \
    -v "$PWD/.lipsync-tmp:/workspace/postprocess_tmp" \
    ghcr.io/agrim-intelligence/lipsync-latentsync:v1.6.2
```
