# onshot-docker

Pre-baked Docker images for OnShot pipeline GPU pod roles. Each image bakes ComfyUI + custom nodes + model checkpoints + patches into a single layer so RunPod pods boot ready-to-serve in ~70 seconds instead of running fragile 30+ minute setup scripts.

## Why this repo exists

The OnShot pipeline (in [`Agrim-Intelligence/kg-storytelling-platform`](https://github.com/Agrim-Intelligence/kg-storytelling-platform)) launches RunPod community GPU pods for each pipeline phase (image generation, audio synthesis, lip sync, video animation). The original deploy-time setup approach hit recurring failures across runs:

- apt-get exit 100 on fresh runpod-slim images
- `git clone` shallow.lock races on shared overlay storage
- 7GB checkpoint downloads that silently no-op'd
- CRLF wrapper-patch silent failures
- Sidecar process start races (`nohup ... & disown` losing state)
- Concurrent deploy invocations stomping each other

Each failure mode burned 15-30 minutes per supervisor run. Pre-baking everything into a docker image eliminates 5 of the 6 failure modes.

## Image catalog

| Role | Subdirectory | Image | Size | Status |
|---|---|---|---|---|
| Lip sync | [`lipsync/`](./lipsync/) | `ghcr.io/agrim-intelligence/lipsync-latentsync` | ~10GB | ✅ scaffolded |
| Image gen (Flux) | `image-flux/` | `ghcr.io/agrim-intelligence/image-flux` | ~25GB | ⏳ planned |
| Audio (chatterbox) | `audio-chatterbox/` | `ghcr.io/agrim-intelligence/audio-chatterbox` | ~3GB | ⏳ planned |
| Audio (fish/MusicGen) | `audio-fish/` | `ghcr.io/agrim-intelligence/audio-fish` | ~15GB | ⏳ planned |
| Video (Wan2.2) | `video-wan22/` | `ghcr.io/agrim-intelligence/video-wan22` | ~35GB | ⏳ planned |

All images publish PUBLIC on GHCR (zero cost). Open-weight models + open-source ComfyUI + small patches that already live publicly — no competitive moat to protect.

## Build pipeline

GitHub Actions workflow auto-builds + pushes when files in a role's subdirectory change on `main`. Workflows live in `.github/workflows/build-{role}.yml`. Manual rebuilds via `workflow_dispatch`.

## Versioning

Each role has a `VERSION` file. Convention: `vX.Y.Z` where:
- **Z** patch — when patches change, when checkpoint version stays same
- **Y** minor — when ComfyUI / wrapper version updates, or new checkpoint variants
- **X** major — when the image's runtime contract changes (port, entrypoint, env vars)

## Independence from `kg-storytelling-platform`

This repo intentionally does not import code from `kg-storytelling-platform`. Each role's subdirectory is self-contained — the Dockerfile, entrypoint, patches, and any required Python sidecars are vendored locally. The kg-storytelling-platform service code only references the published image name (`ghcr.io/agrim-intelligence/{role}:vX.Y.Z`), never the source.

## License

Patches and orchestration code: MIT.
Bundled models retain their upstream licenses (Flux Schnell Apache-2.0, LatentSync, MusicGen, Whisper, etc. — see each role's README for the model license summary).
