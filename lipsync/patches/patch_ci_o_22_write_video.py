#!/usr/bin/env python3
"""CI-O-22: Patch ComfyUI-LatentSyncWrapper/nodes.py for torchvision >= 0.20.

torchvision 0.20 removed torchvision.io.write_video. LatentSyncNode called it
directly and fell through to an `except TypeError` that only triggered for
macro_block_size errors — AttributeError went unhandled, every inference died
with `module 'torchvision.io' has no attribute 'write_video'`.

This patch replaces the torchvision.io.write_video attempt with a direct
imageio.mimsave call (imageio is already a dep of the wrapper).

Run from the deploy script as Step 2d:
    python3 patch_ci_o_22_write_video.py /workspace/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper
"""
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <wrapper_dir>", file=sys.stderr)
        return 2

    wrapper = Path(sys.argv[1])
    nodes_path = wrapper / "nodes.py"
    if not nodes_path.is_file():
        print(f"ERROR: {nodes_path} not found", file=sys.stderr)
        return 1

    code = nodes_path.read_text()

    # CI-O-22m (re-applied 2026-04-28): Normalise CRLF → LF so the in-text
    # matches don't silently miss when the upstream wrapper ships with Windows
    # line endings (verified on a fresh clone of ComfyUI-LatentSyncWrapper —
    # the wrapper has CRLF endings as committed). Without this normalisation
    # the `if old_block in code` check returns False, the patch silently exits
    # rc=1, the deploy script's `grep -q` verification at line 412-419 catches
    # it (good!) but only if exit-11 propagates — which it does when run
    # standalone via remote. CRLF normalisation is required for the patch to
    # succeed on a fresh wrapper clone, so the deploy can proceed past Step 2d.
    if "\r\n" in code:
        code = code.replace("\r\n", "\n")
        nodes_path.write_text(code)
        print("  Patch CI-O-22: normalised CRLF → LF in nodes.py")

    if "CI-O-22 PATCH APPLIED" in code:
        print("  Patch CI-O-22: nodes.py — already patched")
        return 0

    old_block = """            # Move frames to CPU for saving to video
            frames_cpu = frames.cpu()
            try:
                import torchvision.io as io
                io.write_video(temp_video_path, frames_cpu, fps=25, video_codec='h264')
            except TypeError as e:
                # Check if the error is specifically about macro_block_size
                if "macro_block_size" in str(e):
                    import imageio
                    # Use imageio with macro_block_size parameter
                    imageio.mimsave(temp_video_path, frames_cpu.numpy(), fps=25, codec='h264', macro_block_size=1)
                else:"""

    new_block = """            # Move frames to CPU for saving to video
            # CI-O-22 PATCH APPLIED: skip torchvision.io.write_video (removed in tv >= 0.20)
            # and use imageio directly with macro_block_size=1 for non-multiple-of-16 dims.
            frames_cpu = frames.cpu()
            try:
                import imageio
                imageio.mimsave(temp_video_path, frames_cpu.numpy(), fps=25, codec='h264', macro_block_size=1)
            except TypeError as e:
                if "macro_block_size" in str(e):
                    import imageio
                    imageio.mimsave(temp_video_path, frames_cpu.numpy(), fps=25, codec='h264', macro_block_size=1)
                else:"""

    if old_block not in code:
        print("  Patch CI-O-22: WARNING — expected block not found in nodes.py")
        print("  (The wrapper may have changed upstream; verify manually.)")
        return 1

    code = code.replace(old_block, new_block)

    # CI-O-22c: torchvision >= 0.20 also removed io.read_video. Replace the call
    # site with an imageio-based reader that returns [N, H, W, C] uint8 tensors —
    # matching what torchvision.io.read_video()[0] used to return.
    read_old = "            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]"
    read_new = (
        "            # CI-O-22c: torchvision.io.read_video removed in tv >= 0.20 → use imageio\n"
        "            import imageio\n"
        "            import numpy as np, torch as _torch\n"
        "            _reader = imageio.get_reader(output_video_path, 'ffmpeg')\n"
        "            _frames_list = [f for f in _reader]\n"
        "            _reader.close()\n"
        "            processed_frames = _torch.from_numpy(np.stack(_frames_list, axis=0))"
    )
    if "CI-O-22c" not in code and read_old in code:
        code = code.replace(read_old, read_new)
        print("  Patch CI-O-22c: nodes.py — io.read_video → imageio.get_reader")

    nodes_path.write_text(code)
    print("  Patch CI-O-22: nodes.py — torchvision.io.write_video → imageio.mimsave")
    return 0


if __name__ == "__main__":
    sys.exit(main())
