#!/usr/bin/env python3
"""Step 2c — Patch LatentSync nodes.py for read_video cv2 fallback.

torchvision.io.read_video uses PyAV internally. On some RunPod images,
libswscale fails with EAGAIN ("Resource temporarily unavailable") when
converting yuv420p -> rgb24. cv2's ffmpeg backend also fails on these pods.
This patch wraps the read_video call with a try/except that falls back to
opening with cv2.VideoCapture and reading frame-by-frame — works regardless
of system libs.

Note: patch_ci_o_22_write_video.py already replaces the read_video call
site with an imageio reader (CI-O-22c). This patch is the secondary cv2
fallback layer used when the primary CI-O-22c imageio path is not enough
(legacy nodes.py shape). On a freshly-cloned wrapper that has been touched
by the CI-O-22 patch, this script is largely a no-op — it exits 0 cleanly
when it sees the CI-O-22c marker. Kept as a separate file to mirror the
deploy_latentsync_comfyui.sh Step 2c behaviour exactly.

Run from the Dockerfile build:
    python3 patch_latentsync_read_video.py /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper
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
    if "\r\n" in code:
        code = code.replace("\r\n", "\n")
        nodes_path.write_text(code)
        print("  Patch C: normalised CRLF → LF in nodes.py")

    if "cv2 fallback" in code and "cv2.VideoCapture" in code:
        print("  Patch C: nodes.py — already patched (cv2 fallback)")
        return 0

    if "CI-O-22c" in code:
        # The CI-O-22 patch already replaced the read_video line with imageio.
        # No additional cv2 fallback to apply against the legacy line.
        print("  Patch C: CI-O-22c imageio reader already in place — skipping cv2 fallback")
        return 0

    old_line = "            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]"

    new_block = """            try:
                processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]
            except Exception as _av_err:
                print(f"[LatentSync] torchvision read_video failed: {_av_err}, using cv2 fallback")
                import cv2
                _cap = cv2.VideoCapture(output_video_path)
                _frames = []
                while True:
                    _ret, _frame = _cap.read()
                    if not _ret:
                        break
                    _frames.append(torch.from_numpy(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)))
                _cap.release()
                if not _frames:
                    raise RuntimeError(f"cv2 also failed: no frames from {output_video_path}") from _av_err
                processed_frames = torch.stack(_frames)
                print(f"[LatentSync] cv2 fallback: read {len(_frames)} frames")"""

    if old_line not in code:
        print("  Patch C: WARNING — neither CI-O-22c nor legacy read_video line found")
        return 0  # not fatal — CI-O-22 main patch is the load-bearing one

    code = code.replace(old_line, new_block, 1)
    nodes_path.write_text(code)
    print("  Patch C: nodes.py — read_video with cv2 fallback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
