#!/usr/bin/env python3
"""Step 2b — Patch LatentSync for no-face graceful handling.

AI-generated (WAN) videos have frames with no face. Stock LatentSync
crashes with RuntimeError("Face not detected"). Two patches applied:

Patch A — image_processor.py:
    raise RuntimeError("Face not detected") -> return None, None, None

Patch B — lipsync_pipeline.py:
    affine_transform_video: track no_face_indices, use placeholder tensors
    restore_video: skip no-face frames, keep original unchanged
    (Prevents black squares from placeholder tensors being composited back)

Original logic was inlined as a heredoc in deploy_latentsync_comfyui.sh
Step 2b. Lifted out into a standalone file so the docker build can apply it
identically to the deploy.sh pathway.

Run from the Dockerfile build:
    python3 patch_no_face_graceful.py /opt/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper
"""
import sys
from pathlib import Path


def _normalise_crlf(path: Path) -> str:
    """Read file and convert CRLF -> LF in memory; rewrite if needed."""
    code = path.read_text()
    if "\r\n" in code:
        code = code.replace("\r\n", "\n")
        path.write_text(code)
    return code


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <wrapper_dir>", file=sys.stderr)
        return 2

    wrapper = Path(sys.argv[1])
    ip_path = wrapper / "latentsync" / "utils" / "image_processor.py"
    lp_path = wrapper / "latentsync" / "pipelines" / "lipsync_pipeline.py"

    if not ip_path.is_file():
        print(f"ERROR: {ip_path} not found", file=sys.stderr)
        return 1
    if not lp_path.is_file():
        print(f"ERROR: {lp_path} not found", file=sys.stderr)
        return 1

    # ── Patch A: image_processor.py ──
    code = _normalise_crlf(ip_path)
    old_raise = '            raise RuntimeError("Face not detected")'
    new_return = '            return None, None, None  # No face — caller handles gracefully'
    if old_raise in code:
        code = code.replace(old_raise, new_return)
        ip_path.write_text(code)
        print("  Patch A: image_processor.py — raise → return None")
    elif "return None, None, None" in code:
        print("  Patch A: image_processor.py — already patched")
    else:
        print("  Patch A: WARNING — could not find raise RuntimeError in image_processor.py")
        return 1

    # ── Patch B: lipsync_pipeline.py ──
    code = _normalise_crlf(lp_path)

    if "no_face_indices" in code:
        print("  Patch B: lipsync_pipeline.py — already patched")
        return 0

    # B1: Patch affine_transform_video
    old_atv = '''    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices'''

    new_atv = '''    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        no_face_indices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for idx, frame in enumerate(tqdm.tqdm(video_frames)):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                no_face_indices.append(idx)
                h, w = frame.shape[:2]
                placeholder = torch.zeros(3, self.image_processor.resolution, self.image_processor.resolution)
                faces.append(placeholder)
                boxes.append([0, 0, w, h])
                affine_matrices.append(np.eye(2, 3, dtype=np.float64))
            else:
                faces.append(face)
                boxes.append(box)
                affine_matrices.append(affine_matrix)
        self._no_face_indices = set(no_face_indices)
        if no_face_indices:
            print(f"  [LatentSync] No face in {len(no_face_indices)}/{len(video_frames)} frames — passthrough")
        if len(no_face_indices) == len(video_frames):
            raise RuntimeError("Face not detected in ANY frame — cannot apply lip sync")
        faces = torch.stack(faces)
        return faces, boxes, affine_matrices'''

    if old_atv in code:
        code = code.replace(old_atv, new_atv)
        print("  Patch B1: affine_transform_video — no-face tracking")
    else:
        print("  Patch B1: WARNING — could not find affine_transform_video method")
        return 1

    # B2: Patch restore_video to skip no-face frames (prevents black squares)
    old_restore_loop = '''        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]'''

    new_restore_loop = '''        no_face = getattr(self, "_no_face_indices", set())
        for index, face in enumerate(tqdm.tqdm(faces)):
            if index in no_face:
                out_frames.append(video_frames[index])
                continue
            x1, y1, x2, y2 = boxes[index]'''

    if old_restore_loop in code:
        code = code.replace(old_restore_loop, new_restore_loop, 1)
        print("  Patch B2: restore_video — skip no-face frames (no black squares)")
    else:
        print("  Patch B2: WARNING — could not find restore_video loop")
        return 1

    lp_path.write_text(code)
    print("  Patch B: lipsync_pipeline.py — done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
