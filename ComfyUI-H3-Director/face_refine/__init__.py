"""FaceRefine for H3DirectorStudio — 可选修脸，接入 face_refine 口后在最终成片帧上执行。

Ported from ComfyUI_MiniMaxH3_Director director/face_refine（Apache-2.0），
其 track/stitch/inject 又改编自 ComfyUI-H3-FaceRefine（MIT, Carasibana）。
未接线时不导入本包的运行时，零执行影响。
"""

from .pack import (
    MMX_DIR_FACE_REFINE,
    ensure_face_refine_ready,
    face_pack_to_jsonable,
    face_refine_fingerprint,
    normalize_face_refine_pack,
)

__all__ = [
    "MMX_DIR_FACE_REFINE",
    "normalize_face_refine_pack",
    "face_refine_fingerprint",
    "face_pack_to_jsonable",
    "ensure_face_refine_ready",
]
