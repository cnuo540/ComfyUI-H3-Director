# -*- coding: utf-8 -*-
"""Director.face_refine 参数包：校验 / 缓存指纹 / JSON 往返 / 前置检查。

入参 raw 来自 MiniMax H3 Director FaceRefine 节点（其 pack_face_refine 已做过
一轮钳制），这里做 7.0 侧的防御性校验与默认值补齐，不依赖 MiniMax 插件的任何
模块（跨插件只共享 MMX_DIR_FACE_REFINE 类型字符串与字典字段约定）。
"""

from __future__ import annotations

from typing import Any

MMX_DIR_FACE_REFINE = "MMX_DIR_FACE_REFINE"

SELECT_MODES = ("largest_face", "centre_most")
CANVAS_MODES = ("manual", "auto_capped_768")
PASTE_REGIONS = ("face_only", "face_ellipse", "full_crop")
SEED_MODES = ("inherit", "offset")
DEFAULT_DETECTOR = "face_yolov8m.pt"
CANVAS_STRIDE = 32


def _clamp_int(raw: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _clamp_float(raw: Any, lo: float, hi: float, default: float) -> float:
    try:
        n = float(raw)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _snap_canvas(value: int) -> int:
    """Snap 到 32 的倍数（H3：VAE÷16 后再 2×2 patch）。"""
    v = _clamp_int(value, 128, 1344, 768)
    return max(CANVAS_STRIDE, round(v / CANVAS_STRIDE) * CANVAS_STRIDE)


def resolve_detector_path(name: str) -> str:
    """定位人脸检测权重；找不到抛 FileNotFoundError（信息含放置目录）。"""
    import os

    import folder_paths

    name = str(name or DEFAULT_DETECTOR).strip() or DEFAULT_DETECTOR
    path = None
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.isfile(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            "[H3导演台] 修脸检测器 '%s' 未找到。请把 face_yolov8m.pt 放到 "
            "ComfyUI/models/ultralytics/bbox/ 后重试。" % name
        )
    return path


def normalize_face_refine_pack(raw) -> dict[str, Any] | None:
    """未接线 / 无效 → None（Director 直接跳过修脸）。"""
    if raw is None or not isinstance(raw, dict):
        return None
    if raw.get("enabled") is False:
        return None
    canvas_w = _snap_canvas(raw.get("canvas_width", 768))
    canvas_h = _snap_canvas(raw.get("canvas_height", 768))
    mode = str(raw.get("canvas_mode") or "manual").strip().lower()
    if mode not in CANVAS_MODES:
        mode = "manual"
    sel = str(raw.get("select") or "largest_face").strip().lower()
    if sel not in SELECT_MODES:
        sel = "largest_face"
    paste = str(raw.get("paste_region") or "face_only").strip().lower()
    if paste not in PASTE_REGIONS:
        paste = "face_only"
    seed = str(raw.get("seed_mode") or "inherit").strip().lower()
    if seed not in SEED_MODES:
        seed = "inherit"
    sigmas_tensor = raw.get("sigmas_tensor")
    try:
        import torch

        if not torch.is_tensor(sigmas_tensor):
            sigmas_tensor = None
    except Exception:
        sigmas_tensor = None
    parsed = raw.get("sigmas_parsed") or ()
    try:
        sigmas_parsed = tuple(float(x) for x in parsed)
    except (TypeError, ValueError):
        sigmas_parsed = ()
    return {
        "enabled": True,
        "detector": str(raw.get("detector") or DEFAULT_DETECTOR).strip() or DEFAULT_DETECTOR,
        "confidence": _clamp_float(raw.get("confidence", 0.35), 0.05, 0.95, 0.35),
        "crop_factor": _clamp_float(raw.get("crop_factor", 2.5), 1.2, 8.0, 2.5),
        "canvas_width": int(canvas_w),
        "canvas_height": int(canvas_h),
        "canvas_mode": mode,
        "select": sel,
        "denoise": _clamp_float(raw.get("denoise", 0.40), 0.02, 1.0, 0.40),
        "steps": _clamp_int(raw.get("steps", 8), 1, 50, 8),
        "sampler": str(raw.get("sampler") or "euler").strip() or "euler",
        "scheduler": str(raw.get("scheduler") or "simple").strip() or "simple",
        "seed_mode": seed,
        "paste_region": paste,
        "mask_dilation": _clamp_int(raw.get("mask_dilation", 16), 0, 256, 16),
        "feather": _clamp_int(raw.get("feather", 24), 0, 256, 24),
        "colour_match": _clamp_float(raw.get("colour_match", 1.0), 0.0, 1.0, 1.0),
        "blend": _clamp_float(raw.get("blend", 1.0), 0.0, 1.0, 1.0),
        "sigmas_parsed": sigmas_parsed,
        "sigmas_tensor": sigmas_tensor,
        "has_sigmas_tensor": sigmas_tensor is not None,
    }


def face_pack_to_jsonable(pack: dict[str, Any]) -> dict[str, Any]:
    """样本 JSON 往返用：去掉不可序列化的 sigmas_tensor（parsed 足以重建）。"""
    out = {k: v for k, v in (pack or {}).items() if k != "sigmas_tensor"}
    parsed = out.get("sigmas_parsed")
    if parsed is not None:
        out["sigmas_parsed"] = [float(x) for x in parsed]
    return out


def face_refine_fingerprint(pack: dict[str, Any]) -> dict[str, Any] | None:
    """段缓存指纹：未启用 → None（run_cfg 不加键，老缓存哈希保持不变）。"""
    if not isinstance(pack, dict) or not pack.get("enabled"):
        return None
    return {
        "detector": str(pack.get("detector") or DEFAULT_DETECTOR),
        "confidence": round(float(pack.get("confidence") or 0), 4),
        "crop": round(float(pack.get("crop_factor") or 0), 4),
        "canvas": "%dx%d:%s" % (
            int(pack.get("canvas_width") or 0), int(pack.get("canvas_height") or 0),
            str(pack.get("canvas_mode") or "manual")),
        "select": str(pack.get("select") or ""),
        "denoise": round(float(pack.get("denoise") or 0), 4),
        "steps": int(pack.get("steps") or 0),
        "sampler": str(pack.get("sampler") or ""),
        "scheduler": str(pack.get("scheduler") or ""),
        "seed_mode": str(pack.get("seed_mode") or ""),
        "paste": str(pack.get("paste_region") or ""),
        "feather": int(pack.get("feather") or 0),
        "dilation": int(pack.get("mask_dilation") or 0),
        "colour": round(float(pack.get("colour_match") or 0), 4),
        "blend": round(float(pack.get("blend") or 0), 4),
        "sigmas": ",".join("%g" % x for x in (pack.get("sigmas_parsed") or ())),
    }


def ensure_face_refine_ready(pack: dict[str, Any]) -> None:
    """任何采样开始前的快速失败：依赖 + 检测权重 + 预加载（构造在正常模式，避免
    inference 张量权重问题；并让缺依赖在排队后第一秒就报错，而不是生成完才炸）。"""
    if not pack:
        return
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        raise ValueError(
            "[H3导演台] 修脸需要 ultralytics，请先 pip install ultralytics 并重启 ComfyUI。")
    resolve_detector_path(str(pack.get("detector") or DEFAULT_DETECTOR))
    from .track import load_detector

    load_detector(str(pack.get("detector") or DEFAULT_DETECTOR))
