# -*- coding: utf-8 -*-
"""Run one face-refine pass on decoded H3DirectorStudio frames（接线才导入）。

与 MiniMax 版 runtime 的差异（7.0 架构适配）：
- 模型由图内 MiniMaxH3SigmaShift 预移位，这里**不再**二次 apply shift；
- 采样镜像 studio `_first_sample_prepared_latent`（Guider_Basic + sampler_object），
  sigmas 优先用节点 sigmas 口解析值，否则 BasicScheduler(scheduler, steps, denoise)；
- 条件走官方 MiniMaxH3ReferenceToVideo（与主流程同一入口），只送清洗后的段提示词，
  不带参考素材（注入的裁剪 latent 已携带外观，denoise≈0.4 保持结构）；
- 只解码视频流，音频全程不动（段音频由主流程负责）。
"""

from __future__ import annotations

import gc
import logging
import re
from typing import Any

import torch

log = logging.getLogger("ComfyUI-H3-Director.face_refine")

_TAG_RE = re.compile(r"<\s*(?:Picture|Video|Audio)\s*\d+\s*>", re.IGNORECASE)
_FALLBACK_PROMPT = "a person, detailed face, sharp focus, high quality"


def _pad_frames(frames: torch.Tensor, length: int) -> torch.Tensor:
    if frames.shape[0] >= length:
        return frames[:length]
    last = frames[-1:].expand(length - frames.shape[0], *frames.shape[1:])
    return torch.cat([frames, last], dim=0)


def _align_frame_count(frame_count: int) -> int:
    """Round up to MiniMax H3 17k+5 frame grid (5, 22, 39, …)."""
    n = max(5, int(frame_count))
    while n % 17 != 5:
        n += 1
    return n


def _clean_face_prompt(prompt: Any) -> str:
    """去掉 <Picture N>/<Video N>/<Audio N> 等素材标签（修脸条件不带素材块）。"""
    text = str(prompt or "")
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unpack_result(out) -> Any:
    r = getattr(out, "result", out)
    if isinstance(r, tuple):
        return r[0] if len(r) == 1 else r
    return r


def apply_segment_face_refine(
    *,
    frames: torch.Tensor,
    pack: dict[str, Any],
    prompt: str,
    ref_image_size: str,
    model,
    vae,
    audio_vae,
    clip,
    seed: int,
) -> tuple[torch.Tensor, str]:
    """Return (stitched frames [N,H,W,3] float 0..1, report note)。

    ``frames`` 为本段**最终成片帧**（二采关 = 一采解码；二采开 = 二采解码）。
    检测不到人脸时原样返回并在 note 里带 FACE_REFINE_SKIP_NO_FACE 前缀。
    """
    import comfy
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
    from comfy_extras.nodes_custom_sampler import BasicScheduler, Guider_Basic, Noise_RandomNoise
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    from .inject import inject_video_latent
    from .stitch import stitch_faces
    from .track import FACE_REFINE_SKIP_NO_FACE, track_and_crop

    if frames is None or not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("[H3导演台] 修脸需要解码后的视频帧 [N,H,W,C]。")
    base = frames[..., :3].contiguous().float().cpu()
    n_src = int(base.shape[0])
    crops, transform, track_note = track_and_crop(base, pack)
    if crops is None or transform is None:
        return base[:n_src].contiguous(), track_note

    canvas_w, canvas_h = transform["canvas"]
    if canvas_w % 32 or canvas_h % 32:
        raise ValueError(
            "[H3导演台] 修脸画布 %dx%d 不是 32 的倍数（H3 要求）。" % (canvas_w, canvas_h))
    gen_len = _align_frame_count(int(crops.shape[0]))
    crop_in = _pad_frames(crops, gen_len)

    face_prompt = _clean_face_prompt(prompt) or _FALLBACK_PROMPT

    # 与主流程同一官方条件入口；不带参考素材（裁剪 latent 已是外观来源）。
    out = MiniMaxH3ReferenceToVideo.execute(
        clip, vae, audio_vae, face_prompt,
        int(canvas_w), int(canvas_h), int(gen_len),
        ref_image_size=str(ref_image_size or "match"),
        ref_images=None, ref_audios=None, ref_videos=None, ref_video_audios=None)
    res = out.result
    positive, latent = res[0], res[1]

    latent = inject_video_latent(latent, crop_in, vae)
    del crop_in, crops, res, out

    steps = int(pack.get("steps") or 8)
    denoise = float(pack.get("denoise") or 0.40)
    sampler_name = str(pack.get("sampler") or "euler")
    scheduler = str(pack.get("scheduler") or "simple")

    sigma_t = None
    if pack.get("has_sigmas_tensor") and pack.get("sigmas_tensor") is not None:
        sigma_t = pack.get("sigmas_tensor")
    elif pack.get("sigmas_parsed"):
        sigma_t = torch.tensor(
            [float(x) for x in pack["sigmas_parsed"]], dtype=torch.float32)
    if sigma_t is None:
        sigma_t = _unpack_result(
            BasicScheduler.execute(model, scheduler, steps, denoise))
    sigma_t = sigma_t.detach().float().cpu().reshape(-1)

    sampler = comfy.samplers.sampler_object(sampler_name)
    guider = Guider_Basic(model)
    guider.set_conds(positive)
    noise = Noise_RandomNoise(int(seed))
    x0_output = {}
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigma_t.shape[-1] - 1, x0_output)
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher, latent["samples"],
        latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
    sampled = guider.sample(
        noise.generate_noise(latent), latent_image, sampler, sigma_t,
        callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=noise.seed)
    sampled = sampled.to(comfy.model_management.intermediate_device())
    del latent, latent_image, positive, guider, callback, x0_output, noise, sigma_t, sampler

    video_lat = sampled
    if isinstance(sampled, dict):
        video_lat = sampled.get("samples", sampled)
    if isinstance(video_lat, dict):
        video_lat = video_lat.get("samples", video_lat)
    if getattr(video_lat, "is_nested", False):
        video_lat = video_lat.unbind()[0]
    refined = vae.decode(video_lat)
    if refined.dim() == 5:
        refined = refined[0]
    del sampled, video_lat
    refined = refined[: int(n_src), ..., :3].float().cpu()
    if refined.shape[0] < n_src:
        refined = _pad_frames(refined, n_src)

    # 贴回前释放模型（修复A）：贴回是纯张量运算，不需要 UNET/CLIP/VAE；此刻它们
    # 刚被修脸采样+解码用完还驻留显存。8GB 卡上贴回的 CPU→GPU 拷贝曾与 20GB 流式
    # 模型抢位置炸出访问违例（2026-09-24 段2）。先退场，贴回变空场作业。
    try:
        from ..studio_node import _cleanup_runtime_resources
        _cleanup_runtime_resources(deep=True, reason="贴回前释放模型（修脸链路）")
    except Exception:
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

    stitched = stitch_faces(base, refined, transform, pack)
    stitched = stitched[:n_src].contiguous().cpu().float()
    note = (
        f"{track_note}; {sampler_name} {steps}步 denoise={denoise:.2f} "
        f"seed={int(seed)} | 提示词条件(无素材块)"
    )
    del base, refined, transform
    return stitched, note
