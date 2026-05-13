# predenoise_splitter.py
#
# DaC_Predenoise_Splitter — Steudio / Divide and Conquer
#
# Anchors global image structure in latent space BEFORE splitting into per-tile
# latents, preventing content divergence at high denoise (Flux.2 / SD).
#
# Workflow position:
#   DaC_Algorithm → [this node] → KSampler (start_at_step) → VAE Decode → Combine_Tiles
#
# Three modes:
#   latent_interp  — lerp(original, upscaled) latents → crop tiles (fastest)
#   two_phase      — sample N steps on SMALL original latent → upscale latent → crop tiles
#   combined       — sample on small latent → upscale → lerp with upscaled latent → crop tiles
#
# IMPORTANT for two_phase / combined:
#   The downstream KSampler must use "add_noise = disable" (KSamplerAdvanced)
#   because the returned tile_latents are already partially denoised.
#   For latent_interp, use standard KSampler with noise enabled (start_at_step=0).

import os
import sys
import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_latent_factor(vae) -> int:
    """
    Detect VAE spatial downscale factor from the vae object.
    Returns 8 for Flux.2 / SD / SDXL, 4 for Cascade, etc.
    Falls back to 8 if the attribute is absent.
    """
    dsr = getattr(vae, 'downscale_ratio', None)
    if isinstance(dsr, (tuple, list)):
        return int(dsr[1])
    if isinstance(dsr, (int, float)):
        return int(dsr)
    return 8


def _encode_image(vae, image: torch.Tensor) -> torch.Tensor:
    """
    VAE-encode a BHWC float32 [0,1] image tensor.
    Handles both [H,W,C] and [B,H,W,C] inputs.
    Returns the raw latent tensor [1, C, H//lf, W//lf].
    NOTE: vae.encode() returns a tensor, not a dict.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)            # → [1, H, W, C]
    return vae.encode(image[:, :, :, :3])     # drop alpha channel if present


def _encode_resized_original(vae, original_image: torch.Tensor,
                              dac_data: dict) -> torch.Tensor:
    """
    Resize original_image to the upscaled canvas dimensions, then VAE-encode.
    Keeps original_image in float32 [0,1] range throughout.
    """
    H_up = dac_data['upscaled_height']
    W_up = dac_data['upscaled_width']

    orig = original_image.unsqueeze(0) if original_image.ndim == 3 else original_image
    # BHWC → BCHW for interpolate, then back
    resized = F.interpolate(
        orig.permute(0, 3, 1, 2).float(),
        size=(H_up, W_up),
        mode='bilinear',
        align_corners=False,
    ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

    return _encode_image(vae, resized)


def _crop_latents(latent_tensor: torch.Tensor, dac_data: dict,
                  lf: int) -> list:
    """
    Crop a full latent [1, C, H, W] into a list of N tile latents,
    each shaped [1, C, th//lf, tw//lf].
    Returns a list (not a batch) so OUTPUT_IS_LIST can feed each tile
    individually into the downstream per-tile KSampler loop.
    """
    tw_px = dac_data['tile_width']
    th_px = dac_data['tile_height']
    tw = tw_px // lf
    th = th_px // lf

    coords, _ = create_tile_coordinates(
        dac_data['upscaled_width'],
        dac_data['upscaled_height'],
        tw_px, th_px,
        dac_data['overlap_x'],
        dac_data['overlap_y'],
        dac_data['grid_x'],
        dac_data['grid_y'],
        dac_data['tile_order'],
    )

    _, C, H, W = latent_tensor.shape
    tiles = []
    for (x, y) in coords:
        lx = min(x // lf, W - tw)
        ly = min(y // lf, H - th)
        tiles.append(latent_tensor[:, :, ly : ly + th, lx : lx + tw].clone())

    return tiles   # list of [1, C, th, tw] tensors


def _global_sample(model, latent: torch.Tensor, positive, negative,
                   sampler_name: str, scheduler: str,
                   steps: int, split_at_step: int,
                   cfg: float, seed: int) -> torch.Tensor:
    """
    Run KSampler on a latent for `split_at_step` steps only.
    Uses comfy.sample.sample() — the same path as the standard KSampler node.
    Returns the partially-denoised latent tensor.
    """
    noise = comfy.sample.prepare_noise(latent, seed)

    # comfy.sample.sample returns a tensor (already moved to intermediate device)
    predenoised = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=0,
        last_step=split_at_step,
        force_full_denoise=False,
        seed=seed,
    )
    return predenoised


def _upscale_latent(latent: torch.Tensor, target_h: int, target_w: int,
                    lf: int) -> torch.Tensor:
    """
    Upscale a latent tensor [1, C, h, w] to target pixel dimensions / lf.
    Uses bicubic interpolation for smooth latent upscaling.
    """
    th = target_h // lf
    tw = target_w // lf
    return F.interpolate(
        latent.float(), size=(th, tw), mode='bicubic', align_corners=False,
    ).to(latent.dtype)


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _mode_latent_interp(vae, upscaled_image, original_image,
                         alpha, dac_data, lf) -> torch.Tensor:
    """
    Lerp original_latent → upscaled_latent by alpha, then crop into tile latents.
    alpha=0.0 → all original structure (safest).
    alpha=1.0 → all upscaled (same as no anchor).
    start_at_step returned by node = 0  (tile KSampler starts fresh).
    """
    lat_up   = _encode_image(vae, upscaled_image)
    lat_orig = _encode_resized_original(vae, original_image, dac_data)
    mixed    = torch.lerp(lat_orig, lat_up, alpha)
    return _crop_latents(mixed, dac_data, lf)


def _mode_two_phase(vae, model, original_image, positive, negative,
                    sampler_name, scheduler, steps, split_at_step,
                    cfg, seed, dac_data, lf) -> torch.Tensor:
    """
    1. VAE encode the SMALL original image (native model resolution).
    2. Run global KSampler for split_at_step steps on the small latent
       → fast, low VRAM, correct base resolution for global structure.
    3. Bicubic-upscale the predenoised latent to the upscaled canvas size.
    4. Crop the upscaled latent into tile latents.
    start_at_step returned by node = split_at_step.
    DOWNSTREAM KSampler must use add_noise=disable (KSamplerAdvanced).
    """
    small_latent = _encode_image(vae, original_image)
    predenoised  = _global_sample(model, small_latent, positive, negative,
                                  sampler_name, scheduler, steps, split_at_step,
                                  cfg, seed)
    upscaled_lat = _upscale_latent(
        predenoised, dac_data['upscaled_height'], dac_data['upscaled_width'], lf,
    )
    return _crop_latents(upscaled_lat, dac_data, lf)


def _mode_combined(vae, model, upscaled_image, original_image, alpha,
                   positive, negative, sampler_name, scheduler,
                   steps, split_at_step, cfg, seed, dac_data, lf) -> torch.Tensor:
    """
    1. VAE encode the SMALL original image.
    2. Run global KSampler for split_at_step steps on small latent (structure).
    3. Bicubic-upscale the predenoised latent to upscaled canvas size.
    4. Lerp the upscaled predenoised latent with the upscaled_image's latent
       using alpha → blends global structure with upscaled detail.
    5. Crop into tile latents.
    start_at_step returned by node = split_at_step.
    DOWNSTREAM KSampler must use add_noise=disable (KSamplerAdvanced).
    """
    # Phase 1: sample on small original
    small_latent = _encode_image(vae, original_image)
    predenoised  = _global_sample(model, small_latent, positive, negative,
                                  sampler_name, scheduler, steps, split_at_step,
                                  cfg, seed)
    upscaled_lat = _upscale_latent(
        predenoised, dac_data['upscaled_height'], dac_data['upscaled_width'], lf,
    )
    # Phase 2: blend with upscaled image's latent for detail
    lat_up = _encode_image(vae, upscaled_image)
    mixed  = torch.lerp(upscaled_lat, lat_up, alpha)
    return _crop_latents(mixed, dac_data, lf)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class DaC_Predenoise_Splitter:
    """
    Insert between DaC_Algorithm and the per-tile KSampler to prevent tile
    content divergence at high denoise settings.

    Workflow:
        DaC_Algorithm ──► [this node] ──► KSampler ──► VAE Decode ──► Combine_Tiles

    Modes:
      latent_interp  → fastest; lerps original+upscaled latents before splitting.
                        Use standard KSampler with add_noise enabled.
      two_phase      → sample N steps on SMALL original latent → upscale → split.
                        Fast, low VRAM. Use KSamplerAdvanced with add_noise = 'disable'.
      combined       → sample small → upscale → lerp with upscaled latent → split.
                        Use KSamplerAdvanced with add_noise = 'disable'.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_image": ("IMAGE",),
                "vae":            ("VAE",),
                "dac_data":       ("DAC_DATA",),
                "mode": (
                    ["latent_interp", "two_phase", "combined"],
                    {"default": "latent_interp"},
                ),
            },
            "optional": {
                # ── all modes need original_image ────────────────────────
                "original_image": ("IMAGE",),
                "interp_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Blend between original (0.0) and upscaled (1.0) latents. "
                        "Lower values = stronger structural anchor."
                    ),
                }),
                # ── two_phase / combined ──────────────────────────────────
                "model":    ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "euler"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Total steps — must match the downstream KSampler.",
                }),
                "split_at_step": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "tooltip": (
                        "How many global steps to run before splitting into tiles. "
                        "Returned as start_at_step for the downstream KSampler."
                    ),
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
        }

    RETURN_TYPES   = ("LATENT", "INT",           "DAC_DATA")
    RETURN_NAMES   = ("tile_latents", "start_at_step", "dac_data")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION       = "execute"
    CATEGORY       = "Steudio/Divide and Conquer"
    DESCRIPTION    = (
        "Anchors global image structure before splitting into per-tile latents. "
        "Prevents content drift / seams at high denoise (Flux.2). "
        "Replaces Divide_Image_Select + VAE Encode. "
        "tile_latents is a LIST — ComfyUI loops each tile through downstream KSampler. "
        "For two_phase / combined modes, set KSamplerAdvanced add_noise = 'disable'."
    )

    def execute(
        self,
        upscaled_image,
        vae,
        dac_data,
        mode,
        original_image=None,
        interp_alpha=0.5,
        model=None,
        positive=None,
        negative=None,
        sampler_name="euler",
        scheduler="simple",
        steps=20,
        split_at_step=5,
        cfg=1.0,
        seed=0,
    ):
        lf = _get_latent_factor(vae)

        # ── validate mode requirements ────────────────────────────────────
        needs_model = mode in ("two_phase", "combined")

        if original_image is None:
            raise ValueError(
                "[DaC_Predenoise_Splitter] original_image is required for all modes."
            )

        if needs_model and (model is None or positive is None or negative is None):
            raise ValueError(
                f"[DaC_Predenoise_Splitter] mode='{mode}' requires "
                "model, positive, and negative inputs to be connected."
            )

        # ── dispatch ──────────────────────────────────────────────────────
        if mode == "latent_interp":
            tiles = _mode_latent_interp(
                vae, upscaled_image, original_image,
                interp_alpha, dac_data, lf,
            )
            start = 0

        elif mode == "two_phase":
            tiles = _mode_two_phase(
                vae, model, original_image, positive, negative,
                sampler_name, scheduler, steps, split_at_step,
                cfg, seed, dac_data, lf,
            )
            start = split_at_step

        else:   # combined
            tiles = _mode_combined(
                vae, model, upscaled_image, original_image, interp_alpha,
                positive, negative, sampler_name, scheduler,
                steps, split_at_step, cfg, seed, dac_data, lf,
            )
            start = split_at_step

        n = len(tiles)
        th, tw = tiles[0].shape[-2], tiles[0].shape[-1]
        print(
            f"[DaC_Predenoise_Splitter] mode={mode} | "
            f"{n} tiles @ {th}x{tw} latent | "
            f"start_at_step={start}"
        )

        # Wrap each tile tensor as a LATENT dict for ComfyUI
        tile_latent_dicts = [{"samples": t} for t in tiles]
        return (tile_latent_dicts, start, dac_data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS        = {"DaC_Predenoise_Splitter": DaC_Predenoise_Splitter}
NODE_DISPLAY_NAME_MAPPINGS = {"DaC_Predenoise_Splitter": "DaC Predenoise Splitter"}
