# predenoise_splitter.py
#
# DaC_Predenoise_Splitter — Steudio / Divide and Conquer
#
# Anchors global image structure in latent space BEFORE the DaC pipeline
# splits into per-tile processing, preventing content divergence at high
# denoise (Flux.2 / SD).
#
# Workflow position:
#   DaC_Algorithm → [this node] → Divide_Image_Select → KSampler → Combine_Tiles
#
# This node outputs a FULL anchored IMAGE (not tiles). The existing DaC
# pipeline (Divide_Image_Select) handles tile splitting downstream.
#
# Three modes:
#   latent_interp  — lerp(original, upscaled) latents → decode back to image
#   two_phase      — sample N steps on SMALL original latent → upscale → decode
#   combined       — sample small → upscale → lerp with upscaled latent → decode
#
# IMPORTANT for two_phase / combined:
#   The downstream KSampler should use start_at_step from this node's output.
#   For latent_interp, start_at_step = 0 (tile KSampler starts fresh).

import os
import sys
import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers


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
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return vae.encode(image[:, :, :, :3])


def _decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    """
    VAE-decode a latent tensor [1, C, H, W] back to BHWC image [1, H, W, C].
    """
    return vae.decode(latent)


def _resize_image(image: torch.Tensor, target_h: int,
                  target_w: int) -> torch.Tensor:
    """
    Resize a BHWC image tensor to target_h x target_w using bilinear.
    """
    img = image.unsqueeze(0) if image.ndim == 3 else image
    resized = F.interpolate(
        img.permute(0, 3, 1, 2).float(),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    return resized


def _encode_resized_original(vae, original_image: torch.Tensor,
                              dac_data: dict) -> torch.Tensor:
    """
    Resize original_image to the upscaled canvas dimensions, then VAE-encode.
    """
    resized = _resize_image(
        original_image,
        dac_data['upscaled_height'],
        dac_data['upscaled_width'],
    )
    return _encode_image(vae, resized)


def _global_sample(model, latent: torch.Tensor, positive, negative,
                   sampler_name: str, scheduler: str,
                   steps: int, split_at_step: int,
                   cfg: float, seed: int) -> torch.Tensor:
    """
    Run KSampler on a latent for `split_at_step` steps only.
    Returns the partially-denoised latent tensor.
    """
    noise = comfy.sample.prepare_noise(latent, seed)
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
# Mode implementations — all return a full latent [1, C, H, W]
# ---------------------------------------------------------------------------

def _mode_latent_interp(vae, upscaled_image, original_image,
                         alpha, dac_data) -> torch.Tensor:
    """
    Lerp original_latent → upscaled_latent by alpha.
    alpha=0.0 → all original structure (safest).
    alpha=1.0 → all upscaled (same as no anchor).
    """
    lat_up   = _encode_image(vae, upscaled_image)
    lat_orig = _encode_resized_original(vae, original_image, dac_data)
    return torch.lerp(lat_orig, lat_up, alpha)


def _mode_two_phase(vae, model, original_image, positive, negative,
                    sampler_name, scheduler, steps, split_at_step,
                    cfg, seed, dac_data, lf) -> torch.Tensor:
    """
    1. VAE encode the SMALL original image (native model resolution).
    2. Run global KSampler for split_at_step steps on the small latent.
    3. Bicubic-upscale the predenoised latent to the upscaled canvas size.
    """
    small_latent = _encode_image(vae, original_image)
    predenoised  = _global_sample(model, small_latent, positive, negative,
                                  sampler_name, scheduler, steps, split_at_step,
                                  cfg, seed)
    return _upscale_latent(
        predenoised, dac_data['upscaled_height'], dac_data['upscaled_width'], lf,
    )


def _mode_combined(vae, model, upscaled_image, original_image, alpha,
                   positive, negative, sampler_name, scheduler,
                   steps, split_at_step, cfg, seed, dac_data, lf) -> torch.Tensor:
    """
    1. Sample on small original latent for split_at_step steps.
    2. Upscale the predenoised latent to full canvas size.
    3. Lerp with the upscaled_image's latent using alpha.
    """
    small_latent = _encode_image(vae, original_image)
    predenoised  = _global_sample(model, small_latent, positive, negative,
                                  sampler_name, scheduler, steps, split_at_step,
                                  cfg, seed)
    upscaled_lat = _upscale_latent(
        predenoised, dac_data['upscaled_height'], dac_data['upscaled_width'], lf,
    )
    lat_up = _encode_image(vae, upscaled_image)
    return torch.lerp(upscaled_lat, lat_up, alpha)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class DaC_Predenoise_Splitter:
    """
    Anchors global image structure before the DaC pipeline splits into tiles.
    Outputs a FULL IMAGE — the existing Divide_Image_Select handles tiling.

    Workflow:
        DaC_Algorithm ──► [this node] ──► Divide_Image_Select ──► KSampler ──► Combine

    Modes:
      latent_interp  → fastest; lerps original+upscaled latents, decodes to image.
      two_phase      → sample N steps on SMALL original latent → upscale → decode.
                        Fast, low VRAM. Downstream KSampler: start_at_step from output.
      combined       → sample small → upscale → lerp with upscaled → decode.
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

    RETURN_TYPES  = ("IMAGE", "INT",           "DAC_DATA")
    RETURN_NAMES  = ("IMAGE", "start_at_step", "dac_data")
    FUNCTION      = "execute"
    CATEGORY      = "Steudio/Divide and Conquer"
    DESCRIPTION   = (
        "Anchors global image structure before DaC tile splitting. "
        "Outputs a full IMAGE (not tiles) — connect to Divide_Image_Select. "
        "Prevents content drift / seams at high denoise (Flux.2)."
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

        # ── validate ──────────────────────────────────────────────────────
        if original_image is None:
            raise ValueError(
                "[DaC_Predenoise_Splitter] original_image is required for all modes."
            )

        needs_model = mode in ("two_phase", "combined")
        if needs_model and (model is None or positive is None or negative is None):
            raise ValueError(
                f"[DaC_Predenoise_Splitter] mode='{mode}' requires "
                "model, positive, and negative inputs to be connected."
            )

        # ── dispatch — each mode returns a full latent [1, C, H, W] ──────
        if mode == "latent_interp":
            full_latent = _mode_latent_interp(
                vae, upscaled_image, original_image,
                interp_alpha, dac_data,
            )
            start = 0

        elif mode == "two_phase":
            full_latent = _mode_two_phase(
                vae, model, original_image, positive, negative,
                sampler_name, scheduler, steps, split_at_step,
                cfg, seed, dac_data, lf,
            )
            start = split_at_step

        else:   # combined
            full_latent = _mode_combined(
                vae, model, upscaled_image, original_image, interp_alpha,
                positive, negative, sampler_name, scheduler,
                steps, split_at_step, cfg, seed, dac_data, lf,
            )
            start = split_at_step

        # ── decode back to IMAGE for downstream tile splitting ────────────
        anchored_image = _decode_latent(vae, full_latent)

        print(
            f"[DaC_Predenoise_Splitter] mode={mode} | "
            f"output={anchored_image.shape[2]}x{anchored_image.shape[1]} | "
            f"start_at_step={start}"
        )

        return (anchored_image, start, dac_data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS        = {"DaC_Predenoise_Splitter": DaC_Predenoise_Splitter}
NODE_DISPLAY_NAME_MAPPINGS = {"DaC_Predenoise_Splitter": "DaC Predenoise Splitter"}
