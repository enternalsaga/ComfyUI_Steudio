# Design: DaC_Predenoise_Splitter Node

**Date:** 2026-05-12  
**Status:** Approved  
**Target:** `ComfyUI_Steudio` — Divide and Conquer category

---

## Problem

When using Steudio DaC (Divide and Conquer) workflow with Flux.2 at high denoise (>0.8),
each tile is processed independently → tiles diverge in content → visible misalignment when stitched.

Root cause: each tile has no awareness of global image context during its KSampler run.

---

## Solution: DaC_Predenoise_Splitter

A single new node inserted between `DaC_Algorithm` and the per-tile `KSampler`.
It establishes global context in the latent before splitting into tiles, using one of three modes.

---

## Workflow Position

```
[Load Image (original)] ─────────────────────────┐
[DaC_Algorithm] → upscaled_image + dac_data ──── [DaC_Predenoise_Splitter]
[VAE] ────────────────────────────────────────────┘
         ↓ tile_latents (LATENT batch)  +  start_at_step
[KSampler] (start_at_step from node output, high denoise)
         ↓
[VAE Decode] → tile images
         ↓
[Combine_Tiles] (unchanged)
```

---

## Node Interface

### Inputs

| Name | Type | Required | Notes |
|------|------|----------|-------|
| `upscaled_image` | IMAGE | ✅ | Output of DaC_Algorithm |
| `vae` | VAE | ✅ | For encoding |
| `dac_data` | DAC_DATA | ✅ | Tile grid info from DaC_Algorithm |
| `mode` | ENUM | ✅ | `latent_interp`, `two_phase`, `combined` |
| `original_image` | IMAGE | optional | Required for interp modes |
| `interp_alpha` | FLOAT [0,1] | optional | 0=all original, 1=all upscaled. Default 0.5 |
| `model` | MODEL | optional | Required for two_phase/combined |
| `positive` | CONDITIONING | optional | Required for two_phase/combined |
| `negative` | CONDITIONING | optional | Required for two_phase/combined |
| `sampler_name` | ENUM | optional | KSampler sampler |
| `scheduler` | ENUM | optional | KSampler scheduler |
| `steps` | INT | optional | Total steps (same as downstream KSampler) |
| `split_at_step` | INT | optional | How many global steps before split. Default 5 |
| `cfg` | FLOAT | optional | Default 1.0 (Flux guidance) |
| `seed` | INT | optional | Default 0 |

### Outputs

| Name | Type | Notes |
|------|------|-------|
| `tile_latents` | LATENT | Batch tensor [N_tiles, C, th/8, tw/8] |
| `start_at_step` | INT | 0 for interp, split_at_step for two_phase |
| `dac_data` | DAC_DATA | Pass-through for Combine_Tiles |

---

## Mode Details

### `latent_interp` (Recommended default — fastest)

1. VAE encode `upscaled_image` → `upscaled_latent`
2. Resize `original_image` to match upscaled dimensions, VAE encode → `original_latent`
3. `mixed = torch.lerp(original_latent, upscaled_latent, alpha)`
4. Crop `mixed` into tile latents using `dac_data` coordinates
5. `start_at_step = 0`

**Why it works:** Each tile crop already contains original image structure encoded in latent space.
High denoise "refines" rather than "replaces" — the lerp acts as a structural anchor.

### `two_phase` (Most structure-preserving — slower)

1. VAE encode `upscaled_image` → `full_latent`
2. Run global KSampler for `split_at_step` steps on `full_latent` (full resolution)
3. Crop pre-denoised latent into tile latents
4. `start_at_step = split_at_step`

**Why it works:** Global sampling establishes consistent structure across the whole image
before any tile sees the model — each tile then inherits that shared context.

### `combined` (Maximum coherence — slowest)

1. Latent interp (as above)
2. Global sample N steps on the interpolated latent
3. Crop into tile latents
4. `start_at_step = split_at_step`

---

## Latent Crop Helper

```python
def crop_latents(latent_tensor, dac_data):
    """
    Crop a full latent [1, C, H, W] into N tile latents [N, C, th, tw].
    Tile coordinates come from dac_data (pixel space), divided by latent_factor.
    """
    lf = 8  # Flux VAE latent factor — detect from vae.downscale_ratio at runtime
    tile_w = dac_data['tile_width'] // lf
    tile_h = dac_data['tile_height'] // lf
    coords = create_tile_coordinates(...)  # reuse from _utils_.py
    tiles = [latent_tensor[:, :, y//lf : y//lf+tile_h, x//lf : x//lf+tile_w]
             for (x, y) in coords]
    return {"samples": torch.cat(tiles, dim=0)}
```

---

## Edge Cases

| Situation | Handling |
|-----------|----------|
| `original_image=None` with interp mode | Fallback to `two_phase`, print warning |
| `model=None` with `two_phase` | Raise clear ValueError |
| Tile at image edge (undersized) | reflect-pad before encode, crop after |
| Non-standard latent factor | Read from `vae.downscale_ratio` at runtime |
| `tile_width/height` not divisible by 8 | Round down to nearest multiple of 8 |

---

## Files Changed

```
ComfyUI_Steudio/
  nodes/Divide and Conquer/
    predenoise_splitter.py    ← NEW (~200 lines)
    __init__.py               ← add import
  nodes/__init__.py           ← add export
  docs/plans/
    2026-05-12-predenoise-splitter-design.md  ← THIS FILE
```

No changes to: `Combine_Tiles`, `DaC_Algorithm`, `_utils_.py`, `combine_tiles_advanced.py`.
