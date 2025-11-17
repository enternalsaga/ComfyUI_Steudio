# algorithm.py
# Simplified Divide and Conquer Algorithm (No Upscale)
# Inspired by https://github.com/kinfolk0117/ComfyUI_SimpleTiles

import os
import sys
import math
import torch
import comfy.utils

sys.path.append(os.path.dirname(__file__))
from _config_upscale_ import TILE_ORDER_DICT  # we only need tile order now


class DaC_Algorithm_no_upscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024}),
                "tile_height": ("INT", {"default": 1024}),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "spiral"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "DAC_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "dac_data", "ui")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    DESCRIPTION = """
Divide an image into tiles without upscaling.
Overlap is automatically calculated so tiles fit exactly.
Steudio
"""

    def execute(self, image, tile_width, tile_height, tile_order):
        _, height, width, _ = image.shape

        # Validate
        if width < tile_width or height < tile_height:
            raise ValueError(
                f"Image ({width}x{height}) must be at least as large as tile "
                f"({tile_width}x{tile_height}) in both dimensions."
            )

        # Compute grid counts
        grid_x = math.ceil(width / tile_width)
        grid_y = math.ceil(height / tile_height)

        # Compute overlap so tiles fit exactly
        overlap_x = 0 if grid_x == 1 else round((tile_width * grid_x - width) / (grid_x - 1))
        overlap_y = 0 if grid_y == 1 else round((tile_height * grid_y - height) / (grid_y - 1))

        Tiles_Q = grid_x * grid_y

        # Since no upscaling, upscaled dims = original dims
        upscaled_width = width
        upscaled_height = height

        dac_data = {
            "upscaled_width": upscaled_width,
            "upscaled_height": upscaled_height,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "overlap_x": overlap_x,
            "overlap_y": overlap_y,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "tile_order": TILE_ORDER_DICT.get(tile_order, 0),
        }

        algo_ui = f"""Divide and Conquer Algorithm (No Upscale):
Original Image Size: {width}x{height}
Upscaled Image Size: {upscaled_width}x{upscaled_height}
Grid: {grid_x}x{grid_y} ({Tiles_Q} tiles)
Overlap_x: {overlap_x} pixels
Overlap_y: {overlap_y} pixels
"""

        # Return original image unchanged
        return (image, dac_data, algo_ui)


NODE_CLASS_MAPPINGS = {
    "Divide and Conquer Algorithm (no upscale)": DaC_Algorithm_no_upscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Divide and Conquer Algorithm (no upscale)": "Divide and Conquer Algorithm (no upscale)",
}