# combine_tiles_picker.py

import os
import sys
import math
import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates, generate_matrix_ui, generate_tile_mask_np


class Combine_Tiles_Picker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
            },
            "optional": {
                "original_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "UI")
    RETURN_NAMES = ("image", "ui")
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    DESCRIPTION = """Combine processed tiles with original unselected tiles.
Connect processed tiles from Tile Grid Picker pipeline.
Connect original_image for unselected tile fallback.
Original image is used as background canvas."""

    def _blend_tile(self, output, image_tile, x, y,
                    tile_width, tile_height,
                    upscaled_width, upscaled_height,
                    f_overlap_x, f_overlap_y,
                    overlap_x, overlap_y,
                    blend_x, blend_y):
        """Blend a single tile into the output canvas using mask."""
        mask_np = generate_tile_mask_np(
            x, y, tile_width, tile_height,
            upscaled_width, upscaled_height,
            f_overlap_x, f_overlap_y
        )

        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        blur = (ImageFilter.BoxBlur
                if overlap_x <= 64 or overlap_y <= 64
                else ImageFilter.GaussianBlur)
        mask_img = mask_img.filter(blur(radius=(blend_x, blend_y)))
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0

        mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(-1)
        output[:, y:y + tile_height, x:x + tile_width, :] *= (1 - mask_tensor)
        output[:, y:y + tile_height, x:x + tile_width, :] += image_tile * mask_tensor

    def execute(self, images, dac_data, original_image=None):
        if isinstance(dac_data, list):
            dac_data = dac_data[0]

        images = torch.stack(images).squeeze(1)

        selected_indices = dac_data.get('selected_indices', [])

        upscaled_width = dac_data['upscaled_width']
        upscaled_height = dac_data['upscaled_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        tile_height, tile_width = images.shape[1:3]
        f_overlap_x = overlap_x // 4
        f_overlap_y = overlap_y // 4
        blend_x = math.sqrt(overlap_x)
        blend_y = math.sqrt(overlap_y)

        tile_coordinates, matrix = create_tile_coordinates(
            upscaled_width, upscaled_height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        blend_args = dict(
            tile_width=tile_width, tile_height=tile_height,
            upscaled_width=upscaled_width, upscaled_height=upscaled_height,
            f_overlap_x=f_overlap_x, f_overlap_y=f_overlap_y,
            overlap_x=overlap_x, overlap_y=overlap_y,
            blend_x=blend_x, blend_y=blend_y,
        )

        # Use original_image as background canvas
        if original_image is not None:
            if isinstance(original_image, list):
                original_image = original_image[0]
            output = original_image.clone()
        else:
            output = torch.zeros(
                (1, upscaled_height, upscaled_width, 3), dtype=images.dtype
            )

        if not selected_indices:
            # No selection info → fallback to standard combine (all tiles)
            for idx, (x, y) in enumerate(tile_coordinates):
                if idx >= images.shape[0]:
                    break
                self._blend_tile(output, images[idx], x, y, **blend_args)
        else:
            # Only overlay processed tiles at selected positions
            for proc_idx, sel_idx in enumerate(selected_indices):
                if proc_idx >= images.shape[0]:
                    break
                tile_idx = sel_idx - 1  # 1-based → 0-based
                if tile_idx < 0 or tile_idx >= len(tile_coordinates):
                    continue
                x, y = tile_coordinates[tile_idx]
                self._blend_tile(output, images[proc_idx], x, y, **blend_args)

        matrix_ui = generate_matrix_ui(matrix)
        return output, matrix_ui


NODE_CLASS_MAPPINGS = {"Combine Tiles Picker": Combine_Tiles_Picker}
NODE_DISPLAY_NAME_MAPPINGS = {"Combine Tiles Picker": "Combine Tiles Picker"}
