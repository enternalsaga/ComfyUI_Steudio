# tile_grid_picker.py

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from folder_paths import get_temp_directory

sys.path.append(os.path.dirname(__file__))
from _utils_ import create_tile_coordinates, generate_matrix_ui

MAX_PREVIEW_SIZE = 1024


class Tile_Grid_Picker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                "selected_tiles": ("STRING", {"default": "[]"}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "DAC_DATA", "UI")
    RETURN_NAMES = ("TILE(S)", "dac_data", "ui")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "execute"
    CATEGORY = "Steudio/Divide and Conquer"
    DESCRIPTION = """Interactive tile grid picker.
Click tiles in the preview to select/deselect.
Shift+Click to select a range of tiles.
Selected tiles are output as a list."""

    def execute(self, image, dac_data, selected_tiles="[]", node_id=None):
        if isinstance(dac_data, list):
            dac_data = dac_data[0]

        # Parse selected indices (1-based from UI)
        try:
            selected_indices = json.loads(selected_tiles)
            if not isinstance(selected_indices, list):
                selected_indices = []
        except (json.JSONDecodeError, TypeError):
            selected_indices = []


        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_width = dac_data['tile_width']
        tile_height = dac_data['tile_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        tile_coordinates, matrix = create_tile_coordinates(
            image_width, image_height, tile_width, tile_height,
            overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        # Crop all tiles
        all_tiles = []
        for tc in tile_coordinates:
            tile = image[
                :,
                tc[1]:tc[1] + tile_height,
                tc[0]:tc[0] + tile_width,
                :,
            ]
            all_tiles.append(tile)

        # Select tiles — if none selected, output all
        if not selected_indices:
            output_tiles = all_tiles
        else:
            output_tiles = [
                all_tiles[i - 1]
                for i in selected_indices
                if 0 < i <= len(all_tiles)
            ]
            # Fallback: if all indices were invalid, output all
            if not output_tiles:
                output_tiles = all_tiles

        # Save downscaled preview to temp
        preview_filename = None
        if image.shape[0] > 0:
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            w, h = pil_img.size
            if max(w, h) > MAX_PREVIEW_SIZE:
                scale = MAX_PREVIEW_SIZE / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )

            temp_dir = get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            preview_filename = f"tile_picker_{node_id}_{image_width}x{image_height}.png"
            try:
                pil_img.save(os.path.join(temp_dir, preview_filename))
            except Exception as e:
                print(f"[TileGridPicker] Error saving preview: {e}")
                preview_filename = None

        # Enrich dac_data with selected_indices
        enriched_dac_data = dict(dac_data)
        enriched_dac_data['selected_indices'] = selected_indices

        # Build tile_info for frontend grid rendering
        tile_info = []
        for idx, (x, y) in enumerate(tile_coordinates):
            tile_info.append({
                "index": idx + 1,
                "x": x, "y": y,
                "w": tile_width, "h": tile_height,
            })

        # Generate matrix UI text
        matrix_ui = generate_matrix_ui(matrix)

        return {
            "ui": {
                "preview_image": [{
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp",
                }] if preview_filename else [],
                "tile_info": [json.dumps({
                    "tiles": tile_info,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "image_width": image_width,
                    "image_height": image_height,
                })],
            },
            "result": (
                [t for t in output_tiles],
                enriched_dac_data,
                matrix_ui,
            ),
        }


NODE_CLASS_MAPPINGS = {"Tile Grid Picker": Tile_Grid_Picker}
NODE_DISPLAY_NAME_MAPPINGS = {"Tile Grid Picker": "Tile Grid Picker"}
