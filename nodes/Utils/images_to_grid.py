import math
import torch

class ImagesToGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # list of images
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Steudio/Utils"

    def _to_torch_float_hwc(self, img, device=None):
        """Normalize a single image to torch.float32 in [0,1], shape (H, W, C)."""
        if isinstance(img, torch.Tensor):
            t = img
            if device is not None and t.device != device:
                t = t.to(device)
            if t.dim() == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.dim() == 3 and t.shape[-1] not in (1, 3, 4) and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
            if t.dtype == torch.uint8:
                t = t.to(torch.float32) / 255.0
            else:
                t = t.to(torch.float32)
            t = t.clamp(0.0, 1.0)
            return t.contiguous()

        import numpy as np
        np_img = np.array(img)
        np_img = np.squeeze(np_img)
        if np_img.ndim == 3 and np_img.shape[-1] not in (1, 3, 4) and np_img.shape[0] in (1, 3, 4):
            np_img = np_img.transpose(1, 2, 0)
        if np_img.dtype == np.uint8:
            np_img = np_img.astype(np.float32) / 255.0
        elif np.issubdtype(np_img.dtype, np.floating):
            np_img = np.clip(np_img, 0.0, 1.0).astype(np.float32)
        else:
            np_img = np_img.astype(np.float32)
        t = torch.from_numpy(np_img)
        if device is not None:
            t = t.to(device)
        return t.contiguous()

    def execute(self, images):
        if not images:
            empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return (empty,)

        first = images[0]
        device = first.device if isinstance(first, torch.Tensor) else torch.device("cpu")

        norm_imgs = [self._to_torch_float_hwc(img, device=device) for img in images]

        h, w, c = norm_imgs[0].shape
        for t in norm_imgs[1:]:
            if t.shape != (h, w, c):
                raise ValueError(f"All images must have identical shape. Got {t.shape} vs {(h, w, c)}.")

        n = len(norm_imgs)
        if n == 1:
            return norm_imgs[0].unsqueeze(0),

        # --- Orientation-aware scoring ---
        img_ratio = w / h
        if img_ratio > 1:
            # Landscape → prefer more columns
            target_ratio = max(1.0, img_ratio * math.sqrt(n))
        elif img_ratio < 1:
            # Portrait → prefer more rows
            target_ratio = min(1.0, img_ratio * math.sqrt(n))
        else:
            # Square → aim for square grid
            target_ratio = 1.0

        best_rows, best_cols, best_score = None, None, None
        for cols in range(1, n + 1):
            rows = math.ceil(n / cols)
            blanks = rows * cols - n
            grid_ratio = (cols * w) / (rows * h)

            score = (
                blanks * 1000
                + abs(grid_ratio - target_ratio) * 100
                + abs(rows - cols)
            )

            if best_score is None or score < best_score:
                best_rows, best_cols, best_score = rows, cols, score

        rows, cols = best_rows, best_cols
        grid_h = rows * h
        grid_w = cols * w

        output = torch.zeros((1, grid_h, grid_w, c), dtype=torch.float32, device=device)

        for idx, img in enumerate(norm_imgs):
            r, c_idx = divmod(idx, cols)
            y = r * h
            x = c_idx * w
            output[:, y:y + h, x:x + w, :] = img

        return (output,)


NODE_CLASS_MAPPINGS = {"Images to Grid": ImagesToGrid}
NODE_DISPLAY_NAME_MAPPINGS = {"Images to Grid": "Images to Grid"}