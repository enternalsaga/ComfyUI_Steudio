# get_latent_size.py

class Get_Latent_Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {}),  # ComfyUI latent type
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    OUTPUT_NODE = True
    FUNCTION = "get_size"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
Displays the pixel width/height derived from the latent tensor shape.
Pixel size = latent size * 8 (SD1.x/SD2.x).
Supports [B, C, H, W], [C, H, W], and [B, C, D, H, W] with D=1.
"""

    def get_size(self, latent):
        """
        Latent is usually a dict with key 'samples' -> torch.Tensor
        Shape can be:
          - [B, C, H, W]
          - [C, H, W]
          - [B, C, D, H, W] (extra depth/temporal dimension)
        Pixel size = latent size * 8 (for SD1.x/SD2.x).
        """
        if not (isinstance(latent, dict) and "samples" in latent):
            raise ValueError("Invalid latent format: expected dict with 'samples' tensor")

        t = latent["samples"]

        # Handle list/tuple of tensors
        if isinstance(t, (list, tuple)):
            if len(t) == 0:
                raise ValueError("Empty latent samples list")
            t = t[0]

        if hasattr(t, "ndim"):
            ndim = t.ndim
        else:
            raise ValueError("Invalid latent: 'samples' must be a tensor")

        if ndim == 4:
            # [B, C, H, W]
            _, _, h, w = t.shape
        elif ndim == 3:
            # [C, H, W]
            _, h, w = t.shape
        elif ndim == 5:
            # [B, C, D, H, W] → squeeze depth if it's 1
            if t.shape[2] != 1:
                raise ValueError(f"Unexpected depth dimension: {t.shape}")
            _, _, _, h, w = t.shape
        else:
            raise ValueError(f"Unexpected latent tensor shape: {t.shape}")

        width = w * 8
        height = h * 8

        # Build a small UI readout similar to Sequence_Generator
        ui_text = f"Width: {width}px\nHeight: {height}px"

        return {"ui": {"text": (ui_text,)}, "result": (width, height)}


NODE_CLASS_MAPPINGS = {"Get Latent Size": Get_Latent_Size}
NODE_DISPLAY_NAME_MAPPINGS = {"Get Latent Size": "Get Latent Size"}