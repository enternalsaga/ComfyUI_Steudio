import os
import sys

sys.path.append(os.path.dirname(__file__))
from _config_utils_ import _any_


class if_then_else:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": (_any_, {}),
                "input2": (_any_, {}),
                "switch": ("INT", {"default": 0}),
                "match": ("STRING", {"default": "2...6,7...12,else"}),
            },
            "optional": {
                "input3": (_any_, {}),
            }
        }

    RETURN_TYPES = (_any_, "STRING")
    RETURN_NAMES = ("output", "ui")
    FUNCTION = "do_switch"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
Usage examples for match string:

   "8,e"        | if switch == 8 → input1, else → input2
   "2...6,e"    | if switch in [2,3,4,5,6] → input1, else → input2
   "2...6,7...10,e"
                | if switch in [2–6] → input1,
                | if switch in [7–10] → input2,
                | else → input3

Notes:
- Supports 2 or 3 entries (input3 is optional).
- Use "e" or "else" as the fallback branch.
- Ranges are inclusive (start and end included).
"""

    def do_switch(self, input1, input2, switch, match, input3=None):
        tokens = [t.strip().lower() for t in match.split(",")]
        inputs = [input1, input2, input3]

        if len(tokens) not in (2, 3):
            ui = f"Error: Match string must have 2 or 3 entries, got: {tokens}"
            return input1, ui

        def in_range(token, value):
            if "..." in token:
                parts = token.split("...")
                if len(parts) != 2:
                    raise ValueError(f"Invalid range token: {token}")
                try:
                    start, end = int(parts[0]), int(parts[1])
                except Exception:
                    raise ValueError(f"Non-integer range bounds in token: {token}")
                return start <= value <= end
            else:
                try:
                    return int(token) == value
                except Exception:
                    raise ValueError(f"Invalid integer token: {token}")

        # Try matching non-else tokens first
        for idx, token in enumerate(tokens):
            if token in ("e", "else"):
                continue
            try:
                if in_range(token, switch):
                    selected = inputs[idx]
                    ui = (
                        f"Selected: input{idx+1}\n"
                        f"Match token: {token}\n"
                        f"Switch value: {switch}"
                    )
                    return selected, ui
            except ValueError as err:
                ui = f"Error parsing token '{token}': {err}"
                return input1, ui

        # If no explicit match, find the else token (fallback)
        for idx, token in enumerate(tokens):
            if token in ("e", "else"):
                selected = inputs[idx] if inputs[idx] is not None else input1
                ui = (
                    f"Selected: input{idx+1} (else)\n"
                    f"Match token: {token}\n"
                    f"Switch value: {switch}"
                )
                return selected, ui

        # Final fallback
        ui = (
            f"Selected: input1 (default fallback)\n"
            f"Match token: none\n"
            f"Switch value: {switch}"
        )
        return input1, ui


NODE_CLASS_MAPPINGS = {"if then else": if_then_else}
NODE_DISPLAY_NAME_MAPPINGS = {"if then else": "if then else"}