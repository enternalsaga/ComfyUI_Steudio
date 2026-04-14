<center>

# **ComfyUI [**Steudio**](https://linktr.ee/steudio) beta**

</center>

```git clone -b beta https://github.com/Steudio/ComfyUI_Steudio.git```

## New Nodes

### Tile Grid Picker
`Steudio/Divide and Conquer`

Interactive visual tile selector for the Divide and Conquer pipeline. Displays a preview of the tiled image with a clickable grid overlay — click to toggle individual tiles, Shift+Click for range selection. Toolbar provides **All**, **Clear**, **Invert**, and **Play** actions. The Play button re-queues only downstream nodes for quick iteration without re-running the full workflow. Outputs selected tiles as a list and enriched `dac_data` for downstream nodes.

**Inputs:** `image`, `dac_data`
**Outputs:** `TILE(S)` (list), `dac_data`, `ui`

### Combine Tiles Picker
`Steudio/Divide and Conquer`

Reassembles processed tiles back onto the original image. Designed to pair with Tile Grid Picker — only the selected tiles are overlaid while the rest of the image stays untouched. Uses the original image from `dac_data` as background by default; connect `background_image` to override.

**Inputs:** `images`, `dac_data`, `background_image` (optional)
**Outputs:** `image`, `ui`

# Thank you
Copyright (c) 2025, Steudio - https://github.com/steudio

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/steudio)

