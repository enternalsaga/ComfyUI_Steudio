import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── Downstream node walking for Play button ──
function getDownstreamNodes(startNode) {
  const downstreamNodes = new Set();
  function walkForward(node) {
    if (downstreamNodes.has(node.id)) return;
    downstreamNodes.add(node.id);
    if (node.outputs) {
      for (const output of node.outputs) {
        if (output.links) {
          for (const linkId of output.links) {
            const link = app.graph.links[linkId];
            if (link) {
              const targetNode = app.graph._nodes_by_id?.[link.target_id];
              if (targetNode) walkForward(targetNode);
            }
          }
        }
      }
    }
  }
  walkForward(startNode);
  return downstreamNodes;
}

function recursiveAddNodes(nodeId, oldOutput, newOutput) {
  const currentNode = oldOutput[nodeId];
  if (!currentNode || newOutput[nodeId]) return;
  newOutput[nodeId] = currentNode;
  if (currentNode.inputs) {
    for (const inputValue of Object.values(currentNode.inputs)) {
      if (Array.isArray(inputValue) && inputValue.length > 0) {
        recursiveAddNodes(String(inputValue[0]), oldOutput, newOutput);
      }
    }
  }
}

let tilePickerDownstreamNodeIds = null;

const originalApiQueuePrompt = api.queuePrompt;
api.queuePrompt = async function (index, prompt, ...args) {
  if (tilePickerDownstreamNodeIds && prompt.output) {
    const oldOutput = prompt.output;
    const newOutput = {};
    for (const nodeId of tilePickerDownstreamNodeIds) {
      const nodeIdStr = String(nodeId);
      if (oldOutput[nodeIdStr]) {
        recursiveAddNodes(nodeIdStr, oldOutput, newOutput);
      }
    }
    prompt.output = newOutput;
  }
  return originalApiQueuePrompt.apply(this, [index, prompt, ...args]);
};

app.registerExtension({
  name: "Steudio.TileGridPicker",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "Tile Grid Picker") return;

    // ── Helper: get widget by name ──
    nodeType.prototype.getWidget = function (name) {
      return this.widgets?.find((w) => w.name === name);
    };

    // ── Constants ──
    const COLORS = {
      selectedBorder: "#00e5ff",
      selectedFill: "rgba(0, 229, 255, 0.12)",
      unselectedBorder: "#666",
      checkBg: "#00e5ff",
      checkFg: "#000",
      tileLabel: "#fff",
      tileLabelShadow: "rgba(0,0,0,0.7)",
      previewBorder: "#555",
      previewBg: "#222",
      instructionText: "#999",
      statusText: "#aaa",
    };

    // ────────────────────────────────────────────
    // onNodeCreated
    // ────────────────────────────────────────────
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      this.serialize_widgets = true;
      this.properties = this.properties || {};

      // State
      this.selectedTiles = new Set();
      this.lastClickedTile = null;
      this.tileInfo = null;
      this.gridMeta = null;

      // Preview image
      this.previewImage = new Image();
      this.previewImage.src = "";
      this.previewLoaded = false;

      // Hide selected_tiles widget (value still serializes from required)
      const stWidget = this.getWidget("selected_tiles");
      if (stWidget) {
        stWidget.hidden = true;
        stWidget.computeSize = () => [0, -4];
      }

      // Hide play_trigger widget
      const ptWidget = this.getWidget("play_trigger");
      if (ptWidget) {
        ptWidget.hidden = true;
        ptWidget.computeSize = () => [0, -4];
      }

      // ── Toolbar: 3 buttons in 1 row ──
      const node = this;
      const TOOLBAR_BTNS = [
        {
          label: "All",
          action: () => {
            if (!node.tileInfo) return;
            node.tileInfo.forEach((t) => node.selectedTiles.add(t.index));
            node._syncWidgetFromSelection();
            node.setDirtyCanvas(true);
          },
        },
        {
          label: "Clear",
          action: () => {
            node.selectedTiles.clear();
            node._syncWidgetFromSelection();
            node.setDirtyCanvas(true);
          },
        },
        {
          label: "Invert",
          action: () => {
            if (!node.tileInfo) return;
            const all = node.tileInfo.map((t) => t.index);
            const inv = all.filter((i) => !node.selectedTiles.has(i));
            node.selectedTiles.clear();
            inv.forEach((i) => node.selectedTiles.add(i));
            node._syncWidgetFromSelection();
            node.setDirtyCanvas(true);
          },
        },
        {
          label: "Play",
          action: async () => {
            const ptW = node.getWidget("play_trigger");
            if (ptW) {
              ptW.value = (ptW.value + 1) % 1000;
            }
            const downstreamNodeIds = getDownstreamNodes(node);
            if (downstreamNodeIds.size > 0) {
              tilePickerDownstreamNodeIds = downstreamNodeIds;
              try {
                await app.queuePrompt(0);
              } finally {
                tilePickerDownstreamNodeIds = null;
              }
            }
          },
        },
      ];

      const toolbar = this.addCustomWidget({
        name: "tile_toolbar",
        type: "custom",
        value: null,
        computeSize: () => [0, 30],
        draw: function (ctx, _node, w, posY, h) {
          const padding = 6;
          const gap = 4;
          const btnCount = TOOLBAR_BTNS.length;
          const totalW = w - padding * 2;
          const btnW = (totalW - gap * (btnCount - 1)) / btnCount;
          const btnH = h - 6;
          const y = posY + 3;

          for (let i = 0; i < btnCount; i++) {
            const x = padding + i * (btnW + gap);
            // Button background
            ctx.fillStyle = "#333";
            ctx.beginPath();
            ctx.roundRect(x, y, btnW, btnH, 4);
            ctx.fill();
            // Button border
            ctx.strokeStyle = "#666";
            ctx.lineWidth = 1;
            ctx.stroke();
            // Button label
            ctx.fillStyle = "#ddd";
            ctx.font = "11px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(TOOLBAR_BTNS[i].label, x + btnW / 2, y + btnH / 2);
          }
        },
        mouse: function (event, pos, _node) {
          if (event.type !== "pointerdown") return false;
          const [mx, my] = pos;
          const padding = 6;
          const gap = 4;
          const w = _node.size[0];
          const totalW = w - padding * 2;
          const btnW = (totalW - gap * (TOOLBAR_BTNS.length - 1)) / TOOLBAR_BTNS.length;

          for (let i = 0; i < TOOLBAR_BTNS.length; i++) {
            const x = padding + i * (btnW + gap);
            if (mx >= x && mx <= x + btnW) {
              TOOLBAR_BTNS[i].action();
              return true;
            }
          }
          return false;
        },
        serialize: false,
      });

      this.size = this.computeSize();
      this.setDirtyCanvas(true);
    };

    // ────────────────────────────────────────────
    // Sync selection → widget
    // ────────────────────────────────────────────
    nodeType.prototype._syncWidgetFromSelection = function () {
      const sorted = [...this.selectedTiles].sort((a, b) => a - b);
      const stWidget = this.getWidget("selected_tiles");
      if (stWidget) {
        stWidget.value = JSON.stringify(sorted);
        if (stWidget.inputEl) {
          stWidget.inputEl.value = stWidget.value;
        }
      }
      this.properties.selectedTiles = sorted;
    };

    // ────────────────────────────────────────────
    // onExecuted — receive preview + tile_info
    // ────────────────────────────────────────────
    nodeType.prototype.onExecuted = function (message) {
      // Load preview image
      const imageInfo = message?.preview_image?.[0];
      if (imageInfo?.filename) {
        const imageUrl = app.api.apiURL(
          `/view?filename=${imageInfo.filename}&type=${imageInfo.type}&subfolder=${imageInfo.subfolder}&rand=${Date.now()}`
        );
        this.previewImage.onload = () => {
          this.previewLoaded = true;
          // Only auto-size on first preview load; after that user controls size
          if (!this._initialSizeSet) {
            this.size = this.computeSize();
          }
          this.setDirtyCanvas(true);
        };
        this.previewImage.onerror = () => {
          this.previewLoaded = false;
        };
        this.previewImage.src = imageUrl;
      }

      // Parse tile_info
      const tileInfoStr = message?.tile_info?.[0];
      if (tileInfoStr) {
        try {
          this.gridMeta = JSON.parse(tileInfoStr);
          this.tileInfo = this.gridMeta.tiles;
        } catch (e) {
          console.warn("[TileGridPicker] Failed to parse tile_info:", e);
        }
      }

      this.setDirtyCanvas(true);
    };

    // ────────────────────────────────────────────
    // onConfigure — restore state
    // ────────────────────────────────────────────
    nodeType.prototype.onConfigure = function () {
      // Hide selected_tiles widget
      const stWidget = this.getWidget("selected_tiles");
      if (stWidget) {
        stWidget.hidden = true;
        stWidget.computeSize = () => [0, -4];
      }

      const saved = this.properties.selectedTiles;
      if (Array.isArray(saved)) {
        this.selectedTiles = new Set(saved);
      } else {
        this.selectedTiles = new Set();
      }

      // Restore from widget value
      if (stWidget?.value) {
        try {
          const parsed = JSON.parse(stWidget.value);
          if (Array.isArray(parsed)) {
            this.selectedTiles = new Set(parsed);
            this.properties.selectedTiles = parsed;
          }
        } catch (e) {
          // ignore
        }
      }
    };

    // ────────────────────────────────────────────
    // _getWidgetsHeight — actual height of visible widgets
    // ────────────────────────────────────────────
    nodeType.prototype._getWidgetsHeight = function () {
      if (!this.widgets) return 0;
      let h = 0;
      for (const w of this.widgets) {
        if (w.hidden) continue;
        if (w.computeSize) {
          h += w.computeSize()[1] + 4;
        } else {
          h += LiteGraph.NODE_WIDGET_HEIGHT + 4;
        }
      }
      return h;
    };

    // ────────────────────────────────────────────
    // _getTopOffset — header + outputs + widgets
    // ────────────────────────────────────────────
    nodeType.prototype._getTopOffset = function () {
      const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;
      const slotsH = Math.max(
        (this.outputs?.length || 0) * LiteGraph.NODE_SLOT_HEIGHT,
        (this.inputs?.length || 0) * LiteGraph.NODE_SLOT_HEIGHT
      );
      const widgetsH = this._getWidgetsHeight();
      return titleH + slotsH + widgetsH;
    };

    // ────────────────────────────────────────────
    // computeSize — small minimum, user can freely resize
    // ────────────────────────────────────────────
    nodeType.prototype.computeSize = function () {
      const topOffset = this._getTopOffset();
      // Minimum: just enough for toolbar + a small preview area
      const minW = 150;
      const minH = topOffset + 80;

      // On first preview load, suggest a reasonable initial size
      if (this.previewLoaded && this.previewImage.naturalWidth && !this._initialSizeSet) {
        const aspect = this.previewImage.naturalWidth / this.previewImage.naturalHeight;
        const previewW = Math.min(400, this.previewImage.naturalWidth);
        const previewH = previewW / aspect;
        this._initialSizeSet = true;
        return [previewW + 30, topOffset + previewH + 50];
      }

      return [minW, minH];
    };

    // ────────────────────────────────────────────
    // getPreviewArea — compute preview draw region
    // ────────────────────────────────────────────
    nodeType.prototype.getPreviewArea = function () {
      const padding = 15;
      const statusHeight = 30;
      const topOffset = this._getTopOffset();

      const availW = this.size[0] - padding * 2;
      const availH = this.size[1] - topOffset - statusHeight - padding;

      let pw = availW;
      let ph = availH;

      if (
        this.previewLoaded &&
        this.previewImage.naturalWidth &&
        this.previewImage.naturalHeight
      ) {
        const aspect =
          this.previewImage.naturalWidth / this.previewImage.naturalHeight;
        if (availW / availH > aspect) {
          ph = availH;
          pw = ph * aspect;
        } else {
          pw = availW;
          ph = pw / aspect;
        }
      }

      const x = padding + (availW - pw) / 2;
      return { x, y: topOffset, width: pw, height: ph };
    };

    // ────────────────────────────────────────────
    // hitTestTile — which tile was clicked?
    // ────────────────────────────────────────────
    nodeType.prototype._hitTestTile = function (localX, localY) {
      if (!this.tileInfo || !this.gridMeta) return null;

      const preview = this.getPreviewArea();
      const scaleX = preview.width / this.gridMeta.image_width;
      const scaleY = preview.height / this.gridMeta.image_height;

      for (let i = this.tileInfo.length - 1; i >= 0; i--) {
        const t = this.tileInfo[i];
        const tx = t.x * scaleX;
        const ty = t.y * scaleY;
        const tw = t.w * scaleX;
        const th = t.h * scaleY;

        if (
          localX >= tx &&
          localX <= tx + tw &&
          localY >= ty &&
          localY <= ty + th
        ) {
          return t.index;
        }
      }
      return null;
    };

    // ────────────────────────────────────────────
    // onMouseDown — tile click handling
    // ────────────────────────────────────────────
    nodeType.prototype.onMouseDown = function (e, pos, graphCanvas) {
      if (!this.tileInfo) return false;

      const preview = this.getPreviewArea();
      const localX = pos[0] - preview.x;
      const localY = pos[1] - preview.y;

      // Check if click is inside preview area
      if (
        localX < 0 ||
        localY < 0 ||
        localX > preview.width ||
        localY > preview.height
      ) {
        return false;
      }

      const hitTile = this._hitTestTile(localX, localY);
      if (hitTile === null) return false;

      if (e.shiftKey && this.lastClickedTile !== null) {
        // Shift+Click: range selection
        const start = Math.min(this.lastClickedTile, hitTile);
        const end = Math.max(this.lastClickedTile, hitTile);
        for (let i = start; i <= end; i++) {
          this.selectedTiles.add(i);
        }
      } else {
        // Normal click: toggle single tile
        if (this.selectedTiles.has(hitTile)) {
          this.selectedTiles.delete(hitTile);
        } else {
          this.selectedTiles.add(hitTile);
        }
        this.lastClickedTile = hitTile;
      }

      this._syncWidgetFromSelection();
      this.setDirtyCanvas(true);
      return true;
    };

    // ────────────────────────────────────────────
    // onDrawForeground — render preview + grid
    // ────────────────────────────────────────────
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (this.flags.collapsed) return;

      const preview = this.getPreviewArea();

      // Draw preview border
      ctx.strokeStyle = COLORS.previewBorder;
      ctx.lineWidth = 1;
      ctx.strokeRect(preview.x, preview.y, preview.width, preview.height);

      if (this.previewLoaded) {
        // Draw preview image
        ctx.drawImage(
          this.previewImage,
          preview.x,
          preview.y,
          preview.width,
          preview.height
        );
      } else {
        // Placeholder
        ctx.fillStyle = COLORS.previewBg;
        ctx.fillRect(preview.x, preview.y, preview.width, preview.height);
        ctx.fillStyle = COLORS.instructionText;
        ctx.font = "13px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          "Run graph to see preview",
          preview.x + preview.width / 2,
          preview.y + preview.height / 2
        );
      }

      // Draw tile grid overlay
      if (this.tileInfo && this.gridMeta) {
        const scaleX = preview.width / this.gridMeta.image_width;
        const scaleY = preview.height / this.gridMeta.image_height;

        for (const tile of this.tileInfo) {
          const tx = preview.x + tile.x * scaleX;
          const ty = preview.y + tile.y * scaleY;
          const tw = tile.w * scaleX;
          const th = tile.h * scaleY;
          const isSelected = this.selectedTiles.has(tile.index);
          const hasSelection = this.selectedTiles.size > 0;

          ctx.save();

          if (!isSelected && hasSelection) {
            // Unselected: dark overlay 60% to dim
            ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
            ctx.fillRect(tx, ty, tw, th);
          }

          if (isSelected) {
            // Selected: bright border
            ctx.strokeStyle = COLORS.selectedBorder;
            ctx.lineWidth = 2;
            ctx.setLineDash([]);
            ctx.strokeRect(tx + 1, ty + 1, tw - 2, th - 2);
          }

          // Grid line (thin, always visible)
          ctx.strokeStyle = hasSelection && !isSelected
            ? "rgba(100, 100, 100, 0.4)"
            : COLORS.unselectedBorder;
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(tx + 0.5, ty + 0.5, tw - 1, th - 1);

          // Tile number label (center)
          const fontSize = Math.max(
            10,
            Math.min(18, Math.min(tw, th) * 0.25)
          );
          ctx.font = `bold ${fontSize}px Arial`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";

          // Shadow for readability
          ctx.fillStyle = COLORS.tileLabelShadow;
          ctx.fillText(`${tile.index}`, tx + tw / 2 + 1, ty + th / 2 + 1);

          // Label color: bright for selected, dimmed for unselected
          ctx.fillStyle = isSelected
            ? COLORS.selectedBorder
            : (hasSelection ? "rgba(255,255,255,0.4)" : COLORS.tileLabel);
          ctx.fillText(`${tile.index}`, tx + tw / 2, ty + th / 2);

          ctx.restore();
        }
      }

      // Status text at bottom
      ctx.save();
      ctx.fillStyle = COLORS.statusText;
      ctx.font = "10px Arial";
      ctx.textAlign = "center";

      const selectedCount = this.selectedTiles.size;
      const totalCount = this.tileInfo?.length || 0;

      if (totalCount > 0) {
        ctx.fillText(
          `${selectedCount}/${totalCount} tiles selected — Click to toggle, Shift+Click for range`,
          this.size[0] / 2,
          this.size[1] - 10
        );
      } else {
        ctx.fillText(
          "Run graph to load tile grid",
          this.size[0] / 2,
          this.size[1] - 10
        );
      }
      ctx.restore();
    };
  },
});
