"""Optional multimodal context helpers for ARC analyzer prompts."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from inference.agent.runtime_state import Frame


# Exactly midway between the two nearest ARC grays (index 2 = 153, index 3 =
# 102), so it reads as neither and is unambiguous against all 16 ARC colors.
GRID_LINE_COLOR: tuple[int, int, int] = (128, 128, 128)
# Below this, a 1px line would eat too much of the cell to read as a line
# rather than noise, so grid lines are silently skipped.
GRID_LINE_MIN_CELL_PX = 4


ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}


def multimodal_context() -> str:
    return os.environ.get("MULTIMODAL_CONTEXT", "").strip().lower()


def current_grid_image_enabled() -> bool:
    return multimodal_context() == "current_grid"


def current_grid_image_upscale() -> int:
    raw = os.environ.get("MULTIMODAL_UPSCALE", "").strip()
    if not raw:
        return 16
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def multimodal_grid_lines_enabled() -> bool:
    return os.environ.get("MULTIMODAL_GRID_LINES", "").strip() == "1"


def _render_grid_lined_image(frame: Frame, rows: int, cols: int, scale: int) -> Image.Image:
    # Fill the whole canvas with the line color first, then paint each cell
    # (scale-1)px shy of its full span -- the untouched sliver stays the line
    # color, so the target image size never grows past cols*scale x rows*scale.
    # Plain pixel writes rather than ImageDraw: some deployment environments
    # ship a Pillow build where ImageDraw's ImageText submodule fails to
    # import (PIL._typing lacks _Ink), which would otherwise break every
    # current-grid render, not just the grid-lined path.
    image = Image.new("RGB", (cols * scale, rows * scale), GRID_LINE_COLOR)
    pixels = image.load()
    cell_span = scale - 1
    for row_idx, row in enumerate(frame.grid):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else 0
            color = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
            base_x = col_idx * scale
            base_y = row_idx * scale
            for dy in range(cell_span):
                for dx in range(cell_span):
                    pixels[base_x + dx, base_y + dy] = color
    return image


def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None, grid_lines: bool | None = None) -> str:
    rows = len(frame.grid)
    cols = max((len(row) for row in frame.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("Cannot render an empty grid as an image.")

    scale = current_grid_image_upscale() if upscale is None else max(1, int(upscale))
    draw_grid_lines = multimodal_grid_lines_enabled() if grid_lines is None else grid_lines

    if scale >= GRID_LINE_MIN_CELL_PX and draw_grid_lines:
        image = _render_grid_lined_image(frame, rows, cols, scale)
    else:
        image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
        pixels = image.load()
        for row_idx, row in enumerate(frame.grid):
            for col_idx in range(cols):
                value = row[col_idx] if col_idx < len(row) else 0
                pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
        if scale > 1:
            image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def png_data_url_byte_size(data_url: str) -> int:
    """Decoded PNG byte size of a data URL produced by ``frame_to_png_data_url``."""
    _, _, encoded = data_url.partition(",")
    return len(base64.b64decode(encoded)) if encoded else 0


def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None or not current_grid_image_enabled():
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": frame_to_png_data_url(frame),
        },
    }
