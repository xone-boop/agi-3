"""Optional multimodal context helpers for ARC analyzer prompts."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from inference.agent.runtime_state import Frame


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


def current_grid_lines_enabled() -> bool:
    return os.environ.get("MULTIMODAL_GRID_LINES", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Current-grid multimodal rendering requires Pillow's public PIL.Image API."
        ) from exc

    rows = len(frame.grid)
    cols = max((len(row) for row in frame.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("Cannot render an empty grid as an image.")

    scale = current_grid_image_upscale() if upscale is None else max(1, int(upscale))
    if current_grid_lines_enabled() and scale >= 4:
        line_color = (96, 96, 96)
        image = Image.new("RGB", (cols * scale, rows * scale), line_color)
        pixels = image.load()
        content_span = scale - 1
        for row_idx, row in enumerate(frame.grid):
            base_y = row_idx * scale
            for col_idx in range(cols):
                base_x = col_idx * scale
                value = row[col_idx] if col_idx < len(row) else 0
                color = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
                for dy in range(content_span):
                    for dx in range(content_span):
                        pixels[base_x + dx, base_y + dy] = color
    else:
        image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
        pixels = image.load()
        for row_idx, row in enumerate(frame.grid):
            for col_idx in range(cols):
                value = row[col_idx] if col_idx < len(row) else 0
                pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
        if scale > 1:
            image = image.resize(
                (cols * scale, rows * scale), Image.Resampling.NEAREST
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None or not current_grid_image_enabled():
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": frame_to_png_data_url(frame),
        },
    }
