from __future__ import annotations

import base64
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

from inference.agent.runtime_state import Frame
from inference.agent.vision_context import (
    ARC_COLOR_MAP,
    current_grid_image_part,
    frame_to_png_data_url,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PICKLE = BUNDLE_ROOT / "benchmark_initial.pkl"
GRID_LINE_COLOR = (96, 96, 96)
BLOCK_PIL_HELPERS = """
import importlib.abc
import sys

class BlockPillowHelpers(importlib.abc.MetaPathFinder):
    blocked = {"PIL.ImageDraw", "PIL.ImageText", "PIL.ImageFont"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.blocked:
            raise ImportError(f"deliberately blocked optional Pillow helper: {fullname}")
        return None

sys.meta_path.insert(0, BlockPillowHelpers())
"""
BLOCK_PLOTTING_STACK = """
import importlib.abc
import sys

class BlockPlottingStack(importlib.abc.MetaPathFinder):
    blocked_roots = {"matplotlib", "scipy", "imageio"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked_roots:
            raise ImportError(f"deliberately blocked optional plotting dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockPlottingStack())
"""


def _run_fresh_python(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _decode_png(data_url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)
    payload = base64.b64decode(data_url.removeprefix(prefix))
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def _frame() -> Frame:
    return Frame(grid=((8, 9), (14, 15)), step=0, level=1)


def test_vision_import_succeeds_with_pillow_helpers_blocked_in_fresh_process() -> None:
    result = _run_fresh_python(
        "from PIL import Image\n"
        + BLOCK_PIL_HELPERS
        + "\nimport inference.agent.vision_context\n"
        + "assert 'PIL.ImageDraw' not in sys.modules\n"
        + "assert 'PIL.ImageText' not in sys.modules\n"
        + "assert 'PIL.ImageFont' not in sys.modules\n"
    )

    assert result.returncode == 0, result.stderr


def test_solver_import_and_exact_pickle_load_with_pillow_helpers_blocked_in_fresh_process() -> None:
    assert BENCHMARK_PICKLE.is_file()
    result = _run_fresh_python(
        BLOCK_PIL_HELPERS
        + "\nimport pickle\n"
        + "import inference.framework.solver\n"
        + "assert 'inference.agent.tool_agent' not in sys.modules\n"
        + "with open(sys.argv[1], 'rb') as file:\n"
        + "    benchmark = pickle.load(file)\n"
        + "assert type(benchmark).__module__ == 'taaf.benchmark'\n"
        + "assert type(benchmark.solver).__module__ == 'inference.framework.solver'\n"
        + "assert type(benchmark.solver).__name__ == 'HarnessSolver'\n"
        + "assert 'PIL.ImageDraw' not in sys.modules\n"
        + "assert 'PIL.ImageText' not in sys.modules\n"
        + "assert 'PIL.ImageFont' not in sys.modules\n",
        str(BENCHMARK_PICKLE),
    )

    assert result.returncode == 0, result.stderr


def test_inference_agent_public_imports_remain_compatible_in_fresh_process() -> None:
    result = _run_fresh_python(
        BLOCK_PIL_HELPERS
        + "\nimport inference.agent\n"
        + "assert 'inference.agent.tool_agent' not in sys.modules\n"
        + "from inference.agent import Frame, HistoryEntry\n"
        + "assert 'inference.agent.tool_agent' not in sys.modules\n"
        + "from inference.agent import ToolAgent\n"
        + "assert ToolAgent.__module__ == 'inference.agent.tool_agent'\n"
        + "assert Frame.__module__ == 'inference.agent.runtime_state'\n"
        + "assert HistoryEntry.__module__ == 'inference.agent.runtime_state'\n"
        + "assert 'PIL.ImageDraw' not in sys.modules\n"
    )

    assert result.returncode == 0, result.stderr


def test_solver_import_does_not_initialize_optional_plotting_in_fresh_process() -> None:
    result = _run_fresh_python(
        BLOCK_PLOTTING_STACK
        + "\nimport inference.framework.solver\n"
        + "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)\n"
        + "assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)\n"
        + "assert not any(name == 'imageio' or name.startswith('imageio.') for name in sys.modules)\n"
        + "from taaf import game\n"
        + "assert game.Frame.__module__ == 'taaf.game'\n"
    )

    assert result.returncode == 0, result.stderr


def test_optional_frame_plotting_fails_only_when_invoked() -> None:
    result = _run_fresh_python(
        BLOCK_PLOTTING_STACK
        + "\nimport numpy as np\n"
        + "from taaf.game import Frame\n"
        + "frame = Frame(np.array([[0]], dtype=np.int8))\n"
        + "try:\n"
        + "    frame.draw()\n"
        + "except RuntimeError as exc:\n"
        + "    assert 'matplotlib' in str(exc)\n"
        + "else:\n"
        + "    raise AssertionError('Frame.draw() unexpectedly succeeded without matplotlib')\n"
    )

    assert result.returncode == 0, result.stderr


def test_non_grid_rendering_preserves_dimensions_and_nearest_neighbor_cells(monkeypatch) -> None:
    monkeypatch.delenv("MULTIMODAL_GRID_LINES", raising=False)
    image = _decode_png(frame_to_png_data_url(_frame(), upscale=4))

    assert image.size == (8, 8)
    pixels = image.load()
    expected = ((8, 9), (14, 15))
    for row_idx, row in enumerate(expected):
        for col_idx, value in enumerate(row):
            for dy in range(4):
                for dx in range(4):
                    assert pixels[col_idx * 4 + dx, row_idx * 4 + dy] == ARC_COLOR_MAP[value]


def test_grid_rendering_uses_one_pixel_separators_without_changing_dimensions(monkeypatch) -> None:
    monkeypatch.setenv("MULTIMODAL_GRID_LINES", "true")
    image = _decode_png(frame_to_png_data_url(_frame(), upscale=4))

    assert image.size == (8, 8)
    pixels = image.load()
    expected = ((8, 9), (14, 15))
    for row_idx, row in enumerate(expected):
        for col_idx, value in enumerate(row):
            base_x = col_idx * 4
            base_y = row_idx * 4
            for dy in range(3):
                for dx in range(3):
                    assert pixels[base_x + dx, base_y + dy] == ARC_COLOR_MAP[value]

    for separator_x in (3, 7):
        assert all(pixels[separator_x, y] == GRID_LINE_COLOR for y in range(8))
    for separator_y in (3, 7):
        assert all(pixels[x, separator_y] == GRID_LINE_COLOR for x in range(8))


def test_grid_lines_fall_back_to_normal_rendering_below_scale_four(monkeypatch) -> None:
    monkeypatch.setenv("MULTIMODAL_GRID_LINES", "true")
    grid_enabled = _decode_png(frame_to_png_data_url(_frame(), upscale=3))
    monkeypatch.setenv("MULTIMODAL_GRID_LINES", "false")
    grid_disabled = _decode_png(frame_to_png_data_url(_frame(), upscale=3))

    assert grid_enabled.size == (6, 6)
    assert grid_enabled.tobytes() == grid_disabled.tobytes()


def test_current_grid_image_part_contract_is_preserved(monkeypatch) -> None:
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)
    assert current_grid_image_part(_frame()) is None

    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    part = current_grid_image_part(_frame())
    assert part is not None
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
