from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import workshop.composite as composite
from tests.conftest import make_sheet


class CapturingWriter:
    """Stands in for AnimationWriter so tests check pixels, not codecs."""

    def __init__(self, output, width, height, **_):
        self.mp4_path, self.gif_path, self.gif_encoder = Path(output), None, None
        self.size = (width, height)
        self.frames: list[np.ndarray] = []
        CapturingWriter.last = self

    def write(self, frame):
        assert frame.size == self.size
        self.frames.append(np.array(frame))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sheets(tmp_path):
    w, h = 64, 36
    scene_frames = []
    for i in range(4):
        f = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        f.paste((40 * i, 120, 200, 255), (8, 6, 56, 30))  # opaque block w/ transparent margin
        scene_frames.append(f)
    cat_frames = []
    for i in range(3):
        f = Image.new("RGBA", (20, 16), (0, 0, 0, 0))
        f.paste((250, 80 * i, 10, 255), (4, 2, 16, 14))
        cat_frames.append(f)
    return (
        make_sheet(tmp_path / "scene.png", scene_frames, columns=2, interval=0.25),
        make_sheet(tmp_path / "cat.png", cat_frames, columns=3, interval=0.25),
    )


@pytest.mark.parametrize("mode, extra", [
    ("tracking", {"scene_gradient": "#203a55"}),
    ("tracking", {"bg_gradient": ["#142230", "#355a7d"]}),
    ("standard", {"bg_color": "#ffffff"}),
])
def test_render_composite_snapshot(monkeypatch, sheets, snapshot, mode, extra):
    monkeypatch.setattr(composite, "AnimationWriter", CapturingWriter)
    scene, cat = sheets
    composite.render_composite(
        [scene, scene], cat, Path("/tmp/unused.mp4"), mode=mode, scale=1.0, scene_scale=0.5,
        scene_y=4, scene_gap=0.5, floor_y=34, fps=8, char_frames="all", **extra,
    )
    frames = CapturingWriter.last.frames
    assert frames and all(f.shape == (36, 64, 3) for f in frames)
    snapshot(f"composite_{mode}_{'_'.join(extra)}", frames)


def test_tracking_keeps_character_fixed(monkeypatch, sheets):
    monkeypatch.setattr(composite, "AnimationWriter", CapturingWriter)
    scene, cat = sheets
    composite.render_composite([scene], cat, Path("/tmp/u.mp4"), mode="tracking", scale=1.0,
                               scene_scale=0.5, floor_y=34, fps=8, char_frames="all", gif=False)
    # cat body pixel is pure-ish orange-red in every frame at the same position
    cx, cy = (64 - 20) // 2 + 6, 34 - 16 + 6
    assert all(f[cy, cx, 0] == 250 for f in CapturingWriter.last.frames)


def test_resolve_background_variants():
    _, rgb, desc = composite.resolve_background(10, 10, bg_color="#ffffff")
    assert rgb == (255, 255, 255) and desc.startswith("solid")
    canvas, rgb, desc = composite.resolve_background(10, 10, bg_gradient="sunset")
    assert desc.startswith("gradient") and canvas.size == (10, 10)
    _, _, desc = composite.resolve_background(10, 10, bg_color="#000000,#ffffff")
    assert desc.startswith("gradient")
