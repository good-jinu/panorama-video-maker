import json

import numpy as np
import pytest
from PIL import Image

from tests.conftest import make_sheet, numbered_frames
from workshop.utils import (
    create_gradient_image,
    crop_transparent,
    load_sprite_sheet,
    parse_hex_color,
    save_sprite_sheet,
)


@pytest.mark.parametrize(
    "value, expected",
    [("#ffffff", (255, 255, 255, 255)), ("203a55", (32, 58, 85, 255)), ("#11223344", (17, 34, 51, 68))],
)
def test_parse_hex_color(value, expected):
    assert parse_hex_color(value) == expected


def test_parse_hex_color_rejects_bad_length():
    with pytest.raises(ValueError):
        parse_hex_color("#fff")


def test_sprite_sheet_roundtrip(tmp_path):
    frames = numbered_frames(7)
    sheet, meta_path = save_sprite_sheet(frames, tmp_path / "s.png", columns=3, interval=0.5, fps=2)
    meta = json.loads(meta_path.read_text())
    assert meta == {"frameWidth": 8, "frameHeight": 6, "columns": 3, "rows": 3, "frameCount": 7,
                    "intervalSeconds": 0.5, "fps": 2}
    loaded, loaded_meta = load_sprite_sheet(sheet)
    assert len(loaded) == 7 and loaded_meta == meta
    for orig, back in zip(frames, loaded):
        assert np.array_equal(np.array(orig), np.array(back))


def test_load_sprite_sheet_with_grid_and_without_json(tmp_path):
    path = make_sheet(tmp_path / "s.png", numbered_frames(4), columns=2)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError):
        load_sprite_sheet(path)
    frames, meta = load_sprite_sheet(path, grid_str="2x2")
    assert [f.getpixel((0, 0))[0] for f in frames] == [1, 2, 3, 4]
    assert meta["frameCount"] == 4


def test_load_sprite_sheet_single_image_without_json(tmp_path):
    img = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    path = tmp_path / "single.png"
    img.save(path)
    with pytest.raises(ValueError):
        load_sprite_sheet(path)
    frames, meta = load_sprite_sheet(path, allow_single_image=True)
    assert len(frames) == 1
    assert frames[0].size == (100, 50)
    assert meta["columns"] == 1
    assert meta["rows"] == 1
    assert meta["frameCount"] == 1


def test_crop_transparent_uses_union_bbox():
    a = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    b = a.copy()
    a.paste((255, 0, 0, 255), (2, 3, 5, 6))
    b.paste((0, 255, 0, 255), (10, 12, 15, 14))
    out = crop_transparent([a, b])
    assert all(f.size == (13, 11) for f in out)


@pytest.mark.parametrize("direction", ["vertical", "horizontal", "diagonal", "radial"])
def test_gradient_shapes_and_endpoints(direction):
    img = create_gradient_image(40, 30, ["#000000", "#ffffff"], direction=direction)
    arr = np.array(img)
    assert arr.shape == (30, 40, 4) and (arr[..., 3] == 255).all()
    if direction == "vertical":
        assert arr[0, 0, 0] == 0 and arr[-1, 0, 0] == 255
    if direction == "horizontal":
        assert arr[0, 0, 0] == 0 and arr[0, -1, 0] == 255
