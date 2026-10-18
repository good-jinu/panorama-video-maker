import json

import pytest

from tests.conftest import make_sheet, numbered_frames
from workshop.recut import recut_sprite_sheet
from workshop.utils import load_sprite_sheet


def test_recut_selects_range_and_repacks(tmp_path):
    src = make_sheet(tmp_path / "src.png", numbered_frames(28), columns=4, interval=0.25)
    out = tmp_path / "walk.png"
    recut_sprite_sheet(src, out, frames_range="1-19", columns=4)

    meta = json.loads(out.with_suffix(".json").read_text())
    assert (meta["columns"], meta["rows"], meta["frameCount"]) == (4, 5, 19)
    assert meta["intervalSeconds"] == 0.25
    frames, _ = load_sprite_sheet(out)
    assert [f.getpixel((0, 0))[0] for f in frames] == list(range(1, 20))


def test_recut_middle_range(tmp_path):
    src = make_sheet(tmp_path / "src.png", numbered_frames(10), columns=5)
    out = tmp_path / "mid.png"
    recut_sprite_sheet(src, out, frames_range="3-6", columns=2)
    frames, _ = load_sprite_sheet(out)
    assert [f.getpixel((0, 0))[0] for f in frames] == [3, 4, 5, 6]


@pytest.mark.parametrize("bad", ["0-3", "5-11", "6-2"])
def test_recut_rejects_out_of_bounds(tmp_path, bad):
    src = make_sheet(tmp_path / "src.png", numbered_frames(10), columns=5)
    with pytest.raises(ValueError):
        recut_sprite_sheet(src, tmp_path / "x.png", frames_range=bad, columns=2)
