import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from tests.conftest import requires_ffmpeg
from workshop.utils import AnimationWriter, resolve_gif_encoder


def count_frames(path) -> int:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-", "-stats"],
        capture_output=True, text=True, check=True,
    ).stderr
    return int(out.rsplit("frame=", 1)[1].split()[0])


@requires_ffmpeg
@pytest.mark.parametrize("encoder", [
    "ffmpeg",
    pytest.param("gifski", marks=pytest.mark.skipif(shutil.which("gifski") is None, reason="gifski not on PATH")),
])
def test_writer_streams_mp4_and_gif(tmp_path, encoder):
    colors = [(255, 255, 255), (32, 58, 85), (220, 40, 40), (40, 200, 90)]
    with AnimationWriter(tmp_path / "anim.mp4", 65, 33, fps=8, make_gif=True, gif_encoder=encoder) as w:  # odd size on purpose
        for c in colors * 3:
            w.write(Image.new("RGB", (65, 33), c))
    assert w.frame_count == 12
    assert count_frames(w.mp4_path) == 12
    gif = Image.open(w.gif_path)
    assert gif.n_frames == 12
    gif.seek(1)
    assert np.abs(np.array(gif.convert("RGB"))[16, 32].astype(int) - colors[1]).max() <= 6


@requires_ffmpeg
def test_writer_defaults_to_mp4_only(tmp_path):
    with AnimationWriter(tmp_path / "default.mp4", 16, 16, fps=4) as w:
        for _ in range(4):
            w.write(np.zeros((16, 16, 3), np.uint8))
    assert w.gif_path is None
    assert w.mp4_path.exists()
    assert count_frames(w.mp4_path) == 4


@requires_ffmpeg
def test_writer_accepts_numpy_and_skips_gif(tmp_path):
    with AnimationWriter(tmp_path / "a.mp4", 16, 16, fps=4, make_gif=False) as w:
        for _ in range(4):
            w.write(np.zeros((16, 16, 3), np.uint8))
    assert w.gif_path is None and count_frames(w.mp4_path) == 4


def test_writer_rejects_wrong_size(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    with pytest.raises(ValueError):
        with AnimationWriter(tmp_path / "a.mp4", 16, 16, make_gif=False) as w:
            w.write(Image.new("RGB", (8, 8)))


def test_resolve_gif_encoder(monkeypatch):
    monkeypatch.setenv("WORKSHOP_GIF_ENCODER", "ffmpeg")
    assert resolve_gif_encoder() == "ffmpeg"
    with pytest.raises(ValueError):
        resolve_gif_encoder("pngquant")
