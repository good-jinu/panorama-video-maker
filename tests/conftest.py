import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workshop.utils import save_sprite_sheet

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def numbered_frames(n: int, size=(8, 6)) -> list[Image.Image]:
    """Frames whose red channel encodes their 1-based index, so order is checkable."""
    return [Image.new("RGBA", size, (i, 255 - i, 7, 255)) for i in range(1, n + 1)]


def make_sheet(path: Path, frames: list[Image.Image], columns: int, interval: float = 0.25) -> Path:
    save_sprite_sheet(frames, path, columns=columns, interval=interval)
    return path


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def snapshot():
    """Compare a digest of arrays to tests/snapshots/<name>.sha256 (UPDATE_SNAPSHOTS=1 rewrites)."""

    def check(name: str, arrays: list[np.ndarray]) -> None:
        h = hashlib.sha256()
        for a in arrays:
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        digest = h.hexdigest()
        path = SNAPSHOT_DIR / f"{name}.sha256"
        if os.environ.get("UPDATE_SNAPSHOTS") or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(digest + "\n")
            return
        assert digest == path.read_text().strip(), (
            f"snapshot '{name}' changed; rerun with UPDATE_SNAPSHOTS=1 if intended"
        )

    return check
