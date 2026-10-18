from collections import deque

import numpy as np
import pytest
from PIL import Image

from workshop.clean import JEANS_BOX, clean_frame, clean_sprite_sheet
from workshop.utils import load_sprite_sheet, save_sprite_sheet

BG = np.array([0x2C, 0x4D, 0x6B], dtype=np.float32)


def reference_clean(frame, target_rgb, fringe_tol=32.0, gap_tol=22.0, protect_jeans=True):
    """The original pure-Python BFS implementation, kept as the behavioral reference."""
    arr = np.array(frame.convert("RGBA"))
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    dist = np.linalg.norm(rgb.astype(np.float32) - target_rgb, axis=-1)
    fringe = (dist < fringe_tol) & (alpha > 0) & (alpha < 240)
    n_fringe = int(fringe.sum())
    alpha[fringe] = 0
    alpha[alpha < 12] = 0
    cand = (dist < gap_tol) & (alpha >= 240)
    h, w = cand.shape
    seen = np.zeros((h, w), bool)
    n_gaps = 0
    x0, y0, x1, y1 = JEANS_BOX
    for y in range(h):
        for x in range(w):
            if cand[y, x] and not seen[y, x]:
                q, comp = deque([(y, x)]), [(y, x)]
                seen[y, x] = True
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                            comp.append((ny, nx))
                ys, xs = [p[0] for p in comp], [p[1] for p in comp]
                in_jeans = protect_jeans and min(ys) >= y0 and max(ys) <= y1 and min(xs) >= x0 and max(xs) <= x1
                if not in_jeans:
                    n_gaps += len(comp)
                    for py, px in comp:
                        alpha[py, px] = 0
    rgb[alpha == 0] = 0
    return arr, n_fringe, n_gaps


def synthetic_frame(rng, h=300, w=480):
    """Noisy subject with background-colored blobs, semi-transparent edges, and a jeans-box gap."""
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = np.full((h, w), 255, np.uint8)
    for _ in range(40):
        y, x = rng.integers(0, h - 12), rng.integers(0, w - 12)
        rgb[y:y + rng.integers(2, 12), x:x + rng.integers(2, 12)] = BG.astype(np.uint8) + rng.integers(-6, 7, 3)
    alpha[rng.random((h, w)) < 0.15] = rng.integers(1, 240)
    rgb[230:240, 400:420] = BG.astype(np.uint8)  # inside JEANS_BOX -> protected
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


@pytest.mark.parametrize("protect", [True, False])
def test_clean_frame_matches_reference(rng, protect):
    box = JEANS_BOX if protect else None
    for _ in range(3):
        frame = synthetic_frame(rng)
        ref, ref_f, ref_g = reference_clean(frame, BG, protect_jeans=protect)
        out, f, g = clean_frame(frame, BG, protect_box=box)
        assert (f, g) == (ref_f, ref_g)
        assert np.array_equal(np.array(out), ref)


def test_jeans_box_protection():
    arr = np.zeros((300, 480, 4), np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 255
    arr[230:240, 400:420, :3] = BG.astype(np.uint8)  # inside box
    arr[10:20, 10:20, :3] = BG.astype(np.uint8)      # outside box
    out, _, gaps = clean_frame(Image.fromarray(arr), BG)
    a = np.array(out)[..., 3]
    assert gaps == 100
    assert (a[230:240, 400:420] == 255).all() and (a[10:20, 10:20] == 0).all()
    _, _, gaps_unprotected = clean_frame(Image.fromarray(arr), BG, protect_box=None)
    assert gaps_unprotected == 300


def test_transparent_pixels_have_zero_rgb(rng):
    out, _, _ = clean_frame(synthetic_frame(rng), BG)
    arr = np.array(out)
    assert (arr[arr[..., 3] == 0][:, :3] == 0).all()


def test_clean_sprite_sheet_backup_and_snapshot(tmp_path, rng, snapshot):
    frames = [synthetic_frame(rng, 120, 160) for _ in range(3)]
    path = tmp_path / "sheet.png"
    save_sprite_sheet(frames, path, columns=3, interval=0.25)
    clean_sprite_sheet(path)
    assert (tmp_path / "sheet_raw.png").exists()
    cleaned, meta = load_sprite_sheet(path)
    assert meta["intervalSeconds"] == 0.25
    snapshot("clean_sheet", [np.array(f) for f in cleaned])
