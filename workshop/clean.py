"""Post-process a cutout sprite sheet or image to eliminate residual background artifacts:
1. Edge defringing: removes semi-transparent pixels tinted with the background color.
2. Trapped gaps: removes enclosed opaque background patches (e.g. inside chair backs or table legs).
3. Zero RGB: clears color channels of transparent pixels to prevent color bleed during scaling or compositing.

Usage:
    uv run python -m workshop.clean path/to/sprite_sheet.png [--bg-color "#2c4d6b"]
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from workshop.utils import load_sprite_sheet, parse_hex_color, save_sprite_sheet

# Default bounding box (x0, y0, x1, y1), inclusive, for protected shadow regions.
# Gap components lying entirely inside it are kept (they are real foreground shadows, not background).
JEANS_BOX = (370, 210, 460, 280)


def clean_frame(
    frame: Image.Image,
    target_rgb: np.ndarray,
    fringe_tol: float = 32.0,
    gap_tol: float = 22.0,
    protect_box: tuple[int, int, int, int] | None = JEANS_BOX,
) -> tuple[Image.Image, int, int]:
    """Clean a single RGBA frame. Returns (cleaned_frame, fringe_removed_count, gaps_removed_count)."""
    arr = np.array(frame.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    dist = np.linalg.norm(rgb.astype(np.float32) - target_rgb, axis=-1)

    # 1. Defringe: remove semi-transparent edge pixels matching the background color
    fringe_mask = (dist < fringe_tol) & (alpha > 0) & (alpha < 240)
    fringe_count = int(np.sum(fringe_mask))
    alpha[fringe_mask] = 0
    alpha[alpha < 12] = 0

    # 2. Trapped opaque gaps (alpha >= 240): label 4-connected background-colored components
    is_candidate = ((dist < gap_tol) & (alpha >= 240)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_candidate, connectivity=4)

    remove = np.ones(n_labels, dtype=bool)
    remove[0] = False  # label 0 = non-candidate pixels
    if protect_box is not None and n_labels > 1:
        x0, y0, x1, y1 = protect_box
        left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        right = left + stats[:, cv2.CC_STAT_WIDTH] - 1
        bottom = top + stats[:, cv2.CC_STAT_HEIGHT] - 1
        in_box = (left >= x0) & (right <= x1) & (top >= y0) & (bottom <= y1)
        remove &= ~in_box

    gap_mask = remove[labels]
    gaps_count = int(gap_mask.sum())
    alpha[gap_mask] = 0

    # 3. Zero out RGB on transparent pixels
    rgb[alpha == 0] = 0
    return Image.fromarray(arr), fringe_count, gaps_count


def clean_sprite_sheet(
    input_path: Path,
    output_path: Path | None = None,
    target_bg_hex: str = "#2c4d6b",
    fringe_tol: float = 32.0,
    gap_tol: float = 22.0,
    backup: bool = True,
):
    path = Path(input_path)
    out = Path(output_path) if output_path else path

    if backup and out == path:
        backup_path = path.with_name(f"{path.stem}_raw{path.suffix}")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"Backed up original to: {backup_path}")

    frames, meta = load_sprite_sheet(path)
    rgba = parse_hex_color(target_bg_hex)
    target_rgb = np.array(rgba[:3], dtype=np.float32)

    cleaned_frames = []
    total_fringe = 0
    total_gaps = 0

    for f in frames:
        cf, fr, gp = clean_frame(f, target_rgb, fringe_tol=fringe_tol, gap_tol=gap_tol)
        cleaned_frames.append(cf)
        total_fringe += fr
        total_gaps += gp

    save_sprite_sheet(
        cleaned_frames,
        out,
        columns=meta.get("columns"),
        interval=meta.get("intervalSeconds"),
        fps=meta.get("fps"),
    )
    print(f"Cleaned {out}:")
    print(f"  Fringe pixels removed: {total_fringe}")
    print(f"  Trapped gap pixels removed: {total_gaps}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sprite_sheet", type=Path, help="Sprite sheet image to clean")
    ap.add_argument("-o", "--output", type=Path, help="Output path (default: overwrite input)")
    ap.add_argument("--bg-color", type=str, default="#2c4d6b", help="Target background hex color (default: #2c4d6b)")
    ap.add_argument("--fringe-tol", type=float, default=32.0, help="Fringe tolerance (default: 32)")
    ap.add_argument("--gap-tol", type=float, default=22.0, help="Gap tolerance (default: 22)")
    ap.add_argument("--no-backup", action="store_true", help="Skip creating backup file")
    args = ap.parse_args()

    clean_sprite_sheet(
        input_path=args.sprite_sheet,
        output_path=args.output,
        target_bg_hex=args.bg_color,
        fringe_tol=args.fringe_tol,
        gap_tol=args.gap_tol,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
