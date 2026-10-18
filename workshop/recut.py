"""Slice and repack a sprite sheet.

Usage:
    uv run python -m workshop.recut outputs/character/sprite.png --frames 2-29 --columns 4 -o outputs/character/sprite_recut.png
"""
import argparse
from pathlib import Path

from workshop.utils import load_sprite_sheet, save_sprite_sheet


def recut_sprite_sheet(
    sheet_path: Path,
    output_path: Path,
    frames_range: str,
    columns: int,
    grid_str: str = None,
):
    frames, meta = load_sprite_sheet(sheet_path, grid_str=grid_str)
    first, last = map(int, frames_range.split("-"))

    if not (1 <= first <= last <= len(frames)):
        raise ValueError(f"--frames {frames_range} is out of bounds (1-{len(frames)})")

    selected = frames[first - 1:last]
    out, _ = save_sprite_sheet(
        selected,
        output_path,
        columns=columns,
        interval=meta.get("intervalSeconds"),
        fps=meta.get("fps"),
    )
    print(f"Saved recut sprite ({len(selected)} frames, {columns} cols): {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sheet", type=Path, help="Input sprite sheet image")
    ap.add_argument("--frames", required=True, help="1-based inclusive range in reading order, e.g. 2-29")
    ap.add_argument("--columns", type=int, required=True, help="Number of columns in output sheet")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output sprite sheet image path")
    ap.add_argument("--grid", help="Grid dimension as COLSxROWS if companion .json is missing")
    args = ap.parse_args()

    recut_sprite_sheet(
        sheet_path=args.sheet,
        output_path=args.output,
        frames_range=args.frames,
        columns=args.columns,
        grid_str=args.grid,
    )


if __name__ == "__main__":
    main()
