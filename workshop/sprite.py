"""Export a transparent sprite sheet from a video.

Usage:
    uv run python -m workshop.sprite inputs/videos/character.mp4 --crop --keep-frames
    uv run python -m workshop.sprite inputs/videos/scene.mp4 --zoom-start 1.37 --zoom-step 0.0095
    uv run python -m workshop.sprite inputs/videos/character.mp4 --crop --model birefnet
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from workshop.bgremove import MODEL_NAMES, BackgroundRemover
from workshop.clean import clean_frame
from workshop.utils import (
    crop_transparent,
    extract_video_frames,
    parse_hex_color,
    save_sprite_sheet,
)


def zoom_canvas(img: Image.Image, factor: float) -> Image.Image:
    """Scale image about center and crop back to original canvas dimensions."""
    w, h = img.size
    big = img.resize((round(w * factor), round(h * factor)), Image.Resampling.LANCZOS)
    left = (big.width - w) // 2
    top = (big.height - h) // 2
    return big.crop((left, top, left + w, top + h))


def generate_sprite_sheet(
    video_path: Path,
    output_path: Path | None = None,
    interval: float = 0.25,
    columns: int | None = None,
    crop: bool = False,
    scale: float | None = None,
    clean_bg: str | None = None,
    keep_background: bool = False,
    keep_frames: bool = False,
    zoom_start: float | None = None,
    zoom_step: float = 0.01,
    model: str | None = None,
) -> Path:
    if keep_background and crop:
        raise ValueError("--crop requires transparent background; cannot be combined with --keep-background")

    video = Path(video_path)
    tag = ("_scaled" if zoom_start else "") + ("_bg" if keep_background else "")
    default_name = f"sprite{tag}.png"
    out = output_path or Path("outputs") / video.stem / default_name
    out.parent.mkdir(parents=True, exist_ok=True)

    remover = None if keep_background else BackgroundRemover(model)
    target_rgb = np.array(parse_hex_color(clean_bg)[:3], dtype=np.float32) if clean_bg else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = extract_video_frames(video, interval, Path(tmp_dir))
        print(f"Extracted {len(frame_files)} frames (every {interval}s) from {video.name}")

        cutouts = []
        for i, file_path in enumerate(frame_files):
            src = Image.open(file_path).convert("RGB")
            if zoom_start:
                factor = zoom_start - (zoom_step * i)
                src = zoom_canvas(src, factor)

            if remover:
                frame_rgba, category = remover.remove(src)
                print(f"  [{i + 1}/{len(frame_files)}] {remover.name} route={category}")
                if target_rgb is not None:
                    frame_rgba, _, _ = clean_frame(frame_rgba, target_rgb)
            else:
                frame_rgba = src.convert("RGBA")

            if scale and scale != 1.0:
                nw, nh = round(frame_rgba.width * scale), round(frame_rgba.height * scale)
                frame_rgba = frame_rgba.resize((nw, nh), Image.Resampling.LANCZOS)

            cutouts.append(frame_rgba)

    if crop:
        cutouts = crop_transparent(cutouts)

    if keep_frames:
        frames_dir = out.parent / f"frames{tag}"
        frames_dir.mkdir(exist_ok=True)
        for i, c in enumerate(cutouts):
            c.save(frames_dir / f"frame_{i:03d}.png")
        print(f"Saved {len(cutouts)} individual frames to {frames_dir}/")

    saved_sheet, saved_json = save_sprite_sheet(
        cutouts,
        out,
        columns=columns,
        interval=interval,
        fps=round(1.0 / interval, 2) if interval else None,
    )
    print(f"Successfully generated: {saved_sheet} ({saved_sheet.stat().st_size // 1024} KB)")
    return saved_sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video", type=Path, help="Input video file path")
    ap.add_argument("-o", "--output", type=Path, help="Output sprite sheet image path")
    ap.add_argument("--interval", type=float, default=0.25, help="Seconds between frames (default: 0.25)")
    ap.add_argument("--columns", type=int, help="Grid columns (default: automatic square-ish)")
    ap.add_argument("--crop", action="store_true", help="Crop all frames to union bounding box of subject")
    ap.add_argument("--scale", type=float, help="Scale factor for frames (e.g. 0.5 for half resolution)")
    ap.add_argument("--clean-bg", type=str, help="Hex color to defringe and clean (e.g. '#2c4d6b')")
    ap.add_argument("--keep-background", action="store_true", help="Skip background removal")
    ap.add_argument("--keep-frames", action="store_true", help="Also save individual frame images")
    ap.add_argument("--zoom-start", type=float, help="Initial zoom factor for frame 0 (compensates camera zoom)")
    ap.add_argument("--zoom-step", type=float, default=0.01, help="Zoom decrease per frame (default: 0.01)")
    ap.add_argument("--model", choices=MODEL_NAMES, help="Background-removal model (default: $WORKSHOP_BG_MODEL or withoutbg)")
    args = ap.parse_args()

    generate_sprite_sheet(
        video_path=args.video,
        output_path=args.output,
        interval=args.interval,
        columns=args.columns,
        crop=args.crop,
        scale=args.scale,
        clean_bg=args.clean_bg,
        keep_background=args.keep_background,
        keep_frames=args.keep_frames,
        zoom_start=args.zoom_start,
        zoom_step=args.zoom_step,
        model=args.model,
    )


if __name__ == "__main__":
    main()
