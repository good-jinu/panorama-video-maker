"""Build a 2x2 image comparison grid with captions.

Usage:
    uv run python -m workshop.comparison
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

from workshop.utils import parse_hex_color

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "comparisons" / "comparison_2x2.png"

CELL_W, CELL_H = 900, 1130
CAPTION_H = 130
GAP = 0
PAD = 0
BG = (255, 255, 255)
FONT_SIZE = 72

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    ("/Library/Fonts/Arial Bold.ttf", 0),
]


def load_font(size: int):
    for path, index in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


def text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    return (255, 255, 255) if lum < 140 else (20, 20, 20)


def paste_fit(canvas: Image.Image, img_path: Path, x: int, y: int):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, (CELL_W, CELL_H), Image.Resampling.LANCZOS)
    canvas.paste(img, (x + (CELL_W - img.width) // 2, y + (CELL_H - img.height) // 2))


def draw_caption(draw: ImageDraw.ImageDraw, font, x: int, y: int, w: int, text: str, bg_hex: str):
    bg = parse_hex_color(bg_hex)[:3]
    draw.rectangle([x, y, x + w, y + CAPTION_H], fill=bg)
    draw.text((x + w / 2, y + CAPTION_H / 2), text, fill=text_color(bg), font=font, anchor="mm")


def build_comparison_2x2(output_path: Path = DEFAULT_OUT):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = load_font(FONT_SIZE)
    cell_h = CAPTION_H + GAP + CELL_H
    w = PAD * 3 + CELL_W * 2
    h = PAD * 3 + cell_h * 2

    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)

    xl = PAD
    xr = PAD * 2 + CELL_W
    y_top = PAD
    y_bot = PAD * 2 + cell_h

    # Top row
    draw_caption(draw, font, xl, y_top, CELL_W, "nano banana 2", "#90a02f")
    paste_fit(canvas, ROOT / "inputs" / "nanobanana2.jpg", xl, y_top + CAPTION_H + GAP)

    draw_caption(draw, font, xr, y_top, CELL_W, "nano banana 2.1", "#dada0c")
    paste_fit(canvas, ROOT / "inputs" / "nanobanana2_1.jpg", xr, y_top + CAPTION_H + GAP)

    # Bottom row
    draw_caption(draw, font, xl, y_bot, CELL_W * 2 + PAD, "gpt-image", "#10a37f")
    paste_fit(canvas, ROOT / "inputs" / "gpt-image.png", xl, y_bot + CAPTION_H + GAP)
    paste_fit(canvas, ROOT / "inputs" / "gpt-image_variant.png", xr, y_bot + CAPTION_H + GAP)

    canvas.save(output_path)
    print(f"Saved comparison: {output_path} ({w}x{h})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="Output path")
    args = ap.parse_args()
    build_comparison_2x2(args.output)


if __name__ == "__main__":
    main()
