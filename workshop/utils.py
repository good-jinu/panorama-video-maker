"""Shared utilities for image processing, sprite sheet handling, and ffmpeg operations."""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

GRADIENT_PRESETS: dict[str, list[str]] = {
    "scene": ["#142230", "#355a7d"],
    "studio": ["#2d3748", "#1a202c"],
    "twilight": ["#0f172a", "#1e3a5f", "#334155"],
    "sunset": ["#2e1065", "#7c3aed", "#f59e0b"],
    "warm": ["#292524", "#44403c", "#78716c"],
    "light": ["#ffffff", "#e2e8f0"],
}


def parse_hex_color(hex_str: str) -> tuple[int, int, int, int]:
    """Parse a #RRGGBB or #RRGGBBAA hex string into an RGBA tuple."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        return (r, g, b, 255)
    elif len(hex_str) == 8:
        r, g, b, a = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16), int(hex_str[6:8], 16)
        return (r, g, b, a)
    raise ValueError(f"Invalid hex color format: '{hex_str}' (expected #RRGGBB or #RRGGBBAA)")


def create_gradient_image(
    width: int,
    height: int,
    colors: list[tuple[int, int, int, int] | str],
    direction: str = "vertical",
) -> Image.Image:
    """Generate an RGBA gradient image supporting vertical, horizontal, diagonal, and radial patterns."""
    if not colors:
        raise ValueError("At least one color must be provided for gradient")

    rgba_colors = []
    for c in colors:
        if isinstance(c, str):
            rgba_colors.append(parse_hex_color(c))
        else:
            rgba_colors.append(c)

    if len(rgba_colors) == 1:
        return Image.new("RGBA", (width, height), rgba_colors[0])

    stops = np.linspace(0.0, 1.0, len(rgba_colors), dtype=np.float32)
    color_arr = np.array(rgba_colors, dtype=np.float32)

    if direction == "vertical":
        t = np.linspace(0.0, 1.0, height, dtype=np.float32)
        grad_1d = np.zeros((height, 4), dtype=np.float32)
        for ch in range(4):
            grad_1d[:, ch] = np.interp(t, stops, color_arr[:, ch])
        grad = np.tile(grad_1d[:, None, :], (1, width, 1))
    elif direction == "horizontal":
        t = np.linspace(0.0, 1.0, width, dtype=np.float32)
        grad_1d = np.zeros((width, 4), dtype=np.float32)
        for ch in range(4):
            grad_1d[:, ch] = np.interp(t, stops, color_arr[:, ch])
        grad = np.tile(grad_1d[None, :, :], (height, 1, 1))
    elif direction == "radial":
        y, x = np.ogrid[:height, :width]
        cx, cy = width / 2.0, height * 0.45
        max_dist = np.hypot(width / 2.0, height / 2.0)
        dist = np.hypot(x - cx, y - cy) / max_dist
        dist = np.clip(dist, 0.0, 1.0)
        grad = np.zeros((height, width, 4), dtype=np.float32)
        for ch in range(4):
            grad[:, :, ch] = np.interp(dist.ravel(), stops, color_arr[:, ch]).reshape((height, width))
    elif direction == "diagonal":
        y, x = np.ogrid[:height, :width]
        dist = (x / width + y / height) / 2.0
        dist = np.clip(dist, 0.0, 1.0)
        grad = np.zeros((height, width, 4), dtype=np.float32)
        for ch in range(4):
            grad[:, :, ch] = np.interp(dist.ravel(), stops, color_arr[:, ch]).reshape((height, width))
    else:
        raise ValueError(f"Unknown gradient direction: '{direction}'. Expected 'vertical', 'horizontal', 'radial', or 'diagonal'.")

    return Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8), mode="RGBA")


def create_scene_radial_backdrop(
    width: int,
    height: int,
    inner_color: tuple[int, int, int, int] | str = "#203a55",
    radius_scale: float = 2.0,
    radius_x: float | None = None,
    radius_y: float | None = None,
    center_y_ratio: float = 0.52,
) -> tuple[Image.Image, int, int]:
    """Create a soft radial gradient backdrop that fades out smoothly to transparent at the edges.

    Args:
        width: Scene width
        height: Scene height
        inner_color: Center color (e.g. navy #203a55)
        radius_scale: Multiplier for radial gradient size (default: 2.0)
        radius_x: Horizontal radius for fadeout (overrides radius_scale if provided)
        radius_y: Vertical radius for fadeout (overrides radius_scale if provided)
        center_y_ratio: Center Y relative to scene height (0.52 centers on table/couple)

    Returns:
        (backdrop_image, pad_x, pad_y) where (pad_x, pad_y) is offset to align with scene
    """
    if isinstance(inner_color, str):
        rgba = parse_hex_color(inner_color)
    else:
        rgba = inner_color

    rx = radius_x or (width * 0.48 * radius_scale)
    ry = radius_y or (height * 0.52 * radius_scale)

    # Padding ensures the entire falloff fits inside the backdrop image without edge clipping
    pad_x = round(max(rx - width / 2.0 + 40, width * 0.4))
    pad_y = round(max(ry - height / 2.0 + 40, height * 0.4))
    bw = width + 2 * pad_x
    bh = height + 2 * pad_y

    cx = bw / 2.0
    cy = pad_y + height * center_y_ratio

    y, x = np.ogrid[:bh, :bw]
    dist = np.hypot((x - cx) / rx, (y - cy) / ry)
    dist = np.clip(dist, 0.0, 1.0)

    # Smooth cosine falloff with power curve for soft natural fadeout
    alpha_factor = (0.5 * (1.0 + np.cos(dist * np.pi))) ** 1.3

    img_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
    img_arr[:, :, 0] = rgba[0]
    img_arr[:, :, 1] = rgba[1]
    img_arr[:, :, 2] = rgba[2]
    img_arr[:, :, 3] = np.clip(alpha_factor * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr, mode="RGBA"), pad_x, pad_y


def load_sprite_sheet(
    sheet_path: str | Path,
    grid_str: str | None = None,
    allow_single_image: bool = False,
) -> tuple[list[Image.Image], dict]:
    """Load a sprite sheet image and slice it into individual RGBA frames using .json or grid."""
    path = Path(sheet_path)
    sheet = Image.open(path).convert("RGBA")
    json_path = path.with_suffix(".json")

    meta = {}
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        cols = meta.get("columns")
        rows = meta.get("rows")
        count = meta.get("frameCount", cols * rows if cols and rows else None)
        fw = meta.get("frameWidth", sheet.width // cols if cols else None)
        fh = meta.get("frameHeight", sheet.height // rows if rows else None)
    elif grid_str:
        cols, rows = map(int, grid_str.lower().split("x"))
        fw, fh = sheet.width // cols, sheet.height // rows
        count = cols * rows
        meta = {"columns": cols, "rows": rows, "frameWidth": fw, "frameHeight": fh, "frameCount": count}
    elif allow_single_image:
        cols, rows = 1, 1
        fw, fh = sheet.width, sheet.height
        count = 1
        meta = {
            "columns": 1,
            "rows": 1,
            "frameWidth": fw,
            "frameHeight": fh,
            "frameCount": 1,
            "intervalSeconds": 10.0,
            "fps": 1.0,
        }
    else:
        raise ValueError(f"Cannot slice {sheet_path}: no companion .json file found and no --grid specified")

    if not count:
        count = cols * rows

    frames = []
    for i in range(count):
        c, r = i % cols, i // cols
        frames.append(sheet.crop((c * fw, r * fh, (c + 1) * fw, (r + 1) * fh)))

    return frames, meta


def save_sprite_sheet(
    frames: list[Image.Image],
    output_path: str | Path,
    columns: int | None = None,
    interval: float | None = None,
    fps: float | None = None,
) -> tuple[Path, Path]:
    """Pack a list of frames into a grid sprite sheet and save it with a .json metadata file."""
    if not frames:
        raise ValueError("Cannot save empty frames list")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fw, fh = frames[0].size
    count = len(frames)
    cols = columns or math.ceil(math.sqrt(count * fh / fw))
    rows = math.ceil(count / cols)

    sheet = Image.new("RGBA", (cols * fw, rows * fh), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * fw, (i // cols) * fh))

    sheet.save(out_path)

    meta = {
        "frameWidth": fw,
        "frameHeight": fh,
        "columns": cols,
        "rows": rows,
        "frameCount": count,
    }
    if interval is not None:
        meta["intervalSeconds"] = interval
    if fps is not None:
        meta["fps"] = fps

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2))
    return out_path, json_path


def crop_transparent(frames: list[Image.Image]) -> list[Image.Image]:
    """Crop all frames to the union bounding box of non-transparent pixels."""
    boxes = [
        f.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
        for f in frames
        if "A" in f.getbands()
    ]
    boxes = [b for b in boxes if b]
    if not boxes:
        return frames

    union_box = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    return [f.crop(union_box) for f in frames]


def extract_video_frames(video_path: str | Path, interval: float, out_dir: Path) -> list[Path]:
    """Extract frames from a video at a fixed second interval using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-vf", f"fps=1/{interval}",
            str(out_dir / "f_%04d.png"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("f_*.png"))


def resolve_gif_encoder(preference: str | None = None) -> str:
    """Pick the GIF encoder: explicit preference > $WORKSHOP_GIF_ENCODER > gifski if installed > ffmpeg."""
    choice = (preference or os.environ.get("WORKSHOP_GIF_ENCODER") or "auto").lower()
    if choice not in ("auto", "gifski", "ffmpeg"):
        raise ValueError(f"Unknown GIF encoder '{choice}' (expected auto, gifski, or ffmpeg)")
    if choice == "auto":
        return "gifski" if shutil.which("gifski") else "ffmpeg"
    if choice == "gifski" and not shutil.which("gifski"):
        raise RuntimeError("gifski requested but not found on PATH (run `mise install`)")
    return choice


class AnimationWriter:
    """Stream RGB frames directly into ffmpeg (H.264 MP4) and a GIF encoder, without temp files.

    Usage:
        with AnimationWriter("out.mp4", 640, 360, fps=24) as writer:
            for frame in frames:
                writer.write(frame)
        writer.mp4_path, writer.gif_path
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: int = 24,
        make_gif: bool = False,
        gif_encoder: str | None = None,
        gif_quality: int = 90,
    ):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.mp4_path = out.with_suffix(".mp4")
        self.gif_path: Path | None = out.with_suffix(".gif") if make_gif else None
        self.gif_encoder = resolve_gif_encoder(gif_encoder) if make_gif else None
        self.frame_count = 0

        raw_input = [
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        ]
        self._sinks: list[subprocess.Popen] = []  # processes we write frames into
        self._procs: list[subprocess.Popen] = []  # every process to wait on

        mp4 = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error", *raw_input,
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dimensions
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.mp4_path),
            ],
            stdin=subprocess.PIPE,
        )
        self._sinks.append(mp4)
        self._procs.append(mp4)

        if self.gif_encoder == "gifski":
            # ffmpeg converts raw RGB -> full-range yuv444 y4m (lossless chroma) for gifski's stdin
            to_y4m = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-v", "error", *raw_input,
                    "-vf", "scale=out_color_matrix=bt709:out_range=full",
                    "-pix_fmt", "yuvj444p", "-strict", "-1", "-f", "yuv4mpegpipe", "-",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            gifski = subprocess.Popen(
                [
                    "gifski", "--quiet", "--fps", str(fps), "--quality", str(gif_quality),
                    "--y4m-color-override", "bt709", "-o", str(self.gif_path), "-",
                ],
                stdin=to_y4m.stdout,
            )
            to_y4m.stdout.close()  # gifski owns the read end now
            self._sinks.append(to_y4m)
            self._procs += [to_y4m, gifski]
        elif self.gif_encoder == "ffmpeg":
            gif = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-v", "error", *raw_input,
                    "-filter_complex",
                    "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    str(self.gif_path),
                ],
                stdin=subprocess.PIPE,
            )
            self._sinks.append(gif)
            self._procs.append(gif)

    def write(self, frame: Image.Image | np.ndarray) -> None:
        if isinstance(frame, Image.Image):
            if frame.size != self.size:
                raise ValueError(f"Frame size {frame.size} != writer size {self.size}")
            data = frame.convert("RGB").tobytes()
        else:
            data = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).tobytes()
        for proc in self._sinks:
            proc.stdin.write(data)
        self.frame_count += 1

    def close(self) -> tuple[Path, Path | None]:
        for proc in self._sinks:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        failed = [p.args[0] for p in self._procs if p.wait() != 0]
        if failed:
            raise RuntimeError(f"Encoding failed in: {', '.join(failed)}")
        return self.mp4_path, self.gif_path

    def __enter__(self) -> "AnimationWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            for proc in self._procs:
                proc.kill()
                proc.wait()
