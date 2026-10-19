"""Composite two sprite sheets (scene + character) into an animated MP4 and GIF.

Modes:
1. Standard: Background scene is stationary; foreground character walks across.
2. Tracking: Camera is locked to the character (character stays centered); background scene scrolls past right-to-left.

Usage:
    # Tracking shot (camera locked on character):
    uv run python -m workshop.composite \\
        outputs/scene/sprite.png \\
        outputs/character/sprite.png \\
        --tracking --scale 0.45 --bg-color "#ffffff" \\
        -o outputs/composite/animation_tracking.mp4
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from workshop.utils import (
    GRADIENT_PRESETS,
    AnimationWriter,
    create_gradient_image,
    create_scene_radial_backdrop,
    load_sprite_sheet,
    parse_hex_color,
)


def resolve_background(
    width: int,
    height: int,
    bg_color: str = "#ffffff",
    bg_gradient: str | list[str] | None = None,
    direction: str = "vertical",
) -> tuple[Image.Image, tuple[int, int, int], str]:
    """Resolve solid color or gradient background canvas, fallback RGB color, and description."""
    colors = None
    if bg_gradient:
        if isinstance(bg_gradient, str):
            if bg_gradient.lower() in GRADIENT_PRESETS:
                colors = GRADIENT_PRESETS[bg_gradient.lower()]
            elif "," in bg_gradient:
                colors = [c.strip() for c in bg_gradient.split(",")]
            elif "->" in bg_gradient:
                colors = [c.strip() for c in bg_gradient.split("->")]
            elif " " in bg_gradient:
                colors = [c.strip() for c in bg_gradient.split()]
            else:
                colors = [bg_gradient]
        elif isinstance(bg_gradient, list):
            if len(bg_gradient) == 1 and bg_gradient[0].lower() in GRADIENT_PRESETS:
                colors = GRADIENT_PRESETS[bg_gradient[0].lower()]
            elif len(bg_gradient) == 1 and ("," in bg_gradient[0] or "->" in bg_gradient[0]):
                delim = "," if "," in bg_gradient[0] else "->"
                colors = [c.strip() for c in bg_gradient[0].split(delim)]
            else:
                colors = bg_gradient
    elif bg_color and ("," in bg_color or "->" in bg_color or bg_color.lower() in GRADIENT_PRESETS):
        if bg_color.lower() in GRADIENT_PRESETS:
            colors = GRADIENT_PRESETS[bg_color.lower()]
        elif "," in bg_color:
            colors = [c.strip() for c in bg_color.split(",")]
        else:
            colors = [c.strip() for c in bg_color.split("->")]

    if colors and len(colors) > 1:
        canvas = create_gradient_image(width, height, colors, direction=direction)
        fallback_rgb = parse_hex_color(colors[0])[:3]
        desc = f"gradient [{direction}] ({' -> '.join(colors)})"
        return canvas, fallback_rgb, desc

    color_str = colors[0] if colors else bg_color
    rgba = parse_hex_color(color_str)
    return Image.new("RGBA", (width, height), rgba), rgba[:3], f"solid ({color_str})"


def render_composite(
    bg_sheets: Path | list[Path],
    fg_sheet: Path,
    output: Path,
    mode: str = "tracking",
    scale: float = 0.45,
    scene_scale: float | list[float] = 0.70,
    scene_y: int | list[int] = 15,
    scene_gap: float = 2.0,
    floor_y: int = 352,
    fps: int = 24,
    bg_color: str = "#ffffff",
    bg_gradient: str | list[str] | None = None,
    bg_gradient_direction: str = "vertical",
    scene_gradient: str | bool | None = None,
    scene_gradient_scale: float = 2.0,
    char_frames: str | None = "1-15",
    start_x: float | None = None,
    end_x: float | None = None,
    cat_x: float | None = None,
    cat_y: float | None = None,
    flip_fg: bool = False,
    gif: bool = False,
    gif_encoder: str | None = None,
) -> tuple[Path, Path | None]:
    if isinstance(bg_sheets, (str, Path)):
        bg_sheets = [Path(bg_sheets)]

    if isinstance(scene_scale, (int, float)):
        scene_scales = [float(scene_scale)] * len(bg_sheets)
    else:
        scene_scales = [float(s) for s in scene_scale]
        while len(scene_scales) < len(bg_sheets):
            scene_scales.append(scene_scales[-1])

    if isinstance(scene_y, int):
        scene_ys = [int(scene_y)] * len(bg_sheets)
    else:
        scene_ys = [int(y) for y in scene_y]
        while len(scene_ys) < len(bg_sheets):
            scene_ys.append(scene_ys[-1])

    # Load foreground character
    fg_frames, fg_meta = load_sprite_sheet(fg_sheet)
    if char_frames and char_frames.lower() != "all":
        if "-" in char_frames:
            c_first, c_last = map(int, char_frames.split("-"))
            c_first = max(1, c_first)
            c_last = min(len(fg_frames), c_last)
            fg_frames = fg_frames[c_first - 1:c_last]
        else:
            c_count = int(char_frames)
            fg_frames = fg_frames[:c_count]

    fg_w, fg_h = fg_frames[0].size
    cat_w = round(fg_w * scale)
    cat_h = round(fg_h * scale)

    scaled_fg = []
    for f in fg_frames:
        resized = f.resize((cat_w, cat_h), Image.Resampling.LANCZOS)
        if flip_fg:
            resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        scaled_fg.append(resized)

    fg_interval = fg_meta.get("intervalSeconds", 0.25)
    cat_pos_y = round(cat_y) if cat_y is not None else (floor_y - cat_h)
    is_tracking = mode in ("tracking", "track")

    # Load and scale scene background(s)
    scenes = []
    canvas_w, canvas_h = 640, 360
    for idx, sheet_path in enumerate(bg_sheets):
        frames, meta = load_sprite_sheet(sheet_path)
        if idx == 0:
            canvas_w, canvas_h = frames[0].size
        cur_scale = scene_scales[idx]
        cur_y = scene_ys[idx]
        sw = round(canvas_w * cur_scale)
        sh = round(canvas_h * cur_scale)
        scaled_frames = [
            f.resize((sw, sh), Image.Resampling.LANCZOS) if cur_scale != 1.0 else f
            for f in frames
        ]
        interval = meta.get("intervalSeconds", 0.25)
        duration = len(frames) * interval
        scene_dict = {
            "frames": scaled_frames,
            "width": sw,
            "height": sh,
            "y": cur_y,
            "scale": cur_scale,
            "interval": interval,
            "duration": duration,
        }
        has_alpha = "A" in frames[0].getbands() and (np.array(frames[0].getchannel("A")) < 250).any()
        if scene_gradient and has_alpha:
            grad_color = "#203a55" if (scene_gradient is True or scene_gradient == "navy") else scene_gradient
            backdrop, pad_x, pad_y = create_scene_radial_backdrop(
                sw, sh, inner_color=grad_color, radius_scale=scene_gradient_scale
            )
            scene_dict["backdrop"] = backdrop
            scene_dict["pad_x"] = pad_x
            scene_dict["pad_y"] = pad_y
        scenes.append(scene_dict)

    bg_canvas, bg_fallback_rgb, bg_desc = resolve_background(
        canvas_w,
        canvas_h,
        bg_color=bg_color,
        bg_gradient=bg_gradient,
        direction=bg_gradient_direction,
    )

    if is_tracking:
        cat_pos_x = (canvas_w - cat_w) // 2 if cat_x is None else round(cat_x)
        # Compute multi-scene timeline
        # Speed determined by travel distance (canvas_w + sw) / duration
        first_sw = scenes[0]["width"]
        first_duration = scenes[0]["duration"]
        travel_dist = canvas_w + first_sw
        speed = travel_dist / first_duration if first_duration > 0 else 100.0

        current_start_t = 0.0
        for i, s in enumerate(scenes):
            s["start_t"] = current_start_t
            s["travel_dist"] = canvas_w + s["width"]
            s["active_duration"] = s["travel_dist"] / speed
            # world position when scene starts entering from right (x = canvas_w)
            s["world_x_start"] = speed * current_start_t + canvas_w
            current_start_t += s["active_duration"] + (scene_gap if i < len(scenes) - 1 else 0.0)

        total_duration = current_start_t
        total_render_frames = round(total_duration * fps)

        desc_extra = f" + scene-radial ({scene_gradient})" if scene_gradient else ""
        print(f"Rendering [TRACKING MODE] {len(scenes)} scene(s), {total_render_frames} frames ({total_duration:.1f}s @ {fps} fps)...")
        print(f"  Character: fixed at ({cat_pos_x}, {cat_pos_y}), scale={scale:.2f}")
        print(f"  Scene speed: {speed:.1f}px/s on {bg_desc}{desc_extra}")
        for idx, s in enumerate(scenes):
            print(f"    Scene {idx + 1}: t=[{s['start_t']:.1f}s - {s['start_t'] + s['active_duration']:.1f}s], size=({s['width']}x{s['height']}), y={s['y']}, scale={s['scale']:.2f}")
    else:
        # Standard mode: first scene is stationary, character walks across
        scene = scenes[0]
        sw = scene["width"]
        total_duration = scene["duration"]
        total_render_frames = round(total_duration * fps)
        char_start_x = (-cat_w * 0.7) if start_x is None else start_x
        char_end_x = (canvas_w + cat_w * 0.1) if end_x is None else end_x
        scene_x = (canvas_w - sw) // 2
        desc_extra = f" + scene-radial ({scene_gradient})" if scene_gradient else ""
        print(f"Rendering [STANDARD MODE] {total_render_frames} frames ({total_duration:.1f}s @ {fps} fps)...")
        print(f"  Character walk: x={char_start_x:.1f} -> x={char_end_x:.1f}, y={cat_pos_y} on {bg_desc}{desc_extra}")

    with AnimationWriter(output, canvas_w, canvas_h, fps=fps, make_gif=gif, gif_encoder=gif_encoder) as writer:
        for i in range(total_render_frames):
            t = i / fps
            fg_idx = int(t / fg_interval) % len(scaled_fg)
            fg_frame = scaled_fg[fg_idx]

            canvas = bg_canvas.copy()

            if is_tracking:
                # If single scene with explicit start_x or end_x overrides
                if len(scenes) == 1 and (start_x is not None or end_x is not None):
                    s_start = canvas_w if start_x is None else start_x
                    s_end = -scenes[0]["width"] if end_x is None else end_x
                    progress = i / max(1, total_render_frames - 1)
                    sx = round(s_start + (s_end - s_start) * progress)
                    s_idx = int(t / scenes[0]["interval"]) % len(scenes[0]["frames"])
                    sy = scenes[0].get("y", 15)
                    if "backdrop" in scenes[0]:
                        canvas.alpha_composite(scenes[0]["backdrop"], (sx - scenes[0]["pad_x"], sy - scenes[0]["pad_y"]))
                    canvas.alpha_composite(scenes[0]["frames"][s_idx], (sx, sy))
                else:
                    # Draw any scene currently visible in the camera viewport
                    for s in scenes:
                        sx = round(s["world_x_start"] - speed * t)
                        if sx + s["width"] > 0 and sx < canvas_w:
                            # Frame index loops during scene active time
                            time_in_scene = max(0.0, t - s["start_t"])
                            s_idx = int(time_in_scene / s["interval"]) % len(s["frames"])
                            sy = s.get("y", 15)
                            if "backdrop" in s:
                                canvas.alpha_composite(s["backdrop"], (sx - s["pad_x"], sy - s["pad_y"]))
                            canvas.alpha_composite(s["frames"][s_idx], (sx, sy))

                canvas.alpha_composite(fg_frame, (cat_pos_x, cat_pos_y))
            else:
                progress = i / max(1, total_render_frames - 1)
                char_x = round(char_start_x + (char_end_x - char_start_x) * progress)
                s_idx = int(t / scenes[0]["interval"]) % len(scenes[0]["frames"])
                sy = scene.get("y", 15)
                if "backdrop" in scene:
                    canvas.alpha_composite(scene["backdrop"], (scene_x - scene["pad_x"], sy - scene["pad_y"]))
                canvas.alpha_composite(scenes[0]["frames"][s_idx], (scene_x, sy))
                canvas.alpha_composite(fg_frame, (char_x, cat_pos_y))

            frame_rgb = Image.new("RGB", (canvas_w, canvas_h), bg_fallback_rgb)
            frame_rgb.paste(canvas, mask=canvas.getchannel("A") if "A" in canvas.getbands() else None)
            writer.write(frame_rgb)

    print(f"Saved MP4: {writer.mp4_path}")
    if writer.gif_path:
        print(f"Saved GIF: {writer.gif_path} (encoder: {writer.gif_encoder})")
    return writer.mp4_path, writer.gif_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("bg_sheet", type=Path, nargs="?", help="Scene / background sprite sheet")
    ap.add_argument("fg_sheet", type=Path, nargs="?", help="Character / foreground sprite sheet")
    ap.add_argument("--scenes", type=Path, nargs="+", help="One or more scene sprite sheets for multi-scene sequence")
    ap.add_argument("--char", "--fg", dest="fg_arg", type=Path, help="Character sprite sheet")
    ap.add_argument("-o", "--output", type=Path, default=Path("outputs/composite/animation.mp4"), help="Output video path")
    ap.add_argument("--mode", choices=["tracking", "standard"], default="tracking", help="Animation mode: 'tracking' or 'standard' (default: tracking)")
    ap.add_argument("--track-cat", "--tracking", dest="track_alias", action="store_true", help="Alias for --mode tracking")
    ap.add_argument("--scale", "--char-scale", dest="scale", type=float, default=0.45, help="Character scale factor (default: 0.45)")
    ap.add_argument("--scene-scale", "--scene-scales", dest="scene_scale", type=float, nargs="+", default=[0.70], help="Scene scale factor(s). Can provide per-scene values, e.g. 0.70 1.0 (default: 0.70)")
    ap.add_argument("--scene-y", "--scene-ys", dest="scene_y", type=int, nargs="+", default=[15], help="Scene Y position offset(s). Can provide per-scene values, e.g. 15 0 (default: 15)")
    ap.add_argument("--scene-gap", type=float, default=2.0, help="Gap in seconds between scenes in multi-scene tracking mode (default: 2.0)")
    ap.add_argument("--floor-y", type=int, default=352, help="Floor baseline Y coordinate (default: 352)")
    ap.add_argument("--fps", type=int, default=24, help="Framerate (default: 24)")
    ap.add_argument("--bg-color", type=str, default="#ffffff", help="Background hex color (default: #ffffff)")
    ap.add_argument("--bg-gradient", nargs="+", help="Canvas gradation background colors (e.g. '#142230' '#355a7d' or preset: scene, twilight, sunset, studio, warm)")
    ap.add_argument("--bg-gradient-direction", choices=["vertical", "horizontal", "radial", "diagonal"], default="vertical", help="Gradation direction (default: vertical)")
    ap.add_argument("--scene-gradient", nargs="?", const="#203a55", default=None, help="Add radial fade gradient behind scene (navy at center, fades out further from center; default center: #203a55)")
    ap.add_argument("--scene-gradient-scale", type=float, default=2.0, help="Scale multiplier for scene radial gradient (default: 2.0, ~2x bigger)")
    ap.add_argument("--char-frames", "--fg-frames", default="1-15", help="Frame range for character walking loop, e.g. '1-15' or 'all' (default: 1-15)")
    ap.add_argument("--start-x", type=float, help="Start X coordinate of moving layer")
    ap.add_argument("--end-x", type=float, help="End X coordinate of moving layer")
    ap.add_argument("--cat-x", type=float, help="Fixed X position of character in tracking mode")
    ap.add_argument("--cat-y", type=float, help="Fixed Y position of character in tracking mode")
    ap.add_argument("--flip", action="store_true", help="Flip character horizontally")
    ap.add_argument("--gif", action="store_true", help="Also generate GIF version (default: MP4 only)")
    ap.add_argument("--gif-encoder", choices=["auto", "gifski", "ffmpeg"], help="GIF encoder (default: gifski if installed, else ffmpeg)")
    args = ap.parse_args()

    mode = "tracking" if args.track_alias else args.mode

    if args.scenes:
        scenes = args.scenes
        fg = args.fg_arg or args.fg_sheet or args.bg_sheet
    else:
        if not args.bg_sheet or not args.fg_sheet:
            ap.error("Both bg_sheet and fg_sheet are required unless --scenes is used.")
        scenes = [args.bg_sheet]
        fg = args.fg_arg or args.fg_sheet

    render_composite(
        bg_sheets=scenes,
        fg_sheet=fg,
        output=args.output,
        mode=mode,
        scale=args.scale,
        scene_scale=args.scene_scale,
        scene_y=args.scene_y,
        scene_gap=args.scene_gap,
        floor_y=args.floor_y,
        fps=args.fps,
        bg_color=args.bg_color,
        bg_gradient=args.bg_gradient,
        bg_gradient_direction=args.bg_gradient_direction,
        scene_gradient=args.scene_gradient,
        scene_gradient_scale=args.scene_gradient_scale,
        char_frames=args.char_frames,
        start_x=args.start_x,
        end_x=args.end_x,
        cat_x=args.cat_x,
        cat_y=args.cat_y,
        flip_fg=args.flip,
        gif=args.gif,
        gif_encoder=args.gif_encoder,
    )


if __name__ == "__main__":
    main()

