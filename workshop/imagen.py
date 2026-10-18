"""OpenAI GPT Image (`gpt-image-2.5-sunburst-2026-09-08`) pipeline for scene extensions and transition bridges.

Provides:
- Automatic `.env` / `~/.env` loading for `OPENAI_API_KEY`
- Anchor preparation for left/right extensions and two-scene transition bridges
- OpenAI `/v1/images/edits` call with arbitrary divisible-by-16 `WIDTHxHEIGHT` resolutions
- Automatic registration (scale & offset alignment) between generated backgrounds and sprite sheets
"""
import argparse
import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from workshop.utils import load_sprite_sheet

DEFAULT_MODEL = "gpt-image-2.5-sunburst-2026-09-08"


def load_env_key(key_name: str = "OPENAI_API_KEY", root_dir: Optional[Path] = None) -> str:
    """Load `key_name` from environment, `<root>/.env`, or `~/.env`."""
    if os.environ.get(key_name):
        return os.environ[key_name]

    candidates = []
    if root_dir is not None:
        candidates.append(Path(root_dir) / ".env")
    candidates.extend([Path.cwd() / ".env", Path.home() / ".env"])

    for env_path in candidates:
        if env_path.is_file():
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            if os.environ.get(key_name):
                return os.environ[key_name]

    raise RuntimeError(
        f"Missing {key_name}. Please add `{key_name}=sk-...` to `.env` in the project root or `~/.env`."
    )


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_openai_image_edit(
    images: list[Image.Image],
    prompt: str,
    model: str = DEFAULT_MODEL,
    size: str = "1536x512",
    quality: str = "high",
    output_format: str = "png",
    background: Optional[str] = None,
    mask: Optional[Image.Image] = None,
    api_key: Optional[str] = None,
    timeout: int = 180,
) -> Image.Image:
    """Call OpenAI `POST /v1/images/edits` and return the decoded PIL Image."""
    key = api_key or load_env_key("OPENAI_API_KEY")

    payload: dict = {
        "model": model,
        "images": [{"image_url": _image_to_data_url(im)} for im in images],
        "prompt": prompt,
        "quality": quality,
        "size": size,
        "output_format": output_format,
    }
    if background is not None:
        payload["background"] = background
    if mask is not None:
        payload["mask"] = {"image_url": _image_to_data_url(mask)}

    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI /v1/images/edits failed (HTTP {e.code}): {err_msg}") from e

    b64_data = body["data"][0]["b64_json"]
    raw_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGBA")


def call_gemini_omni_video(
    prompt: str,
    output_path: Union[str, Path],
    images: Optional[list[Union[str, Path, Image.Image]]] = None,
    model: str = "gemini-omni-1.1-flash",
    task: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 360,
) -> Path:
    """Generate an MP4 video via Google Gemini Interactions API (`gemini-omni-1.1-flash`)."""
    import time

    key = api_key or load_env_key("GOOGLE_GENERATIVE_AI_API_KEY")
    inputs: list[dict] = []
    for im in images or []:
        pil_im = im.convert("RGB") if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
        buf = io.BytesIO()
        pil_im.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
        inputs.append({"type": "image", "data": b64_str, "mime_type": "image/png"})
    inputs.append({"type": "text", "text": prompt})

    payload: dict = {
        "model": model,
        "input": inputs,
    }
    if task:
        payload["generation_config"] = {"video_config": {"task": task}}

    url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini Interactions API failed (HTTP {e.code}): {err_msg}") from e

    interaction_id = body.get("id")
    deadline = time.time() + timeout
    while body.get("status") not in ("completed", "failed", "cancelled") and interaction_id and time.time() < deadline:
        time.sleep(4)
        poll_url = f"https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}?key={key}"
        with urllib.request.urlopen(poll_url, timeout=60) as r:
            body = json.loads(r.read().decode("utf-8"))

    if body.get("status") != "completed":
        raise RuntimeError(f"Gemini Omni interaction did not complete: {body}")

    video_b64 = None
    for step in body.get("steps", []):
        if step.get("type") == "model_output":
            for part in step.get("content", []):
                if part.get("type") == "video" and part.get("data"):
                    video_b64 = part["data"]
                    break

    if not video_b64:
        raise RuntimeError(f"No video data found in Gemini Omni interaction response: {body}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(video_b64))
    print(f"Saved Gemini Omni video: {out} ({out.stat().st_size // 1024} KB)")
    return out


def build_transition_anchor(
    from_sheet: Union[str, Path, Image.Image],
    to_sheet: Union[str, Path, Image.Image],
    canvas_w: int = 1080,
    canvas_h: int = 360,
    from_scale: float = 0.6806,
    to_scale: float = 0.77,
    left_anchor_w: int = 180,
    include_full_right_scene: bool = True,
    right_anchor_w: int = 180,
) -> tuple[Image.Image, int]:
    """Create a panoramic anchor image combining the right edge of `from_sheet` and `to_sheet`."""
    if isinstance(from_sheet, Image.Image):
        f0 = from_sheet.convert("RGBA")
    else:
        frames_a, _ = load_sprite_sheet(from_sheet, allow_single_image=True)
        f0 = frames_a[0]
        if from_scale != 1.0:
            f0 = f0.resize((round(f0.width * from_scale), round(f0.height * from_scale)), Image.Resampling.LANCZOS)

    if isinstance(to_sheet, Image.Image):
        f1 = to_sheet.convert("RGBA")
    else:
        frames_b, _ = load_sprite_sheet(to_sheet, allow_single_image=True)
        f1 = frames_b[0]
        if to_scale != 1.0:
            f1 = f1.resize((round(f1.width * to_scale), round(f1.height * to_scale)), Image.Resampling.LANCZOS)

    anchor = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # Left anchor: right strip of from_sheet
    lw = min(left_anchor_w, f0.width)
    left_crop = f0.crop((f0.width - lw, 0, f0.width, min(canvas_h, f0.height)))
    anchor.paste(left_crop, (0, 0))

    # Right anchor: full `to_sheet` (so bottom floor is also outpainted) or left strip
    if include_full_right_scene:
        rw = f1.width
        right_crop = f1
    else:
        rw = min(right_anchor_w, f1.width)
        right_crop = f1.crop((0, 0, rw, min(canvas_h, f1.height)))

    right_x = canvas_w - rw
    anchor.paste(right_crop, (right_x, 0))
    return anchor, right_x


def match_sprite_to_background(
    bg_img: Image.Image,
    sprite_frame: Image.Image,
    target_bg_h: int = 360,
    search_side: str = "right",
    expected_scale: Optional[float] = None,
) -> tuple[float, int, int]:
    """Find `(video_scale, offset_x, video_y)` that aligns `sprite_frame` onto `bg_img` (scaled to `target_bg_h`)."""
    raw_w, raw_h = sprite_frame.size
    target_bg_w = round(bg_img.width * (target_bg_h / bg_img.height))
    bg_resized = np.array(bg_img.resize((target_bg_w, target_bg_h), Image.Resampling.LANCZOS).convert("RGB"))
    vid_rgb = np.array(sprite_frame.convert("RGB"))

    # Use the left 25% of sprite_frame when matching on the right of bg_img
    tpl_src = vid_rgb[:, : max(32, raw_w // 4), :] if search_side == "right" else vid_rgb[:, -max(32, raw_w // 4) :, :]

    if expected_scale is not None:
        scales = np.linspace(max(0.3, expected_scale - 0.12), min(1.2, expected_scale + 0.12), 49)
    else:
        scales = np.linspace(0.50, 1.00, 51)

    best_score = -1.0
    best_result = (expected_scale or 0.75, target_bg_w - round(raw_w * 0.25), 0)

    for s in scales:
        th = round(tpl_src.shape[0] * s)
        tw = round(tpl_src.shape[1] * s)
        if th > target_bg_h or tw > target_bg_w or th < 16 or tw < 16:
            continue
        tpl = cv2.resize(tpl_src, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(bg_resized, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            ox, oy = max_loc
            if search_side != "right":
                ox = ox - (round(raw_w * s) - tw)
            best_result = (round(float(s), 4), int(ox), int(oy))

    return best_result


def generate_scene_bridge(
    from_sheet: Path,
    to_sheet: Path,
    output_path: Path,
    from_scale: float = 0.6806,
    to_scale: float = 0.75,
    canvas_w: int = 1080,
    canvas_h: int = 360,
    left_anchor_w: int = 180,
    model: str = DEFAULT_MODEL,
    size: str = "1536x512",
    quality: str = "high",
    prompt: Optional[str] = None,
    save_anchor_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Generate (or reuse if cached and `overwrite=False`) a transition bridge image between two scenes."""
    out = Path(output_path)
    if out.exists() and not overwrite:
        print(f"Using existing bridge image: {out}")
        return out

    anchor_img, right_x = build_transition_anchor(
        from_sheet=from_sheet,
        to_sheet=to_sheet,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        from_scale=from_scale,
        to_scale=to_scale,
        left_anchor_w=left_anchor_w,
        include_full_right_scene=True,
    )

    if save_anchor_path:
        Path(save_anchor_path).parent.mkdir(parents=True, exist_ok=True)
        anchor_img.save(save_anchor_path)

    default_prompt = (
        "Seamlessly fill the white empty middle and bottom-right areas between the left scene and right scene "
        "to create a single continuous horizontal panoramic illustration in the exact same warm anime / webtoon "
        "background art style, clean architectural linework, and coherent ambient lighting. "
        "CRITICAL PRESERVATION REQUIREMENTS: Keep the left scene and right scene anchors completely intact, "
        "unaltered, and unchanged in their exact pixel positions and scale. Do NOT add any extra people, "
        "characters, or pedestrians in the transition walkway. "
        "CRITICAL ARCHITECTURAL DIVIDER: In the middle transition area, construct a natural architectural dividing wall, "
        "entrance facade, or partition (such as a modern glass sliding door, stone entrance wall, or breezeway) "
        "that cleanly and logically separates the two environments to prevent awkward visual jumps between rooms or day/night. "
        "GROUND WALKWAY: Continue the level ground walkway completely flat and smooth across the entire bottom edge, "
        "transitioning seamlessly between the floor materials and extending under any furniture or characters."
    )

    print(f"Generating transition bridge via OpenAI [{model}] ({size}, quality={quality})...")
    generated = call_openai_image_edit(
        images=[anchor_img],
        prompt=prompt or default_prompt,
        model=model,
        size=size,
        quality=quality,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    generated.save(out)
    print(f"Saved bridge image: {out} ({generated.width}x{generated.height}, right_anchor_offset={right_x})")
    return out


def append_scene_to_timeline(
    video_path: Path,
    timeline_path: Path = Path("timelines/timeline.toml"),
    scene_id: Optional[str] = None,
    video_scale: float = 0.75,
    bridge_width: int = 1080,
    overlap_prev: int = 180,
    model: str = DEFAULT_MODEL,
    size: str = "1536x512",
    quality: str = "high",
    prompt: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """End-to-end helper: convert a new MP4 into a sprite sheet, generate the OpenAI bridge from the
    last scene in `timeline_path`, and append the new `[[scenes]]` entry to `timeline_path`."""
    import tomllib
    from workshop.sprite import generate_sprite_sheet

    video = Path(video_path)
    sid = scene_id or video.stem.removeprefix("couple-")
    sprite_out = Path("outputs") / video.stem / "sprite_bg.png"

    # 1. Generate sprite sheet if needed
    if not sprite_out.exists() or overwrite:
        generate_sprite_sheet(
            video_path=video,
            output_path=sprite_out,
            keep_background=True,
        )

    # 2. Read existing timeline to find the preceding scene
    t_path = Path(timeline_path)
    data = tomllib.loads(t_path.read_text())
    scenes = data.get("scenes", [])
    if not scenes:
        raise ValueError(f"No existing [[scenes]] found in {t_path}")

    prev = scenes[-1]
    prev_id = prev.get("id", Path(prev["sheet"]).stem)
    prev_sheet = Path(prev["sheet"])
    prev_scale = float(prev.get("video_scale", prev.get("scale", 1.0)))

    # If the previous scene already has a composed left_bg, use its composed frame as the left anchor
    if prev.get("left_bg"):
        from workshop.scenes import compose_extended_scene
        prev_raw_frames, _ = load_sprite_sheet(prev_sheet, allow_single_image=True)
        prev_left_bg = Image.open(prev["left_bg"])
        composed_prev = compose_extended_scene(
            prev_raw_frames[:1],
            left_bg=prev_left_bg,
            offset=int(prev.get("left_bg_offset", 0)),
            video_scale=prev_scale,
            left_bg_scale=float(prev.get("left_bg_scale", 1.0)),
            video_y=prev.get("video_y", 0),
            seamless=True,
            left_edge_feather=prev.get("left_feather"),
        )
        prev_f0_path = Path("outputs/inpainting_prep") / f"{prev_id}_composed_f0.png"
        prev_f0_path.parent.mkdir(parents=True, exist_ok=True)
        composed_prev[0].save(prev_f0_path)
        source_from_sheet = prev_f0_path
        source_from_scale = 1.0
    else:
        source_from_sheet = prev_sheet
        source_from_scale = prev_scale

    # 3. Generate transition bridge from source_from_sheet -> sprite_out
    bridge_out = Path("inputs/bridges") / f"{prev_id}-to-{sid}-bridge.png"
    anchor_out = Path("outputs/inpainting_prep") / f"bridge_{prev_id}_to_{sid}_anchor.png"

    frames_b, _ = load_sprite_sheet(sprite_out, allow_single_image=True)
    vid_w = round(frames_b[0].width * video_scale)
    left_bg_offset = bridge_width - vid_w

    generate_scene_bridge(
        from_sheet=source_from_sheet,
        to_sheet=sprite_out,
        output_path=bridge_out,
        from_scale=source_from_scale,
        to_scale=video_scale,
        canvas_w=bridge_width,
        canvas_h=int(data.get("camera", {}).get("height", 360)),
        left_anchor_w=overlap_prev,
        model=model,
        size=size,
        quality=quality,
        prompt=prompt,
        save_anchor_path=anchor_out,
        overwrite=overwrite,
    )

    # 4. Append new scene block to timeline TOML if not already present
    existing_ids = {s.get("id") for s in scenes}
    if sid not in existing_ids:
        feather_start = round(overlap_prev * 0.44)
        feather_w = round(overlap_prev * 0.34)
        stride_px = float(data.get("character", {}).get("stride_pixels", 165.8))
        cam_speed = stride_px / 3.75 if stride_px else 44.2
        anim_delay = round(left_bg_offset / cam_speed, 1)

        raw_toml = t_path.read_text()
        # If preceding scenes ended with a gap hold (e.g. gap_after = 3.0), reset to 0.0
        import re
        raw_toml = re.sub(r"gap_after\s*=\s*[1-9]\d*(\.\d+)?", "gap_after = 0.0", raw_toml)

        block = (
            f"\n[[scenes]]\n"
            f'id = "{sid}"\n'
            f'sheet = "{sprite_out.as_posix()}"\n'
            f'left_bg = "{bridge_out.as_posix()}"\n'
            f"left_bg_offset = {left_bg_offset}\n"
            f"left_bg_scale = 1.00\n"
            f"video_scale = {video_scale}\n"
            f"video_y = 0\n"
            f"overlap_prev = {overlap_prev}\n"
            f"left_feather = [{feather_start}, {feather_w}]\n"
            f"anim_delay = {anim_delay}\n"
            f"pingpong = true\n"
            f"y = 0\n"
            f"gap_after = 3.0\n"
        )
        t_path.write_text(raw_toml.rstrip() + "\n" + block)
        print(f"Appended scene [{sid}] to {t_path}")

    return bridge_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-sheet", type=Path, help="Left/preceding scene sprite sheet or image")
    ap.add_argument("--to-sheet", type=Path, help="Right/following scene sprite sheet or image")
    ap.add_argument("-o", "--output", type=Path, help="Output bridge image path")
    ap.add_argument("--add-video", type=Path, help="One-command mode: convert MP4 to sprite, generate bridge, and append to timeline")
    ap.add_argument("--timeline", type=Path, default=Path("timelines/timeline.toml"), help="Timeline TOML to append to (with --add-video)")
    ap.add_argument("--scene-id", type=str, help="Scene ID when using --add-video")
    ap.add_argument("--from-scale", type=float, default=0.6806, help="Scale of left scene")
    ap.add_argument("--to-scale", type=float, default=0.75, help="Scale of right/new scene")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI image model")
    ap.add_argument("--size", type=str, default="1536x512", help="Output size (WIDTHxHEIGHT divisible by 16)")
    ap.add_argument("--quality", type=str, default="high", help="Output quality (low, medium, high, xhigh, max)")
    ap.add_argument("--prompt", type=str, help="Custom prompt for transition generation")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate even if output file exists")
    args = ap.parse_args()

    if args.add_video:
        append_scene_to_timeline(
            video_path=args.add_video,
            timeline_path=args.timeline,
            scene_id=args.scene_id,
            video_scale=args.to_scale,
            model=args.model,
            size=args.size,
            quality=args.quality,
            prompt=args.prompt,
            overwrite=args.overwrite,
        )
        return

    if not (args.from_sheet and args.to_sheet and args.output):
        ap.error("Either --add-video OR (--from-sheet, --to-sheet, and -o/--output) are required.")

    generate_scene_bridge(
        from_sheet=args.from_sheet,
        to_sheet=args.to_sheet,
        output_path=args.output,
        from_scale=args.from_scale,
        to_scale=args.to_scale,
        model=args.model,
        size=args.size,
        quality=args.quality,
        prompt=args.prompt,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
