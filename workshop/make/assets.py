"""Asset generation for `make`: rigid actor sprites, scene backgrounds, bridges, and the stitched world.

Geometry is fixed so every generated image lines up without per-scene tuning. With canvas height H:
- Every scene background is a 3H x H panorama whose road edge sits at `ROAD_TOP` of the height.
- Every bridge is a 3H x H image whose left H pixels repeat the end of the previous scene and whose
  right H pixels repeat the start of the next scene; the middle H pixels are generated.
- The world strip places scene i at x = i * 4H and bridge i at x = i * 4H - 2H, cross-fading each
  H-wide overlap so small drifts in the generated anchors don't show as seams.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from workshop.make.plan import ActorKind
from workshop.utils import save_sprite_sheet

# Generation size for 3:1 panoramas (scene backgrounds and bridges)
PANORAMA_SIZE = "1536x512"
ACTOR_SIZE = "1024x1024"

ROAD_TOP = 0.82  # road edge as a fraction of the height, pinned in every scene prompt
WHEELED_FLOOR = 0.92  # where wheels touch the road
FLOATING_FLOOR = 0.62  # bottom of a floating actor, above the skyline base

ACTOR_SOURCE_HEIGHT = 256  # actor frames are normalized to this height before animation
FRAME_INTERVAL = 1 / 24


class ImageBackend(Protocol):
    def generate(self, prompt: str, size: str, background: str | None = None) -> Image.Image: ...

    def edit(self, image: Image.Image, prompt: str, size: str) -> Image.Image: ...


class OpenAIImageBackend:
    """Default backend: OpenAI image generation and edits."""

    def generate(self, prompt: str, size: str, background: str | None = None) -> Image.Image:
        from workshop.imagen import call_openai_image_generate

        return call_openai_image_generate(prompt, size=size, background=background)

    def edit(self, image: Image.Image, prompt: str, size: str) -> Image.Image:
        from workshop.imagen import call_openai_image_edit

        return call_openai_image_edit(images=[image], prompt=prompt, size=size)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def actor_prompt(style: str, visual: str) -> str:
    return (
        f"{style}. {visual}. A single subject seen in an exact side profile, facing right, "
        "the whole subject fully visible and centered, isolated on a transparent background. "
        "No ground, no shadow, no scenery, no text."
    )


def scene_prompt(style: str, visual: str, ground: str) -> str:
    road_pct = round(ROAD_TOP * 100)
    return (
        f"{style}. A wide side-scrolling 2D game background panorama: {visual}. "
        "Strict side view at street level with no perspective vanishing point. "
        f"A flat {ground.lower().replace('_', ' ')} road runs horizontally across the entire width; "
        f"its top edge is a perfectly straight horizontal line at {road_pct}% of the image height "
        f"and the road fills the bottom {100 - road_pct}% of the image. "
        "Buildings and landmarks stand behind the road. "
        "No vehicles, no people, no animals, no text, no border, no watermark."
    )


def bridge_prompt(style: str, from_name: str, to_name: str) -> str:
    road_pct = round(ROAD_TOP * 100)
    return (
        f"Fill the white middle area so the image becomes one continuous side-scrolling panorama "
        f"in the same {style}, travelling from {from_name} on the left to {to_name} on the right. "
        "Keep the left and right parts exactly as they are. "
        f"The road continues perfectly straight and level at {road_pct}% of the image height across "
        "the whole width. Use a natural in-between landscape such as a highway, countryside, bridge, "
        "or coastline. No vehicles, no people, no text."
    )


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

def ensure_transparent(img: Image.Image) -> Image.Image:
    """Return `img` with a real alpha channel, matting it if the backend returned an opaque image."""
    img = img.convert("RGBA")
    alpha = np.asarray(img.getchannel("A"))
    if alpha.min() < 250:
        return img
    from workshop.bgremove import BackgroundRemover

    cutout, _ = BackgroundRemover().remove(img.convert("RGB"))
    return cutout.convert("RGBA")


@dataclass
class ActorMotion:
    frames: int
    bob_px: float  # peak vertical lift, as a fraction of the actor height
    tilt_deg: float
    bumps_per_loop: int


MOTIONS = {
    ActorKind.WHEELED: ActorMotion(frames=12, bob_px=0.012, tilt_deg=0.4, bumps_per_loop=2),
    ActorKind.FLOATING: ActorMotion(frames=48, bob_px=0.05, tilt_deg=1.5, bumps_per_loop=1),
}


def build_rigid_actor_sheet(source: Image.Image, kind: ActorKind, output_path: Path) -> Path:
    """Animate a single actor image into a looping sprite sheet (bounce for wheels, bob for floaters)."""
    if kind not in MOTIONS:
        raise ValueError(f"Rigid animation supports {list(MOTIONS)}, got {kind}")
    motion = MOTIONS[kind]

    img = ensure_transparent(source)
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox is None:
        raise ValueError("Actor image is fully transparent")
    img = img.crop(bbox)
    scale = ACTOR_SOURCE_HEIGHT / img.height
    img = img.resize((max(1, round(img.width * scale)), ACTOR_SOURCE_HEIGHT), Image.Resampling.LANCZOS)

    lift_max = math.ceil(img.height * motion.bob_px)
    margin = math.ceil(img.width * math.sin(math.radians(motion.tilt_deg))) + 2
    fw, fh = img.width + 2 * margin, img.height + lift_max + margin

    frames: list[Image.Image] = []
    for i in range(motion.frames):
        phase = i / motion.frames
        if kind == ActorKind.WHEELED:
            # Road bumps: short lifts, resting on the road between them
            lift = abs(math.sin(math.pi * motion.bumps_per_loop * phase)) * lift_max
            tilt = math.sin(2 * math.pi * motion.bumps_per_loop * phase) * motion.tilt_deg
        else:
            # Smooth float: gentle sine bob and sway
            lift = (0.5 + 0.5 * math.sin(2 * math.pi * phase)) * lift_max
            tilt = math.sin(2 * math.pi * phase + math.pi / 2) * motion.tilt_deg

        frame = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        frame.paste(img, (margin, fh - img.height - round(lift)), img)
        frame = frame.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(fw / 2, fh))
        frames.append(frame)

    sheet, _ = save_sprite_sheet(frames, output_path, interval=FRAME_INTERVAL, fps=round(1 / FRAME_INTERVAL))
    return sheet


def actor_floor_y(kind: ActorKind, canvas_h: int) -> int:
    return round(canvas_h * (WHEELED_FLOOR if kind == ActorKind.WHEELED else FLOATING_FLOOR))


def actor_scale(sheet_path: Path, kind: ActorKind, canvas_h: int) -> float:
    """Scale so the actor fills a fixed share of the frame height, and fits inside the 9:16 cut."""
    import json

    meta = json.loads(sheet_path.with_suffix(".json").read_text())
    target_h = canvas_h * (0.26 if kind == ActorKind.WHEELED else 0.22)
    max_w = canvas_h * 9 / 16 * 0.6  # leave room around the actor in the vertical crop
    return round(min(target_h / meta["frameHeight"], max_w / meta["frameWidth"]), 4)


# ---------------------------------------------------------------------------
# Scenes, bridges, world
# ---------------------------------------------------------------------------

def fit_panorama(img: Image.Image, canvas_h: int) -> Image.Image:
    """Resize a generated panorama to exactly 3H x H (crop vertically if the aspect differs)."""
    img = img.convert("RGB")
    target_w, target_h = 3 * canvas_h, canvas_h
    scale = max(target_w / img.width, target_h / img.height)
    resized = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    # Keep the bottom (the road) when cropping height
    left = (resized.width - target_w) // 2
    top = resized.height - target_h
    return resized.crop((left, top, left + target_w, top + target_h))


def build_bridge_anchor(prev_scene: Image.Image, next_scene: Image.Image, canvas_h: int) -> Image.Image:
    h = canvas_h
    anchor = Image.new("RGB", (3 * h, h), (255, 255, 255))
    anchor.paste(prev_scene.crop((prev_scene.width - h, 0, prev_scene.width, h)), (0, 0))
    anchor.paste(next_scene.crop((0, 0, h, h)), (2 * h, 0))
    return anchor


def world_layout(num_scenes: int, canvas_h: int) -> tuple[int, list[int], list[int]]:
    """Return (world width, scene x positions, bridge x positions)."""
    h = canvas_h
    scene_xs = [i * 4 * h for i in range(num_scenes)]
    bridge_xs = [x - 2 * h for x in scene_xs[1:]]
    width = scene_xs[-1] + 3 * h
    return width, scene_xs, bridge_xs


def _blend_in(world: np.ndarray, img: Image.Image, x0: int, ramp: int) -> None:
    """Paste `img` at x0, fading it in over its first `ramp` pixels."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    w = arr.shape[1]
    alpha = np.clip(np.arange(w, dtype=np.float32) / max(1, ramp), 0.0, 1.0)[None, :, None]
    region = world[:, x0 : x0 + w]
    world[:, x0 : x0 + w] = region * (1.0 - alpha) + arr * alpha


def compose_world(scenes: list[Image.Image], bridges: list[Image.Image], canvas_h: int) -> Image.Image:
    if len(bridges) != len(scenes) - 1:
        raise ValueError(f"Need {len(scenes) - 1} bridges for {len(scenes)} scenes, got {len(bridges)}")
    h = canvas_h
    width, scene_xs, bridge_xs = world_layout(len(scenes), h)
    world = np.zeros((h, width, 3), dtype=np.float32)
    world[:, : 3 * h] = np.asarray(scenes[0].convert("RGB"), dtype=np.float32)
    for bridge, bx, scene, sx in zip(bridges, bridge_xs, scenes[1:], scene_xs[1:]):
        _blend_in(world, bridge, bx, h)
        _blend_in(world, scene, sx, h)
    return Image.fromarray(np.clip(world, 0, 255).astype(np.uint8))


def segment_world(world: Image.Image, num_scenes: int, canvas_h: int) -> list[Image.Image]:
    """Cut the world strip into one segment per scene, split at each bridge's midpoint.

    Segments after the first start 1px early because the timeline overlaps consecutive scenes by
    `overlap_prev = 1`.
    """
    h = canvas_h
    _, _, bridge_xs = world_layout(num_scenes, h)
    cuts = [0] + [bx + 3 * h // 2 for bx in bridge_xs] + [world.width]
    segments = []
    for i in range(num_scenes):
        start = cuts[i] - (1 if i > 0 else 0)
        segments.append(world.crop((start, 0, cuts[i + 1], h)))
    return segments
