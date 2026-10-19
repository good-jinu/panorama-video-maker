"""Directorial Camera, Pacing Controller, and Multi-Format Framing Engine.

Handles:
- Camera velocity smoothing with ease-in / ease-out deceleration on hold scenes
- Multi-format viewport calculation:
  - 16:9 Landscape (YouTube Lofi, Desktop Wallpaper)
  - 9:16 Vertical (TikTok, Instagram Reels, YouTube Shorts)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from workshop.odyssey.spec import DirectorSpec

if TYPE_CHECKING:
    from workshop.odyssey.actor_fsm import OdysseyActorFSM


@dataclass
class ViewportCrop:
    """Cropping bounding box within the master composite canvas."""

    x: int
    y: int
    w: int
    h: int


class OdysseyDirector:
    """Directorial camera controller for motion pacing and multi-format framing."""

    def __init__(self, spec: DirectorSpec, actor_fsm: OdysseyActorFSM):
        self.spec = spec
        self.actor = actor_fsm
        self.canvas_w = spec.canvas_w
        self.canvas_h = spec.canvas_h
        self.fps = spec.fps

    def compute_vertical_shorts_crop(self, actor_center_x: int) -> ViewportCrop:
        """Calculate 9:16 vertical framing centered around the hero actor with cinematic headroom."""
        target_aspect = 9.0 / 16.0
        # Use full canvas height as the vertical baseline
        crop_h = self.canvas_h
        crop_w = int(crop_h * target_aspect)

        # Center crop horizontally on actor center with screen clamping
        crop_x = actor_center_x - (crop_w // 2)
        crop_x = max(0, min(self.canvas_w - crop_w, crop_x))
        crop_y = 0

        return ViewportCrop(x=crop_x, y=crop_y, w=crop_w, h=crop_h)

    def smooth_hold_speed_multiplier(
        self,
        t: float,
        hold_start: float,
        hold_end: float,
        ease_duration: float = 0.5,
    ) -> float:
        """Compute camera speed multiplier [0.0..1.0] with smooth deceleration into hold and acceleration out."""
        if t < hold_start - ease_duration or t > hold_end + ease_duration:
            return 1.0

        if hold_start <= t <= hold_end:
            return 0.0

        if t < hold_start:
            # Decelerating into hold (1.0 -> 0.0)
            progress = (hold_start - t) / ease_duration
            return 0.5 * (1.0 - progress**2)
        else:
            # Accelerating out of hold (0.0 -> 1.0)
            progress = (t - hold_end) / ease_duration
            return min(1.0, progress**2)
