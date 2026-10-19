"""Finite State Machine (FSM) and gait physics for wandering Odyssey protagonists.

Manages:
- Multi-action states: WALK, PAUSE_SNIFF, SIT_GAZE, SLEEP_COZY, MEOW
- Pacing & Stride Locking: foot-sliding prevention
- Precise footstep event tracking (for procedural foley sound effects)
- Contact ground shadow positioning
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from workshop.character import create_contact_shadow
from workshop.odyssey.spec import ActorAction, GroundMaterial, HeroActorSpec
from workshop.utils import load_sprite_sheet

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class FootstepEvent:
    """Discrete moment a paw contacts the ground during walking."""

    time: float
    frame_index: int
    material: GroundMaterial
    volume_scale: float = 1.0


class OdysseyActorFSM:
    """State machine governing protagonist animations, transitions, and foley triggers."""

    def __init__(
        self,
        spec: HeroActorSpec,
        canvas_w: int = 640,
        root_dir: Path | None = None,
    ):
        self.spec = spec
        self.canvas_w = canvas_w
        self.root = root_dir or Path.cwd()

        # Load walking frames
        p = Path(spec.sheet_path)
        actual_sheet = p if p.is_absolute() else (self.root / p if (self.root / p).exists() else Path.cwd() / p)
        raw_walk, meta = load_sprite_sheet(actual_sheet, allow_single_image=True)

        self.interval = meta.get("intervalSeconds", 0.25)
        self.walk_frame_count = len(raw_walk)
        self.walk_loop_duration = self.walk_frame_count * self.interval

        orig_w, orig_h = raw_walk[0].size
        self.width = round(orig_w * spec.scale)
        self.height = round(orig_h * spec.scale)

        # Scale frames
        self.walk_frames: list[Image.Image] = []
        for f in raw_walk:
            rf = f.resize((self.width, self.height), Image.Resampling.LANCZOS)
            if spec.flip:
                rf = rf.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            self.walk_frames.append(rf)

        # Optional full sheet for extended behaviors (sit, meow, gaze)
        self.action_frames: dict[ActorAction, list[Image.Image]] = {
            ActorAction.WALK: self.walk_frames,
        }

        if spec.full_sheet_path:
            fp = Path(spec.full_sheet_path)
            f_actual = fp if fp.is_absolute() else (self.root / fp if (self.root / fp).exists() else Path.cwd() / fp)
            if f_actual.exists():
                full_raw, _ = load_sprite_sheet(f_actual, allow_single_image=True)
                full_scaled = []
                for f in full_raw:
                    rf = f.resize((self.width, self.height), Image.Resampling.LANCZOS)
                    if spec.flip:
                        rf = rf.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    full_scaled.append(rf)

                # Segment full sheet into standard behavior sets if available
                # (Standard cat sheet layout: 0-14 walk, 15-23 meow/look, 24+ sit/rest)
                if len(full_scaled) >= 28:
                    self.action_frames[ActorAction.MEOW] = full_scaled[15:24]
                    self.action_frames[ActorAction.PAUSE_SNIFF] = full_scaled[15:20]
                    self.action_frames[ActorAction.SIT_GAZE] = full_scaled[20:28]
                    self.action_frames[ActorAction.SLEEP_COZY] = [full_scaled[-1]]

        # Base positioning
        self.pos_x = (canvas_w - self.width) // 2
        self.pos_y = spec.floor_y - self.height
        self.floor_y = spec.floor_y

        # Procedural contact ground shadow
        self.has_shadow = spec.contact_shadow
        self.shadow_image = None
        self.shadow_base_x = 0
        self.shadow_base_y = 0
        if self.has_shadow:
            sw_w = round(self.width * spec.shadow_width_scale)
            sw_h = spec.shadow_height
            self.shadow_image = create_contact_shadow(sw_w, sw_h, opacity=spec.shadow_opacity)
            self.shadow_base_x = self.pos_x + (self.width - sw_w) // 2
            self.shadow_base_y = spec.floor_y - sw_h // 2

    def get_action_frame(self, action: ActorAction, local_t: float) -> Image.Image:
        """Fetch frame for a given action state at local time offset."""
        frames = self.action_frames.get(action)
        if not frames:
            frames = self.walk_frames

        if action in (ActorAction.SIT_GAZE, ActorAction.SLEEP_COZY):
            # Sit down and then hold the final rested pose
            idx = min(int(local_t / self.interval), len(frames) - 1)
        elif action in (ActorAction.PAUSE_SNIFF, ActorAction.MEOW):
            idx = int(local_t / self.interval) % len(frames)
        else:
            # Standard walk cycle loop
            idx = int(local_t / self.interval) % len(frames)
        return frames[idx]

    def compute_footstep_events(
        self,
        active_intervals: Sequence[tuple[float, float, GroundMaterial]],
    ) -> list[FootstepEvent]:
        """Calculate discrete timestamps where paws touch ground during active walk intervals.

        Used by Soundscape engine to produce synchronized footstep foley.
        """
        events: list[FootstepEvent] = []
        contact_frames = self.spec.footstep_contact_frames

        for start_t, end_t, material in active_intervals:
            curr_t = start_t
            # Find the starting cycle offset
            while curr_t < end_t:
                cycle_time = curr_t - start_t
                cycle_frame = int(cycle_time / self.interval) % self.walk_frame_count

                if cycle_frame in contact_frames:
                    events.append(
                        FootstepEvent(
                            time=curr_t,
                            frame_index=cycle_frame,
                            material=material,
                            volume_scale=1.0,
                        )
                    )
                curr_t += self.interval

        return events

    def get_shadow(self) -> tuple[Image.Image | None, tuple[int, int] | None]:
        """Return the contact shadow image and position."""
        if not self.has_shadow or self.shadow_image is None:
            return None, None
        return self.shadow_image, (self.shadow_base_x, self.shadow_base_y)
