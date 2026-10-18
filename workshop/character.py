"""High-level character actor, animation cycle, and contact ground shadow.

Encapsulates:
- Character frame extraction and pacing
- Ground-level alignment and paw contact drop shadow
- Precise stride locking (connecting walk pace to camera tracking speed)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from workshop.utils import load_sprite_sheet


def create_contact_shadow(
    width: int,
    height: int,
    opacity: float = 0.35,
) -> Image.Image:
    """Generate a soft elliptical contact drop-shadow for character grounding."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height / 2.0
    rx, ry = width * 0.46, height * 0.46

    dist = np.hypot((x - cx) / max(1.0, rx), (y - cy) / max(1.0, ry))
    dist = np.clip(dist, 0.0, 1.0)

    # Cosine falloff for smooth grounding
    alpha = np.clip(0.5 * (1.0 + np.cos(dist * np.pi)) * (opacity * 255.0), 0, 255).astype(np.uint8)
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[..., 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


class CharacterActor:
    """High-level foreground animated actor (e.g., walking cat)."""

    def __init__(
        self,
        sheet_path: Union[str, Path],
        scale: float = 0.45,
        frames_spec: str = "all",
        floor_y: int = 352,
        canvas_w: int = 640,
        fixed_x: Optional[int] = None,
        fixed_y: Optional[int] = None,
        flip: bool = False,
        shadow: bool = True,
        shadow_opacity: float = 0.35,
        shadow_width_scale: float = 0.70,
        shadow_height: int = 14,
        stride_pixels: Optional[float] = None,
        full_sheet_path: Optional[Union[str, Path]] = None,
        outro_time: Optional[float] = None,
        action_loop_step: Optional[int] = None,
        meow_loop_step: Optional[int] = None,
        root_dir: Optional[Path] = None,
    ):
        root = root_dir or Path.cwd()
        self.canvas_w = canvas_w
        self.action_loop_step = action_loop_step or meow_loop_step
        self.meow_loop_step = self.action_loop_step
        p = Path(sheet_path)
        actual_path = p if p.is_absolute() else (root / p if (root / p).exists() else Path.cwd() / p)

        raw_frames, meta = load_sprite_sheet(actual_path, allow_single_image=True)

        # Slice frame subset if specified
        if frames_spec and frames_spec.lower() != "all":
            if "-" in frames_spec:
                first, last = map(int, frames_spec.split("-"))
                raw_frames = raw_frames[first - 1 : last]
            else:
                raw_frames = raw_frames[: int(frames_spec)]

        self.interval = meta.get("intervalSeconds", 0.25)
        self.loop_duration = len(raw_frames) * self.interval
        self.stride_pixels = stride_pixels

        # Scale and orient frames
        orig_w, orig_h = raw_frames[0].size
        self.width = round(orig_w * scale)
        self.height = round(orig_h * scale)

        self.frames = []
        for f in raw_frames:
            resized = f.resize((self.width, self.height), Image.Resampling.LANCZOS)
            if flip:
                resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            self.frames.append(resized)

        # Optional full sprite sequence for stopping and walk-off outro
        self.outro_frames = []
        self.outro_time = outro_time
        if full_sheet_path:
            fp = Path(full_sheet_path)
            f_actual = fp if fp.is_absolute() else (root / fp if (root / fp).exists() else Path.cwd() / fp)
            if f_actual.exists():
                full_raw, _ = load_sprite_sheet(f_actual, allow_single_image=True)
                for f in full_raw:
                    rf = f.resize((self.width, self.height), Image.Resampling.LANCZOS)
                    if flip:
                        rf = rf.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    self.outro_frames.append(rf)

        # Positioning on the stage
        self.pos_x = (canvas_w - self.width) // 2 if fixed_x is None else round(fixed_x)
        self.pos_y = round(fixed_y) if fixed_y is not None else (floor_y - self.height)
        self.pos = (self.pos_x, self.pos_y)

        # Ground contact shadow under paws
        self.has_shadow = shadow
        self.shadow_image = None
        self.shadow_pos = (0, 0)
        self.shadow_base_x = 0
        self.shadow_base_y = 0
        if shadow:
            sw_w = round(self.width * shadow_width_scale)
            sw_h = shadow_height
            self.shadow_image = create_contact_shadow(sw_w, sw_h, opacity=shadow_opacity)
            self.shadow_base_x = self.pos_x + (self.width - sw_w) // 2
            self.shadow_base_y = floor_y - sw_h // 2
            self.shadow_pos = (self.shadow_base_x, self.shadow_base_y)

    def _get_active_state(self, t: float) -> tuple[Optional[Image.Image], tuple[int, int], Optional[tuple[int, int]]]:
        """Compute (frame, actor_pos, shadow_pos) at story time `t`."""
        if self.outro_time is not None and self.outro_frames and len(self.outro_frames) >= 40:
            if t >= self.outro_time:
                # Align with stride/loop boundary: wait until current walk cycle reaches index 18 (frame 19)
                loop_len = len(self.frames)
                frame_count_at_outro = int(self.outro_time / self.interval)
                t_trans_frame = ((frame_count_at_outro + loop_len - 1) // loop_len) * loop_len
                t_trans = t_trans_frame * self.interval

                if t >= t_trans:
                    outro_idx = int((t - t_trans) / self.interval)
                    if outro_idx < 21:
                        target_f_idx = 19 + outro_idx
                        frame = self.outro_frames[target_f_idx]
                        if target_f_idx < 28:
                            shift_x = 0
                        else:
                            prog = min(1.0, max(0.0, (outro_idx - 9) / (20 - 9)))
                            exit_dist = (self.canvas_w - (self.pos_x + self.width)) + 30
                            shift_x = round(exit_dist * prog)
                        pos = (self.pos_x + shift_x, self.pos_y)
                        shadow_pos = (self.shadow_base_x + shift_x, self.shadow_base_y) if self.has_shadow else None
                        return frame, pos, shadow_pos
                    else:
                        return None, (self.pos_x, self.pos_y), None

        if self.meow_loop_step and self.meow_loop_step > 0 and self.outro_frames and len(self.outro_frames) >= 28:
            k = self.meow_loop_step
            walk_dur = len(self.frames) * self.interval
            meow_dur = 9 * self.interval
            block_dur = k * walk_dur + meow_dur
            block_idx = int(t / block_dur)
            t_in_block = t - (block_idx * block_dur)
            f_in_block = int(t_in_block / self.interval)

            if f_in_block < len(self.frames):
                frame = self.frames[f_in_block]
            elif f_in_block < len(self.frames) + 9:
                target_f_idx = 19 + (f_in_block - len(self.frames))
                frame = self.outro_frames[target_f_idx]
            else:
                rem_frames = f_in_block - (len(self.frames) + 9)
                walk_idx = rem_frames % len(self.frames)
                frame = self.frames[walk_idx]

            shadow_pos = (self.shadow_base_x, self.shadow_base_y) if self.has_shadow else None
            return frame, (self.pos_x, self.pos_y), shadow_pos

        idx = int(t / self.interval) % len(self.frames)
        shadow_pos = (self.shadow_base_x, self.shadow_base_y) if self.has_shadow else None
        return self.frames[idx], (self.pos_x, self.pos_y), shadow_pos

    def get_frame(self, t: float) -> Image.Image:
        """Return the animated character frame for story timeline time `t`."""
        frame, _, _ = self._get_active_state(t)
        if frame is None:
            return Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        return frame

    def get_actor_frame_and_pos(self, t: float) -> tuple[Optional[Image.Image], tuple[int, int]]:
        frame, pos, _ = self._get_active_state(t)
        return frame, pos

    def get_shadow(self, t: float) -> tuple[Optional[Image.Image], Optional[tuple[int, int]]]:
        if not self.has_shadow:
            return None, None
        _, _, s_pos = self._get_active_state(t)
        if s_pos is None:
            return None, None
        return self.shadow_image, s_pos
