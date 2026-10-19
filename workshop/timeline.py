"""High-level timeline specification, world coordinates, and camera tracking for long-form stories.

The mental model:
- Scenes are placed along a 1D World X axis.
- As time advances (t = 0..total_duration), the camera moves forward at `speed` (pixels/sec).
- The screen X position for any scene at time `t` is:
      screen_x = world_x_start - (camera_speed * parallax * t)
- When `start_on_screen=True`, the first scene opens at `world_x_start = 0.0`,
  so the video starts directly inside the scenery with zero empty lead-in.
"""
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from workshop.character import create_contact_shadow

# Backwards compatibility alias
create_procedural_shadow = create_contact_shadow


@dataclass
class SceneSpec:
    """Specification of an individual scene along the story timeline."""

    id: str
    sheet: Path
    left_bg: Path | None = None
    left_bg_offset: int | None = None
    left_bg_scale: float | None = None
    video_scale: float | None = None
    video_y: int | str = 0
    scale: float = 1.0
    y: int = 0
    duration: float | None = None
    gap_after: float = 2.0
    overlap_prev: int = 0
    left_feather: tuple[int, int] | None = None
    parallax: float = 1.0  # 1.0 = camera speed, < 1.0 = background parallax
    anim_delay: float = 0.0
    loop_start_frame: int | None = None
    pingpong: bool = False
    hold_duration: float = 0.0
    hide_actor_during_hold: bool = False
    hold_frame_segments: list[list[int]] | None = None
    seam_feather_width: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # Calculated when timeline is solved:
    width: int = 0
    height: int = 0
    native_duration: float = 0.0
    start_time: float = 0.0
    active_duration: float = 0.0
    world_x_start: float = 0.0
    hold_start_time: float = 0.0
    hold_end_time: float = 0.0
    hold_cam_x: float = 0.0

    @property
    def effective_video_scale(self) -> float:
        if self.video_scale is not None:
            return float(self.video_scale)
        return float(self.scale)

    @property
    def effective_left_bg_scale(self) -> float:
        if self.left_bg_scale is not None:
            return float(self.left_bg_scale)
        return float(self.scale)

    def screen_x_at(self, t: float, camera_speed: float) -> int:
        """Calculate screen X coordinate of this scene at time `t`."""
        return round(self.world_x_start - camera_speed * self.parallax * t)

    def is_visible_at(self, t: float, camera_speed: float, canvas_w: int) -> bool:
        """Check if this scene currently intersects the camera viewport."""
        sx = self.screen_x_at(t, camera_speed)
        return (sx + self.width > 0) and (sx < canvas_w)


@dataclass
class TimelineConfig:
    """Full configuration for a long-form multi-scene camera tracking story."""

    canvas_w: int = 640
    canvas_h: int = 360
    fps: int = 24
    bg_color: str = "#ffffff"
    bg_gradient: list[str] | None = None
    bg_gradient_direction: str = "vertical"
    start_on_screen: bool = True
    hold_last_scene: bool = False

    # Character actor setup
    char_sheet: Path = Path("outputs/character/sprite.png")
    char_full_sheet: Path | None = None
    char_outro_time: float | None = None
    action_loop_step: int | None = None
    meow_loop_step: int | None = None
    char_scale: float = 0.45
    char_frames: str = "all"
    floor_y: int = 352
    char_x: int | None = None
    char_y: int | None = None
    cat_x: int | None = None
    cat_y: int | None = None
    flip_char: bool = False

    # Ground contact drop shadow
    shadow: bool = True
    shadow_opacity: float = 0.35
    shadow_width_scale: float = 0.70
    shadow_height: int = 14

    # Camera tracking & stride locking
    speed: float | None = None
    stride_pixels: float | None = None

    # Story scenes sequence
    scenes: list[SceneSpec] = field(default_factory=list)


class StoryTimeline:
    """High-level timeline solver and viewport query engine."""

    def __init__(self, config: TimelineConfig, root_dir: Path | None = None):
        self.config = config
        self.root = root_dir or Path.cwd()
        self.speed = self._resolve_speed()
        self.config.speed = self.speed
        self._holds: list[tuple[float, float, float, float, SceneSpec]] = []
        self.total_duration = self._solve_world_positions()

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if (self.root / p).exists():
            return self.root / p
        if (Path.cwd() / p).exists():
            return Path.cwd() / p
        return self.root / p

    def _resolve_speed(self) -> float:
        """Determine camera speed from explicit value, cat stride locking, or first scene pace."""
        if self.config.speed is not None:
            return float(self.config.speed)

        # Stride-locking: speed = stride_pixels / character_loop_duration
        if self.config.stride_pixels is not None:
            char_meta_path = self._resolve_path(self.config.char_sheet).with_suffix(".json")
            if char_meta_path.exists():
                import json
                with open(char_meta_path) as f:
                    cmeta = json.load(f)
                loop_sec = cmeta.get("frameCount", 15) * cmeta.get("intervalSeconds", 0.25)
            else:
                loop_sec = 3.75
            if loop_sec > 0:
                return float(self.config.stride_pixels) / loop_sec

        # Default fallback: speed = (canvas_w + scene_width) / 10s
        return 100.0

    def _get_character_timing(self) -> tuple[float, float]:
        char_meta_path = self._resolve_path(self.config.char_sheet).with_suffix(".json")
        interval = 0.25
        walk_frames = 15
        if char_meta_path.exists():
            import json
            with open(char_meta_path) as f:
                cmeta = json.load(f)
            interval = cmeta.get("intervalSeconds", 0.25)
            walk_frames = cmeta.get("frameCount", 15)
        walk_dur = walk_frames * interval
        meow_dur = 9 * interval
        return walk_dur, meow_dur

    def _dist_to_scroll_time(self, dist: float) -> float:
        if self.speed <= 0:
            return 0.0
        walk_time = dist / self.speed
        loop_step = self.config.action_loop_step or self.config.meow_loop_step
        if not loop_step or loop_step <= 0:
            return walk_time

        walk_dur, action_dur = self._get_character_timing()
        k = loop_step
        block_walk_dur = k * walk_dur
        block_dur = block_walk_dur + action_dur

        num_blocks = int(walk_time // block_walk_dur)
        rem_walk_time = walk_time - num_blocks * block_walk_dur
        if rem_walk_time <= walk_dur:
            elapsed_in_block = rem_walk_time
        else:
            elapsed_in_block = rem_walk_time + action_dur
        return num_blocks * block_dur + elapsed_in_block

    def _scroll_to_real_time(self, scroll_t: float) -> float:
        extra = 0.0
        for h_scroll_t, _, _, h_dur, _ in self._holds:
            if scroll_t > h_scroll_t:
                extra += h_dur
        return scroll_t + extra

    def _dist_to_elapsed_time(self, dist: float) -> float:
        return self._scroll_to_real_time(self._dist_to_scroll_time(dist))

    def scroll_time_at(self, t: float) -> float:
        """Convert real story time `t` to camera/actor scrolling clock time (pausing during scene holds)."""
        paused = 0.0
        for _, h_start, h_end, h_dur, _ in self._holds:
            if t >= h_end:
                paused += h_dur
            elif t > h_start:
                paused += (t - h_start)
        return max(0.0, t - paused)

    def is_actor_hidden_at(self, t: float) -> bool:
        """Return True if the foreground character actor should be hidden at story time `t`."""
        for _, h_start, h_end, _, sc in self._holds:
            if sc.hide_actor_during_hold and h_start <= t < h_end:
                return True
        return False

    def _solve_world_positions(self) -> float:
        """Position every scene along the 1D world timeline."""
        canvas_w = self.config.canvas_w
        start_on_screen = self.config.start_on_screen
        self._holds = []

        current_time = 0.0
        for i, sc in enumerate(self.config.scenes):
            s_path = self._resolve_path(sc.sheet)
            f_json = s_path.with_suffix(".json")

            if f_json.exists():
                import json
                with open(f_json) as f:
                    smeta = json.load(f)
                raw_w = smeta.get("frameWidth", canvas_w)
                raw_h = smeta.get("frameHeight", round(canvas_w * 9 / 16))
                sc.native_duration = smeta.get("frameCount", 40) * smeta.get("intervalSeconds", 0.25)
            elif s_path.exists():
                with Image.open(s_path) as im:
                    raw_w, raw_h = im.size
                sc.native_duration = sc.duration or 10.0
            else:
                raw_w, raw_h = canvas_w, round(canvas_w * 9 / 16)
                sc.native_duration = 10.0

            vid_w = round(raw_w * sc.effective_video_scale)
            vid_h = round(raw_h * sc.effective_video_scale)

            # Measure total width and height if a left background extension is attached
            if sc.left_bg:
                l_path = self._resolve_path(sc.left_bg)
                if l_path.exists():
                    with Image.open(l_path) as lim:
                        target_bg_h = round(raw_h * sc.effective_left_bg_scale)
                        l_w = round(lim.width * (target_bg_h / lim.height))
                        l_h = target_bg_h
                    offset = sc.left_bg_offset if sc.left_bg_offset is not None else (l_w - round(vid_w * 0.25))
                    sc.width = max(l_w, offset + vid_w)
                    if isinstance(sc.video_y, str):
                        v = sc.video_y.strip().lower()
                        if v == "bottom":
                            actual_vy = max(0, l_h - vid_h)
                        elif v == "center":
                            actual_vy = max(0, (l_h - vid_h) // 2)
                        else:
                            try:
                                actual_vy = int(sc.video_y)
                            except ValueError:
                                actual_vy = 0
                    else:
                        actual_vy = int(sc.video_y or 0)
                    sc.height = max(l_h, actual_vy + vid_h)
                else:
                    sc.width = vid_w
                    sc.height = vid_h
            else:
                sc.width = vid_w
                sc.height = vid_h

            # First scene opens directly on screen if start_on_screen is enabled
            if i == 0 and start_on_screen:
                sc.world_x_start = 0.0
                sc.start_time = 0.0
            elif i > 0 and sc.overlap_prev > 0:
                prev = self.config.scenes[i - 1]
                sc.world_x_start = prev.world_x_start + prev.width - sc.overlap_prev
                sc.start_time = max(0.0, self._dist_to_elapsed_time(sc.world_x_start - canvas_w))
            else:
                sc.start_time = current_time
                cam_x_now = self.cam_x_at(current_time)
                sc.world_x_start = cam_x_now + canvas_w

            if sc.hold_duration and sc.hold_duration > 0:
                hold_cam_x = max(0.0, sc.world_x_start + sc.width - canvas_w)
                h_scroll_t = self._dist_to_scroll_time(hold_cam_x)
                h_real_start = self._scroll_to_real_time(h_scroll_t)
                h_real_end = h_real_start + float(sc.hold_duration)
                sc.hold_cam_x = hold_cam_x
                sc.hold_start_time = h_real_start
                sc.hold_end_time = h_real_end
                self._holds.append((h_scroll_t, h_real_start, h_real_end, float(sc.hold_duration), sc))

            end_world_x = sc.world_x_start + sc.width
            end_time = self._dist_to_elapsed_time(end_world_x)
            sc.active_duration = end_time - sc.start_time
            current_time = end_time + (sc.gap_after if i < len(self.config.scenes) - 1 else 0.0)

            if i == len(self.config.scenes) - 1 and self.config.hold_last_scene:
                t_flush = max(0.0, self._dist_to_elapsed_time(sc.world_x_start + sc.width - canvas_w))
                if sc.hold_duration and sc.hold_duration > 0:
                    t_flush = sc.hold_end_time
                sc.active_duration = t_flush - sc.start_time
                current_time = t_flush + sc.gap_after

        return current_time

    def cam_x_at(self, t: float) -> float:
        """Calculate the world camera X position at story time t."""
        if not self.config.scenes or self.speed <= 0:
            return 0.0

        scroll_t = self.scroll_time_at(t)

        loop_step = self.config.action_loop_step or self.config.meow_loop_step
        if loop_step and loop_step > 0:
            walk_dur, action_dur = self._get_character_timing()
            k = loop_step
            block_walk_dur = k * walk_dur
            block_dur = block_walk_dur + action_dur

            block_idx = int(scroll_t / block_dur)
            t_in_block = scroll_t - (block_idx * block_dur)
            if t_in_block < walk_dur:
                walk_in_block = t_in_block
            elif t_in_block < walk_dur + action_dur:
                walk_in_block = walk_dur
            else:
                walk_in_block = walk_dur + (t_in_block - (walk_dur + action_dur))

            total_walk_time = block_idx * block_walk_dur + walk_in_block
            cam_x = self.speed * total_walk_time
        else:
            cam_x = self.speed * scroll_t

        if self.config.hold_last_scene:
            last = self.config.scenes[-1]
            max_cam = max(0.0, last.world_x_start + last.width - self.config.canvas_w)
            cam_x = min(cam_x, max_cam)
        return cam_x

    def visible_scenes(self, t: float, canvas_w: int) -> Iterator[tuple[SceneSpec, int]]:
        """Yield (scene_spec, screen_x) for every scene visible in the camera at time `t`."""
        cam_x = self.cam_x_at(t)
        active_hold_idx = None
        for idx, s in enumerate(self.config.scenes):
            if s.hold_duration > 0 and s.hold_start_time <= t < s.hold_end_time:
                active_hold_idx = idx
                break
        for idx, s in enumerate(self.config.scenes):
            if active_hold_idx is not None and idx > active_hold_idx:
                continue
            sx = round(s.world_x_start - cam_x * s.parallax)
            if sx + s.width > 0 and sx < canvas_w:
                yield s, sx


def build_timeline(config: TimelineConfig, root_dir: Path | None = None) -> float:
    """Solve the world-coordinate timeline for all scenes. Returns total duration in seconds."""
    solver = StoryTimeline(config, root_dir=root_dir)
    return solver.total_duration


def load_timeline_toml(path: str | Path, root_dir: Path | None = None) -> TimelineConfig:
    """Load and solve a TimelineConfig from a TOML file."""
    path = Path(path)
    root = root_dir or Path.cwd()
    data = tomllib.loads(path.read_text())

    cam = data.get("camera", {})
    char = data.get("character", {})

    scenes = [
        SceneSpec(
            id=sc.get("id", Path(sc["sheet"]).stem),
            sheet=Path(sc["sheet"]),
            left_bg=Path(sc["left_bg"]) if sc.get("left_bg") else None,
            left_bg_offset=int(sc["left_bg_offset"]) if sc.get("left_bg_offset") is not None else None,
            left_bg_scale=float(sc["left_bg_scale"]) if sc.get("left_bg_scale") is not None else (float(sc["bg_scale"]) if sc.get("bg_scale") is not None else None),
            video_scale=float(sc["video_scale"]) if sc.get("video_scale") is not None else None,
            video_y=sc.get("video_y", sc.get("video_valign", 0)),
            scale=float(sc.get("scale", 1.0)),
            y=int(sc.get("y", 0)),
            duration=float(sc["duration"]) if sc.get("duration") is not None else None,
            gap_after=float(sc.get("gap_after", 2.0)),
            overlap_prev=int(sc.get("overlap_prev", 0)),
            left_feather=tuple(sc["left_feather"]) if sc.get("left_feather") else None,
            parallax=float(sc.get("parallax", 1.0)),
            anim_delay=float(sc.get("anim_delay", 0.0)),
            loop_start_frame=int(sc["loop_start_frame"]) if sc.get("loop_start_frame") is not None else None,
            pingpong=bool(sc.get("pingpong", False)),
            hold_duration=float(sc.get("hold_duration", 0.0)),
            hide_actor_during_hold=bool(sc.get("hide_actor_during_hold", False)),
            hold_frame_segments=sc.get("hold_frame_segments"),
            seam_feather_width=int(sc["seam_feather_width"]) if sc.get("seam_feather_width") is not None else None,
            extra=sc,
        )
        for sc in data.get("scenes", [])
    ]

    config = TimelineConfig(
        canvas_w=int(cam.get("width", 640)),
        canvas_h=int(cam.get("height", 360)),
        fps=int(cam.get("fps", 24)),
        bg_color=cam.get("bg_color", "#ffffff"),
        bg_gradient=cam.get("bg_gradient"),
        bg_gradient_direction=cam.get("bg_gradient_direction", "vertical"),
        start_on_screen=bool(cam.get("start_on_screen", True)),
        hold_last_scene=bool(cam.get("hold_last_scene", False)),
        char_sheet=Path(char.get("sheet", "outputs/character/sprite.png")),
        char_full_sheet=Path(char["full_sheet"]) if char.get("full_sheet") else None,
        char_outro_time=float(char["outro_time"]) if char.get("outro_time") is not None else None,
        action_loop_step=int(char["action_loop_step"]) if char.get("action_loop_step") else (int(char["meow_loop_step"]) if char.get("meow_loop_step") else None),
        meow_loop_step=int(char["meow_loop_step"]) if char.get("meow_loop_step") else None,
        char_scale=float(char.get("scale", 0.45)),
        char_frames=char.get("frames", "all"),
        floor_y=int(char.get("floor_y", 352)),
        char_x=char.get("x", char.get("char_x", char.get("cat_x"))),
        char_y=char.get("y", char.get("char_y", char.get("cat_y"))),
        cat_x=char.get("x", char.get("char_x", char.get("cat_x"))),
        cat_y=char.get("y", char.get("char_y", char.get("cat_y"))),
        flip_char=bool(char.get("flip", False)),
        shadow=bool(char.get("shadow", True)),
        shadow_opacity=float(char.get("shadow_opacity", 0.35)),
        speed=cam.get("speed"),
        stride_pixels=char.get("stride_pixels"),
        scenes=scenes,
    )

    build_timeline(config, root_dir=root)
    return config
