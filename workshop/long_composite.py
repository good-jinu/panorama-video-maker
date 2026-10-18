"""Long-form multi-scene streaming compositor with parallax, lazy caching, and ground shadows."""
import argparse
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from workshop.character import CharacterActor
from workshop.composite import resolve_background
from workshop.scenes import SceneCache
from workshop.timeline import StoryTimeline, TimelineConfig, load_timeline_toml
from workshop.utils import AnimationWriter


class StoryStage:
    """High-level stage manager that composites scenery, character, and shadows for each frame."""

    def __init__(self, config: TimelineConfig, root_dir: Optional[Path] = None):
        self.config = config
        self.root = root_dir or Path.cwd()
        self.timeline = StoryTimeline(config, root_dir=self.root)
        self.cache = SceneCache(root_dir=self.root, max_cached=3)

        # Foreground actor (e.g., walking cat)
        self.actor = CharacterActor(
            sheet_path=config.char_sheet,
            scale=config.char_scale,
            frames_spec=config.char_frames,
            floor_y=config.floor_y,
            canvas_w=config.canvas_w,
            fixed_x=config.cat_x,
            fixed_y=config.cat_y,
            flip=config.flip_char,
            shadow=config.shadow,
            shadow_opacity=config.shadow_opacity,
            shadow_width_scale=config.shadow_width_scale,
            shadow_height=config.shadow_height,
            stride_pixels=config.stride_pixels,
            full_sheet_path=config.char_full_sheet,
            outro_time=config.char_outro_time,
            action_loop_step=config.action_loop_step or config.meow_loop_step,
            meow_loop_step=config.meow_loop_step,
            root_dir=self.root,
        )

        # Stage backdrop (solid color or gradient canvas)
        self.bg_canvas, self.bg_fallback_rgb, self.bg_desc = resolve_background(
            config.canvas_w,
            config.canvas_h,
            bg_color=config.bg_color,
            bg_gradient=config.bg_gradient,
            direction=config.bg_gradient_direction,
        )

    def render_frame(self, t: float) -> Image.Image:
        """Compose a single frame at story time `t`."""
        canvas = self.bg_canvas.copy()

        # 1. Background scenery layers currently visible in camera viewport
        visible_entries = list(self.timeline.visible_scenes(t, self.config.canvas_w))
        self.cache.prune_unneeded({s.id for s, _ in visible_entries})

        for spec, screen_x in visible_entries:
            scene = self.cache.get(spec)
            if scene.backdrop is not None:
                canvas.alpha_composite(scene.backdrop, (screen_x - scene.pad_x, scene.y - scene.pad_y))
            canvas.alpha_composite(scene.get_frame(t), (screen_x, scene.y))

        # 2. Ground contact drop-shadow and foreground actor (unless hidden by scene hold)
        if not self.timeline.is_actor_hidden_at(t):
            actor_t = self.timeline.scroll_time_at(t)
            if self.actor.has_shadow:
                shadow_img, shadow_pos = self.actor.get_shadow(actor_t)
                if shadow_img is not None and shadow_pos is not None:
                    canvas.alpha_composite(shadow_img, shadow_pos)

            # 3. Foreground actor
            actor_frame, actor_pos = self.actor.get_actor_frame_and_pos(actor_t)
            if actor_frame is not None:
                canvas.alpha_composite(actor_frame, actor_pos)

        # 4. Flatten onto solid background for video encoder
        frame_rgb = Image.new("RGB", (self.config.canvas_w, self.config.canvas_h), self.bg_fallback_rgb)
        frame_rgb.paste(canvas, mask=canvas.getchannel("A") if "A" in canvas.getbands() else None)
        return frame_rgb


def render_long_animation(
    config: TimelineConfig,
    output_path: Union[str, Path],
    root_dir: Optional[Path] = None,
    make_gif: bool = False,
    gif_encoder: Optional[str] = None,
) -> tuple[Path, Optional[Path]]:
    """Stream-render a long-form tracking animation across a multi-scene timeline."""
    stage = StoryStage(config, root_dir=root_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fps = config.fps
    total_duration = stage.timeline.total_duration
    total_frames = round(total_duration * fps)

    print(f"Rendering [STORY TIMELINE] {len(config.scenes)} scene(s), {total_frames} frames ({total_duration:.1f}s @ {fps} fps)...")
    print(f"  Actor: fixed at {stage.actor.pos}, scale={config.char_scale:.2f}")
    print(f"  Ground: y={config.floor_y}, shadow={'on' if config.shadow else 'off'}")
    print(f"  Camera velocity: {stage.timeline.speed:.1f}px/s on {stage.bg_desc}")
    for idx, s in enumerate(config.scenes):
        print(f"    Scene {idx + 1} [{s.id}]: t=[{s.start_time:.1f}s - {s.start_time + s.active_duration:.1f}s], size=({s.width}x{s.height}), y={s.y}")

    with AnimationWriter(output, config.canvas_w, config.canvas_h, fps=fps, make_gif=make_gif, gif_encoder=gif_encoder) as writer:
        for i in range(total_frames):
            t = i / fps
            writer.write(stage.render_frame(t))

    print(f"Saved MP4: {writer.mp4_path}")
    if writer.gif_path:
        print(f"Saved GIF: {writer.gif_path}")
    return writer.mp4_path, writer.gif_path


def render_timeline_job(
    timeline_path: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    make_gif: bool = False,
    gif_encoder: Optional[str] = None,
) -> tuple[Path, Optional[Path]]:
    """Entry point for the job runner (jobs/*.toml)."""
    p = Path(timeline_path)
    root = Path.cwd()
    config = load_timeline_toml(p, root_dir=root)
    out = Path(output) if output else root / "outputs" / "long-story" / "animation.mp4"
    return render_long_animation(config, output_path=out, root_dir=root, make_gif=make_gif, gif_encoder=gif_encoder)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("timeline", type=Path, help="Path to timeline TOML file")
    ap.add_argument("-o", "--output", type=Path, help="Output MP4 path")
    ap.add_argument("--gif", action="store_true", help="Also generate GIF")
    args = ap.parse_args()

    render_timeline_job(args.timeline, output=args.output, make_gif=args.gif)


if __name__ == "__main__":
    main()
