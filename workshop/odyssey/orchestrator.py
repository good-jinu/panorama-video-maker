"""A-to-Z End-to-End Orchestrator for Odyssey Journey Productions.

Compiles a high-level JourneySpec:
1. Validates and stages scene assets and transition bridges
2. Sets up Actor FSM, Stride-Locking, and Directorial Pacing
3. Stream-renders constant-memory composite video frames
4. Synthesizes material-specific footstep foley and weather soundscapes
5. Multiplexes master audio and video via FFmpeg
6. Produces simultaneous 16:9 (YouTube) and 9:16 (Shorts/TikTok) deliveries
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from workshop.long_composite import StoryStage
from workshop.odyssey.actor_fsm import OdysseyActorFSM
from workshop.odyssey.director import OdysseyDirector
from workshop.odyssey.soundscape import SoundscapeEngine
from workshop.odyssey.spec import GroundMaterial, JourneySpec
from workshop.timeline import SceneSpec, TimelineConfig
from workshop.utils import AnimationWriter


class OdysseyOrchestrator:
    """Full-stack vertical production engine compiling JourneySpec into finished master videos."""

    def __init__(self, journey: JourneySpec, root_dir: Path | None = None):
        self.journey = journey
        self.root = root_dir or Path.cwd()

    def compile_timeline_config(self) -> TimelineConfig:
        """Translate high-level JourneySpec into low-level TimelineConfig."""
        j = self.journey
        actor = j.actor
        director = j.director

        scenes: list[SceneSpec] = []
        for s in j.scenes:
            scenes.append(
                SceneSpec(
                    id=s.id,
                    sheet=Path(s.sheet_path),
                    left_bg=Path(s.left_bg_path) if s.left_bg_path else None,
                    left_bg_offset=s.left_bg_offset,
                    scale=s.scale,
                    y=s.y,
                    duration=s.duration,
                    hold_duration=s.hold_duration,
                    parallax=s.parallax,
                    gap_after=s.gap_after,
                    overlap_prev=s.overlap_prev,
                )
            )

        return TimelineConfig(
            canvas_w=director.canvas_w,
            canvas_h=director.canvas_h,
            fps=director.fps,
            bg_color=director.bg_color,
            hold_last_scene=director.hold_last_scene,
            speed=director.speed,
            char_sheet=Path(actor.sheet_path),
            char_full_sheet=Path(actor.full_sheet_path) if actor.full_sheet_path else None,
            char_scale=actor.scale,
            floor_y=actor.floor_y,
            stride_pixels=actor.stride_pixels,
            shadow=actor.contact_shadow,
            shadow_opacity=actor.shadow_opacity,
            shadow_width_scale=actor.shadow_width_scale,
            shadow_height=actor.shadow_height,
            flip_char=actor.flip,
            scenes=scenes,
        )

    def run(self, output_dir: str | Path = "outputs/odyssey") -> dict[str, Any]:
        """Execute end-to-end A-to-Z production pipeline."""
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        config = self.compile_timeline_config()
        stage = StoryStage(config, root_dir=self.root)
        fps = config.fps
        total_duration = stage.timeline.total_duration
        total_frames = round(total_duration * fps)

        print(f"🎬 [PANORAMA VIDEO MAKER] Initiating A-to-Z Production: '{self.journey.title}'")
        print(f"   Duration: {total_duration:.1f}s ({total_frames} frames @ {fps} fps)")
        print(f"   Protagonist: {self.journey.actor.name} (Stride={self.journey.actor.stride_pixels}px, Scale={self.journey.actor.scale})")
        print(f"   Scenes: {len(self.journey.scenes)} chapter(s)")

        # 1. Render raw composite video (constant-memory streaming)
        raw_video_path = out_d / "temp_silent_master.mp4"
        with AnimationWriter(raw_video_path, config.canvas_w, config.canvas_h, fps=fps) as writer:
            for i in range(total_frames):
                t = i / fps
                writer.write(stage.render_frame(t))

        # 2. Extract scene active intervals & ground materials for soundscape
        actor_fsm = OdysseyActorFSM(self.journey.actor, canvas_w=config.canvas_w, root_dir=self.root)
        active_intervals: list[tuple[float, float, GroundMaterial]] = []
        scene_moods: list[tuple[float, float, Any]] = []

        for j_scene, s_spec in zip(self.journey.scenes, config.scenes):
            start_t = s_spec.start_time
            end_t = start_t + s_spec.active_duration
            active_intervals.append((start_t, end_t, j_scene.ground_material))
            scene_moods.append((start_t, end_t, j_scene.ambient_mood))

        # 3. Compute foley footstep events synchronized to stride
        footsteps = actor_fsm.compute_footstep_events(active_intervals)
        print(f"🐾 [SOUNDSCAPE] Synthesized {len(footsteps)} synchronized paw-step foley event(s)")

        # 4. Synthesize master procedural audio
        sound_engine = SoundscapeEngine(self.journey.soundscape)
        audio_wav_path = out_d / "master_soundscape.wav"
        sound_engine.render_master_audio(
            duration=total_duration,
            footsteps=footsteps,
            scene_moods=scene_moods,
            out_wav_path=audio_wav_path,
        )

        # 5. Multiplex 16:9 Master Output
        master_16_9 = out_d / "odyssey_master_16_9.mp4"
        sound_engine.multiplex_audio_video(
            video_path=raw_video_path,
            foley_wav_path=audio_wav_path,
            out_mp4_path=master_16_9,
            music_path=self.journey.soundscape.music_track,
        )
        print(f"✅ [MASTER 16:9] Widescreen video ready: {master_16_9}")

        results: dict[str, Any] = {
            "title": self.journey.title,
            "duration": total_duration,
            "total_frames": total_frames,
            "widescreen_16_9": str(master_16_9),
            "soundscape_wav": str(audio_wav_path),
            "footstep_count": len(footsteps),
        }

        # 6. Render 9:16 Vertical Shorts if specified
        if "9:16" in self.journey.director.output_formats:
            director = OdysseyDirector(self.journey.director, actor_fsm)
            actor_center = stage.actor.pos_x + (stage.actor.width // 2)
            crop = director.compute_vertical_shorts_crop(actor_center)
            master_9_16 = out_d / "odyssey_shorts_9_16.mp4"

            # Execute vertical crop using ffmpeg
            crop_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(master_16_9),
                "-filter:v",
                f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}",
                "-c:a",
                "copy",
                str(master_9_16),
            ]
            subprocess.check_call(crop_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ [SHORTS 9:16] Vertical mobile cut ready: {master_9_16}")
            results["vertical_9_16"] = str(master_9_16)

        # Clean up temporary silent video
        if raw_video_path.exists():
            raw_video_path.unlink()

        return results


def main():
    parser = argparse.ArgumentParser(description="Panorama Video Maker: render an odyssey journey from a TOML spec")
    parser.add_argument("journey_toml", type=Path, help="Path to Journey TOML specification")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/odyssey"), help="Output directory")
    args = parser.parse_args()

    journey = JourneySpec.from_toml_file(args.journey_toml)
    # Asset paths resolve against the TOML's folder first, then the current directory
    orchestrator = OdysseyOrchestrator(journey, root_dir=args.journey_toml.resolve().parent)
    orchestrator.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
