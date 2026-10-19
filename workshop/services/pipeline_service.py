"""High-level pipeline operations for video studio production.

Includes:
1. Video-to-sprite extraction with ONNX matting
2. AI panoramic bridge outpainting (OpenAI)
3. Keyframe-conditioned AI video generation (Gemini Omni)
4. One-command automated video-scene append with bridge synthesis
5. Audio playlist assembly and looped video muxing
"""
import logging
from collections.abc import Sequence
from pathlib import Path

from sqlmodel import Session

from workshop.config import settings
from workshop.db.models import Asset, Project, Scene
from workshop.imagen import (
    DEFAULT_MODEL as DEFAULT_IMAGEN_MODEL,
)
from workshop.imagen import (
    call_gemini_omni_video,
    generate_scene_bridge,
)
from workshop.music_station import build_music_station
from workshop.sprite import generate_sprite_sheet
from workshop.utils import load_sprite_sheet

logger = logging.getLogger(__name__)



def extract_sprite_pipeline(
    video_path: Path,
    output_sheet: Path | None = None,
    crop: bool = False,
    keep_frames: bool = False,
    keep_background: bool = False,
    model_name: str = "rmbg",
    zoom_start: float = 1.0,
    zoom_step: float = 0.0,
) -> Path:
    """Extract frames from an MP4 into an RGBA sprite sheet."""
    stem = video_path.stem
    out = output_sheet or (settings.outputs_dir / stem / "sprite_bg.png" if keep_background else settings.outputs_dir / stem / "sprite.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    generate_sprite_sheet(
        video_path=video_path,
        output_path=out,
        crop=crop,
        keep_frames=keep_frames,
        keep_background=keep_background,
        model_name=model_name,
        zoom_start=zoom_start,
        zoom_step=zoom_step,
    )
    return out


def generate_bridge_pipeline(
    from_sheet: Path,
    to_sheet: Path,
    from_scale: float = 0.6806,
    to_scale: float = 0.75,
    output_path: Path | None = None,
    prompt: str | None = None,
    model: str = DEFAULT_IMAGEN_MODEL,
    size: str = "1536x512",
    quality: str = "high",
    overwrite: bool = False,
) -> Path:
    """Synthesize a panoramic transition bridge between two sprite sheets via OpenAI."""
    if output_path is None:
        bridges_dir = settings.bridges_dir
        bridges_dir.mkdir(parents=True, exist_ok=True)
        output_path = bridges_dir / f"{from_sheet.parent.name}-to-{to_sheet.parent.name}-bridge.png"


    output_path.parent.mkdir(parents=True, exist_ok=True)
    return generate_scene_bridge(
        from_sheet=from_sheet,
        to_sheet=to_sheet,
        from_scale=from_scale,
        to_scale=to_scale,
        output=output_path,
        prompt=prompt,
        model=model,
        size=size,
        quality=quality,
        overwrite=overwrite,
    )


def generate_ai_video_pipeline(
    prompt: str,
    output_path: Path,
    image_paths: Sequence[Path] | None = None,
    model: str = "gemini-omni-1.1-flash",
) -> Path:
    """Generate a keyframe-conditioned video clip using Gemini Omni."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return call_gemini_omni_video(
        prompt=prompt,
        output_path=output_path,
        image_paths=list(image_paths) if image_paths else None,
        model=model,
    )


def append_video_scene_to_project(
    session: Session,
    project_id: int,
    video_path: Path,
    scene_name: str | None = None,
    video_scale: float = 0.75,
    prompt: str | None = None,
    bridge_model: str = DEFAULT_IMAGEN_MODEL,
    overwrite_bridge: bool = False,
    seam_feather_width: int | None = None,
    hold_duration: float = 0.0,
    hide_actor_during_hold: bool = False,
) -> Scene:
    """End-to-End One-Command Scene Append:

    1. Extracts sprite sheet from video_path into outputs/<stem>/sprite_bg.png
    2. Synthesizes transition bridge from the project's preceding scene (if exists)
    3. Calculates scale, offset, and anim_delay automatically
    4. Registers the new Scene and generative Assets in the database
    """
    root = settings.root_dir
    project = session.get(Project, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    video_path = Path(video_path)
    stem = video_path.stem
    scene_id = scene_name or stem

    # 1. Extract sprite sheet
    sheet_path = settings.outputs_dir / stem / "sprite_bg.png"
    if not sheet_path.exists() or overwrite_bridge:
        sheet_path = extract_sprite_pipeline(video_path, output_sheet=sheet_path, keep_background=True)

    # 2. Inspect video dimensions for auto-offset
    frames, meta = load_sprite_sheet(sheet_path)
    fw, _fh = meta["frameWidth"], meta["frameHeight"]

    # Calculate offset
    left_bg_offset = round(1080 - fw * video_scale)
    # Stride speed calculation
    speed = (project.stride_pixels or 165.8) / 3.75
    anim_delay = round(left_bg_offset / speed, 1) if speed > 0 else 0.0

    left_bg_path = None
    left_feather_l = None
    left_feather_r = None
    overlap_prev = 0

    # 3. Check previous scene to build bridge
    sorted_scenes = sorted(project.scenes, key=lambda s: s.order_idx)
    if sorted_scenes:
        prev_scene = sorted_scenes[-1]
        prev_sheet = Path(prev_scene.sheet_path)
        if not prev_sheet.is_absolute():
            prev_sheet = root / prev_sheet

        prev_scale = prev_scene.video_scale or prev_scene.scale or 0.75
        bridge_file = settings.bridges_dir / f"{prev_scene.name}-to-{scene_id}-bridge.png"

        try:
            generate_bridge_pipeline(
                from_sheet=prev_sheet,
                to_sheet=sheet_path,
                from_scale=prev_scale,
                to_scale=video_scale,
                output_path=bridge_file,
                prompt=prompt,
                model=bridge_model,
                overwrite=overwrite_bridge,
            )
            left_bg_path = str(bridge_file.relative_to(root)) if bridge_file.is_relative_to(root) else str(bridge_file)
            overlap_prev = 180
            left_feather_l = 80
            left_feather_r = 60
        except Exception as e:
            logger.warning("Could not automatically generate bridge: %s", e)

    # 4. Insert Scene record
    new_order = len(sorted_scenes)
    scene = Scene(
        project_id=project_id,
        order_idx=new_order,
        name=scene_id,
        sheet_path=str(sheet_path.relative_to(root)) if sheet_path.is_relative_to(root) else str(sheet_path),
        left_bg_path=left_bg_path,
        left_bg_offset=left_bg_offset,
        left_bg_scale=1.0,
        video_scale=video_scale,
        video_y=0,
        scale=1.0,
        y=0,
        overlap_prev=overlap_prev,
        left_feather_l=left_feather_l,
        left_feather_r=left_feather_r,
        anim_delay=anim_delay,
        pingpong=True,
        seam_feather_width=seam_feather_width,
        hold_duration=hold_duration,
        hide_actor_during_hold=hide_actor_during_hold,
    )
    session.add(scene)

    # Register assets
    session.add(
        Asset(
            project_id=project_id,
            asset_type="sprite",
            file_path=scene.sheet_path,
            model_name="workshop.sprite",
            prompt=f"Extracted sprite sheet from {video_path.name}",
        )
    )
    if left_bg_path:
        session.add(
            Asset(
                project_id=project_id,
                asset_type="image",
                file_path=left_bg_path,
                model_name=bridge_model,
                prompt=prompt or f"Transition bridge from {sorted_scenes[-1].name} to {scene_id}",
            )
        )

    session.commit()
    session.refresh(scene)
    return scene


def build_music_station_pipeline(
    video_path: Path,
    youtube_urls: Sequence[str],
    output_path: Path | None = None,
) -> Path:
    """Download YouTube audio, concatenate, and loop video over the full soundtrack."""
    root = settings.root_dir
    out = output_path or (settings.outputs_dir / "music_station" / "music_station.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    return build_music_station(
        video_path=video_path,
        playlist_urls=list(youtube_urls),
        output=out,
        root_dir=root,
    )

