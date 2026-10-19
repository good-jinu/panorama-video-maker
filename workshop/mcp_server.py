"""Model Context Protocol (MCP) Server for AI-assisted video editing and compositing.

Implements end-to-end video platform capabilities:
1. Project & Timeline management in SQLite
2. Sprite Sheet Extraction with deep-learning matting (BiRefNet, RMBG)
3. OpenAI Panoramic Transition Bridge synthesis
4. Gemini Omni keyframe-conditioned video generation
5. One-command automated video-scene append with bridge outpainting
6. Motion-preserving seam blending and piecewise-linear camera holds
7. Stride-locked streaming video compositing
8. Audio download & music station packaging
"""
import json
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from sqlmodel import Session, select

from workshop.config import ROOT_DIR
from workshop.db.database import get_engine, init_db
from workshop.db.models import (
    Asset,
    Project,
    RenderJob,
    Scene,
)
from workshop.services.pipeline_service import (
    append_video_scene_to_project,
    build_music_station_pipeline,
    extract_sprite_pipeline,
    generate_ai_video_pipeline,
    generate_bridge_pipeline,
)
from workshop.services.project_service import (
    get_project_with_relations,
    import_timeline_toml_to_project,
)
from workshop.services.render_service import run_render_job, start_render_job

# Initialize database schema
init_db()

mcp = MCPServer(
    name="panorama-video-maker",
    description="Full-stack AI video production platform: manage multi-scene projects, tracking shots, asset lineage, bridge synthesis, and background rendering.",
    instructions="Use these tools to plan, structure, generate assets, and render multi-scene videos.",
)


# ==============================================================================
# Project & Timeline Tools
# ==============================================================================

@mcp.tool()
def create_project(
    title: str,
    description: str | None = None,
    canvas_w: int = 640,
    canvas_h: int = 360,
    fps: int = 24,
    bg_color: str = "#ffffff",
    bg_gradient: str | None = None,
    bg_gradient_direction: str = "vertical",
    char_sheet: str | None = None,
    char_scale: float = 0.45,
    floor_y: int = 352,
    stride_pixels: float | None = None,
    action_loop_step: int | None = None,
) -> str:
    """Create a new video editing project in SQLite."""
    engine = get_engine()
    with Session(engine) as session:
        project = Project(
            title=title,
            description=description,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            fps=fps,
            bg_color=bg_color,
            bg_gradient=bg_gradient,
            bg_gradient_direction=bg_gradient_direction,
            char_sheet=char_sheet,
            char_scale=char_scale,
            floor_y=floor_y,
            stride_pixels=stride_pixels,
            action_loop_step=action_loop_step,
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        return json.dumps({
            "status": "success",
            "project_id": project.id,
            "title": project.title,
            "canvas": f"{project.canvas_w}x{project.canvas_h}@{project.fps}fps",
        })


@mcp.tool()
def list_projects(limit: int = 20) -> str:
    """List recent video projects from the database."""
    engine = get_engine()
    with Session(engine) as session:
        statement = select(Project).order_by(Project.created_at.desc()).limit(limit)
        projects = session.exec(statement).all()
        result = [
            {
                "id": p.id,
                "title": p.title,
                "scenes_count": len(p.scenes),
                "created_at": p.created_at.isoformat(),
            }
            for p in projects
        ]
        return json.dumps(result, indent=2)


@mcp.tool()
def get_project_details(project_id: int) -> str:
    """Inspect full project metadata, scene timeline order, and registered assets."""
    engine = get_engine()
    with Session(engine) as session:
        project = get_project_with_relations(session, project_id)
        if not project:
            return json.dumps({"error": f"Project {project_id} not found"})

        scenes = [
            {
                "id": s.id,
                "order_idx": s.order_idx,
                "name": s.name,
                "sheet_path": s.sheet_path,
                "left_bg_path": s.left_bg_path,
                "left_bg_offset": s.left_bg_offset,
                "scale": s.scale,
                "video_scale": s.video_scale,
                "overlap_prev": s.overlap_prev,
                "anim_delay": s.anim_delay,
                "hold_duration": s.hold_duration,
                "hide_actor_during_hold": s.hide_actor_during_hold,
                "hold_frame_segments": s.hold_frame_segments,
                "seam_feather_width": s.seam_feather_width,
            }
            for s in sorted(project.scenes, key=lambda s: s.order_idx)
        ]

        assets = [
            {
                "id": a.id,
                "type": a.asset_type,
                "path": a.file_path,
                "model": a.model_name,
                "prompt": a.prompt,
                "version": a.version,
            }
            for a in project.assets
        ]

        jobs = [
            {
                "id": j.id,
                "status": j.status,
                "progress": j.progress,
                "output_path": j.output_path,
                "error": j.error_message,
            }
            for j in project.render_jobs
        ]

        return json.dumps({
            "id": project.id,
            "title": project.title,
            "description": project.description,
            "canvas": {"width": project.canvas_w, "height": project.canvas_h, "fps": project.fps},
            "scenes": scenes,
            "assets": assets,
            "render_jobs": jobs,
        }, indent=2)


@mcp.tool()
def add_scene(
    project_id: int,
    sheet_path: str,
    name: str = "scene",
    order_idx: int = 0,
    left_bg_path: str | None = None,
    left_bg_offset: int | None = None,
    scale: float = 1.0,
    video_scale: float | None = None,
    video_y: int = 0,
    y: int = 0,
    overlap_prev: int = 0,
    left_feather_l: int | None = None,
    left_feather_r: int | None = None,
    anim_delay: float = 0.0,
    pingpong: bool = False,
    loop_start_frame: int | None = None,
    hold_duration: float = 0.0,
    hide_actor_during_hold: bool = False,
    hold_frame_segments: str | None = None,
    seam_feather_width: int | None = None,
) -> str:
    """Add a scene to a project timeline with complete placement, loop, and hold controls."""
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return json.dumps({"error": f"Project {project_id} not found"})

        scene = Scene(
            project_id=project_id,
            order_idx=order_idx,
            name=name,
            sheet_path=sheet_path,
            left_bg_path=left_bg_path,
            left_bg_offset=left_bg_offset,
            scale=scale,
            video_scale=video_scale,
            video_y=video_y,
            y=y,
            overlap_prev=overlap_prev,
            left_feather_l=left_feather_l,
            left_feather_r=left_feather_r,
            anim_delay=anim_delay,
            pingpong=pingpong,
            loop_start_frame=loop_start_frame,
            hold_duration=hold_duration,
            hide_actor_during_hold=hide_actor_during_hold,
            hold_frame_segments=hold_frame_segments,
            seam_feather_width=seam_feather_width,
        )
        session.add(scene)
        session.commit()
        session.refresh(scene)
        return json.dumps({
            "status": "success",
            "scene_id": scene.id,
            "order_idx": scene.order_idx,
            "name": scene.name,
        })


@mcp.tool()
def set_character_actor(
    project_id: int,
    char_sheet: str,
    char_full_sheet: str | None = None,
    char_scale: float = 0.45,
    floor_y: int = 352,
    stride_pixels: float | None = None,
    action_loop_step: int | None = None,
) -> str:
    """Configure the foreground actor, walk cycle sheet, and camera stride-locking pace."""
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return json.dumps({"error": f"Project {project_id} not found"})

        project.char_sheet = char_sheet
        project.char_full_sheet = char_full_sheet
        project.char_scale = char_scale
        project.floor_y = floor_y
        project.stride_pixels = stride_pixels
        project.action_loop_step = action_loop_step
        project.updated_at = datetime.now(UTC)
        session.add(project)
        session.commit()
        return json.dumps({
            "status": "success",
            "project_id": project_id,
            "char_sheet": char_sheet,
            "stride_pixels": stride_pixels,
        })


@mcp.tool()
def record_asset(
    project_id: int,
    asset_type: str,
    file_path: str,
    model_name: str | None = None,
    prompt: str | None = None,
    version: int = 1,
) -> str:
    """Record a media asset (video, image, sprite) with prompt and model lineage."""
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return json.dumps({"error": f"Project {project_id} not found"})

        asset = Asset(
            project_id=project_id,
            asset_type=asset_type,
            file_path=file_path,
            model_name=model_name,
            prompt=prompt,
            version=version,
        )
        session.add(asset)
        session.commit()
        session.refresh(asset)
        return json.dumps({
            "status": "success",
            "asset_id": asset.id,
            "file_path": asset.file_path,
            "version": asset.version,
        })


# ==============================================================================
# Advanced Pipeline Tools (Extraction, Bridges, AI Video & Music Station)
# ==============================================================================

@mcp.tool()
def extract_sprite_from_video(
    video_path: str,
    output_sheet: str | None = None,
    crop: bool = False,
    keep_frames: bool = False,
    keep_background: bool = False,
    model_name: str | None = None,
    zoom_start: float = 1.0,
    zoom_step: float = 0.0,
) -> str:
    """Extract frames from an MP4 video into an RGBA sprite sheet using ONNX background matting (RMBG / BiRefNet)."""
    try:
        vp = Path(video_path)
        out = Path(output_sheet) if output_sheet else None
        res_path = extract_sprite_pipeline(
            video_path=vp,
            output_sheet=out,
            crop=crop,
            keep_frames=keep_frames,
            keep_background=keep_background,
            model_name=model_name,
            zoom_start=zoom_start,
            zoom_step=zoom_step,
        )
        return json.dumps({
            "status": "success",
            "sprite_sheet": str(res_path),
            "keep_background": keep_background,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def generate_transition_bridge(
    from_sheet: str,
    to_sheet: str,
    from_scale: float = 0.6806,
    to_scale: float = 0.75,
    output_path: str | None = None,
    prompt: str | None = None,
    model: str = "gpt-image-2.5-sunburst-2026-09-08",
    size: str = "1536x512",
    quality: str = "high",
    overwrite: bool = False,
) -> str:
    """Generate a seamless panoramic transition bridge image between two sprite sheets via OpenAI inpainting/outpainting."""
    try:
        out = Path(output_path) if output_path else None
        bridge = generate_bridge_pipeline(
            from_sheet=Path(from_sheet),
            to_sheet=Path(to_sheet),
            from_scale=from_scale,
            to_scale=to_scale,
            output_path=out,
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            overwrite=overwrite,
        )
        return json.dumps({
            "status": "success",
            "bridge_path": str(bridge),
            "model": model,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def generate_ai_video_clip(
    prompt: str,
    output_path: str,
    image_paths: list[str] | None = None,
    model: str = "gemini-omni-1.1-flash",
) -> str:
    """Generate a 10-second keyframe-conditioned animated MP4 clip via Gemini Omni."""
    try:
        imgs = [Path(p) for p in image_paths] if image_paths else None
        res_video = generate_ai_video_pipeline(
            prompt=prompt,
            output_path=Path(output_path),
            image_paths=imgs,
            model=model,
        )
        return json.dumps({
            "status": "success",
            "video_path": str(res_video),
            "model": model,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def append_video_scene(
    project_id: int,
    video_path: str,
    scene_name: str | None = None,
    video_scale: float = 0.75,
    prompt: str | None = None,
    overwrite_bridge: bool = False,
    seam_feather_width: int | None = None,
    hold_duration: float = 0.0,
    hide_actor_during_hold: bool = False,
) -> str:
    """One-Command End-to-End Scene Append:

    Extracts sprite sheet from video, synthesizes transition bridge from the preceding scene via OpenAI,
    calculates scale/offset/anim_delay, and registers the Scene in the database.
    """
    engine = get_engine()
    with Session(engine) as session:
        try:
            scene = append_video_scene_to_project(
                session=session,
                project_id=project_id,
                video_path=Path(video_path),
                scene_name=scene_name,
                video_scale=video_scale,
                prompt=prompt,
                overwrite_bridge=overwrite_bridge,
                seam_feather_width=seam_feather_width,
                hold_duration=hold_duration,
                hide_actor_during_hold=hide_actor_during_hold,
            )
            return json.dumps({
                "status": "success",
                "scene_id": scene.id,
                "name": scene.name,
                "sheet_path": scene.sheet_path,
                "left_bg_path": scene.left_bg_path,
                "anim_delay": scene.anim_delay,
                "left_bg_offset": scene.left_bg_offset,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


@mcp.tool()
def build_music_station_video(
    video_path: str,
    youtube_urls: list[str],
    output_path: str | None = None,
) -> str:
    """Download YouTube audio playlist tracks via yt-dlp, concatenate into 48kHz AAC, and stream-loop the video."""
    try:
        out = Path(output_path) if output_path else None
        res_mp4 = build_music_station_pipeline(
            video_path=Path(video_path),
            youtube_urls=youtube_urls,
            output_path=out,
        )
        return json.dumps({
            "status": "success",
            "music_station_path": str(res_mp4),
            "tracks_count": len(youtube_urls),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ==============================================================================
# Rendering Tools
# ==============================================================================

@mcp.tool()
def render_project_video(project_id: int) -> str:
    """Trigger background video compositing for a project and return a tracking Job ID."""
    engine = get_engine()
    with Session(engine) as session:
        try:
            job = start_render_job(session, project_id)
            import threading
            threading.Thread(target=run_render_job, args=(job.id, ROOT_DIR), daemon=True).start()
            return json.dumps({
                "status": "QUEUED",
                "job_id": job.id,
                "project_id": project_id,
                "message": "Render job started in background.",
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


@mcp.tool()
def check_render_status(job_id: str) -> str:
    """Check status (QUEUED, PROCESSING, COMPLETED, FAILED) and output path of a render job."""
    engine = get_engine()
    with Session(engine) as session:
        job = session.get(RenderJob, job_id)
        if not job:
            return json.dumps({"error": f"Job {job_id} not found"})

        return json.dumps({
            "job_id": job.id,
            "project_id": job.project_id,
            "status": job.status,
            "progress": job.progress,
            "output_path": job.output_path,
            "error_message": job.error_message,
        })


@mcp.tool()
def import_timeline_toml(toml_path: str = "timelines/timeline.example.toml", title: str | None = None) -> str:
    """Import or reload any timeline TOML file into SQLite as a Project with all scenes, bridges, and parameters."""
    engine = get_engine()
    with Session(engine) as session:
        try:
            path = ROOT_DIR / toml_path if not Path(toml_path).is_absolute() else Path(toml_path)
            proj = import_timeline_toml_to_project(path, session, title=title, root_dir=ROOT_DIR)
            return json.dumps({
                "status": "success",
                "project_id": proj.id,
                "title": proj.title,
                "scenes_count": len(proj.scenes),
                "fps": proj.fps,
                "stride_pixels": proj.stride_pixels,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


@mcp.tool()
def render_odyssey_journey(
    journey_toml_path: str = "timelines/cat_odyssey.example.toml",
    output_dir: str = "outputs/odyssey",
) -> str:
    """Run an A-to-Z vertical production for a side-scrolling Cat Odyssey journey.

    Compiles the JourneySpec, renders continuous tracking shots with stride-locking,
    synthesizes material-specific paw footstep foley, generates environmental audio,
    and produces both 16:9 widescreen and 9:16 vertical mobile cuts.
    """
    try:
        from workshop.odyssey.orchestrator import OdysseyOrchestrator
        from workshop.odyssey.spec import JourneySpec

        p = ROOT_DIR / journey_toml_path if not Path(journey_toml_path).is_absolute() else Path(journey_toml_path)
        if not p.exists():
            return json.dumps({"error": f"Journey file {p} not found"})

        out_d = ROOT_DIR / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        journey = JourneySpec.from_toml_file(p)
        orchestrator = OdysseyOrchestrator(journey, root_dir=ROOT_DIR)
        result = orchestrator.run(output_dir=out_d)

        return json.dumps({
            "status": "success",
            "production": result,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _make_project_dir(project_dir: str) -> Path:
    p = Path(project_dir)
    return p if p.is_absolute() else ROOT_DIR / p


@mcp.tool()
def plan_prompt_journey(prompt: str, num_scenes: int = 5, project_dir: str | None = None) -> str:
    """Step 1 of prompt-to-video: plan a journey from one sentence (one cheap LLM call, no images).

    Returns the plan and the image API calls `make_journey` would make, so the cost can be shown
    to the user before anything is generated. Edit `<project_dir>/plan.json` to change the plan.
    """
    try:
        from workshop.make import MakeOptions, MakeProject
        from workshop.make.plan import slugify

        root = _make_project_dir(project_dir or f"outputs/make/{slugify(prompt)[:48]}")
        project = MakeProject(root, MakeOptions(num_scenes=num_scenes))
        plan = project.load_or_plan(prompt)
        project.check_supported(plan)
        return json.dumps({
            "project_dir": str(root),
            "plan": plan.model_dump(mode="json"),
            "pending_image_calls": project.pending_image_calls(plan),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def make_journey(project_dir: str, seconds_per_scene: float = 12.0) -> str:
    """Step 2 of prompt-to-video: generate every asset for a planned journey and render it.

    Runs in the background (several minutes, and it spends image API credits for each pending
    call reported by `plan_prompt_journey`). Poll `get_journey_status` for progress.
    """
    import threading

    from workshop.make import MakeOptions, MakeProject

    root = _make_project_dir(project_dir)
    if not (root / "plan.json").exists():
        return json.dumps({"error": f"No plan.json in {root}; call plan_prompt_journey first"})
    status_path = root / "status.json"

    def write_status(**fields) -> None:
        status_path.write_text(json.dumps({"updated_at": datetime.now(UTC).isoformat(), **fields}, ensure_ascii=False))

    def work() -> None:
        project = MakeProject(
            root,
            MakeOptions(seconds_per_scene=seconds_per_scene),
            log=lambda msg: write_status(status="PROCESSING", step=msg),
        )
        try:
            result = project.run(prompt=None)
            write_status(status="COMPLETED", result=result)
        except Exception as e:
            write_status(status="FAILED", error=str(e))

    write_status(status="QUEUED")
    threading.Thread(target=work, daemon=True).start()
    return json.dumps({"status": "QUEUED", "project_dir": str(root)})


@mcp.tool()
def get_journey_status(project_dir: str) -> str:
    """Check progress of `make_journey` (QUEUED, PROCESSING with current step, COMPLETED, FAILED)."""
    status_path = _make_project_dir(project_dir) / "status.json"
    if not status_path.exists():
        return json.dumps({"error": f"No make_journey run found in {project_dir}"})
    return status_path.read_text()


def main():
    """Run MCP server in stdio mode."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

