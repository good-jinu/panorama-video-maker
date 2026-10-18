import json
from pathlib import Path
from typing import Optional, Union
import tomllib

from sqlmodel import Session, select

from workshop.db.models import Asset, Project, ProjectCreate, RenderJob, Scene
from workshop.timeline import SceneSpec, TimelineConfig, build_timeline


def create_project(session: Session, project_in: ProjectCreate) -> Project:
    project = Project.model_validate(project_in)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def get_project_with_relations(session: Session, project_id: int) -> Optional[Project]:
    statement = select(Project).where(Project.id == project_id)
    return session.exec(statement).first()


def project_to_timeline_config(project: Project, root_dir: Optional[Path] = None) -> TimelineConfig:
    """Transform database Project and attached Scenes into the engine's TimelineConfig."""
    root = root_dir or Path.cwd()

    # Parse gradient colors if stored as JSON or comma-separated
    bg_gradient = None
    if project.bg_gradient:
        try:
            bg_gradient = json.loads(project.bg_gradient)
        except Exception:
            bg_gradient = [c.strip() for c in project.bg_gradient.split(",") if c.strip()]

    scenes_spec: list[SceneSpec] = []
    # Sort scenes by order_idx
    sorted_scenes = sorted(project.scenes, key=lambda s: s.order_idx)

    for sc in sorted_scenes:
        left_feather = None
        if sc.left_feather_l is not None and sc.left_feather_r is not None:
            left_feather = (sc.left_feather_l, sc.left_feather_r)

        scenes_spec.append(
            SceneSpec(
                id=sc.name or f"scene_{sc.order_idx}",
                sheet=Path(sc.sheet_path),
                left_bg=Path(sc.left_bg_path) if sc.left_bg_path else None,
                left_bg_offset=sc.left_bg_offset,
                left_bg_scale=sc.left_bg_scale,
                video_scale=sc.video_scale,
                video_y=sc.video_y,
                scale=sc.scale,
                y=sc.y,
                duration=sc.duration,
                gap_after=sc.gap_after,
                overlap_prev=sc.overlap_prev,
                left_feather=left_feather,
                parallax=sc.parallax,
                pingpong=sc.pingpong,
                loop_start_frame=sc.loop_start_frame,
                anim_delay=sc.anim_delay,
                hold_duration=sc.hold_duration,
                hide_actor_during_hold=sc.hide_actor_during_hold,
                hold_frame_segments=json.loads(sc.hold_frame_segments) if sc.hold_frame_segments else None,
                seam_feather_width=sc.seam_feather_width,
            )
        )

    char_sheet_path = Path(project.char_sheet) if project.char_sheet else Path("outputs/character/sprite.png")
    char_full_sheet = Path(project.char_full_sheet) if project.char_full_sheet else None

    config = TimelineConfig(
        canvas_w=project.canvas_w,
        canvas_h=project.canvas_h,
        fps=project.fps,
        bg_color=project.bg_color,
        bg_gradient=bg_gradient,
        bg_gradient_direction=project.bg_gradient_direction,
        char_sheet=char_sheet_path,
        char_full_sheet=char_full_sheet,
        char_scale=project.char_scale,
        floor_y=project.floor_y,
        stride_pixels=project.stride_pixels,
        action_loop_step=project.action_loop_step,
        shadow=project.shadow,
        shadow_opacity=project.shadow_opacity,
        scenes=scenes_spec,
    )

    if scenes_spec:
        build_timeline(config, root_dir=root)

    return config


def import_timeline_toml_to_project(
    toml_path: Union[str, Path],
    session: Session,
    title: Optional[str] = None,
    root_dir: Optional[Path] = None,
) -> Project:
    """Import or sync a timeline TOML file (e.g. timelines/timeline.example.toml) into SQLite database."""
    root = root_dir or Path.cwd()
    toml_p = Path(toml_path)
    if not toml_p.is_absolute():
        toml_p = root / toml_p

    if not toml_p.exists():
        raise FileNotFoundError(f"Timeline TOML file not found: {toml_p}")

    data = tomllib.loads(toml_p.read_text())
    cam = data.get("camera", {})
    char = data.get("character", {})

    proj_title = (
        title
        or data.get("title")
        or toml_p.name.replace(".example.toml", "").replace(".toml", "").replace("-", " ").replace(".", " ").title()
    )

    # Check if project already exists
    project = session.exec(select(Project).where(Project.title == proj_title)).first()
    if not project:
        project = Project(
            title=proj_title,
            description="Continuous multi-scene tracking and compositing video project",
            canvas_w=int(cam.get("width", 640)),
            canvas_h=int(cam.get("height", 360)),
            fps=int(cam.get("fps", 24)),
            bg_color=cam.get("bg_color", "#ffffff"),
            bg_gradient=json.dumps(cam["bg_gradient"]) if cam.get("bg_gradient") else None,
            bg_gradient_direction=cam.get("bg_gradient_direction", "vertical"),
            char_sheet=char.get("sheet", "outputs/character/sprite.png"),
            char_full_sheet=char.get("full_sheet"),
            char_scale=float(char.get("scale", 0.45)),
            floor_y=int(char.get("floor_y", 352)),
            stride_pixels=float(char["stride_pixels"]) if char.get("stride_pixels") is not None else None,
            action_loop_step=int(char["action_loop_step"]) if char.get("action_loop_step") else None,
            shadow=bool(char.get("shadow", True)),
            shadow_opacity=float(char.get("shadow_opacity", 0.35)),
        )
        session.add(project)
        session.commit()
        session.refresh(project)
    else:
        project.canvas_w = int(cam.get("width", 640))
        project.canvas_h = int(cam.get("height", 360))
        project.fps = int(cam.get("fps", 24))
        project.bg_color = cam.get("bg_color", "#ffffff")
        if cam.get("bg_gradient"):
            project.bg_gradient = json.dumps(cam["bg_gradient"])
        project.bg_gradient_direction = cam.get("bg_gradient_direction", "vertical")
        if char.get("sheet"):
            project.char_sheet = char.get("sheet")
        if char.get("full_sheet"):
            project.char_full_sheet = char.get("full_sheet")
        project.char_scale = float(char.get("scale", 0.45))
        project.floor_y = int(char.get("floor_y", 352))
        if char.get("stride_pixels") is not None:
            project.stride_pixels = float(char["stride_pixels"])
        if char.get("action_loop_step") or char.get("meow_loop_step"):
            project.action_loop_step = int(char.get("action_loop_step") or char.get("meow_loop_step"))
        project.shadow = bool(char.get("shadow", True))
        project.shadow_opacity = float(char.get("shadow_opacity", 0.35))
        session.commit()
        session.refresh(project)

    # Sync scenes: delete existing scenes to match TOML order exactly
    for old_scene in list(project.scenes):
        session.delete(old_scene)
    session.commit()

    # Re-insert scenes from TOML
    for idx, sc in enumerate(data.get("scenes", [])):
        left_feather = sc.get("left_feather")
        l_f_l = left_feather[0] if left_feather and len(left_feather) > 0 else None
        l_f_r = left_feather[1] if left_feather and len(left_feather) > 1 else None

        bg_scale = sc.get("left_bg_scale")
        if bg_scale is None:
            bg_scale = sc.get("bg_scale")

        v_y = sc.get("video_y", sc.get("video_valign", 0))
        try:
            v_y = int(v_y)
        except Exception:
            v_y = 0

        segments = sc.get("hold_frame_segments")
        hold_segments_json = json.dumps(segments) if segments else None

        scene = Scene(
            project_id=project.id,
            order_idx=idx,
            name=sc.get("id", f"scene_{idx}"),
            sheet_path=sc["sheet"],
            left_bg_path=sc.get("left_bg"),
            left_bg_offset=int(sc["left_bg_offset"]) if sc.get("left_bg_offset") is not None else None,
            left_bg_scale=float(bg_scale) if bg_scale is not None else None,
            video_scale=float(sc["video_scale"]) if sc.get("video_scale") is not None else None,
            video_y=v_y,
            scale=float(sc.get("scale", 1.0)),
            y=int(sc.get("y", 0)),
            duration=float(sc["duration"]) if sc.get("duration") is not None else None,
            gap_after=float(sc.get("gap_after", 0.0)),
            overlap_prev=int(sc.get("overlap_prev", 0)),
            left_feather_l=l_f_l,
            left_feather_r=l_f_r,
            parallax=float(sc.get("parallax", 1.0)),
            pingpong=bool(sc.get("pingpong", False)),
            loop_start_frame=int(sc["loop_start_frame"]) if sc.get("loop_start_frame") is not None else None,
            anim_delay=float(sc.get("anim_delay", 0.0)),
            hold_duration=float(sc.get("hold_duration", 0.0)),
            hide_actor_during_hold=bool(sc.get("hide_actor_during_hold", False)),
            hold_frame_segments=hold_segments_json,
            seam_feather_width=int(sc["seam_feather_width"]) if sc.get("seam_feather_width") is not None else None,
        )
        session.add(scene)

    session.commit()
    session.refresh(project)

    session.refresh(project)
    return project
