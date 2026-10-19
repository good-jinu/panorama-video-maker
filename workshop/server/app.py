"""FastAPI application providing REST APIs for video project editing and asset tracking."""
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

import workshop.services.pipeline_service as pipe_svc
from workshop.config import ROOT_DIR, settings
from workshop.db.database import get_session, init_db
from workshop.db.models import (
    Asset,
    AssetCreate,
    AssetRead,
    Project,
    ProjectCreate,
    ProjectDetailRead,
    ProjectRead,
    ProjectUpdate,
    RenderJob,
    RenderJobRead,
    Scene,
    SceneCreate,
    SceneRead,
    SceneUpdate,
)
from workshop.mcp_server import mcp
from workshop.services.project_service import (
    get_project_with_relations,
    import_timeline_toml_to_project,
)
from workshop.services.render_service import run_render_job, start_render_job

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure standard workspace directories exist
    settings.ensure_directories()
    # Initialize SQLite database schema
    init_db()
    yield


app = FastAPI(
    title="Panorama Video Maker API",
    description="Backend API for managing video projects, multi-scene timelines, generative assets, and render jobs.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount MCP SSE app for live agent tool-calling over HTTP/SSE
app.mount("/mcp", mcp.sse_app())

# Serve web player and rendered output media
web_dir = settings.web_dir
outputs_dir = settings.outputs_dir
outputs_dir.mkdir(parents=True, exist_ok=True)

if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")



@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "time": datetime.now(UTC).isoformat()}


# ==============================================================================
# Projects
# ==============================================================================

@app.post("/api/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED, tags=["Projects"])
def create_new_project(project_in: ProjectCreate, session: Session = Depends(get_session)):
    project = Project.model_validate(project_in)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@app.get("/api/projects", response_model=list[ProjectRead], tags=["Projects"])
def list_projects(
    skip: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session),
):
    projects = session.exec(select(Project).offset(skip).limit(limit)).all()
    return projects


@app.get("/api/projects/{project_id}", response_model=ProjectDetailRead, tags=["Projects"])
def get_project_detail(project_id: int, session: Session = Depends(get_session)):
    project = get_project_with_relations(session, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.put("/api/projects/{project_id}", response_model=ProjectRead, tags=["Projects"])
def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = project_update.model_dump(exclude_unset=True)
    for key, val in update_data.items():
        setattr(project, key, val)

    project.updated_at = datetime.now(UTC)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@app.delete("/api/projects/{project_id}", status_code=status.HTTP_200_OK, tags=["Projects"])
def delete_project(project_id: int, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    session.delete(project)
    session.commit()
    return {"ok": True, "deleted_id": project_id}


# ==============================================================================
# Scenes
# ==============================================================================

@app.post("/api/projects/{project_id}/scenes", response_model=SceneRead, status_code=status.HTTP_201_CREATED, tags=["Scenes"])
def add_scene_to_project(
    project_id: int,
    scene_in: SceneCreate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    scene = Scene.model_validate(scene_in)
    scene.project_id = project_id
    session.add(scene)
    session.commit()
    session.refresh(scene)
    return scene


@app.put("/api/scenes/{scene_id}", response_model=SceneRead, tags=["Scenes"])
def update_scene(
    scene_id: int,
    scene_update: SceneUpdate,
    session: Session = Depends(get_session),
):
    scene = session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    update_data = scene_update.model_dump(exclude_unset=True)
    for key, val in update_data.items():
        setattr(scene, key, val)

    session.add(scene)
    session.commit()
    session.refresh(scene)
    return scene


@app.delete("/api/scenes/{scene_id}", status_code=status.HTTP_200_OK, tags=["Scenes"])
def delete_scene(scene_id: int, session: Session = Depends(get_session)):
    scene = session.get(Scene, scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    session.delete(scene)
    session.commit()
    return {"ok": True, "deleted_id": scene_id}


# ==============================================================================
# Assets (Lineage & Provenance)
# ==============================================================================

@app.post("/api/projects/{project_id}/assets", response_model=AssetRead, status_code=status.HTTP_201_CREATED, tags=["Assets"])
def register_asset(
    project_id: int,
    asset_in: AssetCreate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    asset = Asset.model_validate(asset_in)
    asset.project_id = project_id
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


@app.get("/api/projects/{project_id}/assets", response_model=list[AssetRead], tags=["Assets"])
def list_project_assets(project_id: int, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    statement = select(Asset).where(Asset.project_id == project_id).order_by(Asset.created_at.desc())
    return session.exec(statement).all()


# ==============================================================================
# Render Jobs (Asynchronous execution)
# ==============================================================================

@app.post("/api/projects/{project_id}/render", response_model=RenderJobRead, status_code=status.HTTP_202_ACCEPTED, tags=["Render"])
def trigger_render(
    project_id: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    try:
        job = start_render_job(session, project_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Trigger background thread execution
    background_tasks.add_task(run_render_job, job.id, ROOT_DIR)
    return job


@app.get("/api/jobs/{job_id}", response_model=RenderJobRead, tags=["Render"])
def get_job_status(job_id: str, session: Session = Depends(get_session)):
    job = session.get(RenderJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


@app.get("/api/jobs", response_model=list[RenderJobRead], tags=["Render"])
def list_jobs(skip: int = 0, limit: int = 50, session: Session = Depends(get_session)):
    jobs = session.exec(select(RenderJob).order_by(RenderJob.created_at.desc()).offset(skip).limit(limit)).all()
    return jobs


# ==============================================================================
# Pipeline Operations
# ==============================================================================

class ExtractSpriteRequest(BaseModel):
    video_path: str
    output_sheet: str | None = None
    crop: bool = False
    keep_frames: bool = False
    keep_background: bool = False
    model_name: str | None = None
    zoom_start: float = 1.0
    zoom_step: float = 0.0


@app.post("/api/pipeline/extract-sprite", tags=["Pipeline"])
def api_extract_sprite(req: ExtractSpriteRequest):
    try:
        res = pipe_svc.extract_sprite_pipeline(
            video_path=Path(req.video_path),
            output_sheet=Path(req.output_sheet) if req.output_sheet else None,
            crop=req.crop,
            keep_frames=req.keep_frames,
            keep_background=req.keep_background,
            model_name=req.model_name,
            zoom_start=req.zoom_start,
            zoom_step=req.zoom_step,
        )
        return {"status": "success", "sprite_sheet": str(res)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class GenerateBridgeRequest(BaseModel):
    from_sheet: str
    to_sheet: str
    from_scale: float = 0.6806
    to_scale: float = 0.75
    output_path: str | None = None
    prompt: str | None = None
    model: str = "gpt-image-2.5-sunburst-2026-09-08"
    size: str = "1536x512"
    quality: str = "high"
    overwrite: bool = False


@app.post("/api/pipeline/generate-bridge", tags=["Pipeline"])
def api_generate_bridge(req: GenerateBridgeRequest):
    try:
        bridge = pipe_svc.generate_bridge_pipeline(
            from_sheet=Path(req.from_sheet),
            to_sheet=Path(req.to_sheet),
            from_scale=req.from_scale,
            to_scale=req.to_scale,
            output_path=Path(req.output_path) if req.output_path else None,
            prompt=req.prompt,
            model=req.model,
            size=req.size,
            quality=req.quality,
            overwrite=req.overwrite,
        )
        return {"status": "success", "bridge_path": str(bridge)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class GenerateAIVideoRequest(BaseModel):
    prompt: str
    output_path: str
    image_paths: list[str] | None = None
    model: str = "gemini-omni-1.1-flash"


@app.post("/api/pipeline/generate-ai-video", tags=["Pipeline"])
def api_generate_ai_video(req: GenerateAIVideoRequest):
    try:
        imgs = [Path(p) for p in req.image_paths] if req.image_paths else None
        video = pipe_svc.generate_ai_video_pipeline(
            prompt=req.prompt,
            output_path=Path(req.output_path),
            image_paths=imgs,
            model=req.model,
        )
        return {"status": "success", "video_path": str(video)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class AddVideoSceneRequest(BaseModel):
    video_path: str
    scene_name: str | None = None
    video_scale: float = 0.75
    prompt: str | None = None
    overwrite_bridge: bool = False
    seam_feather_width: int | None = None
    hold_duration: float = 0.0
    hide_actor_during_hold: bool = False


@app.post("/api/projects/{project_id}/add-video-scene", response_model=SceneRead, tags=["Pipeline"])
def api_add_video_scene(
    project_id: int,
    req: AddVideoSceneRequest,
    session: Session = Depends(get_session),
):
    try:
        scene = pipe_svc.append_video_scene_to_project(
            session=session,
            project_id=project_id,
            video_path=Path(req.video_path),
            scene_name=req.scene_name,
            video_scale=req.video_scale,
            prompt=req.prompt,
            overwrite_bridge=req.overwrite_bridge,
            seam_feather_width=req.seam_feather_width,
            hold_duration=req.hold_duration,
            hide_actor_during_hold=req.hide_actor_during_hold,
        )
        return scene
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MusicStationRequest(BaseModel):
    video_path: str
    youtube_urls: list[str]
    output_path: str | None = None


@app.post("/api/pipeline/music-station", tags=["Pipeline"])
def api_music_station(req: MusicStationRequest):
    try:
        out = Path(req.output_path) if req.output_path else None
        res = pipe_svc.build_music_station_pipeline(
            video_path=Path(req.video_path),
            youtube_urls=req.youtube_urls,
            output_path=out,
        )
        return {"status": "success", "music_station_path": str(res)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# Timeline TOML Import
# ==============================================================================

class TimelineImportRequest(BaseModel):
    toml_path: str = "timelines/timeline.example.toml"
    title: str | None = None


@app.post("/api/timeline/import", tags=["Projects"])
def api_import_timeline(req: TimelineImportRequest, session: Session = Depends(get_session)):
    """Import or reload any timeline TOML file into SQLite as a Project and Scenes."""
    try:
        p = import_timeline_toml_to_project(ROOT_DIR / req.toml_path, session, title=req.title, root_dir=ROOT_DIR)
        return {
            "status": "success",
            "project_id": p.id,
            "title": p.title,
            "scenes_count": len(p.scenes),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


