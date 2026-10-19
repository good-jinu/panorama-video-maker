"""REST API for prompt-to-video projects (used by the chat app in `chat/`).

Projects live in `outputs/make/<project_id>/`; see `workshop.make.pipeline` for the folder layout.
"""
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from workshop.config import settings
from workshop.make import MakeProject
from workshop.make.plan import (
    MAX_SCENES,
    MIN_SCENES,
    JourneyPlan,
    PlannedActor,
    PlannedScene,
    slugify,
)

router = APIRouter(prefix="/api/make", tags=["Make"])

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _projects_dir() -> Path:
    return settings.outputs_dir / "make"


def _open(project_id: str) -> MakeProject:
    if not _ID_RE.match(project_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid project id {project_id!r}")
    project = MakeProject(_projects_dir() / project_id)
    if not project.plan_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Project {project_id!r} not found")
    return project


def _media_url(path: str | None) -> str | None:
    """Map a rendered file to its `/outputs/...` URL (served by the static mount in app.py)."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(settings.outputs_dir.resolve())
    except ValueError:
        return None
    return f"/outputs/{rel.as_posix()}"


def _status_view(project: MakeProject) -> dict[str, Any] | None:
    st = project.read_status()
    if st and st.get("status") == "COMPLETED":
        result = st.get("result", {})
        st["videos"] = {
            "16:9": _media_url(result.get("widescreen_16_9")),
            "9:16": _media_url(result.get("vertical_9_16")),
        }
        st["duration"] = result.get("duration")
    return st


def _view(project: MakeProject, **extra: Any) -> dict[str, Any]:
    plan = project.load_plan()
    return {
        "project_id": project.root.name,
        "plan": plan.model_dump(mode="json"),
        "scene_ids": [s.id for s in plan.scenes],
        "pending_image_calls": project.pending_image_calls(plan),
        "options": {
            "seconds_per_scene": project.options.seconds_per_scene,
            "actor_size": project.options.actor_size,
        },
        "status": _status_view(project),
        **extra,
    }


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=1)
    num_scenes: int = Field(default=5, ge=MIN_SCENES, le=MAX_SCENES)


class PlanUpdate(BaseModel):
    title: str
    style: str
    actor: PlannedActor
    scenes: list[PlannedScene]


class RedoRequest(BaseModel):
    target: str = Field(description='"actor" or a scene id')


class OptionsUpdate(BaseModel):
    seconds_per_scene: float | None = Field(default=None, ge=2, le=120)
    actor_size: float | None = Field(default=None, ge=0.3, le=3)


@router.post("/plan", status_code=status.HTTP_201_CREATED)
def create_plan(req: PlanRequest) -> dict[str, Any]:
    """Plan a new project from a prompt (one LLM call, no images)."""
    base = slugify(req.prompt)[:48].strip("-") or "journey"
    project_id, n = base, 2
    while (_projects_dir() / project_id).exists():
        project_id, n = f"{base}-{n}", n + 1

    project = MakeProject(_projects_dir() / project_id)
    project.options.num_scenes = req.num_scenes
    try:
        plan = project.load_or_plan(req.prompt)
        project.check_supported(plan)
    except NotImplementedError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Planner returned an invalid plan: {e}")
    return _view(project)


@router.get("/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return _view(_open(project_id))


@router.put("/{project_id}/plan")
def update_plan(project_id: str, update: PlanUpdate) -> dict[str, Any]:
    """Replace the plan; cached images that no longer match are deleted so the next run remakes them."""
    project = _open(project_id)
    if project.is_running():
        raise HTTPException(status.HTTP_409_CONFLICT, "Wait for the current run to finish")
    try:
        new = JourneyPlan(prompt=project.load_plan().prompt, **update.model_dump())
        project.check_supported(new)
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e))
    removed = project.update_plan(new)
    return _view(project, removed=removed)


@router.post("/{project_id}/redo")
def redo(project_id: str, req: RedoRequest) -> dict[str, Any]:
    project = _open(project_id)
    if project.is_running():
        raise HTTPException(status.HTTP_409_CONFLICT, "Wait for the current run to finish")
    try:
        removed = project.redo(req.target)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e))
    return _view(project, removed=removed)


@router.put("/{project_id}/options")
def update_options(project_id: str, update: OptionsUpdate) -> dict[str, Any]:
    project = _open(project_id)
    if update.seconds_per_scene is not None:
        project.options.seconds_per_scene = update.seconds_per_scene
    if update.actor_size is not None:
        project.options.actor_size = update.actor_size
    project.save_options()
    return _view(project)


@router.post("/{project_id}/run", status_code=status.HTTP_202_ACCEPTED)
def run(project_id: str) -> dict[str, Any]:
    """Generate missing images and render, in the background. Poll `/status`."""
    project = _open(project_id)
    try:
        project.start_background_run()
    except RuntimeError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    except NotImplementedError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e))
    return _view(project)


@router.get("/{project_id}/status")
def get_status(project_id: str) -> dict[str, Any]:
    project = _open(project_id)
    return _status_view(project) or {"status": "NOT_STARTED"}
