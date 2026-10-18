"""SQLModel data schemas for projects, timeline scenes, media assets, and render jobs."""
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, Relationship, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ==============================================================================
# Project
# ==============================================================================

class ProjectBase(SQLModel):
    title: str = Field(index=True, max_length=128)
    description: Optional[str] = Field(default=None, max_length=512)
    canvas_w: int = Field(default=640)
    canvas_h: int = Field(default=360)
    fps: int = Field(default=24)
    bg_color: str = Field(default="#ffffff")
    bg_gradient: Optional[str] = Field(default=None)  # JSON or comma-separated hex colors
    bg_gradient_direction: str = Field(default="vertical")

    # Character actor configuration
    char_sheet: Optional[str] = Field(default=None)
    char_full_sheet: Optional[str] = Field(default=None)
    char_scale: float = Field(default=0.45)
    floor_y: int = Field(default=352)
    stride_pixels: Optional[float] = Field(default=None)
    action_loop_step: Optional[int] = Field(default=None)
    shadow: bool = Field(default=True)
    shadow_opacity: float = Field(default=0.35)


class Project(ProjectBase, table=True):
    __tablename__ = "projects"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    scenes: list["Scene"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "Scene.order_idx"},
    )
    assets: list["Asset"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    render_jobs: list["RenderJob"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    canvas_w: Optional[int] = None
    canvas_h: Optional[int] = None
    fps: Optional[int] = None
    bg_color: Optional[str] = None
    bg_gradient: Optional[str] = None
    bg_gradient_direction: Optional[str] = None
    char_sheet: Optional[str] = None
    char_full_sheet: Optional[str] = None
    char_scale: Optional[float] = None
    floor_y: Optional[int] = None
    stride_pixels: Optional[float] = None
    action_loop_step: Optional[int] = None
    shadow: Optional[bool] = None
    shadow_opacity: Optional[float] = None


class ProjectRead(ProjectBase):
    id: int
    created_at: datetime
    updated_at: datetime


# ==============================================================================
# Scene
# ==============================================================================

class SceneBase(SQLModel):
    project_id: int = Field(foreign_key="projects.id", index=True)
    order_idx: int = Field(default=0, index=True)
    name: str = Field(default="scene", max_length=64)
    sheet_path: str = Field(description="Path to scene sprite sheet or video")
    left_bg_path: Optional[str] = Field(default=None, description="Path to transition bridge or left background")
    left_bg_offset: Optional[int] = Field(default=None)
    left_bg_scale: Optional[float] = Field(default=None)
    video_scale: Optional[float] = Field(default=None)
    video_y: int = Field(default=0)
    scale: float = Field(default=1.0)
    y: int = Field(default=0)
    duration: Optional[float] = Field(default=None)
    gap_after: float = Field(default=0.0)
    overlap_prev: int = Field(default=0)
    left_feather_l: Optional[int] = Field(default=None)
    left_feather_r: Optional[int] = Field(default=None)
    parallax: float = Field(default=1.0)
    pingpong: bool = Field(default=False)
    loop_start_frame: Optional[int] = Field(default=None)
    anim_delay: float = Field(default=0.0)
    hold_duration: float = Field(default=0.0)
    hide_actor_during_hold: bool = Field(default=False)
    hold_frame_segments: Optional[str] = Field(default=None, description="JSON encoded list of [start, end, repeats]")
    seam_feather_width: Optional[int] = Field(default=None, description="Narrow left feather width overriding DP seam")


class Scene(SceneBase, table=True):
    __tablename__ = "scenes"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)

    project: Optional[Project] = Relationship(back_populates="scenes")


class SceneCreate(SceneBase):
    pass


class SceneUpdate(SQLModel):
    order_idx: Optional[int] = None
    name: Optional[str] = None
    sheet_path: Optional[str] = None
    left_bg_path: Optional[str] = None
    left_bg_offset: Optional[int] = None
    left_bg_scale: Optional[float] = None
    video_scale: Optional[float] = None
    video_y: Optional[int] = None
    scale: Optional[float] = None
    y: Optional[int] = None
    duration: Optional[float] = None
    gap_after: Optional[float] = None
    overlap_prev: Optional[int] = None
    left_feather_l: Optional[int] = None
    left_feather_r: Optional[int] = None
    parallax: Optional[float] = None
    pingpong: Optional[bool] = None
    loop_start_frame: Optional[int] = None
    anim_delay: Optional[float] = None
    hold_duration: Optional[float] = None
    hide_actor_during_hold: Optional[bool] = None
    hold_frame_segments: Optional[str] = None
    seam_feather_width: Optional[int] = None


class SceneRead(SceneBase):
    id: int
    created_at: datetime


# ==============================================================================
# Asset (Lineage & Provenance Tracking)
# ==============================================================================

class AssetBase(SQLModel):
    project_id: Optional[int] = Field(default=None, foreign_key="projects.id", index=True)
    asset_type: str = Field(description="video, image, sprite, audio", index=True)
    file_path: str = Field(description="Path to stored asset file")
    version: int = Field(default=1)
    model_name: Optional[str] = Field(default=None, description="e.g. gpt-image-2.5, gemini-omni-1.1-flash, birefnet")
    prompt: Optional[str] = Field(default=None, description="Generation prompt or source market data summary")
    source_metadata: Optional[str] = Field(default=None, description="JSON string with custom params or provenance")


class Asset(AssetBase, table=True):
    __tablename__ = "assets"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)

    project: Optional[Project] = Relationship(back_populates="assets")


class AssetCreate(AssetBase):
    pass


class AssetRead(AssetBase):
    id: int
    created_at: datetime


# ==============================================================================
# RenderJob (Long-running asynchronous execution state)
# ==============================================================================

class RenderJobBase(SQLModel):
    project_id: int = Field(foreign_key="projects.id", index=True)
    status: str = Field(default="QUEUED", index=True)  # QUEUED, PROCESSING, COMPLETED, FAILED
    progress: float = Field(default=0.0)  # 0.0 to 1.0
    output_path: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)


class RenderJob(RenderJobBase, table=True):
    __tablename__ = "render_jobs"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = Field(default=None)

    project: Optional[Project] = Relationship(back_populates="render_jobs")


class RenderJobRead(RenderJobBase):
    id: str
    created_at: datetime
    completed_at: Optional[datetime]


# Combined schema for rich Project inspection
class ProjectDetailRead(ProjectRead):
    scenes: list[SceneRead] = []
    assets: list[AssetRead] = []
    render_jobs: list[RenderJobRead] = []
