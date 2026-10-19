"""SQLModel data schemas for projects, timeline scenes, media assets, and render jobs."""
from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Field, Relationship, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


# ==============================================================================
# Project
# ==============================================================================

class ProjectBase(SQLModel):
    title: str = Field(index=True, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    canvas_w: int = Field(default=640)
    canvas_h: int = Field(default=360)
    fps: int = Field(default=24)
    bg_color: str = Field(default="#ffffff")
    bg_gradient: str | None = Field(default=None)  # JSON or comma-separated hex colors
    bg_gradient_direction: str = Field(default="vertical")

    # Character actor configuration
    char_sheet: str | None = Field(default=None)
    char_full_sheet: str | None = Field(default=None)
    char_scale: float = Field(default=0.45)
    floor_y: int = Field(default=352)
    stride_pixels: float | None = Field(default=None)
    action_loop_step: int | None = Field(default=None)
    shadow: bool = Field(default=True)
    shadow_opacity: float = Field(default=0.35)


class Project(ProjectBase, table=True):
    __tablename__ = "projects"

    id: int | None = Field(default=None, primary_key=True)
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
    title: str | None = None
    description: str | None = None
    canvas_w: int | None = None
    canvas_h: int | None = None
    fps: int | None = None
    bg_color: str | None = None
    bg_gradient: str | None = None
    bg_gradient_direction: str | None = None
    char_sheet: str | None = None
    char_full_sheet: str | None = None
    char_scale: float | None = None
    floor_y: int | None = None
    stride_pixels: float | None = None
    action_loop_step: int | None = None
    shadow: bool | None = None
    shadow_opacity: float | None = None


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
    left_bg_path: str | None = Field(default=None, description="Path to transition bridge or left background")
    left_bg_offset: int | None = Field(default=None)
    left_bg_scale: float | None = Field(default=None)
    video_scale: float | None = Field(default=None)
    video_y: int = Field(default=0)
    scale: float = Field(default=1.0)
    y: int = Field(default=0)
    duration: float | None = Field(default=None)
    gap_after: float = Field(default=0.0)
    overlap_prev: int = Field(default=0)
    left_feather_l: int | None = Field(default=None)
    left_feather_r: int | None = Field(default=None)
    parallax: float = Field(default=1.0)
    pingpong: bool = Field(default=False)
    loop_start_frame: int | None = Field(default=None)
    anim_delay: float = Field(default=0.0)
    hold_duration: float = Field(default=0.0)
    hide_actor_during_hold: bool = Field(default=False)
    hold_frame_segments: str | None = Field(default=None, description="JSON encoded list of [start, end, repeats]")
    seam_feather_width: int | None = Field(default=None, description="Narrow left feather width overriding DP seam")


class Scene(SceneBase, table=True):
    __tablename__ = "scenes"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)

    project: Project | None = Relationship(back_populates="scenes")


class SceneCreate(SceneBase):
    pass


class SceneUpdate(SQLModel):
    order_idx: int | None = None
    name: str | None = None
    sheet_path: str | None = None
    left_bg_path: str | None = None
    left_bg_offset: int | None = None
    left_bg_scale: float | None = None
    video_scale: float | None = None
    video_y: int | None = None
    scale: float | None = None
    y: int | None = None
    duration: float | None = None
    gap_after: float | None = None
    overlap_prev: int | None = None
    left_feather_l: int | None = None
    left_feather_r: int | None = None
    parallax: float | None = None
    pingpong: bool | None = None
    loop_start_frame: int | None = None
    anim_delay: float | None = None
    hold_duration: float | None = None
    hide_actor_during_hold: bool | None = None
    hold_frame_segments: str | None = None
    seam_feather_width: int | None = None


class SceneRead(SceneBase):
    id: int
    created_at: datetime


# ==============================================================================
# Asset (Lineage & Provenance Tracking)
# ==============================================================================

class AssetBase(SQLModel):
    project_id: int | None = Field(default=None, foreign_key="projects.id", index=True)
    asset_type: str = Field(description="video, image, sprite, audio", index=True)
    file_path: str = Field(description="Path to stored asset file")
    version: int = Field(default=1)
    model_name: str | None = Field(default=None, description="e.g. gpt-image-2.5, gemini-omni-1.1-flash, birefnet")
    prompt: str | None = Field(default=None, description="Generation prompt or source market data summary")
    source_metadata: str | None = Field(default=None, description="JSON string with custom params or provenance")


class Asset(AssetBase, table=True):
    __tablename__ = "assets"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)

    project: Project | None = Relationship(back_populates="assets")


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
    output_path: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class RenderJob(RenderJobBase, table=True):
    __tablename__ = "render_jobs"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = Field(default=None)

    project: Project | None = Relationship(back_populates="render_jobs")


class RenderJobRead(RenderJobBase):
    id: str
    created_at: datetime
    completed_at: datetime | None


# Combined schema for rich Project inspection
class ProjectDetailRead(ProjectRead):
    scenes: list[SceneRead] = []
    assets: list[AssetRead] = []
    render_jobs: list[RenderJobRead] = []
