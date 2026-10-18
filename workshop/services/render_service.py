"""Asynchronous render job execution and lifecycle management."""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Session

from workshop.db.database import get_engine
from workshop.db.models import Asset, Project, RenderJob
from workshop.long_composite import render_long_animation
from workshop.services.project_service import (
    get_project_with_relations,
    project_to_timeline_config,
)

logger = logging.getLogger(__name__)


def start_render_job(session: Session, project_id: int) -> RenderJob:
    """Register a new render job in QUEUED status."""
    project = session.get(Project, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    job = RenderJob(
        project_id=project_id,
        status="QUEUED",
        progress=0.0,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def run_render_job(job_id: str, root_dir: Optional[Path] = None) -> None:
    """Execute the render job synchronously or inside a background thread/task."""
    engine = get_engine()
    root = root_dir or Path.cwd()

    with Session(engine) as session:
        job = session.get(RenderJob, job_id)
        if not job:
            logger.error("Job %s not found", job_id)
            return

        job.status = "PROCESSING"
        job.progress = 0.1
        session.commit()

        try:
            project = get_project_with_relations(session, job.project_id)
            if not project:
                raise ValueError(f"Project {job.project_id} not found")

            if not project.scenes:
                raise ValueError(f"Project {job.project_id} has no scenes configured")

            out_dir = root / "outputs" / "rendered"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"project_{project.id}_{job_id[:8]}.mp4"

            job.progress = 0.3
            session.commit()

            timeline_config = project_to_timeline_config(project, root_dir=root)

            # Render the multi-scene animation
            render_long_animation(
                timeline_config,
                output_path=output_path,
                root_dir=root,
                make_gif=False,
            )

            job.status = "COMPLETED"
            job.progress = 1.0
            job.output_path = str(output_path.relative_to(root)) if output_path.is_relative_to(root) else str(output_path)
            job.completed_at = datetime.now(timezone.utc)

            # Register as an Asset for lineage & provenance tracking
            asset = Asset(
                project_id=project.id,
                asset_type="video",
                file_path=job.output_path,
                version=len(project.assets) + 1,
                model_name="workshop.composite",
                prompt=f"Rendered video for project: {project.title}",
            )
            session.add(asset)
            session.commit()

        except Exception as exc:
            logger.exception("Render job %s failed: %s", job_id, exc)
            job.status = "FAILED"
            job.error_message = str(exc)
            job.completed_at = datetime.now(timezone.utc)
            session.commit()
