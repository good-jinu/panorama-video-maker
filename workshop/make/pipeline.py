"""Resumable prompt-to-video pipeline.

Every step writes its result into the project folder and is skipped when that file already exists,
so a re-run after a failure (or after editing `plan.json`) only pays for what is missing:

    <project>/
      plan.json                 planner output (edit it and re-run to change the journey)
      options.json              render settings from the last run (speed, actor size)
      status.json               progress of a background run
      actor/source.png          generated actor image
      actor/sprite.png/.json    animated loop
      scenes/<id>.png           3H x H scene backgrounds
      bridges/<a>--<b>.png      3H x H transitions
      world/segment_<n>.png     stitched world, cut per scene
      journey.toml              JourneySpec for the existing odyssey renderer
      render/                   16:9 and 9:16 videos
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from workshop.make import assets
from workshop.make.plan import ActorKind, JourneyPlan, Planner, plan_journey
from workshop.odyssey.spec import (
    DirectorSpec,
    HeroActorSpec,
    JourneySceneSpec,
    JourneySpec,
    SoundscapeSpec,
)


@dataclass
class MakeOptions:
    num_scenes: int = 5
    seconds_per_scene: float = 12.0
    canvas_h: int = 360
    fps: int = 24
    output_formats: tuple[str, ...] = ("16:9", "9:16")
    actor_size: float = 1.0  # multiplier on the default actor height


class MakeProject:
    def __init__(
        self,
        root: Path,
        options: MakeOptions | None = None,
        images: assets.ImageBackend | None = None,
        planner: Planner | None = None,
        log: Callable[[str], None] = print,
    ):
        self.root = Path(root).resolve()
        self.options = options or self.saved_options() or MakeOptions()
        self.images = images or assets.OpenAIImageBackend()
        self.planner = planner
        self.log = log

    # -- paths -------------------------------------------------------------
    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def actor_source_path(self) -> Path:
        return self.root / "actor" / "source.png"

    @property
    def actor_sheet_path(self) -> Path:
        return self.root / "actor" / "sprite.png"

    def scene_path(self, scene_id: str) -> Path:
        return self.root / "scenes" / f"{scene_id}.png"

    def bridge_path(self, from_id: str, to_id: str) -> Path:
        return self.root / "bridges" / f"{from_id}--{to_id}.png"

    def segment_path(self, index: int) -> Path:
        return self.root / "world" / f"segment_{index:02d}.png"

    @property
    def journey_path(self) -> Path:
        return self.root / "journey.toml"

    @property
    def options_path(self) -> Path:
        return self.root / "options.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    def saved_options(self) -> MakeOptions | None:
        if not self.options_path.exists():
            return None
        data = json.loads(self.options_path.read_text())
        data["output_formats"] = tuple(data.get("output_formats", MakeOptions.output_formats))
        return MakeOptions(**data)

    def save_options(self) -> None:
        self.options_path.parent.mkdir(parents=True, exist_ok=True)
        self.options_path.write_text(json.dumps(asdict(self.options), indent=2))

    def _rel(self, path: Path) -> str:
        """Journey paths are relative to the project folder so the folder can be moved."""
        return path.relative_to(self.root).as_posix()

    # -- steps -------------------------------------------------------------
    def load_or_plan(self, prompt: str | None) -> JourneyPlan:
        if self.plan_path.exists():
            plan = JourneyPlan.model_validate_json(self.plan_path.read_text())
            self.log(f"Using existing plan: {self.plan_path}")
            return plan
        if not prompt:
            raise ValueError(f"No plan at {self.plan_path}; a prompt is required")
        self.log("Planning journey...")
        plan = plan_journey(prompt, self.options.num_scenes, planner=self.planner)
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text(plan.model_dump_json(indent=2))
        return plan

    def check_supported(self, plan: JourneyPlan) -> None:
        if plan.actor.kind == ActorKind.WALKER:
            raise NotImplementedError(
                f"'{plan.actor.name}' is a walking actor. `make` only animates WHEELED and FLOATING actors "
                "for now; walkers need a real walk cycle (see `workshop.sprite`). Edit actor.kind in "
                f"{self.plan_path} to override."
            )

    def pending_image_calls(self, plan: JourneyPlan) -> list[str]:
        """List the image API calls a run would still make."""
        calls = []
        if not self.actor_source_path.exists():
            calls.append(f"actor: {plan.actor.name}")
        for s in plan.scenes:
            if not self.scene_path(s.id).exists():
                calls.append(f"scene: {s.name}")
        for a, b in zip(plan.scenes, plan.scenes[1:]):
            if not self.bridge_path(a.id, b.id).exists():
                calls.append(f"bridge: {a.name} -> {b.name}")
        return calls

    def load_plan(self) -> JourneyPlan:
        if not self.plan_path.exists():
            raise FileNotFoundError(f"No plan at {self.plan_path}")
        return JourneyPlan.model_validate_json(self.plan_path.read_text())

    def update_plan(self, new: JourneyPlan) -> list[str]:
        """Replace the plan and delete only the cached images it makes stale. Returns what was removed."""
        old = self.load_plan()
        removed: list[str] = []

        if new.style != old.style or new.actor.visual != old.actor.visual or new.actor.kind != old.actor.kind:
            removed += self._remove_actor()

        old_scenes = {s.id: s for s in old.scenes}
        for s in new.scenes:
            prev = old_scenes.get(s.id)
            changed = prev is not None and (
                new.style != old.style or prev.visual != s.visual or prev.ground_material != s.ground_material
            )
            if changed:
                removed += self._remove_scene(s.id)
        # Bridges are keyed by neighbour pairs, so new orders or new scenes simply miss the cache.

        self.plan_path.write_text(new.model_dump_json(indent=2))
        return removed

    def redo(self, target: str) -> list[str]:
        """Delete one cached image ("actor" or a scene id, which also drops its bridges) so the next run remakes it."""
        if target == "actor":
            return self._remove_actor()
        plan = self.load_plan()
        if target not in {s.id for s in plan.scenes}:
            raise ValueError(f"Unknown target {target!r}; use 'actor' or one of {[s.id for s in plan.scenes]}")
        return self._remove_scene(target)

    def _remove(self, paths: list[Path]) -> list[str]:
        removed = []
        for p in paths:
            if p.exists():
                p.unlink()
                removed.append(p.relative_to(self.root).as_posix())
        return removed

    def _remove_actor(self) -> list[str]:
        sheet = self.actor_sheet_path
        return self._remove([self.actor_source_path, sheet, sheet.with_suffix(".json")])

    def _remove_scene(self, scene_id: str) -> list[str]:
        bridges = [p for p in (self.root / "bridges").glob("*.png") if scene_id in p.stem.split("--")]
        return self._remove([self.scene_path(scene_id), *bridges])

    def build_actor(self, plan: JourneyPlan) -> Path:
        if not self.actor_source_path.exists():
            self.log(f"Generating actor: {plan.actor.name}")
            img = self.images.generate(
                assets.actor_prompt(plan.style, plan.actor.visual), size=assets.ACTOR_SIZE, background="transparent"
            )
            self.actor_source_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(self.actor_source_path)
        if not self.actor_sheet_path.exists():
            assets.build_rigid_actor_sheet(Image.open(self.actor_source_path), plan.actor.kind, self.actor_sheet_path)
        return self.actor_sheet_path

    def build_scenes(self, plan: JourneyPlan) -> list[Image.Image]:
        h = self.options.canvas_h
        scenes = []
        for s in plan.scenes:
            path = self.scene_path(s.id)
            if not path.exists():
                self.log(f"Generating scene: {s.name}")
                img = self.images.generate(
                    assets.scene_prompt(plan.style, s.visual, s.ground_material), size=assets.PANORAMA_SIZE
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                assets.fit_panorama(img, h).save(path)
            scenes.append(assets.fit_panorama(Image.open(path), h))
        return scenes

    def build_bridges(self, plan: JourneyPlan, scenes: list[Image.Image]) -> list[Image.Image]:
        h = self.options.canvas_h
        bridges = []
        for (a, img_a), (b, img_b) in zip(zip(plan.scenes, scenes), zip(plan.scenes[1:], scenes[1:])):
            path = self.bridge_path(a.id, b.id)
            if not path.exists():
                self.log(f"Generating bridge: {a.name} -> {b.name}")
                anchor = assets.build_bridge_anchor(img_a, img_b, h)
                img = self.images.edit(anchor, assets.bridge_prompt(plan.style, a.name, b.name), size=assets.PANORAMA_SIZE)
                path.parent.mkdir(parents=True, exist_ok=True)
                img.convert("RGB").resize((3 * h, h), Image.Resampling.LANCZOS).save(path)
            bridges.append(Image.open(path).convert("RGB").resize((3 * h, h), Image.Resampling.LANCZOS))
        return bridges

    def build_world(self, plan: JourneyPlan, scenes: list[Image.Image], bridges: list[Image.Image]) -> list[Path]:
        # Cheap to recompute, and must follow any edited scene or bridge image
        h = self.options.canvas_h
        world = assets.compose_world(scenes, bridges, h)
        paths = []
        for i, seg in enumerate(assets.segment_world(world, len(scenes), h)):
            path = self.segment_path(i)
            path.parent.mkdir(parents=True, exist_ok=True)
            seg.save(path)
            paths.append(path)
        return paths

    def build_journey(self, plan: JourneyPlan, actor_sheet: Path, segments: list[Path]) -> JourneySpec:
        opt = self.options
        h = opt.canvas_h
        kind = plan.actor.kind
        journey = JourneySpec(
            title=plan.title,
            description=plan.prompt,
            actor=HeroActorSpec(
                name=plan.actor.name,
                sheet_path=self._rel(actor_sheet),
                scale=round(assets.actor_scale(actor_sheet, kind, h) * opt.actor_size, 4),
                floor_y=assets.actor_floor_y(kind, h),
                stride_pixels=None,
                contact_shadow=kind == ActorKind.WHEELED,
                footstep_contact_frames=[],
            ),
            director=DirectorSpec(
                canvas_w=round(h * 16 / 9),
                canvas_h=h,
                fps=opt.fps,
                # One scene plus its bridge is 4H of world width
                speed=round(4 * h / opt.seconds_per_scene, 3),
                hold_last_scene=True,
                output_formats=list(opt.output_formats),
            ),
            soundscape=SoundscapeSpec(music_track=None, foley_footsteps=False),
            scenes=[
                JourneySceneSpec(
                    id=s.id,
                    name=s.name,
                    prompt=s.visual,
                    sheet_path=self._rel(seg),
                    ground_material=s.ground_material,
                    ambient_mood=s.ambient_mood,
                    gap_after=0.0,
                    overlap_prev=1 if i > 0 else 0,
                )
                for i, (s, seg) in enumerate(zip(plan.scenes, segments))
            ],
        )
        self.journey_path.write_text(journey_to_toml(journey))
        return journey

    def render(self, journey: JourneySpec) -> dict[str, Any]:
        from workshop.odyssey.orchestrator import OdysseyOrchestrator

        self.log("Rendering video...")
        return OdysseyOrchestrator(journey, root_dir=self.root).run(output_dir=self.root / "render")

    def run(self, prompt: str | None, confirm: Callable[[JourneyPlan, list[str]], bool] | None = None) -> dict[str, Any]:
        plan = self.load_or_plan(prompt)
        self.check_supported(plan)
        pending = self.pending_image_calls(plan)
        if pending and confirm is not None and not confirm(plan, pending):
            return {"status": "cancelled", "plan": str(self.plan_path), "pending_image_calls": pending}

        self.save_options()
        actor_sheet = self.build_actor(plan)
        scenes = self.build_scenes(plan)
        bridges = self.build_bridges(plan, scenes)
        segments = self.build_world(plan, scenes, bridges)
        journey = self.build_journey(plan, actor_sheet, segments)
        result = self.render(journey)
        return {"status": "completed", "project": str(self.root), "journey": str(self.journey_path), **result}

    # -- background runs (web API, MCP) ------------------------------------
    def read_status(self) -> dict[str, Any] | None:
        if not self.status_path.exists():
            return None
        return json.loads(self.status_path.read_text())

    def _write_status(self, **fields: Any) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": datetime.now(UTC).isoformat(), **fields}
        # Write then rename, so a reader polling mid-write never sees a half-written file
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(self.status_path)

    def is_running(self) -> bool:
        status = self.read_status()
        return status is not None and status.get("status") in ("QUEUED", "PROCESSING")

    def start_background_run(self) -> None:
        """Run the whole pipeline in a thread, reporting progress through status.json."""
        if self.is_running():
            raise RuntimeError(f"A run is already in progress for {self.root.name}")
        self.check_supported(self.load_plan())
        self._write_status(status="QUEUED")
        log = self.log

        def report(msg: str) -> None:
            self._write_status(status="PROCESSING", step=msg)
            log(msg)

        def work() -> None:
            self.log = report
            try:
                result = self.run(prompt=None)
                self._write_status(status="COMPLETED", result=result)
            except Exception as e:
                self._write_status(status="FAILED", error=str(e))
            finally:
                self.log = log

        threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------------
# TOML output (stdlib has a reader only)
# ---------------------------------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"Unsupported TOML value: {v!r}")


def _toml_table(d: dict[str, Any]) -> list[str]:
    return [f"{k} = {_toml_value(v)}" for k, v in d.items() if v is not None]


def journey_to_toml(journey: JourneySpec) -> str:
    data = journey.to_toml_dict()
    lines = _toml_table({k: data[k] for k in ("title", "description")})
    for table in ("actor", "director", "soundscape"):
        lines += ["", f"[{table}]", *_toml_table(data[table])]
    for scene in data["scenes"]:
        lines += ["", "[[scenes]]", *_toml_table(scene)]
    return "\n".join(lines) + "\n"
