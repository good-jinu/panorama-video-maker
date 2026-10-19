"""Prompt-to-plan step: turn one sentence into a validated journey plan.

The planner LLM only decides *what* to make (actor, art style, list of scenes, sound moods).
All geometry (sizes, ground line, speed) is fixed by the asset pipeline so the plan stays small
and every generated image lines up.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from workshop.odyssey.spec import AmbientMood, GroundMaterial

MIN_SCENES = 2
MAX_SCENES = 8


class ActorKind(StrEnum):
    """How the actor moves, which decides how its animation is built."""

    WHEELED = "WHEELED"  # cars, bikes, trains: ride on the road with a small bounce
    FLOATING = "FLOATING"  # balloons, boats, planes, birds: drift above the road with a slow bob
    WALKER = "WALKER"  # legged characters: need a real walk cycle (not supported by `make` yet)


class PlannedActor(BaseModel):
    name: str
    kind: ActorKind
    visual: str = Field(description="Visual description used as the actor image prompt.")


class PlannedScene(BaseModel):
    name: str
    visual: str = Field(description="Visual description used as the scene background prompt.")
    ground_material: GroundMaterial = GroundMaterial.ASPHALT
    ambient_mood: AmbientMood = AmbientMood.SILENT

    @property
    def id(self) -> str:
        return slugify(self.name)


class JourneyPlan(BaseModel):
    prompt: str
    title: str
    style: str = Field(description="One shared art style applied to every generated image.")
    actor: PlannedActor
    scenes: list[PlannedScene]

    @field_validator("scenes")
    @classmethod
    def _check_scenes(cls, scenes: list[PlannedScene]) -> list[PlannedScene]:
        if len(scenes) < MIN_SCENES:
            raise ValueError(f"A journey needs at least {MIN_SCENES} scenes, got {len(scenes)}")
        ids = [s.id for s in scenes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Scene names must be unique, got {ids}")
        return scenes[:MAX_SCENES]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "scene"


def _enum_values(enum: type[StrEnum]) -> list[str]:
    return [e.value for e in enum]


# Gemini `responseSchema` (OpenAPI subset)
PLAN_RESPONSE_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "style": {"type": "STRING"},
        "actor": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "kind": {"type": "STRING", "enum": _enum_values(ActorKind)},
                "visual": {"type": "STRING"},
            },
            "required": ["name", "kind", "visual"],
        },
        "scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "visual": {"type": "STRING"},
                    "ground_material": {"type": "STRING", "enum": _enum_values(GroundMaterial)},
                    "ambient_mood": {"type": "STRING", "enum": _enum_values(AmbientMood)},
                },
                "required": ["name", "visual", "ground_material", "ambient_mood"],
            },
        },
    },
    "required": ["title", "style", "actor", "scenes"],
}


def build_planner_prompt(prompt: str, num_scenes: int) -> str:
    return f"""You are the director of a side-scrolling panorama video.
The camera tracks one actor moving left-to-right through a sequence of scenes, filmed from a pure side view.

User request: "{prompt}"

Plan exactly {num_scenes} scenes in travel order.
- title: a short video title.
- style: ONE art style for every image (for example "soft watercolor storybook illustration"). Keep it consistent.
- actor.kind: WHEELED if it rolls on the ground (car, bike, train, skateboard), FLOATING if it flies or floats
  (balloon, plane, bird, boat, cloud), WALKER if it walks on legs (person, cat, dog).
- actor.visual: what the actor looks like, seen from the side, facing right. Describe only the actor.
- scenes[].visual: what this place looks like as a wide street-level backdrop with famous landmarks.
  Do not mention the actor, vehicles or people.
- ground_material and ambient_mood: pick the closest options for the place.
"""


Planner = Callable[[str, int], dict]


def gemini_planner(prompt: str, num_scenes: int) -> dict:
    from workshop.imagen import call_gemini_json

    return call_gemini_json(build_planner_prompt(prompt, num_scenes), PLAN_RESPONSE_SCHEMA)


def plan_journey(prompt: str, num_scenes: int = 5, planner: Planner | None = None) -> JourneyPlan:
    """Ask the planner LLM for a journey plan and validate it."""
    num_scenes = max(MIN_SCENES, min(MAX_SCENES, num_scenes))
    raw = (planner or gemini_planner)(prompt, num_scenes)
    return JourneyPlan(prompt=prompt, **raw)
