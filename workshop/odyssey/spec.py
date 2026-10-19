"""High-level Journey & Odyssey domain specifications.

Defines the vertical schema for infinite-walk and side-scrolling odyssey productions:
- Hero Actor specifications & gait physics
- Story Scenes with ambient moods and ground surface materials
- Directorial camera pacing (16:9 widescreen & 9:16 vertical shorts)
- Multi-track procedural soundscape (footstep foley, environmental ambience, lofi music)
"""
from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ActorAction(StrEnum):
    """Behavioral state of the wandering protagonist."""

    WALK = "WALK"
    PAUSE_SNIFF = "PAUSE_SNIFF"
    SIT_GAZE = "SIT_GAZE"
    SLEEP_COZY = "SLEEP_COZY"
    DASH = "DASH"
    MEOW = "MEOW"


class GroundMaterial(StrEnum):
    """Ground surface for audio foley footsteps and contact aesthetics."""

    ASPHALT = "ASPHALT"
    GRASS = "GRASS"
    WOODEN_FLOOR = "WOODEN_FLOOR"
    WATER_PUDDLE = "WATER_PUDDLE"
    SNOW = "SNOW"
    STONE = "STONE"


class AmbientMood(StrEnum):
    """Environmental soundscape and weather ambience."""

    RAINY_LOFI = "RAINY_LOFI"
    NIGHT_CRICKETS = "NIGHT_CRICKETS"
    FOREST_WIND = "FOREST_WIND"
    COZY_FIREPLACE = "COZY_FIREPLACE"
    CYBERPUNK_HUM = "CYBERPUNK_HUM"
    OCEAN_WAVES = "OCEAN_WAVES"
    SILENT = "SILENT"


class HeroActorSpec(BaseModel):
    """Protagonist actor identity, walking gait, and contact physics."""

    name: str = "Wandering Cat"
    sheet_path: str = "outputs/character/sprite.png"
    full_sheet_path: str | None = None
    scale: float = 0.45
    floor_y: int = 352
    stride_pixels: float | None = 180.0
    contact_shadow: bool = True
    shadow_opacity: float = 0.35
    shadow_width_scale: float = 0.70
    shadow_height: int = 14
    flip: bool = False
    footstep_contact_frames: list[int] = Field(
        default_factory=lambda: [3, 10],
        description="Frame indices in walk cycle where paws make solid ground contact (triggers foley).",
    )


class JourneySceneSpec(BaseModel):
    """A chapter or world section in the continuous journey."""

    id: str
    name: str = "scene"
    prompt: str | None = None
    sheet_path: str
    left_bg_path: str | None = None
    left_bg_offset: int | None = None
    ground_material: GroundMaterial = GroundMaterial.ASPHALT
    ambient_mood: AmbientMood = AmbientMood.RAINY_LOFI
    scale: float = 1.0
    y: int = 0
    duration: float | None = None
    hold_duration: float = 0.0
    hold_action: ActorAction = ActorAction.SIT_GAZE
    parallax: float = 1.0
    gap_after: float = 0.0
    overlap_prev: int = 0


class SoundscapeSpec(BaseModel):
    """Multi-track procedural audio configuration."""

    music_track: str | None = None
    music_volume: float = 0.70
    ambient_volume: float = 0.50
    foley_footsteps: bool = True
    foley_volume: float = 0.60
    ducking: bool = True


class DirectorSpec(BaseModel):
    """Directorial camera, pacing, and framing parameters."""

    canvas_w: int = 640
    canvas_h: int = 360
    fps: int = 24
    speed: float | None = None
    camera_easing: bool = True
    bg_color: str = "#000000"
    output_formats: list[str] = Field(
        default_factory=lambda: ["16:9"],
        description="Target formats: '16:9' (Widescreen), '9:16' (Shorts/TikTok/Reels).",
    )


class JourneySpec(BaseModel):
    """Top-level A-to-Z production blueprint for an Endless Journey / Odyssey video."""

    title: str = "Cat Odyssey: The Infinite Journey"
    description: str | None = None
    actor: HeroActorSpec = Field(default_factory=HeroActorSpec)
    director: DirectorSpec = Field(default_factory=DirectorSpec)
    soundscape: SoundscapeSpec = Field(default_factory=SoundscapeSpec)
    scenes: list[JourneySceneSpec] = Field(default_factory=list)

    @classmethod
    def from_toml_file(cls, path: str | Path) -> JourneySpec:
        """Parse high-level Odyssey specification from a TOML configuration."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JourneySpec:
        """Construct JourneySpec from dictionary payload."""
        actor_data = data.get("actor", {})
        director_data = data.get("director", {})
        soundscape_data = data.get("soundscape", {})

        # Handle scenes list
        raw_scenes = data.get("scenes", [])
        scenes: list[JourneySceneSpec] = []
        for s in raw_scenes:
            scenes.append(JourneySceneSpec(**s))

        return cls(
            title=data.get("title", "Cat Odyssey"),
            description=data.get("description"),
            actor=HeroActorSpec(**actor_data),
            director=DirectorSpec(**director_data),
            soundscape=SoundscapeSpec(**soundscape_data),
            scenes=scenes,
        )

    def to_toml_dict(self) -> dict[str, Any]:
        """Convert JourneySpec to nested dictionary ready for TOML serialization."""
        return self.model_dump(mode="json")
