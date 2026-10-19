"""Tests for the Odyssey Studio vertical deep pipeline."""
from pathlib import Path

import numpy as np
import pytest

from workshop.odyssey.actor_fsm import OdysseyActorFSM
from workshop.odyssey.director import OdysseyDirector
from workshop.odyssey.orchestrator import OdysseyOrchestrator
from workshop.odyssey.soundscape import SoundscapeEngine
from workshop.odyssey.spec import (
    ActorAction,
    AmbientMood,
    DirectorSpec,
    GroundMaterial,
    HeroActorSpec,
    JourneySceneSpec,
    JourneySpec,
    SoundscapeSpec,
)


def test_journey_spec_parsing(tmp_path: Path):
    """Verify JourneySpec parses from TOML and serializes back."""
    toml_content = """
    title = "Test Cat Walk"
    description = "Test Description"

    [actor]
    name = "Test Cat"
    sheet_path = "outputs/munchkin-walking/sprite_walk_4x5.png"
    scale = 0.5
    floor_y = 350
    stride_pixels = 160.0

    [director]
    canvas_w = 640
    canvas_h = 360
    fps = 24
    output_formats = ["16:9", "9:16"]

    [soundscape]
    foley_footsteps = true
    foley_volume = 0.5
    ambient_volume = 0.3

    [[scenes]]
    id = "scene-1"
    name = "Sunny Path"
    sheet_path = "outputs/couple-picnic-river/sprite_bg.png"
    ground_material = "GRASS"
    ambient_mood = "FOREST_WIND"
    duration = 5.0
    """
    toml_file = tmp_path / "journey.toml"
    toml_file.write_text(toml_content)

    spec = JourneySpec.from_toml_file(toml_file)
    assert spec.title == "Test Cat Walk"
    assert spec.actor.name == "Test Cat"
    assert spec.actor.scale == 0.5
    assert len(spec.scenes) == 1
    assert spec.scenes[0].ground_material == GroundMaterial.GRASS
    assert spec.scenes[0].ambient_mood == AmbientMood.FOREST_WIND
    assert "9:16" in spec.director.output_formats


def test_actor_fsm_frames_and_foley():
    """Verify Actor FSM loads frames, calculates actions, and generates footstep triggers."""
    sheet_path = "outputs/munchkin-walking/sprite_walk_4x5.png"
    if not Path(sheet_path).exists():
        pytest.skip(f"Test sprite {sheet_path} not found")

    actor_spec = HeroActorSpec(
        name="Munchkin",
        sheet_path=sheet_path,
        full_sheet_path="outputs/munchkin-walking/sprite_4x7.png",
        scale=0.45,
        floor_y=352,
        footstep_contact_frames=[3, 10],
    )
    fsm = OdysseyActorFSM(actor_spec, canvas_w=640)

    assert fsm.width > 0
    assert fsm.height > 0
    assert len(fsm.walk_frames) > 0

    # Test actions
    walk_frame = fsm.get_action_frame(ActorAction.WALK, 0.5)
    assert walk_frame.size == (fsm.width, fsm.height)

    sit_frame = fsm.get_action_frame(ActorAction.SIT_GAZE, 2.0)
    assert sit_frame.size == (fsm.width, fsm.height)

    # Test shadow
    shadow_img, shadow_pos = fsm.get_shadow()
    assert shadow_img is not None
    assert shadow_pos is not None

    # Test footstep event computation
    intervals = [(0.0, 5.0, GroundMaterial.GRASS)]
    events = fsm.compute_footstep_events(intervals)
    assert len(events) > 0
    assert events[0].material == GroundMaterial.GRASS
    assert events[0].time >= 0.0


def test_soundscape_synthesis(tmp_path: Path):
    """Verify procedural foley and weather ambience synthesis."""
    sound_spec = SoundscapeSpec(
        foley_footsteps=True,
        foley_volume=0.5,
        ambient_volume=0.3,
    )
    engine = SoundscapeEngine(sound_spec, sample_rate=22050)

    # Synthesize footsteps for multiple materials
    for mat in [GroundMaterial.ASPHALT, GroundMaterial.GRASS, GroundMaterial.WATER_PUDDLE, GroundMaterial.WOODEN_FLOOR]:
        sample = engine.synthesize_footstep_sample(mat)
        assert isinstance(sample, np.ndarray)
        assert len(sample) > 0
        assert np.max(np.abs(sample)) <= 1.0

    # Synthesize ambient track
    moods = [(0.0, 2.0, AmbientMood.RAINY_LOFI), (2.0, 4.0, AmbientMood.FOREST_WIND)]
    ambience = engine.synthesize_ambient_track(4.0, moods)
    assert len(ambience) == int(4.0 * 22050)

    # Render master WAV file
    out_wav = tmp_path / "test_sound.wav"
    from workshop.odyssey.actor_fsm import FootstepEvent
    test_events = [
        FootstepEvent(time=0.5, frame_index=3, material=GroundMaterial.GRASS),
        FootstepEvent(time=1.2, frame_index=10, material=GroundMaterial.ASPHALT),
    ]
    rendered_wav = engine.render_master_audio(
        duration=2.0,
        footsteps=test_events,
        scene_moods=moods,
        out_wav_path=out_wav,
    )
    assert rendered_wav.exists()
    assert rendered_wav.stat().st_size > 1000


def test_director_framing_and_easing():
    """Verify director camera easing and vertical 9:16 shorts crop calculation."""
    dir_spec = DirectorSpec(canvas_w=640, canvas_h=360)
    actor_spec = HeroActorSpec(
        sheet_path="outputs/munchkin-walking/sprite_walk_4x5.png" if Path("outputs/munchkin-walking/sprite_walk_4x5.png").exists() else "tests/fixtures/dummy.png",
    )
    # Stub actor FSM if sheet missing
    if Path(actor_spec.sheet_path).exists():
        fsm = OdysseyActorFSM(actor_spec, canvas_w=640)
    else:
        pytest.skip("Test sprite not available for full director test")

    director = OdysseyDirector(dir_spec, fsm)

    # Test vertical crop for Shorts (9:16)
    crop = director.compute_vertical_shorts_crop(actor_center_x=320)
    assert crop.h == 360
    assert crop.w == int(360 * (9.0 / 16.0))  # 202
    assert crop.x >= 0
    assert crop.y == 0

    # Test camera easing multiplier
    speed_mult_normal = director.smooth_hold_speed_multiplier(1.0, hold_start=5.0, hold_end=8.0)
    assert speed_mult_normal == 1.0

    speed_mult_held = director.smooth_hold_speed_multiplier(6.0, hold_start=5.0, hold_end=8.0)
    assert speed_mult_held == 0.0


def test_orchestrator_compilation():
    """Verify JourneySpec compiles to low-level TimelineConfig."""
    journey = JourneySpec(
        title="Test Journey",
        actor=HeroActorSpec(name="Cat", scale=0.4, stride_pixels=150.0),
        director=DirectorSpec(canvas_w=640, canvas_h=360, fps=24),
        scenes=[
            JourneySceneSpec(
                id="s1",
                sheet_path="outputs/scene1.png",
                duration=4.0,
                ground_material=GroundMaterial.WOODEN_FLOOR,
            )
        ],
    )
    orch = OdysseyOrchestrator(journey)
    config = orch.compile_timeline_config()

    assert config.canvas_w == 640
    assert config.canvas_h == 360
    assert config.fps == 24
    assert config.char_scale == 0.4
    assert len(config.scenes) == 1
    assert config.scenes[0].id == "s1"
