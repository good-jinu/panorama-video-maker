from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.conftest import make_sheet
from workshop.character import CharacterActor, create_contact_shadow
from workshop.scenes import compose_extended_scene
from workshop.timeline import SceneSpec, StoryTimeline, TimelineConfig


@pytest.fixture
def sample_assets(tmp_path):
    # 4 frames of 40x30
    frames = [Image.new("RGBA", (40, 30), (10 * i, 50, 100, 255)) for i in range(4)]
    sheet = make_sheet(tmp_path / "sprite.png", frames, columns=2, interval=0.25)
    # Character sheet: 3 frames of 20x15
    char_frames = [Image.new("RGBA", (20, 15), (200, 100, 50, 255)) for _ in range(3)]
    char_sheet = make_sheet(tmp_path / "cat.png", char_frames, columns=3, interval=0.20)
    # Left background image: 80x30
    left_bg = Image.new("RGBA", (80, 30), (30, 40, 50, 255))
    left_bg_path = tmp_path / "left_bg.png"
    left_bg.save(left_bg_path)

    return sheet, char_sheet, left_bg_path


def test_compose_extended_scene():
    sprite_frames = [Image.new("RGBA", (50, 40), (255, 0, 0, 255)) for _ in range(3)]
    left_bg = Image.new("RGBA", (100, 40), (0, 255, 0, 255))

    offset = 80
    stitched = compose_extended_scene(sprite_frames, left_bg, offset=offset, seamless=False)

    assert len(stitched) == 3
    # Combined width = offset + sprite_width = 80 + 50 = 130
    assert stitched[0].size == (130, 40)
    arr = np.array(stitched[0])
    # Left part (0..80) should be from left_bg (green)
    assert (arr[:, :75, 1] == 255).all()
    # Right part (80..130) should be from sprite_frames (red)
    assert (arr[:, 85:, 0] == 255).all()


def test_compose_extended_scene_separate_scale():
    # Sprite: 100x80 -> scaled by 0.5 becomes 50x40
    sprite_frames = [Image.new("RGBA", (100, 80), (255, 0, 0, 255))]
    # Left bg: 80x80 -> scaled by 1.0 (relative to raw_h 80) becomes 80x80
    left_bg = Image.new("RGBA", (80, 80), (0, 255, 0, 255))

    # video_scale=0.5, left_bg_scale=1.0, video_y="bottom", offset=60
    stitched = compose_extended_scene(
        sprite_frames,
        left_bg,
        offset=60,
        video_scale=0.5,
        left_bg_scale=1.0,
        video_y="bottom",
        seamless=False,
    )
    # comb_w = max(80, 60 + 50) = 110, comb_h = max(80, 40) = 80
    assert stitched[0].size == (110, 80)
    arr = np.array(stitched[0])
    # Video is at y = 80 - 40 = 40..80, x = 60..110
    # Above video (y < 40, x > 80) is transparent
    assert (arr[:30, 90:, 3] == 0).all()
    # Inside video (y in 45..75, x in 65..105) is red
    assert (arr[45:75, 65:105, 0] == 255).all()


def test_compose_extended_scene_seamless():
    sprite_frames = [Image.new("RGBA", (60, 40), (255, 0, 0, 255))]
    left_bg = Image.new("RGBA", (100, 60), (0, 255, 0, 255))

    stitched = compose_extended_scene(
        sprite_frames,
        left_bg,
        offset=70,
        video_scale=1.0,
        left_bg_scale=1.5,
        video_y=0,
        seamless=True,
    )
    assert stitched[0].size == (130, 60)
    arr = np.array(stitched[0])
    # Far left is green, far right-top is red, and bottom-right floor is extended green
    assert (arr[:, :60, 1] == 255).all()
    assert (arr[:30, 110:, 0] == 255).all()
    assert (arr[45:, 110:, 1] == 255).all()


def test_story_timeline_start_on_screen(sample_assets):
    sheet, char_sheet, _ = sample_assets
    scene1 = SceneSpec(id="s1", sheet=sheet, scale=1.0)
    scene2 = SceneSpec(id="s2", sheet=sheet, scale=1.0, gap_after=1.0)

    config = TimelineConfig(
        canvas_w=100,
        canvas_h=60,
        fps=10,
        speed=20.0,
        char_sheet=char_sheet,
        start_on_screen=True,
        scenes=[scene1, scene2],
    )
    timeline = StoryTimeline(config)

    # First scene should start directly at world_x_start = 0.0
    assert config.scenes[0].world_x_start == 0.0
    # First scene travels its width (40px) at 20px/s -> active_duration = 2.0s
    assert config.scenes[0].active_duration == pytest.approx(2.0)

    # At t = 0, first scene should be on screen at screen_x = 0
    visible_t0 = list(timeline.visible_scenes(0.0, 100))
    assert len(visible_t0) == 1
    assert visible_t0[0][0].id == "s1"
    assert visible_t0[0][1] == 0


def test_character_actor_placement_and_shadow(sample_assets):
    _, char_sheet, _ = sample_assets
    actor = CharacterActor(
        sheet_path=char_sheet,
        scale=1.0,
        floor_y=50,
        canvas_w=100,
        shadow=True,
    )
    # Actor height is 15, floor_y is 50 -> pos_y should be 50 - 15 = 35
    assert actor.pos_y == 35
    assert actor.has_shadow
    assert actor.shadow_image is not None
    assert actor.shadow_pos[1] == 50 - 14 // 2


def test_story_timeline_overlap_prev(sample_assets):
    sheet, char_sheet, left_bg_path = sample_assets
    scene1 = SceneSpec(id="s1", sheet=sheet, left_bg=left_bg_path, left_bg_offset=60, scale=1.0)
    scene2 = SceneSpec(
        id="s2",
        sheet=sheet,
        left_bg=left_bg_path,
        left_bg_offset=60,
        scale=1.0,
        overlap_prev=20,
        left_feather=(5, 10),
    )

    config = TimelineConfig(
        canvas_w=100,
        canvas_h=60,
        fps=10,
        speed=20.0,
        char_sheet=char_sheet,
        start_on_screen=True,
        scenes=[scene1, scene2],
    )
    StoryTimeline(config)

    # scene1 width = 60 + 40 = 100, world_x_start = 0.0
    assert config.scenes[0].width == 100
    assert config.scenes[0].world_x_start == 0.0
    # scene2 overlaps scene1 by 20px -> world_x_start = 100 - 20 = 80.0
    assert config.scenes[1].world_x_start == pytest.approx(80.0)


def test_build_transition_anchor():
    from workshop.imagen import build_transition_anchor

    img_a = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    img_b = Image.new("RGBA", (150, 80), (0, 0, 255, 255))

    anchor, right_x = build_transition_anchor(
        img_a,
        img_b,
        canvas_w=500,
        canvas_h=100,
        left_anchor_w=60,
        include_full_right_scene=True,
    )
    assert anchor.size == (500, 100)
    assert right_x == 350
    arr = np.array(anchor)
    # Left 60px is red, middle is white, right 150px (y < 80) is blue
    assert (arr[:, :60, 0] == 255).all() and (arr[:, :60, 1] == 0).all()
    assert (arr[:, 100:300, :3] == 255).all()
    assert (arr[:80, 350:, 2] == 255).all() and (arr[:80, 350:, 0] == 0).all()


def test_loaded_scene_anim_delay_and_loop_start():
    from workshop.scenes import LoadedScene

    # 4 frames: 0, 1 (video 1) and 2, 3 (video 2)
    frames = [Image.new("RGBA", (10, 10), (i, 0, 0, 255)) for i in range(4)]
    spec = SceneSpec(
        id="test",
        sheet=Path("test.png"),
        anim_delay=2.0,
        loop_start_frame=2,
    )
    spec.start_time = 5.0
    scene = LoadedScene(spec=spec, frames=frames, interval=1.0)

    # Before start_time + anim_delay (t < 7.0): clamped to frame 0
    assert scene.get_frame(5.0) == frames[0]
    assert scene.get_frame(6.5) == frames[0]

    # At t = 7.0 (time_in_scene = 0.0): frame 0
    assert scene.get_frame(7.0) == frames[0]
    # At t = 8.0 (time_in_scene = 1.0): frame 1
    assert scene.get_frame(8.0) == frames[1]
    # At t = 9.0 (time_in_scene = 2.0): frame 2 (video 2 starts)
    assert scene.get_frame(9.0) == frames[2]
    # At t = 10.0 (time_in_scene = 3.0): frame 3
    assert scene.get_frame(10.0) == frames[3]

    # At t = 11.0 (time_in_scene = 4.0): loops back to loop_start_frame=2 (video 2 replay!)
    assert scene.get_frame(11.0) == frames[2]
    # At t = 12.0 (time_in_scene = 5.0): frame 3
    assert scene.get_frame(12.0) == frames[3]
    # At t = 13.0 (time_in_scene = 6.0): frame 2 again
    assert scene.get_frame(13.0) == frames[2]


def test_loaded_scene_pingpong():
    from workshop.scenes import LoadedScene

    # 4 frames: 0, 1, 2, 3 -> pingpong: 0, 1, 2, 3, 2, 1, 0, 1, ...
    frames = [Image.new("RGBA", (10, 10), (i, 0, 0, 255)) for i in range(4)]
    spec = SceneSpec(
        id="test_pingpong",
        sheet=Path("test.png"),
        pingpong=True,
    )
    spec.start_time = 0.0
    scene = LoadedScene(spec=spec, frames=frames, interval=1.0)

    expected_indices = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]
    for step, expected_idx in enumerate(expected_indices):
        assert scene.get_frame(float(step)) == frames[expected_idx]


def test_loaded_scene_pingpong_with_loop_start():
    from workshop.scenes import LoadedScene

    # 5 frames: 0, 1 (video 1) and 2, 3, 4 (video 2)
    # loop_start_frame=2, pingpong=True
    # video 1 plays once: 0, 1
    # video 2 pingpongs: 2, 3, 4, 3, 2, 3, 4, 3, 2...
    frames = [Image.new("RGBA", (10, 10), (i, 0, 0, 255)) for i in range(5)]
    spec = SceneSpec(
        id="test_pingpong_loop_start",
        sheet=Path("test.png"),
        loop_start_frame=2,
        pingpong=True,
    )
    spec.start_time = 0.0
    scene = LoadedScene(spec=spec, frames=frames, interval=1.0)

    expected_indices = [0, 1, 2, 3, 4, 3, 2, 3, 4, 3, 2]
    for step, expected_idx in enumerate(expected_indices):
        assert scene.get_frame(float(step)) == frames[expected_idx]


def test_character_actor_outro(tmp_path):
    # 2 frames walk sheet
    walk_frames = [Image.new("RGBA", (20, 20), (10 * i, 0, 0, 255)) for i in range(2)]
    walk_sheet = make_sheet(tmp_path / "walk.png", walk_frames, columns=2, interval=1.0)

    # 40 frames full sheet
    full_frames = [Image.new("RGBA", (20, 20), (0, i, 0, 255)) for i in range(40)]
    full_sheet = make_sheet(tmp_path / "full.png", full_frames, columns=5, interval=1.0)

    actor = CharacterActor(
        sheet_path=walk_sheet,
        full_sheet_path=full_sheet,
        outro_time=4.0,
        scale=1.0,
        canvas_w=100,
        floor_y=50,
        shadow=True,
    )

    # Before outro_time (t=0..3): loops walk_frames (0, 1, 0, 1)
    f0, pos0 = actor.get_actor_frame_and_pos(0.0)
    assert np.array(f0)[0, 0, 0] == 0
    f1, pos1 = actor.get_actor_frame_and_pos(1.0)
    assert np.array(f1)[0, 0, 0] == 10
    assert pos0 == pos1

    # At t=4.0 (outro_time, and loop boundary 4.0 % 2.0 == 0):
    # plays full_frames[19] (outro_idx=0)
    f4, pos4 = actor.get_actor_frame_and_pos(4.0)
    assert np.array(f4)[0, 0, 1] == 19

    # At t=24.0 (outro_idx=20):
    # plays full_frames[39] (final frame) with shifted position
    f24, pos24 = actor.get_actor_frame_and_pos(24.0)
    assert np.array(f24)[0, 0, 1] == 39
    assert pos24[0] > pos0[0]

    # After outro (t=25.0, outro_idx=21): actor and shadow exit
    f25, _ = actor.get_actor_frame_and_pos(25.0)
    assert f25 is None
    s_img, s_pos = actor.get_shadow(25.0)
    assert s_img is None and s_pos is None


def test_character_actor_meow_loop_step(tmp_path):
    # 15 frames walk sheet
    walk_frames = [Image.new("RGBA", (20, 20), (i, 0, 0, 255)) for i in range(15)]
    walk_sheet = make_sheet(tmp_path / "walk.png", walk_frames, columns=4, interval=0.25)

    # 40 frames full sheet
    full_frames = [Image.new("RGBA", (20, 20), (0, i, 0, 255)) for i in range(40)]
    full_sheet = make_sheet(tmp_path / "full.png", full_frames, columns=5, interval=0.25)

    actor = CharacterActor(
        sheet_path=walk_sheet,
        full_sheet_path=full_sheet,
        meow_loop_step=12,
        scale=1.0,
        canvas_w=100,
        floor_y=50,
        shadow=True,
    )

    # Loop 1 walk (0.0s .. 3.5s): frames 0..14
    f0, _ = actor.get_actor_frame_and_pos(0.0)
    assert np.array(f0)[0, 0, 0] == 0
    f14, _ = actor.get_actor_frame_and_pos(14 * 0.25)
    assert np.array(f14)[0, 0, 0] == 14

    # Loop 1 meow (3.75s .. 5.75s, 9 frames from full sheet: 19..27)
    f_meow_start, _ = actor.get_actor_frame_and_pos(15 * 0.25)
    assert np.array(f_meow_start)[0, 0, 1] == 19
    f_meow_end, _ = actor.get_actor_frame_and_pos(23 * 0.25)
    assert np.array(f_meow_end)[0, 0, 1] == 27

    # Loop 2 walk (6.00s .. 9.50s): resumes walk loop from frame 0
    f_loop2_start, _ = actor.get_actor_frame_and_pos(24 * 0.25)
    assert np.array(f_loop2_start)[0, 0, 0] == 0
    f_loop2_mid, _ = actor.get_actor_frame_and_pos((24 + 5) * 0.25)
    assert np.array(f_loop2_mid)[0, 0, 0] == 5

    # Loop 13 walk and meow (starts at 47.25s = 189 * 0.25)
    # Walk: 189..203 -> frames 0..14
    f_loop13_start, _ = actor.get_actor_frame_and_pos(189 * 0.25)
    assert np.array(f_loop13_start)[0, 0, 0] == 0
    # Meow: 204..212 -> full sheet frames 19..27
    f_loop13_meow, _ = actor.get_actor_frame_and_pos(204 * 0.25)
    assert np.array(f_loop13_meow)[0, 0, 1] == 19


def test_timeline_meow_pauses(tmp_path):
    walk_frames = [Image.new("RGBA", (20, 20), (i, 0, 0, 255)) for i in range(19)]
    walk_sheet = make_sheet(tmp_path / "walk.png", walk_frames, columns=5, interval=0.25)

    scene_frames = [Image.new("RGBA", (100, 100), (255, 255, 255, 255))]
    scene_sheet = make_sheet(tmp_path / "scene.png", scene_frames, columns=1, interval=1.0)

    config = TimelineConfig(
        canvas_w=100,
        canvas_h=100,
        fps=24,
        speed=100.0,
        char_sheet=walk_sheet,
        meow_loop_step=4,
        scenes=[SceneSpec(id="s1", sheet=scene_sheet, scale=1.0, duration=30.0)],
    )

    tl = StoryTimeline(config, root_dir=tmp_path)

    # During walk loop 1 (0 to 4.75s), camera moves at speed (100 px/s)
    cam0 = tl.cam_x_at(0.0)
    assert cam0 == 0.0
    cam_walk = tl.cam_x_at(4.75)
    assert cam_walk == pytest.approx(475.0)

    # During meow pause (4.75s to 7.00s = 2.25s pause), camera is paused!
    cam_pause1 = tl.cam_x_at(5.0)
    cam_pause2 = tl.cam_x_at(6.99)
    assert cam_pause1 == pytest.approx(475.0)
    assert cam_pause2 == pytest.approx(475.0)

    # After meow pause (t > 7.00s), camera resumes moving
    cam_resume = tl.cam_x_at(8.0)
    assert cam_resume == pytest.approx(575.0)

    # Test inverse function _dist_to_elapsed_time
    assert tl._dist_to_elapsed_time(475.0) == pytest.approx(4.75)
    assert tl._dist_to_elapsed_time(575.0) == pytest.approx(8.0)




