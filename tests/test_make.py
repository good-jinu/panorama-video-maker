"""Tests for prompt-to-video `make`, with the planner and image APIs mocked (no credits spent)."""
import io
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from workshop.make import assets
from workshop.make.pipeline import MakeOptions, MakeProject
from workshop.make.plan import ActorKind, JourneyPlan, plan_journey
from workshop.odyssey.spec import JourneySpec

H = 36  # tiny canvas keeps the end-to-end render fast

PLAN = {
    "title": "Tiny Car World Tour",
    "style": "flat vector illustration",
    "actor": {"name": "Tiny Car", "kind": "WHEELED", "visual": "a small yellow car"},
    "scenes": [
        {"name": "Paris", "visual": "Eiffel tower", "ground_material": "STONE", "ambient_mood": "SILENT"},
        {"name": "Tokyo", "visual": "neon streets", "ground_material": "ASPHALT", "ambient_mood": "CYBERPUNK_HUM"},
        {"name": "New York", "visual": "skyline", "ground_material": "ASPHALT", "ambient_mood": "RAINY_LOFI"},
    ],
}


def fake_planner(prompt: str, num_scenes: int) -> dict:
    return json.loads(json.dumps(PLAN))


class FakeImages:
    """Returns flat images in the sizes the real API would, and records every call."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, size: str, background: str | None = None) -> Image.Image:
        self.calls.append(f"generate:{size}:{background}")
        w, h = (int(v) for v in size.split("x"))
        if background == "transparent":
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            img.paste((240, 200, 20, 255), (w // 4, h // 3, 3 * w // 4, 2 * h // 3))
            return img
        return Image.new("RGBA", (w, h), (40 * len(self.calls) % 255, 120, 200, 255))

    def edit(self, image: Image.Image, prompt: str, size: str) -> Image.Image:
        self.calls.append(f"edit:{size}")
        w, h = (int(v) for v in size.split("x"))
        return Image.new("RGBA", (w, h), (90, 160, 90, 255))


def make_project(tmp_path: Path, images: FakeImages | None = None) -> MakeProject:
    return MakeProject(
        tmp_path / "project",
        MakeOptions(canvas_h=H, fps=4, seconds_per_scene=2.0),
        images=images or FakeImages(),
        planner=fake_planner,
        log=lambda msg: None,
    )


# -- planning ------------------------------------------------------------------

def test_plan_validates_and_slugs_scene_ids():
    plan = plan_journey("a small car", 3, planner=fake_planner)
    assert plan.actor.kind == ActorKind.WHEELED
    assert [s.id for s in plan.scenes] == ["paris", "tokyo", "new-york"]


def test_plan_rejects_duplicate_scene_names():
    raw = fake_planner("", 3)
    raw["scenes"][1]["name"] = "Paris"
    with pytest.raises(ValueError, match="unique"):
        JourneyPlan(prompt="x", **raw)


def test_walker_actor_is_refused_before_any_image_call(tmp_path: Path):
    images = FakeImages()
    project = make_project(tmp_path, images)
    project.planner = lambda p, n: {**fake_planner(p, n), "actor": {"name": "Cat", "kind": "WALKER", "visual": "a cat"}}
    with pytest.raises(NotImplementedError, match="walking actor"):
        project.run("a cat walking")
    assert images.calls == []


def test_gemini_planner_request_and_parsing(monkeypatch):
    from workshop import imagen

    captured = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(PLAN)}]}}]}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(imagen.urllib.request, "urlopen", fake_urlopen)
    result = imagen.call_gemini_json("plan it", {"type": "OBJECT"}, model="test-model", api_key="k")
    assert result["title"] == PLAN["title"]
    assert captured["url"].endswith("/models/test-model:generateContent")
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"


# -- assets --------------------------------------------------------------------

def test_rigid_actor_sheet_loops_and_rests_on_floor(tmp_path: Path):
    src = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    src.paste((255, 0, 0, 255), (50, 80, 150, 120))
    sheet = assets.build_rigid_actor_sheet(src, ActorKind.WHEELED, tmp_path / "sprite.png")
    meta = json.loads(sheet.with_suffix(".json").read_text())
    assert meta["frameCount"] == assets.MOTIONS[ActorKind.WHEELED].frames
    assert meta["intervalSeconds"] == pytest.approx(1 / 24)

    from workshop.utils import load_sprite_sheet

    frames, _ = load_sprite_sheet(sheet)
    # Frame 0 has no lift: the actor's bottom edge touches the bottom of the frame
    assert frames[0].getchannel("A").getbbox()[3] == frames[0].height


def test_world_layout_and_segments_cover_the_strip():
    scenes = [Image.new("RGB", (3 * H, H), c) for c in ("red", "green", "blue")]
    bridges = [Image.new("RGB", (3 * H, H), "white")] * 2
    world = assets.compose_world(scenes, bridges, H)
    assert world.size == (3 * H + 2 * 4 * H, H)
    # Scene centres are untouched by the cross-fades
    assert world.getpixel((4 * H + 3 * H // 2, H // 2)) == (0, 128, 0)

    segments = assets.segment_world(world, 3, H)
    # Each later segment starts 1px early to match overlap_prev = 1
    assert sum(s.width for s in segments) == world.width + 2


def test_bridge_anchor_repeats_scene_edges():
    prev = Image.new("RGB", (3 * H, H), "red")
    nxt = Image.new("RGB", (3 * H, H), "blue")
    anchor = assets.build_bridge_anchor(prev, nxt, H)
    assert anchor.getpixel((0, 0)) == (255, 0, 0)
    assert anchor.getpixel((3 * H // 2, 0)) == (255, 255, 255)
    assert anchor.getpixel((3 * H - 1, 0)) == (0, 0, 255)


# -- pipeline ------------------------------------------------------------------

def test_declining_confirmation_spends_nothing(tmp_path: Path):
    images = FakeImages()
    project = make_project(tmp_path, images)
    result = project.run("a small car", confirm=lambda plan, pending: False)
    assert result["status"] == "cancelled"
    assert len(result["pending_image_calls"]) == 1 + 3 + 2
    assert images.calls == []
    assert project.plan_path.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_end_to_end_render_and_resume(tmp_path: Path):
    images = FakeImages()
    project = make_project(tmp_path, images)
    result = project.run("a small car riding through world's cities")

    assert result["status"] == "completed"
    assert images.calls.count("generate:1024x1024:transparent") == 1
    assert images.calls.count(f"generate:{assets.PANORAMA_SIZE}:None") == 3
    assert images.calls.count(f"edit:{assets.PANORAMA_SIZE}") == 2
    assert Path(result["widescreen_16_9"]).stat().st_size > 0
    assert Path(result["vertical_9_16"]).stat().st_size > 0

    # journey.toml round-trips through the existing renderer's parser, with project-relative paths
    journey = JourneySpec.from_toml_file(project.journey_path)
    assert [s.id for s in journey.scenes] == ["paris", "tokyo", "new-york"]
    assert journey.actor.sheet_path == "actor/sprite.png"
    assert journey.director.hold_last_scene is True

    # Second run reuses every generated image
    images.calls.clear()
    assert project.pending_image_calls(project.load_or_plan(None)) == []
    project.run(None)
    assert images.calls == []


def test_extract_sprite_pipeline_forwards_model(monkeypatch, tmp_path: Path):
    """Regression: the service passed `model_name=` to a function that only accepts `model=`."""
    from workshop.services import pipeline_service

    seen = {}
    monkeypatch.setattr(pipeline_service, "generate_sprite_sheet", lambda **kw: seen.update(kw))
    pipeline_service.extract_sprite_pipeline(tmp_path / "clip.mp4", output_sheet=tmp_path / "s.png", model_name="birefnet")
    assert seen["model"] == "birefnet"
    # Every forwarded keyword must exist on the real function
    import inspect

    from workshop.sprite import generate_sprite_sheet

    inspect.signature(generate_sprite_sheet).bind(**seen)
