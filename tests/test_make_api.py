"""Tests for the /api/make REST routes and plan editing, with planner, images and render mocked."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_make import FakeImages, fake_planner
from workshop.config import settings
from workshop.make import assets, pipeline, plan
from workshop.server.app import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
    monkeypatch.setattr(plan, "gemini_planner", fake_planner)
    images = FakeImages()
    monkeypatch.setattr(assets, "OpenAIImageBackend", lambda: images)

    def fake_render(self, journey):
        out = self.root / "render"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("odyssey_master_16_9.mp4", "odyssey_shorts_9_16.mp4"):
            (out / name).write_bytes(b"mp4")
        return {"duration": 30.0, "widescreen_16_9": str(out / "odyssey_master_16_9.mp4"),
                "vertical_9_16": str(out / "odyssey_shorts_9_16.mp4")}

    monkeypatch.setattr(pipeline.MakeProject, "render", fake_render)
    c = TestClient(app)
    c.images = images
    return c


def wait_done(client: TestClient, project_id: str) -> dict:
    for _ in range(200):
        st = client.get(f"/api/make/{project_id}/status").json()
        if st["status"] in ("COMPLETED", "FAILED"):
            return st
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_plan_run_edit_and_rerun(client: TestClient):
    res = client.post("/api/make/plan", json={"prompt": "a small car riding through cities", "num_scenes": 3})
    assert res.status_code == 201
    view = res.json()
    pid = view["project_id"]
    assert view["scene_ids"] == ["paris", "tokyo", "new-york"]
    assert len(view["pending_image_calls"]) == 6

    assert client.post(f"/api/make/{pid}/run").status_code == 202
    st = wait_done(client, pid)
    assert st["status"] == "COMPLETED", st
    assert st["videos"]["16:9"] == f"/outputs/make/{pid}/render/odyssey_master_16_9.mp4"
    assert len(client.images.calls) == 6

    # Change Tokyo's look and add Rome at the front: Tokyo + its bridges, Rome + its bridge are stale
    p = client.get(f"/api/make/{pid}").json()["plan"]
    p["scenes"][1]["visual"] = "neon streets at night"
    p["scenes"].insert(0, {"name": "Rome", "visual": "Colosseum", "ground_material": "STONE", "ambient_mood": "SILENT"})
    res = client.put(f"/api/make/{pid}/plan", json=p)
    assert res.status_code == 200
    view = res.json()
    assert sorted(view["removed"]) == ["bridges/paris--tokyo.png", "bridges/tokyo--new-york.png", "scenes/tokyo.png"]
    assert sorted(view["pending_image_calls"]) == sorted([
        "scene: Rome", "scene: Tokyo", "bridge: Rome -> Paris", "bridge: Paris -> Tokyo", "bridge: Tokyo -> New York",
    ])

    # Render-only change costs nothing
    client.images.calls.clear()
    client.put(f"/api/make/{pid}/options", json={"seconds_per_scene": 20})
    client.post(f"/api/make/{pid}/run")
    assert wait_done(client, pid)["status"] == "COMPLETED"
    assert len(client.images.calls) == 5
    assert client.get(f"/api/make/{pid}").json()["options"]["seconds_per_scene"] == 20


def test_redo_actor_and_unknown_target(client: TestClient):
    pid = client.post("/api/make/plan", json={"prompt": "car", "num_scenes": 3}).json()["project_id"]
    client.post(f"/api/make/{pid}/run")
    wait_done(client, pid)

    res = client.post(f"/api/make/{pid}/redo", json={"target": "actor"})
    assert res.json()["pending_image_calls"] == ["actor: Tiny Car"]
    assert client.post(f"/api/make/{pid}/redo", json={"target": "mars"}).status_code == 422


def test_new_prompt_never_reuses_an_existing_project(client: TestClient):
    a = client.post("/api/make/plan", json={"prompt": "car", "num_scenes": 3}).json()["project_id"]
    b = client.post("/api/make/plan", json={"prompt": "car", "num_scenes": 3}).json()["project_id"]
    assert (a, b) == ("car", "car-2")


def test_invalid_and_missing_project_ids(client: TestClient):
    assert client.get("/api/make/..%2Fsecrets").status_code in (400, 404)
    assert client.get("/api/make/Bad_ID").status_code == 400
    assert client.get("/api/make/nope").status_code == 404
