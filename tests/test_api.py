"""Automated integration tests for the FastAPI web server and database layer."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from workshop.db.database import get_session
from workshop.server.app import app


@pytest.fixture(name="client")
def client_fixture(tmp_path: Path):
    # Use isolated temporary SQLite database for tests
    test_db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(test_engine)

    def override_get_session():
        with Session(test_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_project_crud_lifecycle(client: TestClient):
    # 1. Create project
    create_payload = {
        "title": "KOSPI Daily Morning Briefing",
        "description": "Automated 60s financial recap video",
        "canvas_w": 1080,
        "canvas_h": 1920,
        "fps": 30,
        "bg_color": "#121b28",
        "char_scale": 0.5,
    }
    res = client.post("/api/projects", json=create_payload)
    assert res.status_code == 201
    data = res.json()
    project_id = data["id"]
    assert data["title"] == "KOSPI Daily Morning Briefing"
    assert data["canvas_w"] == 1080

    # 2. List projects
    res = client.get("/api/projects")
    assert res.status_code == 200
    assert len(res.json()) >= 1

    # 3. Update project
    update_res = client.put(f"/api/projects/{project_id}", json={"title": "Updated Briefing"})
    assert update_res.status_code == 200
    assert update_res.json()["title"] == "Updated Briefing"

    # 4. Get detail with empty relations
    detail_res = client.get(f"/api/projects/{project_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["title"] == "Updated Briefing"
    assert detail["scenes"] == []
    assert detail["assets"] == []


def test_scene_and_asset_management(client: TestClient):
    # Setup project
    p = client.post("/api/projects", json={"title": "Tech Sector Analysis"}).json()
    project_id = p["id"]

    # Add scene 1
    scene1_payload = {
        "project_id": project_id,
        "order_idx": 0,
        "name": "opening_indices",
        "sheet_path": "outputs/sector/scene1.png",
        "scale": 1.0,
    }
    s1 = client.post(f"/api/projects/{project_id}/scenes", json=scene1_payload)
    assert s1.status_code == 201
    assert s1.json()["name"] == "opening_indices"

    # Add scene 2
    scene2_payload = {
        "project_id": project_id,
        "order_idx": 1,
        "name": "semiconductor_breakdown",
        "sheet_path": "outputs/sector/scene2.png",
        "overlap_prev": 120,
        "scale": 0.85,
    }
    s2 = client.post(f"/api/projects/{project_id}/scenes", json=scene2_payload)
    assert s2.status_code == 201

    # Register an asset with prompt provenance
    asset_payload = {
        "project_id": project_id,
        "asset_type": "image",
        "file_path": "outputs/sector/scene1.png",
        "model_name": "gpt-image-2.5",
        "prompt": "Bullish semiconductor trading floor infographic",
        "version": 1,
    }
    a1 = client.post(f"/api/projects/{project_id}/assets", json=asset_payload)
    assert a1.status_code == 201
    assert a1.json()["model_name"] == "gpt-image-2.5"

    # Verify project details now include scenes and assets
    detail = client.get(f"/api/projects/{project_id}").json()
    assert len(detail["scenes"]) == 2
    assert len(detail["assets"]) == 1
    assert detail["scenes"][0]["name"] == "opening_indices"
    assert detail["scenes"][1]["name"] == "semiconductor_breakdown"


def test_render_job_trigger_and_status(client: TestClient):
    p = client.post("/api/projects", json={"title": "Render Pipeline Test"}).json()
    project_id = p["id"]

    # Trigger render (no scenes should still queue or fail gracefully)
    render_res = client.post(f"/api/projects/{project_id}/render")
    assert render_res.status_code == 202
    job_data = render_res.json()
    job_id = job_data["id"]
    assert job_data["status"] == "QUEUED"

    # Check status endpoint
    status_res = client.get(f"/api/jobs/{job_id}")
    assert status_res.status_code == 200
    assert status_res.json()["id"] == job_id


def test_pipeline_routes(client: TestClient, monkeypatch, tmp_path):
    import workshop.services.pipeline_service as pipe_svc

    # Mock underlying pipeline functions
    fake_sprite = tmp_path / "sprite.png"
    fake_sprite.write_text("fake")
    monkeypatch.setattr(pipe_svc, "extract_sprite_pipeline", lambda **k: fake_sprite)
    monkeypatch.setattr(pipe_svc, "generate_bridge_pipeline", lambda **k: tmp_path / "bridge.png")
    monkeypatch.setattr(pipe_svc, "generate_ai_video_pipeline", lambda **k: tmp_path / "ai_vid.mp4")
    monkeypatch.setattr(pipe_svc, "build_music_station_pipeline", lambda **k: tmp_path / "music.mp4")

    # 1. extract-sprite
    res = client.post("/api/pipeline/extract-sprite", json={"video_path": "fake.mp4", "keep_background": True})
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 2. generate-bridge
    res = client.post("/api/pipeline/generate-bridge", json={"from_sheet": "a.png", "to_sheet": "b.png"})
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 3. generate-ai-video
    res = client.post("/api/pipeline/generate-ai-video", json={"prompt": "animate cat", "output_path": "out.mp4"})
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 4. music-station
    res = client.post("/api/pipeline/music-station", json={"video_path": "in.mp4", "youtube_urls": ["https://youtube.com/watch?v=123"]})
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_timeline_import_and_render_routes(client: TestClient, monkeypatch, tmp_path):
    # Test timeline import endpoint with generic example
    res = client.post("/api/timeline/import", json={"toml_path": "timelines/timeline.example.toml"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["title"] == "Timeline Example"
    assert data["scenes_count"] >= 2
    project_id = data["project_id"]

    # Test project render trigger
    render_res = client.post(f"/api/projects/{project_id}/render")
    assert render_res.status_code == 202
    assert render_res.json()["status"] == "QUEUED"


