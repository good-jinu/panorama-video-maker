"""Tests for Model Context Protocol (MCP) server tools."""
import json
from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

import workshop.db.database as db_mod
from workshop.mcp_server import (
    add_scene,
    create_project,
    get_project_details,
    record_asset,
    set_character_actor,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "mcp_test.db"
    test_engine = create_engine(f"sqlite:///{test_db}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(test_engine)

    monkeypatch.setattr(db_mod, "get_engine", lambda: test_engine)
    import workshop.mcp_server as mcp_mod
    monkeypatch.setattr(mcp_mod, "get_engine", lambda: test_engine)
    yield


def test_mcp_workflow_end_to_end():
    # 1. Create project via MCP
    created_raw = create_project(
        title="Weekly Financial Market Wrap",
        description="Auto-generated video via AI agent tool calling",
        canvas_w=1920,
        canvas_h=1080,
        fps=30,
    )
    created = json.loads(created_raw)
    assert created["status"] == "success"
    p_id = created["project_id"]

    # 2. Add scenes via MCP
    s1_raw = add_scene(p_id, sheet_path="outputs/market/intro.png", name="intro", order_idx=0)
    assert json.loads(s1_raw)["status"] == "success"

    s2_raw = add_scene(p_id, sheet_path="outputs/market/chart.png", name="kospi_chart", order_idx=1, overlap_prev=100)
    assert json.loads(s2_raw)["status"] == "success"

    # 3. Configure Character via MCP
    char_raw = set_character_actor(p_id, char_sheet="outputs/actors/anchor.png", stride_pixels=140.0)
    assert json.loads(char_raw)["status"] == "success"

    # 4. Record Asset Lineage via MCP
    asset_raw = record_asset(
        project_id=p_id,
        asset_type="image",
        file_path="outputs/market/chart.png",
        model_name="gemini-omni-1.1-flash",
        prompt="KOSPI and NASDAQ weekly comparison chart",
    )
    assert json.loads(asset_raw)["status"] == "success"

    # 5. Inspect full details via MCP
    details_raw = get_project_details(p_id)
    details = json.loads(details_raw)
    assert details["title"] == "Weekly Financial Market Wrap"
    assert len(details["scenes"]) == 2
    assert len(details["assets"]) == 1
    assert details["assets"][0]["model"] == "gemini-omni-1.1-flash"


def test_mcp_advanced_pipeline_tools(monkeypatch, tmp_path):
    import workshop.mcp_server as mcp_mod

    # Mock services
    fake_sprite = tmp_path / "sprite.png"
    fake_sprite.write_text("fake")
    monkeypatch.setattr(mcp_mod, "extract_sprite_pipeline", lambda **k: fake_sprite)
    monkeypatch.setattr(mcp_mod, "generate_bridge_pipeline", lambda **k: tmp_path / "bridge.png")
    monkeypatch.setattr(mcp_mod, "generate_ai_video_pipeline", lambda **k: tmp_path / "ai_vid.mp4")
    monkeypatch.setattr(mcp_mod, "build_music_station_pipeline", lambda **k: tmp_path / "music.mp4")

    # 1. extract_sprite_from_video
    res = json.loads(mcp_mod.extract_sprite_from_video("video.mp4", keep_background=True))
    assert res["status"] == "success"

    # 2. generate_transition_bridge
    res = json.loads(mcp_mod.generate_transition_bridge("a.png", "b.png", prompt="divider"))
    assert res["status"] == "success"

    # 3. generate_ai_video_clip
    res = json.loads(mcp_mod.generate_ai_video_clip("animate", "out.mp4"))
    assert res["status"] == "success"

    # 4. build_music_station_video
    res = json.loads(mcp_mod.build_music_station_video("vid.mp4", ["https://youtube.com/watch?v=abc"]))
    assert res["status"] == "success"
    assert res["tracks_count"] == 1


def test_mcp_import_timeline_example():
    import workshop.mcp_server as mcp_mod

    # Test import_timeline_toml with generic example
    import_res = json.loads(mcp_mod.import_timeline_toml("timelines/timeline.example.toml"))
    assert import_res["status"] == "success"
    assert import_res["title"] == "Timeline Example"
    assert import_res["scenes_count"] >= 2
    assert "project_id" in import_res


def test_mcp_render_odyssey_journey(monkeypatch, tmp_path):
    import workshop.mcp_server as mcp_mod

    # Mock OdysseyOrchestrator to avoid full rendering in unit test
    class MockOrchestrator:
        def __init__(self, journey, root_dir=None):
            self.journey = journey

        def run(self, output_dir):
            return {
                "title": self.journey.title,
                "duration": 12.0,
                "widescreen_16_9": str(tmp_path / "master.mp4"),
                "vertical_9_16": str(tmp_path / "shorts.mp4"),
                "footstep_count": 14,
            }

    import workshop.odyssey.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "OdysseyOrchestrator", MockOrchestrator)

    res_raw = mcp_mod.render_odyssey_journey("timelines/cat_odyssey.example.toml", output_dir=str(tmp_path))
    res = json.loads(res_raw)
    assert res["status"] == "success"
    assert res["production"]["title"] == "Cat Odyssey: Journey Across Horizons"
    assert res["production"]["footstep_count"] == 14


