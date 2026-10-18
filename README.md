# AI Video Studio & Compositing Platform

A full-stack, modular AI video production platform and compositing engine featuring **FastAPI**, an **SQLite RDB** storage layer for multi-scene projects and asset lineage, an **MCP (Model Context Protocol)** server for autonomous AI agent tool-calling, and deep-learning media processing pipelines (ONNX background matting, panoramic transition bridge synthesis, DP seam blending, and ffmpeg pipe-streaming).

```
data/             SQLite database storage (studio.db) [gitignored]
inputs/           raw source video footage (.mp4) and bridges (.png) [gitignored]
jobs/             declarative job configurations (*.toml, *.example.toml)
timelines/        multi-scene continuous timeline configurations (*.toml, *.example.toml)
outputs/          rendered media results and project assets [gitignored]
tests/            test suite (API, MCP, engine unit/integration, visual regression)
web/              local sprite & timeline preview player (served at /web)
workshop/         core platform modules:
  ├── db/         SQLModel schemas (Project, Scene, Asset, RenderJob) & SQLite session
  ├── server/     FastAPI REST API server with static web player mount
  ├── services/   timeline configuration builders & asynchronous render runner
  ├── mcp_server.py  Model Context Protocol (MCP) server for AI agent tool calling
  └── [engine]    bgremove, character, clean, composite, imagen, jobs, long_composite,
                  music_station, recut, scenes, sprite, timeline, utils
```

---

## Quick Start (mise)

```bash
# 1. Install toolchain and dependencies
mise install && mise run setup

# 2. Run automated test suite (API, MCP, and engine)
mise run test

# 3. Launch the FastAPI Web Server (Interactive Swagger docs at http://localhost:8000/docs)
mise run server

# 4. Launch the Model Context Protocol (MCP) server for AI agents
mise run mcp
```

---

## Platform Architecture & Features

### 1. Web REST API & SQLite Database Layer (`workshop.server`, `workshop.db`)
Built with **FastAPI** and **SQLModel** (SQLAlchemy 2.0 + Pydantic v2):
- **Projects**: Manage multi-scene video timeline specifications (resolution, framerate, background color/gradients, character actor parameters).
- **Scenes**: Sequential timeline scenes with scale, vertical alignment, overlap transitions, feathering, and camera hold durations.
- **Asset Lineage & Provenance**: Track generated media assets (video, image, sprite, audio) with model names (`gpt-image-2.5`, `gemini-omni-1.1-flash`), generation prompts, version numbers, and parent project references.
- **Asynchronous Render Jobs**: State machine (`QUEUED` ➔ `PROCESSING` ➔ `COMPLETED` / `FAILED`) executing composite rendering in background worker threads without blocking API requests.

Interactive documentation is accessible at `http://localhost:8000/docs`.

### 2. Model Context Protocol (MCP) Server (`workshop.mcp_server`)
Equips AI agents (Claude Desktop, Cursor, Antigravity, etc.) with standardized tool-calling capabilities to orchestrate the video production pipeline:
- `create_project(title, width, height, fps, ...)`: Initialize a new video project.
- `add_scene(project_id, sheet_path, order_idx, ...)`: Insert a scene clip into the timeline.
- `set_character_actor(project_id, char_sheet, stride_pixels, ...)`: Configure character tracking.
- `record_asset(project_id, asset_type, file_path, model_name, prompt)`: Log generative provenance.
- `render_project_video(project_id)`: Trigger background composite rendering.
- `check_render_status(job_id)`: Monitor progress (0~100%) and retrieve the final video path.

Configure with Claude Desktop (`claude_desktop_config.json`) or Cursor MCP settings:
```json
{
  "mcpServers": {
    "video-studio": {
      "command": "uv",
      "args": ["run", "python", "-m", "workshop.mcp_server"],
      "cwd": "/path/to/video-compositing-studio"
    }
  }
}
```

### 3. Core Media Engine (`workshop`)
- **Sprite Extraction & Matting (`workshop.sprite`, `workshop.bgremove`)**: Deep learning background matting with ONNX Runtime (RMBG 2.0, BiRefNet) and transparent border normalization.
- **Recutting & Grid Repacking (`workshop.recut`, `workshop.clean`)**: Repack arbitrary frame sequences into neat walk/action cycle sprite grids.
- **AI Scene Inpainting & Bridge Synthesis (`workshop.imagen`)**: Panoramic horizontal scene outpainting and keyframe video synthesis via OpenAI and Gemini Omni.
- **Dynamic Seam Blending (`workshop.scenes`)**: Dynamic programming minimum-error boundary seam stitching with motion-preservation penalties.
- **1D World Timeline Solver & Pipe-Streaming Renderer (`workshop.timeline`, `workshop.long_composite`)**: Stride-locking (camera velocity matched to actor paw/foot pace), contact ground drop shadows, and constant-memory ffmpeg pipe streaming.
- **Declarative Job Runner (`workshop.jobs`)**: TOML-based declarative reproducible pipelines (`jobs/*.toml`).

---

## Testing

```bash
mise run test
```
The test suite includes 72 automated unit and integration tests covering:
- FastAPI REST endpoints and SQLite database transactions (`test_api.py`)
- MCP tool-calling workflows and schema serialization (`test_mcp.py`)
- Sprite slicing, metadata loading, and grid packing (`test_recut.py`, `test_clean.py`)
- Minimum-error DP seam finding and image feathering (`test_timeline.py`)
- Pipe-streaming `AnimationWriter` (`test_writer.py`)
- Declarative job validation and execution (`test_jobs.py`)
