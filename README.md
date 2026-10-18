# Panorama Video Maker

A full-stack, modular AI video production platform and compositing engine featuring **FastAPI**, an **SQLite RDB** storage layer for multi-scene projects and asset lineage, an **MCP (Model Context Protocol)** server for autonomous AI agent tool-calling, and deep-learning media processing pipelines (ONNX background matting, panoramic transition bridge synthesis, DP seam blending, and ffmpeg pipe-streaming).

```
data/             SQLite database storage (studio.db) [gitignored]
inputs/           raw source video footage (.mp4) and bridges (.png) [gitignored]
jobs/             declarative job configurations (*.example.toml)
timelines/        multi-scene continuous timeline configurations (*.example.toml)
outputs/          rendered media results and project assets [gitignored]
tests/            automated test suite (API, MCP, engine unit/integration)
web/              local sprite & timeline studio player (served at /web)
workshop/         core platform modules:
  ├── db/         SQLModel schemas (Project, Scene, Asset, RenderJob) & SQLite session
  ├── server/     FastAPI REST API server with static web player mount
  ├── services/   timeline configuration builders & asynchronous render runner
  ├── mcp_server.py  Model Context Protocol (MCP) server for AI agent tool calling
  └── [engine]    bgremove, character, clean, composite, imagen, jobs, long_composite,
                  music_station, recut, scenes, sprite, timeline, utils
```

---

## Prerequisites

- **Python**: `>= 3.13`
- **FFmpeg**: Required on system `$PATH` for video decoding, encoding, and stream piping.
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **uv** (recommended) or **mise** for dependency and environment management.

---

## Quick Start

### Option A: Using `uv` (Recommended)

```bash
# 1. Install dependencies
uv sync

# 2. Run automated test suite
uv run pytest

# 3. Launch FastAPI REST Server & Studio Web UI
uv run uvicorn workshop.server.app:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Studio Preview & Timeline Player**: [http://localhost:8000/web](http://localhost:8000/web)

### Option B: Using `mise`

```bash
# Install toolchain and Python virtual environment
mise install && mise run setup

# Run test suite
mise run test

# Launch server
mise run server
```

---

## CLI Rendering & Workflows

The platform provides command-line utilities for video processing, sprite sheet generation, and long-form continuous timeline compositing:

### 1. Multi-Scene Continuous Timeline Rendering
Render a multi-scene tracking shot defined in a TOML specification:
```bash
uv run python -m workshop.long_composite timelines/timeline.example.toml -o outputs/timeline_render.mp4
```

### 2. Declarative Job Runner
Execute reproducible video processing pipelines (slicing, matting, cleaning, compositing):
```bash
# Validate without running
uv run python -m workshop.jobs jobs/pipeline.example.toml --dry-run

# Execute pipeline
uv run python -m workshop.jobs jobs/pipeline.example.toml
```

### 3. Video to Cutout Sprite Sheet Extraction
Extract walking or action cycles from video clips with automatic background matting:
```bash
uv run python -m workshop.sprite inputs/sample.mp4 -o outputs/sprite.png --crop
```

---

## Platform Architecture & Features

### 1. Web REST API & SQLite Persistence Layer (`workshop.server`, `workshop.db`)
Built with **FastAPI** and **SQLModel** (SQLAlchemy 2.0 + Pydantic v2):
- **Projects**: Manage multi-scene video timeline specifications (resolution, framerate, background colors/gradients, character tracking parameters).
- **Scenes**: Sequential timeline scenes with custom scales, vertical alignment, overlap transitions, feathering, and camera hold durations.
- **Asset Lineage & Provenance**: Track generated media assets (video, image, sprite, audio) with model names (`gpt-image-2.5`, `gemini-omni-1.1-flash`), generation prompts, version numbers, and project associations.
- **Asynchronous Render Worker**: Background execution (`QUEUED` ➔ `PROCESSING` ➔ `COMPLETED` / `FAILED`) executing composite rendering without blocking API requests.

### 2. Model Context Protocol (MCP) Server (`workshop.mcp_server`)
Equips AI agents (Claude Desktop, Cursor, Antigravity, etc.) with standardized tool-calling capabilities to programmatically orchestrate video production:
- `create_project(title, width, height, fps, ...)`: Initialize a new video project.
- `add_scene(project_id, sheet_path, order_idx, ...)`: Insert a scene clip into the timeline.
- `set_character_actor(project_id, char_sheet, stride_pixels, ...)`: Configure character tracking.
- `record_asset(project_id, asset_type, file_path, model_name, prompt)`: Log generative provenance.
- `render_project_video(project_id)`: Trigger background composite rendering.
- `check_render_status(job_id)`: Monitor render progress (0~100%) and retrieve the final video path.

#### Example MCP Configuration
Add to your agent configuration (e.g. `claude_desktop_config.json` or Cursor MCP settings):
```json
{
  "mcpServers": {
    "panorama-video-maker": {
      "command": "uv",
      "args": ["run", "python", "-m", "workshop.mcp_server"],
      "cwd": "/path/to/panorama-video-maker"
    }
  }
}
```

### 3. Core Media Engine (`workshop`)
- **Sprite Extraction & Matting (`workshop.sprite`, `workshop.bgremove`)**: Deep learning background matting with ONNX Runtime (RMBG 2.0, BiRefNet) and transparent border normalization.
- **Recutting & Grid Repacking (`workshop.recut`, `workshop.clean`)**: Repack arbitrary frame sequences into neat walk/action cycle sprite grids.
- **Panoramic Bridge Synthesis (`workshop.imagen`)**: Seamless horizontal scene outpainting and keyframe video synthesis via OpenAI and Gemini Omni.
- **Dynamic Seam Blending (`workshop.scenes`)**: Dynamic programming minimum-error boundary seam stitching with motion-preservation penalties.
- **1D World Timeline Solver & Pipe-Streaming Renderer (`workshop.timeline`, `workshop.long_composite`)**: Stride-locking (camera velocity matched to actor pace), contact ground drop shadows, and constant-memory ffmpeg pipe streaming.

---

## Environment Variables

For generative AI features (scene outpainting and transition bridge synthesis) or Hugging Face model downloads, configure a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

| Variable | Description |
| :--- | :--- |
| `OPENAI_API_KEY` | *(Optional)* OpenAI API key for DALL-E / GPT image outpainting and transition bridges. |
| `GOOGLE_GENERATIVE_AI_API_KEY` | *(Optional)* Google Gemini API key for generative video and image synthesis. |
| `HF_TOKEN` | *(Optional)* Hugging Face access token for downloading gated ONNX matting models (e.g. RMBG 2.0). |

---

## Testing

```bash
# Run unit and integration tests
uv run pytest
```

The test suite covers:
- FastAPI REST endpoints and SQLite database transactions (`test_api.py`)
- MCP tool-calling workflows and schema serialization (`test_mcp.py`)
- Sprite slicing, metadata loading, and grid packing (`test_recut.py`, `test_clean.py`)
- Minimum-error DP seam finding and image feathering (`test_timeline.py`)
- Pipe-streaming `AnimationWriter` (`test_writer.py`)
- Declarative job validation and execution (`test_jobs.py`)

---

## License

This project is licensed under the [MIT License](LICENSE).
