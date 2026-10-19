# Panorama Video Maker

A vertically integrated engine for side-scrolling odyssey videos, continuous tracking shots, and endless wandering animations with stride-locked physics and procedural soundscapes.

## Quick Start

### 1. Prerequisites
- Python `>= 3.13`, [FFmpeg](https://ffmpeg.org/), and [uv](https://github.com/astral-sh/uv).

### 2. Run Odyssey Production (A-to-Z)
Render full 16:9 widescreen and 9:16 vertical mobile cuts with synchronized foley audio:
```bash
uv sync
uv run python -m workshop.odyssey.orchestrator timelines/cat_odyssey.example.toml -o outputs/odyssey
```

### 3. Make a Video from One Prompt
Plans the journey with Gemini, generates the actor, one background per place, and the bridges between them with OpenAI, then renders 16:9 and 9:16 cuts:
```bash
uv run python -m workshop.make "a small car riding through world's cities"
```
- Needs `GOOGLE_GENERATIVE_AI_API_KEY` (planning) and `OPENAI_API_KEY` (images) in `.env`.
- Shows the plan and the number of image API calls before spending anything (`--yes` skips the prompt, `--plan-only` stops after planning).
- Everything lands in `outputs/make/<prompt>/`. Re-running reuses existing files, so edit `plan.json` or delete one scene image and re-run to change only that part.
- Supports rolling and floating actors (cars, trains, balloons, boats). Walking characters still need the video-to-sprite workflow.
- MCP: `plan_prompt_journey` → `make_journey` → `get_journey_status`.

### 4. Studio Server & Web UI
```bash
uv run uvicorn workshop.server.app:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Web Studio: `http://localhost:8000/web`

### 5. MCP Agent Server
Add to your agent configuration (`claude_desktop_config.json`, Cursor, etc.):
```json
{
  "mcpServers": {
    "panorama-video-maker": {
      "command": "uv",
      "args": ["run", "python", "-m", "workshop.mcp_server"],
      "cwd": "/path/to/image-generating"
    }
  }
}
```

### 6. Testing
```bash
uv run pytest
```

## License
[MIT](LICENSE)
