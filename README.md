# Odyssey Studio

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

### 3. Studio Server & Web UI
```bash
uv run uvicorn workshop.server.app:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Web Studio: `http://localhost:8000/web`

### 4. MCP Agent Server
Add to your agent configuration (`claude_desktop_config.json`, Cursor, etc.):
```json
{
  "mcpServers": {
    "odyssey-studio": {
      "command": "uv",
      "args": ["run", "python", "-m", "workshop.mcp_server"],
      "cwd": "/path/to/image-generating"
    }
  }
}
```

### 5. Testing
```bash
uv run pytest
```

## License
[MIT](LICENSE)
