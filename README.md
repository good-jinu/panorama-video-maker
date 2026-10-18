# Video & Sprite Compositing Framework

A modular Python framework and engine for video frame extraction, ONNX-based background removal, sprite sheet recutting, AI-assisted panoramic transition bridge synthesis, and continuous camera-tracking long-form video compositing.

```
inputs/
  videos/      raw source video footage (.mp4) [gitignored]
  bridges/     panoramic bridge / background extensions (.png) [gitignored]
jobs/          declarative job configurations (*.toml, *.example.toml)
timelines/     multi-scene continuous timeline configurations (*.toml, *.example.toml)
outputs/       rendered media results (fully reproducible via jobs) [gitignored]
tests/         test suite (unit, integration, visual regression snapshots)
workshop/      core engine modules:
                 bgremove, character, clean, comparison, composite,
                 imagen, jobs, long_composite, music_station,
                 recut, scenes, sprite, timeline, utils
web/           local sprite & timeline preview player
```

---

## Quick Start (mise)

```bash
# Install toolchain and dependencies
mise install && mise run setup

# List available declarative jobs
mise run job

# Run a declarative job (e.g. tracking.example)
mise run job tracking.example --dry-run

# Run automated tests
mise run test

# Launch local preview web player
mise run web     # http://localhost:8765/web/index.html
```

---

## Core Capabilities & Modules

### 1. Sprite Extraction & Background Removal (`workshop.sprite`, `workshop.bgremove`)
Extracts frames from video clips into compact RGBA sprite sheets using deep learning matting models:
- **ONNX Background Matting**: Supports RMBG 2.0 and BiRefNet with automatic category routing.
- **Cropping & Normalization**: Automatically trims transparent borders and normalizes frame bounds.

```bash
# Extract transparent sprite sheet with BiRefNet
uv run python -m workshop.sprite inputs/videos/character.mp4 --crop --model birefnet

# Extract background with camera motion compensation
uv run python -m workshop.sprite inputs/videos/scene.mp4 --zoom-start 1.37 --zoom-step 0.0095
```

### 2. Recutting & Frame Repacking (`workshop.recut`, `workshop.clean`)
Slices arbitrary frame ranges and repacks them into regular grids for clean animation walk/action cycles:

```bash
# Slice specific frames into a 4-column sprite sheet
uv run python -m workshop.recut outputs/character/sprite.png --frames 1-16 --columns 4 -o outputs/character/sprite_walk.png

# Alpha thresholding and edge cleaning
uv run python -m workshop.clean outputs/character/sprite.png -o outputs/character/sprite_cleaned.png
```

### 3. AI Scene Extension & Transition Bridges (`workshop.imagen`)
Synthesizes panoramic extensions and seamless bridge images between scenes using multimodal generative models:
- OpenAI Image Inpainting/Outpainting (`gpt-image-2.5-sunburst-2026-09-08`)
- Gemini Omni Video Generation (`gemini-omni-1.1-flash`)

```bash
# Generate panoramic transition bridge between two scene sheets
uv run python -m workshop.imagen \
  --from-sheet outputs/scene1/sprite_bg.png \
  --to-sheet outputs/scene2/sprite_bg.png \
  --from-scale 0.70 --to-scale 0.75 \
  -o inputs/bridges/scene1-to-scene2-bridge.png
```

### 4. Dynamic Seam Stitching & Blending (`workshop.scenes`)
Aligns and blends static extended backgrounds with animated video loops using dynamic programming minimum-error seam boundaries with motion penalties, preventing visual jitter.

### 5. Multi-Scene Timeline Solving & Streaming Compositor (`workshop.timeline`, `workshop.long_composite`)
- **1D World Coordinate Solver**: Solves scene entry/exit, overlap feathering, and piecewise-linear camera holds.
- **Stride-Locking**: Locks camera movement velocity to the character actor's walk pace to prevent foot/ground sliding.
- **Contact Ground Shadow**: Real-time perspective-correct contact shadow generation.
- **Pipe-Streaming Video Render**: Directly streams synthesized RGB frames to `ffmpeg` via standard pipe, keeping memory footprint constant regardless of video length.

```bash
# Render timeline video from a TOML configuration
uv run python -m workshop.long_composite timelines/timeline.example.toml -o outputs/story/animation.mp4
```

### 6. Declarative Job Runner (`workshop.jobs`)
Define reproducible multi-step pipelines in simple TOML files in `jobs/`:

```bash
# List discovered jobs
uv run python -m workshop.jobs

# Validate a job without executing
uv run python -m workshop.jobs tracking.example --dry-run
```

---

## Testing

```bash
mise run test
```
The test suite covers:
- Sprite slicing, metadata loading, and grid packing
- Minimum-error DP seam finding and image feathering
- Timeline world-coordinate solving and camera positioning
- Pipe-streaming `AnimationWriter` (MP4/GIF)
- Declarative job validation and execution
