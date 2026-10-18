"""High-level scene composition, extension, and lazy caching.

Provides self-contained abstractions for scenes:
- compose_extended_scene: cleanly stitches a static wide background with an animated sprite sheet
- LoadedScene: encapsulates ready-to-render frames with time-based frame retrieval
- SceneCache: keeps only viewport-visible scenes in RAM during long timeline tracking shots
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from workshop.utils import create_scene_radial_backdrop, load_sprite_sheet


def _compute_overlap_seam_mask(
    bg_rgba: Image.Image,
    vid_frames: list[Image.Image],
    offset: int,
    actual_video_y: int,
    feather_px: float = 8.0,
) -> Optional[np.ndarray]:
    """Compute a 2D alpha mask (vh, vw) that smoothly transitions across the minimum-error seam."""
    vid_frame = vid_frames[0]
    vw, vh = vid_frame.size
    bw, bh = bg_rgba.size

    # Restrict seam search to the left anchor zone (at most 28% of video width) so we never
    # mask out animated characters in the middle/right of the scene when left_bg spans full width.
    max_anchor_w = max(16, round(vw * 0.28))
    ov_x0 = max(0, offset)
    ov_x1 = min(bw, offset + max_anchor_w)
    ov_y0 = max(0, actual_video_y)
    ov_y1 = min(bh, actual_video_y + vh)

    ov_w = ov_x1 - ov_x0
    ov_h = ov_y1 - ov_y0
    if ov_w < 16 or ov_h < 16:
        return None

    vy0 = ov_y0 - actual_video_y
    vy1 = ov_y1 - actual_video_y
    vx0 = ov_x0 - offset
    vx1 = ov_x1 - offset

    bg_ov = np.array(bg_rgba)[ov_y0:ov_y1, ov_x0:ov_x1, :3].astype(np.float32)
    vid_ov = np.array(vid_frame)[vy0:vy1, vx0:vx1, :3].astype(np.float32)

    diff = np.abs(bg_ov - vid_ov).mean(axis=2)
    grad_bg = np.abs(np.gradient(bg_ov, axis=1)).mean(axis=2)
    grad_vid = np.abs(np.gradient(vid_ov, axis=1)).mean(axis=2)
    cost = diff + 2.0 * (grad_bg + grad_vid)

    # Penalize placing the seam to the right of animated pixels so motion is always preserved
    if len(vid_frames) > 1:
        step = max(1, len(vid_frames) // 10)
        stack = np.stack(
            [np.array(f)[vy0:vy1, vx0:vx1, :3].astype(np.float32) for f in vid_frames[::step]],
            axis=0,
        )
        motion_std = stack.std(axis=0).mean(axis=2)
        cum_motion = np.cumsum(np.maximum(0.0, motion_std - 3.0), axis=1) * 0.15
        cost = cost + cum_motion

    left_margin = max(2, min(ov_w // 4, round(ov_w * 0.10)))
    right_margin = max(2, min(ov_w // 4, round(ov_w * 0.18)))
    if ov_w - left_margin - right_margin <= 2:
        left_margin = right_margin = 1

    cost[:, :left_margin] = 1e9
    cost[:, ov_w - right_margin :] = 1e9

    dp = np.full_like(cost, 1e12)
    back = np.zeros_like(cost, dtype=np.int32)
    dp[0] = cost[0]

    for y in range(1, ov_h):
        prev = dp[y - 1]
        for x in range(left_margin, ov_w - right_margin):
            c0 = prev[x]
            cm1 = prev[x - 1] + 2.0
            cp1 = prev[x + 1] + 2.0
            if c0 <= cm1 and c0 <= cp1:
                dp[y, x] = cost[y, x] + c0
                back[y, x] = x
            elif cm1 <= cp1:
                dp[y, x] = cost[y, x] + cm1
                back[y, x] = x - 1
            else:
                dp[y, x] = cost[y, x] + cp1
                back[y, x] = x + 1

    seam = np.zeros(ov_h, dtype=np.float32)
    seam[-1] = float(np.argmin(dp[-1]))
    for y in range(ov_h - 2, -1, -1):
        seam[y] = float(back[y + 1, int(seam[y + 1])])

    k = min(17, max(3, (ov_h // 8) | 1))
    pad = k // 2
    seam_smooth = np.convolve(np.pad(seam, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")

    # Map seam into full video coordinates
    seam_full = np.full(vh, float(vx0 + ov_w // 2), dtype=np.float32)
    seam_full[vy0:vy1] = vx0 + seam_smooth
    if vy0 > 0:
        seam_full[:vy0] = seam_full[vy0]
    if vy1 < vh:
        seam_full[vy1:] = seam_full[vy1 - 1]

    yy, xx = np.mgrid[:vh, :vw]
    alpha_x = np.clip((xx - seam_full[:, None]) / max(1.0, feather_px) + 0.5, 0.0, 1.0)
    return alpha_x


def compose_extended_scene(
    sprite_frames: list[Image.Image],
    left_bg: Image.Image,
    offset: int,
    video_scale: float = 1.0,
    left_bg_scale: float = 1.0,
    video_y: Union[int, str] = 0,
    seamless: bool = True,
    left_edge_feather: Optional[tuple[int, int]] = None,
    seam_feather_width: Optional[int] = None,
) -> list[Image.Image]:
    """Stitch a static left background extension onto an animated sprite sheet.

    Args:
        sprite_frames: List of animated frames (e.g. couple eating dinner).
        left_bg: Static panoramic background (e.g. outdoor patio + restaurant).
        offset: X-coordinate in the combined canvas where sprite_frames are pasted.
                The left_bg is pasted at (0, 0), and sprite_frames are pasted at (offset, video_y).
        video_scale: Scale factor applied to the animated video/sprite frames.
        left_bg_scale: Scale factor applied to the static background image (relative to sprite frame height).
        video_y: Vertical offset for the animated frames. Can be an integer (pixels from top),
                 or string "top" (0), "bottom" (comb_h - vid_h), or "center".
        seamless: When True, blends the overlap seam along the minimum-error path and extends
                  any lower foreground floor from left_bg underneath shorter video frames.
        left_edge_feather: Optional `(start_x, width_px)` alpha ramp applied to the left edge
                           of `base_canvas` so this extended scene blends seamlessly over a
                           preceding scene when `overlap_prev > 0`.
        seam_feather_width: Optional explicit pixel width for linear left-edge feather on the
                            sprite frames (overrides DP seam search when the sprite was cropped
                            directly from `left_bg`).

    Returns:
        A list of new composited frames of size (comb_w, comb_h).
    """
    if not sprite_frames:
        return []

    raw_w, raw_h = sprite_frames[0].size
    bg_rgba = left_bg.convert("RGBA")

    # Target video frame size
    vid_w = round(raw_w * video_scale)
    vid_h = round(raw_h * video_scale)
    if (vid_w, vid_h) != (raw_w, raw_h):
        resized_sprite_frames = [
            sf.resize((vid_w, vid_h), Image.Resampling.LANCZOS)
            for sf in sprite_frames
        ]
    else:
        resized_sprite_frames = sprite_frames

    # Target left_bg size based on left_bg_scale (relative to raw_h)
    target_bg_h = round(raw_h * left_bg_scale)
    target_bg_w = round(bg_rgba.width * (target_bg_h / bg_rgba.height))
    if (target_bg_w, target_bg_h) != bg_rgba.size:
        bg_rgba = bg_rgba.resize((target_bg_w, target_bg_h), Image.Resampling.LANCZOS)

    comb_w = max(target_bg_w, offset + vid_w)
    comb_h = max(target_bg_h, vid_h)

    # Resolve vertical placement of video
    if isinstance(video_y, str):
        v = video_y.strip().lower()
        if v == "bottom":
            actual_video_y = comb_h - vid_h
        elif v == "center":
            actual_video_y = (comb_h - vid_h) // 2
        elif v == "top":
            actual_video_y = 0
        else:
            try:
                actual_video_y = int(video_y)
            except ValueError:
                actual_video_y = 0
    else:
        actual_video_y = int(video_y or 0)

    # Prepare static base canvas with left_bg and optional foreground floor continuation
    base_canvas = Image.new("RGBA", (comb_w, comb_h), (0, 0, 0, 0))
    base_canvas.paste(bg_rgba, (0, 0))

    vid_bottom = actual_video_y + vid_h
    has_floor_extension = False
    if seamless and vid_bottom < target_bg_h and comb_w > target_bg_w and target_bg_w >= 64:
        has_floor_extension = True
        src_x0 = max(0, round(target_bg_w * 0.60))
        overlap_y0 = max(0, vid_bottom - 6)
        floor_src = bg_rgba.crop((src_x0, overlap_y0, target_bg_w, target_bg_h))
        floor_stretched = floor_src.resize(
            (comb_w - src_x0, target_bg_h - overlap_y0), Image.Resampling.LANCZOS
        )
        f_arr = np.array(floor_stretched).copy()
        blend_start_x = max(src_x0, target_bg_w - 80)
        blend_span = max(1.0, float(target_bg_w - blend_start_x))
        xs = np.arange(f_arr.shape[1], dtype=np.float32) + src_x0
        a_f = np.clip((xs - blend_start_x) / blend_span, 0.0, 1.0)
        f_arr[:, :, 3] = np.round(f_arr[:, :, 3].astype(np.float32) * a_f[None, :]).astype(np.uint8)
        base_canvas.alpha_composite(Image.fromarray(f_arr), (src_x0, overlap_y0))

    if left_edge_feather is not None:
        f_start, f_width = left_edge_feather
        b_arr = np.array(base_canvas).copy()
        xs_b = np.arange(comb_w, dtype=np.float32)
        a_left = np.clip((xs_b - float(f_start)) / max(1.0, float(f_width)), 0.0, 1.0)
        b_arr[:, :, 3] = np.round(b_arr[:, :, 3].astype(np.float32) * a_left[None, :]).astype(np.uint8)
        base_canvas = Image.fromarray(b_arr)

    alpha_x = None
    if seam_feather_width is not None and seam_feather_width > 0:
        xs_v = np.arange(vid_w, dtype=np.float32)[None, :]
        alpha_x = np.broadcast_to(np.clip(xs_v / float(seam_feather_width), 0.0, 1.0), (vid_h, vid_w)).copy()
    elif seamless and offset < target_bg_w:
        alpha_x = _compute_overlap_seam_mask(bg_rgba, resized_sprite_frames, offset, actual_video_y)

    stitched_frames = []
    yy = np.arange(vid_h, dtype=np.float32)[:, None]
    for sf in resized_sprite_frames:
        frame = base_canvas.copy()
        if alpha_x is not None:
            v_arr = np.array(sf.convert("RGBA")).copy()
            alpha = alpha_x
            if has_floor_extension or (seamless and vid_bottom < target_bg_h):
                lum = v_arr[:, :, :3].mean(axis=2)
                is_not_shoe = np.clip((lum - 75.0) / 45.0, 0.0, 1.0)
                alpha_y_soft = np.clip((vid_h - 1.0 - yy) / 6.0, 0.0, 1.0)
                alpha_y_crisp = np.clip((vid_h - 1.0 - yy) / 1.5, 0.0, 1.0)
                alpha_bottom = is_not_shoe * alpha_y_soft + (1.0 - is_not_shoe) * alpha_y_crisp
                alpha = alpha * alpha_bottom
            v_arr[:, :, 3] = np.round(v_arr[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
            frame.alpha_composite(Image.fromarray(v_arr), (offset, actual_video_y))
        else:
            frame.paste(sf, (offset, actual_video_y))
        stitched_frames.append(frame)

    return stitched_frames


@dataclass
class LoadedScene:
    """A fully loaded, scaled, ready-to-render scene in memory."""

    spec: "SceneSpec"
    frames: list[Image.Image]
    interval: float
    backdrop: Optional[Image.Image] = None
    pad_x: int = 0
    pad_y: int = 0

    @property
    def width(self) -> int:
        return self.spec.width

    @property
    def height(self) -> int:
        return self.spec.height

    @property
    def y(self) -> int:
        return self.spec.y

    def get_frame(self, t: float) -> Image.Image:
        """Get the animation frame for a given absolute story timeline time `t`."""
        segments = getattr(self.spec, "hold_frame_segments", None)
        if segments and getattr(self.spec, "hold_duration", 0.0) > 0:
            h_start = getattr(self.spec, "hold_start_time", 0.0)
            h_end = getattr(self.spec, "hold_end_time", 0.0)
            if t < h_start or t >= h_end:
                return self.frames[0]
            hold_step = int((t - h_start) / self.interval)
            cursor = 0
            for seg in segments:
                s_idx, e_idx, reps = int(seg[0]), int(seg[1]), int(seg[2])
                seg_len = max(1, e_idx - s_idx + 1)
                span = seg_len * max(1, reps)
                if hold_step < cursor + span:
                    local_step = hold_step - cursor
                    frame_idx = min(len(self.frames) - 1, s_idx + (local_step % seg_len))
                    return self.frames[frame_idx]
                cursor += span
            last_seg = segments[-1]
            return self.frames[min(len(self.frames) - 1, int(last_seg[1]))]

        anim_delay = getattr(self.spec, "anim_delay", 0.0) or 0.0
        time_in_scene = max(0.0, t - self.spec.start_time - anim_delay)
        raw_idx = int(time_in_scene / self.interval)
        loop_start = getattr(self.spec, "loop_start_frame", None)
        pingpong = getattr(self.spec, "pingpong", False)

        if loop_start is not None and len(self.frames) > loop_start:
            if raw_idx < loop_start:
                frame_idx = raw_idx
            else:
                loop_len = len(self.frames) - loop_start
                step = raw_idx - loop_start
                if pingpong and loop_len > 1:
                    cycle = 2 * (loop_len - 1)
                    pos = step % cycle
                    sub_idx = pos if pos < loop_len else cycle - pos
                else:
                    sub_idx = step % loop_len
                frame_idx = loop_start + sub_idx
        else:
            n = len(self.frames)
            if pingpong and n > 1:
                cycle = 2 * (n - 1)
                pos = raw_idx % cycle
                frame_idx = pos if pos < n else cycle - pos
            else:
                frame_idx = raw_idx % n
        return self.frames[frame_idx]


class SceneCache:
    """Dynamic viewport-based loader for scenes along a long continuous timeline."""

    def __init__(self, root_dir: Optional[Path] = None, max_cached: int = 3):
        self.root = root_dir or Path.cwd()
        self.max_cached = max_cached
        self._cache: dict[str, LoadedScene] = {}

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if (self.root / p).exists():
            return self.root / p
        if (Path.cwd() / p).exists():
            return Path.cwd() / p
        return self.root / p

    def get(self, spec: "SceneSpec") -> LoadedScene:
        """Fetch from cache or lazily load, stitch, and scale the scene."""
        if spec.id in self._cache:
            return self._cache[spec.id]

        sheet_path = self._resolve_path(spec.sheet)
        raw_frames, meta = load_sprite_sheet(sheet_path, allow_single_image=True)
        interval = meta.get("intervalSeconds", 0.25)

        # 1. Stitch static left background extension if configured
        if spec.left_bg:
            l_bg_path = self._resolve_path(spec.left_bg)
            with Image.open(l_bg_path) as lim:
                lim_rgba = lim.convert("RGBA")

            target_bg_h = round(raw_frames[0].height * spec.effective_left_bg_scale)
            l_w = round(lim_rgba.width * (target_bg_h / lim_rgba.height))
            vid_w = round(raw_frames[0].width * spec.effective_video_scale)
            default_offset = l_w - round(vid_w * 0.25)
            offset = spec.left_bg_offset if spec.left_bg_offset is not None else default_offset

            left_feather = spec.left_feather
            if left_feather is None and getattr(spec, "overlap_prev", 0) > 0:
                ov = spec.overlap_prev
                left_feather = (round(ov * 0.44), round(ov * 0.34))

            scaled_frames = compose_extended_scene(
                raw_frames,
                lim_rgba,
                offset=offset,
                video_scale=spec.effective_video_scale,
                left_bg_scale=spec.effective_left_bg_scale,
                video_y=spec.video_y,
                left_edge_feather=left_feather,
                seam_feather_width=getattr(spec, "seam_feather_width", None),
            )
        else:
            # 2. Scale frames to target scene dimensions
            target_size = (spec.width, spec.height)
            scaled_frames = [
                f.resize(target_size, Image.Resampling.LANCZOS) if f.size != target_size else f
                for f in raw_frames
            ]

        # 3. Optional soft radial vignette / backdrop
        backdrop, pad_x, pad_y = None, 0, 0
        scene_grad = spec.extra.get("scene_gradient")
        if scene_grad:
            grad_color = "#203a55" if (scene_grad is True or scene_grad == "navy") else scene_grad
            scale_mult = float(spec.extra.get("scene_gradient_scale", 2.0))
            backdrop, pad_x, pad_y = create_scene_radial_backdrop(
                spec.width, spec.height, inner_color=grad_color, radius_scale=scale_mult
            )

        loaded = LoadedScene(
            spec=spec,
            frames=scaled_frames,
            interval=interval,
            backdrop=backdrop,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        self._cache[spec.id] = loaded
        return loaded

    def prune_unneeded(self, visible_scene_ids: set[str]) -> None:
        """Evict off-screen scenes to free memory."""
        to_delete = [sid for sid in self._cache if sid not in visible_scene_ids]
        for sid in to_delete:
            del self._cache[sid]
