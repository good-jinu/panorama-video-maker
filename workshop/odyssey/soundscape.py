"""Procedural Soundscape Engine for Odyssey Journey productions.

Integrates:
- Stride-synchronized paw footstep foley (procedural synthesis adapted to ground material)
- Multi-scene environmental weather & ambience crossfader
- Lofi music track ducking and master audio export via numpy & ffmpeg
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from workshop.odyssey.spec import AmbientMood, GroundMaterial, SoundscapeSpec

if TYPE_CHECKING:
    from workshop.odyssey.actor_fsm import FootstepEvent


class SoundscapeEngine:
    """Orchestrates procedural footsteps, environmental ambience, and background music."""

    def __init__(self, spec: SoundscapeSpec, sample_rate: int = 44100):
        self.spec = spec
        self.sample_rate = sample_rate

    def synthesize_footstep_sample(self, material: GroundMaterial) -> np.ndarray:
        """Synthesize a short, realistic paw-step audio burst based on surface physics."""
        sr = self.sample_rate
        duration = 0.08  # 80ms paw impact
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Base decay envelope (exponential decay)
        envelope = np.exp(-t * 60.0)

        # Procedural acoustic signature by material
        if material == GroundMaterial.ASPHALT:
            # Crisp high-frequency click + quick damp
            click = np.sin(2 * np.pi * 900.0 * t) * envelope
            noise = np.random.normal(0, 0.15, n_samples) * envelope
            sig = click * 0.7 + noise * 0.3
        elif material == GroundMaterial.WOODEN_FLOOR:
            # Warm hollow resonance (low frequency tap)
            resonance = np.sin(2 * np.pi * 240.0 * t) * np.exp(-t * 35.0)
            sig = resonance * 0.8 + np.random.normal(0, 0.05, n_samples) * envelope
        elif material == GroundMaterial.WATER_PUDDLE:
            # Wet splash (frequency glide + noise)
            freq = np.linspace(1200.0, 400.0, n_samples)
            splash = np.sin(2 * np.pi * freq * t) * envelope
            noise = np.random.normal(0, 0.25, n_samples) * (envelope**0.5)
            sig = splash * 0.5 + noise * 0.5
        elif material == GroundMaterial.GRASS:
            # Soft friction swish
            noise = np.random.normal(0, 0.2, n_samples) * (envelope**0.8)
            sig = noise
        elif material == GroundMaterial.SNOW:
            # Low crunch
            crunch = np.sin(2 * np.pi * 180.0 * t) * envelope
            noise = np.random.normal(0, 0.2, n_samples) * envelope
            sig = crunch * 0.4 + noise * 0.6
        else:
            # Generic stone / default
            sig = np.sin(2 * np.pi * 600.0 * t) * envelope

        # Normalize and apply foley volume
        peak = np.max(np.abs(sig)) + 1e-6
        return (sig / peak) * self.spec.foley_volume

    def synthesize_ambient_track(
        self,
        duration: float,
        scene_moods: list[tuple[float, float, AmbientMood]],
    ) -> np.ndarray:
        """Synthesize smooth environmental ambient noise with crossfading across scenes."""
        sr = self.sample_rate
        total_samples = int(duration * sr)
        audio = np.zeros(total_samples, dtype=np.float32)

        for start_t, end_t, mood in scene_moods:
            s_idx = max(0, int(start_t * sr))
            e_idx = min(total_samples, int(end_t * sr))
            if e_idx <= s_idx:
                continue

            seg_len = e_idx - s_idx
            t_seg = np.linspace(0, seg_len / sr, seg_len, endpoint=False)

            if mood == AmbientMood.RAINY_LOFI:
                # Continuous rain pink noise + distant soft roll
                rain = np.random.normal(0, 0.08, seg_len)
                audio[s_idx:e_idx] += rain.astype(np.float32) * self.spec.ambient_volume
            elif mood == AmbientMood.COZY_FIREPLACE:
                # Soft low rumble + random crackle pops
                rumble = np.sin(2 * np.pi * 60.0 * t_seg) * 0.04
                crackle = np.zeros(seg_len, dtype=np.float32)
                pop_indices = np.random.choice(seg_len, size=max(1, seg_len // 4000), replace=False)
                crackle[pop_indices] = np.random.uniform(0.1, 0.3, len(pop_indices))
                audio[s_idx:e_idx] += (rumble + crackle) * self.spec.ambient_volume
            elif mood == AmbientMood.NIGHT_CRICKETS:
                # Pulsing cricket chirp bursts (4.5 kHz sine modulated)
                chirp = np.sin(2 * np.pi * 4500.0 * t_seg) * (np.sin(2 * np.pi * 3.0 * t_seg) > 0.7) * 0.05
                audio[s_idx:e_idx] += chirp.astype(np.float32) * self.spec.ambient_volume
            elif mood == AmbientMood.FOREST_WIND:
                # Slow undulating low-pass wind swell
                wind = np.sin(2 * np.pi * 0.2 * t_seg) * np.random.normal(0, 0.06, seg_len)
                audio[s_idx:e_idx] += wind.astype(np.float32) * self.spec.ambient_volume

        return audio

    def render_master_audio(
        self,
        duration: float,
        footsteps: list[FootstepEvent],
        scene_moods: list[tuple[float, float, AmbientMood]],
        out_wav_path: Path | str,
    ) -> Path:
        """Mix foley footsteps, environmental ambience, and export master uncompressed WAV."""
        sr = self.sample_rate
        total_samples = int(duration * sr)
        master = np.zeros(total_samples, dtype=np.float32)

        # 1. Layer environmental ambience
        if self.spec.ambient_volume > 0:
            ambience = self.synthesize_ambient_track(duration, scene_moods)
            master += ambience[:total_samples]

        # 2. Layer footstep foley
        if self.spec.foley_footsteps and footsteps:
            for ev in footsteps:
                sample = self.synthesize_footstep_sample(ev.material)
                start_idx = int(ev.time * sr)
                if start_idx < total_samples:
                    valid_len = min(len(sample), total_samples - start_idx)
                    master[start_idx : start_idx + valid_len] += sample[:valid_len]

        # 3. Soft clipping / Limiter to prevent distortion
        master = np.tanh(master)

        # Convert float32 [-1.0, 1.0] to int16
        out_path = Path(out_wav_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        int16_audio = (master * 32767.0).astype(np.int16)

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)  # Mono track
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(int16_audio.tobytes())

        return out_path

    def multiplex_audio_video(
        self,
        video_path: Path | str,
        foley_wav_path: Path | str,
        out_mp4_path: Path | str,
        music_path: Path | str | None = None,
    ) -> Path:
        """Combine silent video with master foley audio and optional background music via FFmpeg."""
        out_p = Path(out_mp4_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(foley_wav_path),
        ]

        if music_path and Path(music_path).exists():
            cmd.extend([
                "-i",
                str(music_path),
                "-filter_complex",
                f"[1:a]volume=1.0[foley];[2:a]volume={self.spec.music_volume}[music];[foley][music]amix=inputs=2:duration=first[aout]",
                "-map",
                "0:v",
                "-map",
                "[aout]",
            ])
        else:
            cmd.extend([
                "-map",
                "0:v",
                "-map",
                "1:a",
            ])

        cmd.extend([
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_p),
        ])

        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_p
