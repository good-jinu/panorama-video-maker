"""Build a long-form music station MP4 by downloading YouTube audio tracks and looping a video."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union


def _extract_youtube_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from a watch or short URL."""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"Could not extract YouTube ID from URL: {url}")
    return m.group(1)


def _probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def download_youtube_audio(
    urls: Sequence[str],
    music_dir: Union[str, Path] = Path("inputs/music"),
) -> list[Path]:
    """Download YouTube audio tracks as .m4a files in order, reusing cached files if present."""
    out_dir = Path(music_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks: list[Path] = []
    missing: list[tuple[int, str, Path]] = []

    for idx, url in enumerate(urls, start=1):
        vid = _extract_youtube_id(url)
        target = out_dir / f"{idx:02d}_{vid}.m4a"
        tracks.append(target)
        if not target.exists():
            existing = list(out_dir.glob(f"*_{vid}.m4a"))
            if existing:
                print(f"Reusing cached audio for {vid}: {existing[0].name} -> {target.name}")
                shutil.copy2(existing[0], target)
            else:
                missing.append((idx, url, target))

    for idx, url, target in missing:
        print(f"Downloading track {idx}/{len(urls)}: {url} -> {target.name}")
        template = str(out_dir / f"{idx:02d}_%(id)s.%(ext)s")
        base_args = [
            "-f",
            "ba[ext=m4a]/ba",
            "-x",
            "--audio-format",
            "m4a",
            "--audio-quality",
            "0",
            "-o",
            template,
            url,
        ]
        # Prefer uvx yt-dlp@latest to avoid HTTP 403 from outdated local yt-dlp builds
        runner = ["uvx", "yt-dlp@latest"] if shutil.which("uvx") else ["yt-dlp"]
        cmd = [*runner, *base_args]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("Retrying download with fallback audio streams (HLS/m3u8)...")
            fallback_args = [
                "-f",
                "234/233/ba/b",
                "-x",
                "--audio-format",
                "m4a",
                "--audio-quality",
                "0",
                "-o",
                template,
                url,
            ]
            subprocess.run([*runner, *fallback_args], check=True)
        if not target.exists():
            raise FileNotFoundError(f"Expected downloaded audio file not found: {target}")

    return tracks


def concat_audio_tracks(tracks: Sequence[Path], output_audio: Path) -> Path:
    """Concatenate multiple audio files into a unified stereo 48kHz AAC .m4a file."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    filter_parts = []
    concat_inputs = []
    for i, track in enumerate(tracks):
        cmd.extend(["-i", str(track)])
        filter_parts.append(
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]"
        )
        concat_inputs.append(f"[a{i}]")

    filter_complex = (
        ";".join(filter_parts)
        + ";"
        + "".join(concat_inputs)
        + f"concat=n={len(tracks)}:v=0:a=1[outa]"
    )
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_audio),
        ]
    )
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_audio


def build_music_station(
    video_path: Union[str, Path],
    output: Union[str, Path],
    urls: list[str],
    music_dir: Union[str, Path] = Path("inputs/music"),
    playlist_loops: int = 1,
) -> Path:
    """Download playlist tracks, concatenate audio, and loop `video_path` over the full soundtrack."""
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    out_mp4 = Path(output)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    tracks = download_youtube_audio(urls, music_dir=music_dir)
    if playlist_loops > 1:
        tracks = tracks * playlist_loops

    print(f"Concatenating {len(tracks)} audio track(s)...")
    for i, t in enumerate(tracks, 1):
        dur = _probe_duration(t)
        print(f"  Track {i:02d}: {t.name} ({dur:.1f}s)")

    merged_audio = out_mp4.with_name(f"{out_mp4.stem}_audio.m4a")
    concat_audio_tracks(tracks, merged_audio)

    audio_dur = _probe_duration(merged_audio)
    video_dur = _probe_duration(video)
    loops_count = audio_dur / video_dur
    print(
        f"Muxing looped video ({video_dur:.1f}s x {loops_count:.2f} loops) with playlist audio ({audio_dur:.1f}s / {audio_dur / 60:.1f}m)..."
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(video),
        "-i",
        str(merged_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    final_dur = _probe_duration(out_mp4)
    print(f"Saved music station MP4: {out_mp4} ({final_dur:.1f}s / {final_dur / 60:.2f} min)")
    return out_mp4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="Looping source MP4 video")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output music station MP4 path")
    ap.add_argument("--music-dir", type=Path, default=Path("inputs/music"), help="Directory for downloaded audio")
    ap.add_argument("urls", nargs="+", help="YouTube URLs in playlist order")
    args = ap.parse_args()

    build_music_station(
        video_path=args.video,
        output=args.output,
        urls=args.urls,
        music_dir=args.music_dir,
    )


if __name__ == "__main__":
    main()
