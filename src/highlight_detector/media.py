"""Decode audio and low-resolution video with ffmpeg, streamed so long VODs never sit in memory."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

AUDIO_RATE = 16_000
FRAME_W, FRAME_H = 64, 36


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(f"{tool} not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html")


def probe(path: str | Path) -> dict:
    """Duration and which streams exist."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration:stream=codec_type",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise MediaError(f"ffprobe could not read {path}: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout or "{}")
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    try:
        duration = float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration": duration, "has_audio": "audio" in kinds, "has_video": "video" in kinds}


def _stream(cmd: list[str], block_bytes: int) -> Iterator[bytes]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(block_bytes)
            if not chunk:
                break
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()


def audio_blocks(path: str | Path, seconds: int = 60) -> Iterator[np.ndarray]:
    """Mono float32 samples in [-1, 1] at 16 kHz, one block at a time."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_RATE),
        "-f",
        "s16le",
        "-",
    ]
    carry = b""
    for chunk in _stream(cmd, AUDIO_RATE * 2 * seconds):
        data = carry + chunk
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            yield np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def video_frames(path: str | Path, fps: float) -> Iterator[np.ndarray]:
    """Tiny grayscale frames (36x64, uint8) sampled at `fps`."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-an",
        "-vf",
        f"fps={fps},scale={FRAME_W}:{FRAME_H}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    size = FRAME_W * FRAME_H
    carry = b""
    for chunk in _stream(cmd, size * 256):
        data = carry + chunk
        n = len(data) // size
        carry = data[n * size :]
        for i in range(n):
            yield np.frombuffer(data[i * size : (i + 1) * size], dtype=np.uint8).reshape(FRAME_H, FRAME_W)
