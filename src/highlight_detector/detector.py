from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import media, sidecar
from .scoring import Highlight, Options, pick
from .signals import audio_features, video_features


def detect(
    path: str | Path,
    options: Options | None = None,
    *,
    hop: float = 0.5,
    fps: float = 4.0,
    use_video: bool = True,
    chat: str | Path | None = None,
    subtitles: str | Path | None = None,
) -> dict:
    """Analyse one video and return a JSON-serialisable result with ranked highlights."""
    opts = options or Options()
    media.require_ffmpeg()
    path = Path(path)
    if not path.exists():
        raise media.MediaError(f"file not found: {path}")
    info = media.probe(path)
    t0 = time.time()
    raw: dict[str, np.ndarray] = {}
    if info["has_audio"]:
        raw.update(audio_features(media.audio_blocks(path), hop))
    if use_video and info["has_video"]:
        raw.update(video_features(media.video_frames(path, fps), fps, hop))
    n = max((len(v) for v in raw.values()), default=int(info["duration"] / hop))
    if chat:
        raw["chat"] = sidecar.chat_rate(sidecar.load_chat(chat), n, hop)
    if subtitles:
        raw["speech_hype"] = sidecar.speech_hype(sidecar.load_subtitles(subtitles), n, hop)
    if not raw:
        raise media.MediaError("no audio or video stream found")
    highlights, curve, norm = pick(raw, hop, opts)
    return {
        "tool": "highlight-detector",
        "version": __import__("highlight_detector").__version__,
        "source": path.name,
        "duration_sec": round(info["duration"] or len(curve) * hop, 2),
        "hop_sec": hop,
        "features": sorted(k for k in norm if opts.weights.get(k, 0) > 0),
        "weights": {k: v for k, v in opts.weights.items() if k in norm},
        "analysis_sec": round(time.time() - t0, 2),
        "highlights": [h.to_dict() for h in highlights],
        "timeline": [round(float(v), 3) for v in curve],
    }


__all__ = ["detect", "Highlight", "Options"]
