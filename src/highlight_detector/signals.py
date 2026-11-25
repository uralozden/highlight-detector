"""Per-bin raw features. Every feature is a 1-D array with one value per `hop` seconds.

These are cheap, language-independent cues that tend to line up with action in gameplay:
loud moments, sharp transients (gunfire, explosions, impacts), fast motion and hard cuts,
and bright flashes. None of them knows what "a good clip" is; they only say *something
happened here*. Scoring turns them into candidates.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .media import AUDIO_RATE

FFT_SIZE = 1024


def audio_features(blocks: Iterable[np.ndarray], hop: float) -> dict[str, np.ndarray]:
    """loudness_db and spectral flux (onset strength) per hop."""
    hop_n = int(round(AUDIO_RATE * hop))
    win = np.hanning(FFT_SIZE).astype(np.float32)
    loud: list[float] = []
    flux: list[float] = []
    prev_spec: np.ndarray | None = None
    buf = np.zeros(0, dtype=np.float32)
    for block in blocks:
        buf = np.concatenate([buf, block])
        n_bins = len(buf) // hop_n
        if n_bins == 0:
            continue
        frames = buf[: n_bins * hop_n].reshape(n_bins, hop_n)
        buf = buf[n_bins * hop_n :]
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        loud.extend((20 * np.log10(np.maximum(rms, 1e-5))).tolist())
        # Spectral flux on non-overlapping FFT frames inside each hop, averaged per hop.
        per_hop = hop_n // FFT_SIZE
        if per_hop < 1:
            raise ValueError("hop too short for audio analysis")
        sub = frames[:, : per_hop * FFT_SIZE].reshape(n_bins * per_hop, FFT_SIZE) * win
        spec = np.log1p(np.abs(np.fft.rfft(sub, axis=1)) * 10.0)
        if prev_spec is None:
            prev_spec = spec[:1]
        stacked = np.vstack([prev_spec, spec])
        diff = np.maximum(np.diff(stacked, axis=0), 0).mean(axis=1)
        prev_spec = spec[-1:]
        flux.extend(diff.reshape(n_bins, per_hop).max(axis=1).tolist())
    return {"loudness_db": np.asarray(loud), "transients": np.asarray(flux)}


def video_features(frames: Iterable[np.ndarray], fps: float, hop: float) -> dict[str, np.ndarray]:
    """motion (mean abs frame difference), cuts (hard scene changes) and flashes per hop."""
    per_hop = max(1, int(round(fps * hop)))
    motion: list[float] = []
    cuts: list[float] = []
    flash: list[float] = []
    prev: np.ndarray | None = None
    acc_m: list[float] = []
    acc_c = 0.0
    acc_f = 0.0
    for frame in frames:
        f = frame.astype(np.float32)
        if prev is not None:
            d = np.abs(f - prev)
            m = float(d.mean())
            acc_m.append(m)
            # A cut changes most of the picture at once; motion changes parts of it.
            if m > 40 and float((d > 30).mean()) > 0.6:
                acc_c += 1
            acc_f = max(acc_f, max(0.0, float(f.mean() - prev.mean())))
        else:
            acc_m.append(0.0)
        prev = f
        if len(acc_m) == per_hop:
            motion.append(float(np.mean(acc_m)))
            cuts.append(acc_c)
            flash.append(acc_f)
            acc_m, acc_c, acc_f = [], 0.0, 0.0
    if acc_m:
        motion.append(float(np.mean(acc_m)))
        cuts.append(acc_c)
        flash.append(acc_f)
    return {"motion": np.asarray(motion), "cuts": np.asarray(cuts), "flash": np.asarray(flash)}
