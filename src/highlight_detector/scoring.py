"""Turn raw per-hop features into ranked highlight windows.

Approach (deliberately simple and inspectable):
1. Each feature is normalised against the video itself (robust z-score: median / MAD), so a
   quiet strategy game and a loud shooter are judged against their own baseline. Loudness is
   compared with the *last two minutes* instead, so a streamer getting louder over a session
   doesn't flatten every later peak.
2. Normalised features are mapped to [0, 1] and blended with public, tunable weights into one
   "intensity" curve, which is smoothed over a couple of seconds.
3. Greedy peak picking: take the highest point, build a clip around it (with build-up before
   the peak and a short tail after), mask it out, repeat.

The score is a relative "how much is happening here compared with the rest of this video"
number. It is not a prediction of views, retention or virality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

DEFAULT_WEIGHTS: dict[str, float] = {
    "loudness": 0.30,
    "transients": 0.20,
    "motion": 0.20,
    "cuts": 0.05,
    "flash": 0.05,
    "speech_hype": 0.15,
    "chat": 0.35,
}

LABELS = {
    "loudness": "loud spike",
    "transients": "sharp sound hits (shots, impacts, explosions)",
    "motion": "fast on-screen motion",
    "cuts": "hard scene changes",
    "flash": "bright flashes",
    "speech_hype": "hype words in speech",
    "chat": "chat burst",
}


@dataclass
class Options:
    top: int = 10
    min_len: float = 15.0
    max_len: float = 45.0
    pre_roll: float = 5.0
    post_roll: float = 3.0
    smooth: float = 2.0
    min_score: float = 15.0
    min_gap: float = 20.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class Highlight:
    rank: int
    start_sec: float
    end_sec: float
    peak_sec: float
    score: int
    signals: dict[str, float]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _moving(x: np.ndarray, n: int, fn: str = "mean") -> np.ndarray:
    n = max(1, int(n))
    if n == 1 or len(x) == 0:
        return x.astype(np.float64)
    pad = n // 2
    xp = np.pad(x.astype(np.float64), (pad, n - 1 - pad), mode="edge")
    if fn == "mean":
        return np.convolve(xp, np.ones(n) / n, mode="valid")
    w = np.lib.stride_tricks.sliding_window_view(xp, n)
    return np.median(w, axis=1)


def _robust_unit(x: np.ndarray, baseline: np.ndarray | None = None, spread: float | None = None) -> np.ndarray:
    """Map to [0, 1]: 0 at/below the baseline, 1 at ~3 robust SDs above it."""
    if len(x) == 0:
        return x
    base = np.median(x) if baseline is None else baseline
    mad = spread if spread is not None else float(np.median(np.abs(x - np.median(x))))
    scale = max(1.4826 * mad, 1e-6)
    return np.clip((x - base) / (3.0 * scale), 0.0, 1.0)


def normalise(raw: dict[str, np.ndarray], hop: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "loudness_db" in raw and len(raw["loudness_db"]):
        db = raw["loudness_db"]
        # Ignore digital silence when estimating the spread.
        active = db[db > np.percentile(db, 10)] if len(db) > 10 else db
        spread = float(np.median(np.abs(active - np.median(active)))) if len(active) else 1.0
        local = _moving(db, int(120 / hop), "median") if len(db) > 4 * int(120 / hop) else np.full_like(db, np.median(db))
        out["loudness"] = np.clip((db - local) / max(3.0 * 1.4826 * spread, 3.0), 0, 1)
    if "transients" in raw:
        out["transients"] = _robust_unit(raw["transients"])
    if "motion" in raw:
        out["motion"] = _robust_unit(raw["motion"])
    if "cuts" in raw:
        out["cuts"] = np.clip(_moving(raw["cuts"], int(3 / hop), "mean") * (3 / hop) / 3.0, 0, 1)
    if "flash" in raw:
        out["flash"] = np.clip(raw["flash"] / 60.0, 0, 1)
    if "speech_hype" in raw:
        out["speech_hype"] = np.clip(_moving(raw["speech_hype"], int(4 / hop)) / 1.5, 0, 1)
    if "chat" in raw:
        # Chat is bursty and heavy-tailed, so compare on a log scale: 2x the usual rate is
        # notable, 8x or more is as strong as it gets.
        rate = _moving(raw["chat"], int(5 / hop))
        typical = max(float(np.median(rate)), 0.05)
        out["chat"] = np.clip(np.log2((rate + 0.05) / typical) / 3.0, 0, 1)
    return out


def align(feats: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    n = max((len(v) for v in feats.values()), default=0)
    out = {}
    for k, v in feats.items():
        if len(v) < n:
            v = np.pad(v, (0, n - len(v)), mode="constant", constant_values=float(np.median(v)) if len(v) else 0.0)
        out[k] = v
    return out, n


def intensity(norm: dict[str, np.ndarray], weights: dict[str, float], hop: float, smooth: float) -> np.ndarray:
    used = {k: w for k, w in weights.items() if k in norm and w > 0}
    if not used:
        raise ValueError("no usable features: the file has neither audio nor video, or all weights are 0")
    total = sum(used.values())
    # Weighted quadratic mean: one very strong signal (a chat explosion, a huge explosion) can
    # carry a moment on its own, while weak noise in many signals stays low.
    curve = np.sqrt(sum((norm[k] ** 2) * (w / total) for k, w in used.items()))
    return _moving(np.asarray(curve), int(round(smooth / hop)))


def _reason(norm: dict[str, np.ndarray], raw: dict[str, np.ndarray], i0: int, i1: int, peak: int, hop: float, weights: dict[str, float]) -> tuple[str, dict[str, float]]:
    sig = {k: round(float(np.max(v[i0:i1])) if i1 > i0 else 0.0, 3) for k, v in norm.items()}
    ranked = sorted((k for k in sig if weights.get(k, 0) > 0 and sig[k] >= 0.35), key=lambda k: -sig[k] * weights[k])
    parts: list[str] = []
    for k in ranked[:3]:
        if k == "loudness" and "loudness_db" in raw:
            db = raw["loudness_db"]
            lo = max(0, peak - int(60 / hop))
            delta = float(np.max(db[i0:i1]) - np.median(db[lo : peak + 1] if peak + 1 > lo else db))
            if delta < 3:
                continue
            parts.append(f"loud spike (+{delta:.0f} dB over the previous minute)")
        elif k == "chat" and "chat" in raw:
            c = raw["chat"]
            per_min = float(np.max(_moving(c, int(10 / hop))[i0:i1]) * 60 / hop)
            typical = float(np.median(_moving(c, int(10 / hop))) * 60 / hop)
            parts.append(f"chat burst ({per_min:.0f} msg/min vs {typical:.0f} typical)")
        elif k == "cuts" and "cuts" in raw:
            parts.append(f"{int(raw['cuts'][i0:i1].sum())} hard scene changes")
        else:
            parts.append(LABELS[k])
    if not parts:
        parts.append("mild activity above this video's baseline")
    return "; ".join(parts), sig


def pick(raw: dict[str, np.ndarray], hop: float, opts: Options) -> tuple[list[Highlight], np.ndarray, dict[str, np.ndarray]]:
    raw_aligned, n = align(raw)
    norm = normalise(raw_aligned, hop)
    norm, n = align(norm)
    if n == 0:
        return [], np.zeros(0), norm
    curve = intensity(norm, opts.weights, hop, opts.smooth)
    duration = n * hop
    work = curve.copy()
    taken = np.zeros(n, dtype=bool)
    hl: list[tuple[int, int, int, float]] = []
    min_n, max_n = int(opts.min_len / hop), int(opts.max_len / hop)
    median = float(np.median(curve))
    top_level = float(np.percentile(curve, 99.5)) or 1.0
    while len(hl) < opts.top:
        p = int(np.argmax(work))
        if work[p] <= 0 or taken[p]:
            break
        peak_v = float(curve[p])
        # Follow the action while the curve stays well above the baseline, then add a little
        # context before it (the setup) and a short tail after it (the reaction).
        lo_thr = median + 0.35 * (peak_v - median)
        hi_thr = median + 0.35 * (peak_v - median)
        s = p
        while s > 0 and p - s < max_n and curve[s - 1] > lo_thr and not taken[s - 1]:
            s -= 1
        e = p
        while e < n - 1 and e - s < max_n and curve[e + 1] > hi_thr and not taken[e + 1]:
            e += 1
        for _ in range(int(opts.pre_roll / hop)):
            if s > 0 and not taken[s - 1]:
                s -= 1
        for _ in range(int(opts.post_roll / hop)):
            if e < n - 1 and not taken[e + 1]:
                e += 1
        # Enforce length: earlier context first (viewers need the setup), then the tail.
        while e - s + 1 < min_n and (s > 0 and not taken[s - 1] or e < n - 1 and not taken[e + 1]):
            if s > 0 and not taken[s - 1] and (p - s) <= 2 * (e - p) + 1:
                s -= 1
            elif e < n - 1 and not taken[e + 1]:
                e += 1
            else:
                s -= 1
        while e - s + 1 > max_n:
            if p - s > e - p:
                s += 1
            else:
                e -= 1
        if e - s + 1 < min_n * 0.6:
            work[p] = 0
            taken[p] = True
            continue
        seg = curve[s : e + 1]
        k = max(1, int(len(seg) * 0.3))
        top_mean = float(np.mean(np.sort(seg)[-k:]))
        raw_score = 0.6 * peak_v + 0.4 * top_mean
        hl.append((s, e, p, raw_score))
        taken[s : e + 1] = True
        # Keep picks apart so ten clips don't all come from one fight.
        gap = int(max(opts.min_gap, 0) / hop)
        work[max(0, s - gap) : e + 1 + gap] = 0
    out: list[Highlight] = []
    for s, e, p, rs in hl:
        score = int(round(100 * min(1.0, rs / max(top_level, 1e-6)) * min(1.0, rs / 0.25)))
        if score < opts.min_score:
            continue
        reason, sig = _reason(norm, raw_aligned, s, e + 1, p, hop, opts.weights)
        out.append(Highlight(0, round(s * hop, 2), round(min(duration, (e + 1) * hop), 2), round(p * hop, 2), score, sig, reason))
    out.sort(key=lambda h: (-h.score, h.start_sec))
    for i, h in enumerate(out, 1):
        h.rank = i
    return out, curve, norm
