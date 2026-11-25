"""Optional side inputs: chat replays and subtitles.

Chat is the strongest public signal a streamer has: when a moment lands, chat explodes.
Subtitles (SRT/VTT, e.g. from Whisper) add hype words like "let's go", "no way", "yok artık".
Both are turned into one value per hop.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

HYPE_WORDS = {
    # English
    "let's go",
    "lets go",
    "no way",
    "oh my god",
    "omg",
    "what",
    "wtf",
    "bro",
    "dude",
    "insane",
    "clutch",
    "holy",
    "yes",
    "run",
    "go go",
    "oh no",
    "nooo",
    "dead",
    "ace",
    "gg",
    "wow",
    "crazy",
    # Turkish
    "yok artık",
    "abi",
    "kanka",
    "oha",
    "vay",
    "hadi",
    "koş",
    "öldü",
    "inanılmaz",
    "efsane",
    "aman",
    "nasıl ya",
    "bitti",
    "olamaz",
    "helal",
}
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d+))?")
_SRT_RANGE = re.compile(r"([\d:.,]+)\s*-->\s*([\d:.,]+)")


def _to_seconds(text: str) -> float | None:
    m = _TS.search(text)
    if not m:
        return None
    h, mnt, s, frac = m.groups()
    sec = int(h or 0) * 3600 + int(mnt) * 60 + int(s)
    if frac:
        sec += float("0." + frac)
    return float(sec)


def load_chat(path: str | Path) -> list[float]:
    """Message timestamps (seconds from the start of the video).

    Accepted: TwitchDownloader / chat-downloader JSON (`content_offset_seconds`,
    `time_in_seconds` or `offset`), CSV with a `time`/`offset`/`seconds` column, or text lines
    that start with a timestamp such as `[1:02:03] user: message` or `754.2 message`.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    times: list[float] = []
    data = None
    if p.suffix.lower() == ".json" or raw.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            if p.suffix.lower() == ".json":
                raise
    if data is not None:
        items = data.get("comments", data.get("messages", [])) if isinstance(data, dict) else data
        for it in items:
            if not isinstance(it, dict):
                continue
            for key in ("content_offset_seconds", "time_in_seconds", "offset", "seconds", "t"):
                if key in it:
                    try:
                        times.append(float(it[key]))
                    except (TypeError, ValueError):
                        pass
                    break
    elif p.suffix.lower() == ".csv":
        for row in csv.DictReader(raw.splitlines()):
            for key in ("seconds", "offset", "time", "timestamp"):
                if row.get(key):
                    v = row[key]
                    t = _to_seconds(v) if ":" in v else float(v)
                    if t is not None:
                        times.append(t)
                    break
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            head = line.split()[0].strip("[]")
            t = _to_seconds(head) if ":" in head else None
            if t is None:
                try:
                    t = float(head)
                except ValueError:
                    continue
            times.append(t)
    return sorted(t for t in times if t >= 0)


def chat_rate(times: list[float], n_bins: int, hop: float) -> np.ndarray:
    out = np.zeros(n_bins)
    for t in times:
        i = int(t / hop)
        if 0 <= i < n_bins:
            out[i] += 1
    return out


def load_subtitles(path: str | Path) -> list[tuple[float, float, str]]:
    """(start, end, text) cues from SRT or WebVTT."""
    cues: list[tuple[float, float, str]] = []
    block: list[str] = []
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines() + [""]
    for line in lines:
        if line.strip():
            block.append(line.strip())
            continue
        for i, b in enumerate(block):
            m = _SRT_RANGE.search(b)
            if m:
                a, z = _to_seconds(m.group(1)), _to_seconds(m.group(2))
                text = " ".join(block[i + 1 :])
                text = re.sub(r"<[^>]+>", "", text)
                if a is not None and z is not None and text:
                    cues.append((a, max(z, a), text))
                break
        block = []
    return cues


def hype_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for w in HYPE_WORDS if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low))
    hits += min(3, text.count("!"))
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        hits += 2  # SHOUTING
    if re.search(r"(\w)\1{3,}", low):  # "noooo", "yesssss"
        hits += 1
    return float(hits)


def speech_hype(cues: list[tuple[float, float, str]], n_bins: int, hop: float) -> np.ndarray:
    out = np.zeros(n_bins)
    for a, z, text in cues:
        s = hype_score(text)
        if s <= 0:
            continue
        i0, i1 = int(a / hop), max(int(a / hop) + 1, int(np.ceil(z / hop)))
        out[max(0, i0) : min(n_bins, i1)] += s / max(1, i1 - i0) * 2
    return out
