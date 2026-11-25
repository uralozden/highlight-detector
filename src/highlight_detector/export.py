"""Writers (JSON, CSV, YouTube chapters, EDL, table) and clip cutting."""

from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path


def ts(sec: float, frac: bool = False) -> str:
    sec = max(0.0, sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    base = f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
    return base + (f".{int(round((sec % 1) * 10)) % 10}" if frac else "")


def to_json(result: dict, timeline: bool = False) -> str:
    data = dict(result)
    if not timeline:
        data.pop("timeline", None)
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_csv(result: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "start_sec", "end_sec", "peak_sec", "score", "reason"])
    for h in result["highlights"]:
        w.writerow([h["rank"], h["start_sec"], h["end_sec"], h["peak_sec"], h["score"], h["reason"]])
    return buf.getvalue()


def to_chapters(result: dict) -> str:
    """YouTube-style chapter list in time order (YouTube needs a 0:00 entry first)."""
    lines = ["0:00 Start"]
    for h in sorted(result["highlights"], key=lambda h: h["start_sec"]):
        if h["start_sec"] < 10:
            continue
        lines.append(f"{ts(h['start_sec'])} Highlight #{h['rank']} ({h['score']})")
    return "\n".join(lines) + "\n"


def to_edl(result: dict, fps: int = 30) -> str:
    """CMX3600 EDL so the picks open as a timeline in Resolve / Premiere."""

    def tc(sec: float) -> str:
        f = int(round(sec * fps))
        return f"{f // (3600 * fps):02d}:{f // (60 * fps) % 60:02d}:{f // fps % 60:02d}:{f % fps:02d}"

    out = ["TITLE: highlights", "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    for i, h in enumerate(sorted(result["highlights"], key=lambda h: h["start_sec"]), 1):
        d = h["end_sec"] - h["start_sec"]
        out.append(f"{i:03d}  AX       AA/V  C        {tc(h['start_sec'])} {tc(h['end_sec'])} {tc(rec)} {tc(rec + d)}")
        out.append(f"* FROM CLIP NAME: {result['source']}")
        out.append(f"* COMMENT: #{h['rank']} score {h['score']} - {h['reason']}")
        out.append("")
        rec += d
    return "\n".join(out)


def to_table(result: dict) -> str:
    rows = [f"{result['source']}  ·  {ts(result['duration_sec'])}  ·  analysed in {result['analysis_sec']}s  ·  signals: {', '.join(result['features'])}", ""]
    if not result["highlights"]:
        rows.append("No highlights above the threshold. Try --min-score 0 or add --chat / --subs.")
        return "\n".join(rows) + "\n"
    rows.append(f"{'#':>2}  {'start':>8}  {'end':>8}  {'len':>4}  {'score':>5}  reason")
    for h in result["highlights"]:
        bar = "█" * (h["score"] // 10)
        rows.append(f"{h['rank']:>2}  {ts(h['start_sec']):>8}  {ts(h['end_sec']):>8}  {h['end_sec'] - h['start_sec']:>3.0f}s  {h['score']:>5}  {bar:<10} {h['reason']}")
    return "\n".join(rows) + "\n"


def cut_clips(src: str | Path, result: dict, out_dir: str | Path, copy: bool = False) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    paths = []
    for h in result["highlights"]:
        dst = out / f"{stem}_hl{h['rank']:02d}_{ts(h['start_sec']).replace(':', '-')}.mp4"
        codec = ["-c", "copy"] if copy else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k"]
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{h['start_sec']:.2f}",
            "-to",
            f"{h['end_sec']:.2f}",
            "-i",
            str(src),
            *codec,
            "-movflags",
            "+faststart",
            str(dst),
        ]
        # -ss/-to before -i is fast; with re-encoding it is still frame accurate.
        if copy:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{h['start_sec']:.2f}",
                "-i",
                str(src),
                "-t",
                f"{h['end_sec'] - h['start_sec']:.2f}",
                *codec,
                str(dst),
            ]
        subprocess.run(cmd, check=True)
        paths.append(dst)
    return paths
