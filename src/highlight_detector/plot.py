"""Render the intensity timeline with picked highlights as a PNG (needs Pillow)."""

from __future__ import annotations

from pathlib import Path


def plot_timeline(result: dict, out: str | Path, width: int = 1200, height: int = 260) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:  # pragma: no cover
        raise SystemExit("--plot needs Pillow: pip install 'highlight-detector[plot]'") from e
    from .export import ts

    curve = result.get("timeline") or []
    hop = result["hop_sec"]
    bg, grid, line, hl_fill, text = (14, 14, 18), (40, 40, 48), (120, 220, 255), (255, 90, 60, 70), (230, 230, 235)
    img = Image.new("RGBA", (width, height), bg)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    left, right, top, bottom = 40, 34, 46, 30
    w, h = width - left - right, height - top - bottom
    total = max(len(curve) * hop, 1e-6)
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = top + h * (1 - frac)
        d.line([(left, y), (left + w, y)], fill=grid)
    for hlt in result["highlights"]:
        x0 = left + w * hlt["start_sec"] / total
        x1 = left + w * hlt["end_sec"] / total
        d.rectangle([x0, top, x1, top + h], fill=hl_fill)
        d.text((x0 + 3, top - 18), f"#{hlt['rank']}", fill=(255, 140, 110), font=font)
    if curve:
        peak = max(max(curve), 1e-6)
        step = max(1, len(curve) // w)
        pts = []
        for i in range(0, len(curve), step):
            v = max(curve[i : i + step])
            pts.append((left + w * i / len(curve), top + h * (1 - v / peak)))
        d.line(pts, fill=line, width=2)
    img = Image.alpha_composite(img, layer)
    d = ImageDraw.Draw(img)
    for k in range(6):
        t = total * k / 5
        d.text((left + w * k / 5 - 14, top + h + 8), ts(t), fill=text, font=font)
    d.text((left, 6), f"{result['source']}  ·  intensity", fill=text, font=font)
    img.convert("RGB").save(out)
    return Path(out)
