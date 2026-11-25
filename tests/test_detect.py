import json

import numpy as np
from conftest import ACTION

from highlight_detector import Options, detect
from highlight_detector.cli import main
from highlight_detector.scoring import pick


def _covers(h, window):
    a, b = window
    return h["start_sec"] <= a + 2 and h["end_sec"] >= a + 2 or a <= h["peak_sec"] <= b + 2


def test_finds_both_action_windows(synthetic_video):
    res = detect(synthetic_video, Options(top=5))
    top2 = res["highlights"][:2]
    assert len(top2) == 2
    for window in ACTION:
        assert any(_covers(h, window) for h in top2), (window, top2)
    for h in res["highlights"]:
        assert 0 <= h["start_sec"] < h["end_sec"] <= res["duration_sec"] + 0.5
        assert 0 <= h["score"] <= 100
        assert h["reason"]


def test_length_limits_respected(synthetic_video):
    res = detect(synthetic_video, Options(top=5, min_len=20, max_len=25))
    for h in res["highlights"]:
        assert 20 * 0.6 <= h["end_sec"] - h["start_sec"] <= 25.01


def test_audio_only(synthetic_video):
    res = detect(synthetic_video, Options(top=3), use_video=False)
    assert "motion" not in res["features"]
    assert any(_covers(h, ACTION[0]) for h in res["highlights"])


def test_chat_signal_alone_finds_burst():
    hop = 0.5
    n = 1200  # 10 minutes
    chat = np.random.default_rng(0).poisson(0.3, n).astype(float)
    chat[700:720] += 15  # 10 second chat explosion at 5:50
    res, _, _ = pick({"chat": chat}, hop, Options(top=1))
    assert res and res[0].start_sec <= 352 <= res[0].end_sec
    assert "chat burst" in res[0].reason


def test_no_overlap_and_gap(synthetic_video):
    res = detect(synthetic_video, Options(top=10, min_gap=10))
    spans = sorted((h["start_sec"], h["end_sec"]) for h in res["highlights"])
    for (_, a1), (b0, _) in zip(spans, spans[1:], strict=False):
        assert b0 >= a1


def test_cli_json_and_formats(synthetic_video, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main([str(synthetic_video), "-f", "json", "-o", str(out), "-n", "3"]) == 0
    data = json.loads(out.read_text())
    assert data["tool"] == "highlight-detector" and data["highlights"]
    for fmt in ("csv", "chapters", "edl", "table"):
        assert main([str(synthetic_video), "-f", fmt, "-n", "2", "--audio-only"]) == 0
    text = capsys.readouterr().out
    assert "0:00 Start" in text and "TITLE: highlights" in text


def test_cli_export_and_plot(synthetic_video, tmp_path):
    clips = tmp_path / "clips"
    png = tmp_path / "t.png"
    assert main([str(synthetic_video), "-n", "1", "-f", "json", "-o", str(tmp_path / "x.json"), "--export", str(clips), "--plot", str(png)]) == 0
    files = list(clips.glob("*.mp4"))
    assert len(files) == 1 and files[0].stat().st_size > 1000
    assert png.stat().st_size > 1000


def test_missing_file_is_clean_error(capsys):
    assert main(["/nope/missing.mp4"]) == 1
    assert "not found" in capsys.readouterr().err
