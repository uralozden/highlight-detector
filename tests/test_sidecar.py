import json

from highlight_detector.sidecar import hype_score, load_chat, load_subtitles


def test_chat_formats(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"comments": [{"content_offset_seconds": 5.5}, {"content_offset_seconds": 1}]}))
    assert load_chat(j) == [1.0, 5.5]
    t = tmp_path / "c.txt"
    t.write_text("[0:01:02] bob: LETS GO\n[1:00:00] amy: lol\n12.5 raw\nnot a line\n")
    assert load_chat(t) == [12.5, 62.0, 3600.0]
    c = tmp_path / "c.csv"
    c.write_text("time,user,msg\n0:10,a,hi\n0:20,b,yo\n")
    assert load_chat(c) == [10.0, 20.0]


def test_subtitles_and_hype(tmp_path):
    s = tmp_path / "s.srt"
    s.write_text("1\n00:00:01,000 --> 00:00:03,500\nok so we go left\n\n2\n00:01:00,000 --> 00:01:02,000\nNO WAY BRO!!\n")
    cues = load_subtitles(s)
    assert cues[0][:2] == (1.0, 3.5) and cues[1][0] == 60.0
    assert hype_score(cues[1][2]) > hype_score(cues[0][2])
    assert hype_score("yok artık abi") >= 2
    assert hype_score("noooooo") >= 1
