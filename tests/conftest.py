import shutil
import subprocess

import pytest

ACTION = [(60.0, 68.0), (140.0, 146.0)]  # loud + fast-moving windows in the synthetic video
DURATION = 180


def make_video(path, duration=DURATION, action=ACTION):
    enable = "+".join(f"between(t,{a},{b})" for a, b in action)
    vf = f"[0:v][1:v]overlay=x='mod(t*900,320)':y='mod(t*500,180)':enable='{enable}'[v]"
    af = f"[2:a]volume=0.03[bg];[3:a]volume='if({enable},1,0)':eval=frame[burst];[bg][burst]amix=inputs=2:normalize=0[a]"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x303030:s=320x180:r=24:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=s=160x90:r=24:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=c=pink:a=0.5:d={duration}:r=16000",
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=c=white:a=0.8:d={duration}:r=16000",
        "-filter_complex",
        vf + ";" + af,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return make_video(tmp_path_factory.mktemp("media") / "synthetic.mp4")
