"""Generate a reproducible 10-minute fake "gameplay session" plus a chat log, for demos and tests.

    python examples/make_sample.py            # writes sample.mp4 and sample_chat.txt

The video is synthetic (no game footage): a quiet background with three staged events:
  2:10  firefight   - rapid loud bursts and fast motion
  5:05  explosion   - one huge hit and a bright flash
  8:20  chat moment - nothing on screen, but chat explodes (only found with --chat)
"""

from __future__ import annotations

import random
import subprocess
import sys

DUR = 600


def main(out: str = "sample.mp4", chat: str = "sample_chat.txt") -> None:
    fight = "between(t,130,141)"
    boom = "between(t,305,307)"
    move = f"{fight}+{boom}"
    vf = (
        f"[0:v][1:v]overlay=x='mod(t*1400,480)':y='mod(t*700,270)':enable='{move}',"
        f"drawbox=c=white@0.9:t=fill:enable='between(t,305,305.6)'[v]"
    )
    af = (
        "[2:a]volume=0.05[bg];"
        f"[3:a]volume='if({fight},if(lt(mod(t,0.35),0.08),1.0,0.05),0)':eval=frame[gun];"
        f"[4:a]volume='if({boom},1.4*exp(-(t-305)*1.5),0)':eval=frame[ex];"
        "[bg][gun][ex]amix=inputs=3:normalize=0[a]"
    )
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=0x2a2d34:s=640x360:r=30:d={DUR}",
        "-f", "lavfi", "-i", f"testsrc2=s=160x90:r=30:d={DUR}",
        "-f", "lavfi", "-i", f"anoisesrc=c=pink:a=0.4:d={DUR}:r=44100",
        "-f", "lavfi", "-i", f"anoisesrc=c=white:a=0.9:d={DUR}:r=44100",
        "-f", "lavfi", "-i", f"anoisesrc=c=brown:a=0.9:d={DUR}:r=44100",
        "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-c:a", "aac", "-shortest", out,
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    rnd = random.Random(7)
    lines = []
    t = 0.0
    while t < DUR:
        rate = 0.4
        if 500 <= t <= 512:
            rate = 12.0  # chat goes wild at 8:20
        elif 131 <= t <= 145 or 305 <= t <= 312:
            rate = 3.0
        t += rnd.expovariate(rate)
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        lines.append(f"[{h}:{m:02d}:{s:02d}] viewer{rnd.randint(1, 400)}: {rnd.choice(['lol', 'gg', 'KEKW', 'no way', 'LETS GO', 'hi'])}")
    with open(chat, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out} and {chat}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
