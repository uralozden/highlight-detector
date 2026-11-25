from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, export
from .detector import detect
from .media import MediaError
from .scoring import DEFAULT_WEIGHTS, Options

EPILOG = """examples:
  highlight-detector session.mp4
  highlight-detector stream.mkv --chat chat.json -n 15 --export clips/
  highlight-detector vod.mp4 --format chapters > chapters.txt
  highlight-detector vod.mp4 --format edl -o picks.edl      # open in Resolve / Premiere
  highlight-detector vod.mp4 --weights motion=0.4,loudness=0.2 --plot timeline.png

Need vertical 9:16 clips with captions and face cam layouts? https://hookharvest.co
"""


def _weights(spec: str | None) -> dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if not spec:
        return w
    for part in spec.split(","):
        if not part.strip():
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        if key not in DEFAULT_WEIGHTS:
            raise SystemExit(f"unknown weight '{key}'. Known: {', '.join(DEFAULT_WEIGHTS)}")
        w[key] = float(val)
    return w


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="highlight-detector",
        description="Find the action in long gameplay recordings and streams. Runs locally with FFmpeg; no GPU, no API keys.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="video or audio file (anything FFmpeg can read)")
    p.add_argument("-n", "--top", type=int, default=10, help="max highlights to return (default 10)")
    p.add_argument("--min-len", type=float, default=15, help="shortest clip in seconds (default 15)")
    p.add_argument("--max-len", type=float, default=45, help="longest clip in seconds (default 45)")
    p.add_argument("--pre-roll", type=float, default=5, help="context kept before the action starts (default 5)")
    p.add_argument("--post-roll", type=float, default=3, help="tail kept after the action ends (default 3)")
    p.add_argument("--min-gap", type=float, default=20, help="min seconds between two highlights (default 20)")
    p.add_argument("--min-score", type=float, default=15, help="drop picks below this score 0-100 (default 15)")
    p.add_argument("--chat", help="chat replay with timestamps (TwitchDownloader JSON, CSV, or '[h:mm:ss] msg' text)")
    p.add_argument("--subs", help="subtitles (SRT/VTT) to detect hype words like 'no way', 'let's go', 'yok artık'")
    p.add_argument("--weights", help="override signal weights, e.g. motion=0.4,chat=0.5 (" + ", ".join(DEFAULT_WEIGHTS) + ")")
    p.add_argument("--audio-only", action="store_true", help="skip video analysis (faster)")
    p.add_argument("--fps", type=float, default=4, help="frames per second sampled for motion (default 4)")
    p.add_argument("-f", "--format", choices=["table", "json", "csv", "chapters", "edl"], help="output format (default: table in a terminal, json when piped)")
    p.add_argument("-o", "--output", help="write the output here instead of stdout")
    p.add_argument("--timeline", action="store_true", help="include the per-0.5s intensity curve in JSON")
    p.add_argument("--plot", help="save a timeline PNG (needs Pillow)")
    p.add_argument("--export", metavar="DIR", help="cut every highlight into an MP4 in DIR")
    p.add_argument("--copy", action="store_true", help="with --export: stream copy (instant, cuts snap to keyframes)")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_len > args.max_len:
        print("error: --min-len is longer than --max-len", file=sys.stderr)
        return 2
    opts = Options(
        top=args.top,
        min_len=args.min_len,
        max_len=args.max_len,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        min_gap=args.min_gap,
        min_score=args.min_score,
        weights=_weights(args.weights),
    )
    if sys.stderr.isatty():
        print(f"analysing {Path(args.video).name} ...", file=sys.stderr)
    try:
        result = detect(args.video, opts, fps=args.fps, use_video=not args.audio_only, chat=args.chat, subtitles=args.subs)
    except (MediaError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    fmt = args.format or ("table" if sys.stdout.isatty() and not args.output else "json")
    text = {
        "table": lambda: export.to_table(result),
        "json": lambda: export.to_json(result, timeline=args.timeline) + "\n",
        "csv": lambda: export.to_csv(result),
        "chapters": lambda: export.to_chapters(result),
        "edl": lambda: export.to_edl(result),
    }[fmt]()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    if args.plot:
        from .plot import plot_timeline

        print(f"wrote {plot_timeline(result, args.plot)}", file=sys.stderr)
    if args.export:
        for pth in export.cut_clips(args.video, result, args.export, copy=args.copy):
            print(f"wrote {pth}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
