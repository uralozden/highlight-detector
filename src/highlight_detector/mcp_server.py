"""MCP server: lets Claude, Cursor or any MCP client find highlights in local videos.

Run:  highlight-detector-mcp      (stdio transport)
"""

from __future__ import annotations

from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    try:
        from mcp.server.fastmcp import FastMCP as _Server
    except ImportError as e:  # pragma: no cover
        raise SystemExit("The MCP server needs the extra: pip install 'highlight-detector[mcp]'") from e

from . import export
from .detector import detect
from .scoring import Options

server = _Server("highlight-detector")


@server.tool()
def find_highlights(
    video_path: str,
    top: int = 10,
    min_len: float = 12,
    max_len: float = 45,
    chat_path: str | None = None,
    subtitles_path: str | None = None,
) -> dict:
    """Find the most intense moments in a local gameplay video or stream recording.

    Returns ranked highlights with start/end seconds, a 0-100 score (relative to this video,
    not a view prediction) and a plain-language reason. chat_path accepts a chat replay with
    timestamps; subtitles_path accepts SRT/VTT.
    """
    result = detect(
        Path(video_path).expanduser(),
        Options(top=top, min_len=min_len, max_len=max_len),
        chat=chat_path,
        subtitles=subtitles_path,
    )
    result.pop("timeline", None)
    return result


@server.tool()
def export_highlights(video_path: str, out_dir: str, top: int = 5, copy: bool = False) -> list[str]:
    """Find highlights and cut each one into an MP4 file in out_dir. Returns the file paths."""
    src = Path(video_path).expanduser()
    result = detect(src, Options(top=top))
    return [str(p) for p in export.cut_clips(src, result, Path(out_dir).expanduser(), copy=copy)]


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
