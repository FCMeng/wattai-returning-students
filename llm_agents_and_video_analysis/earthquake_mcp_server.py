"""Small student-facing MCP server backed by a packaged USGS data snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer


COURSE_ROOT = Path(__file__).resolve().parent
DATA_PATH = COURSE_ROOT / "data" / "usgs_earthquakes_2025_01.jsonl"

mcp = MCPServer(
    "USGS Earthquakes",
    instructions="Look up a reviewed USGS earthquake by its stable event ID.",
)


@mcp.tool()
def lookup_usgs_earthquake(event_id: str) -> dict[str, Any]:
    """Return one earthquake from the January 2025 USGS classroom snapshot."""
    clean_id = event_id.strip()
    if not clean_id:
        raise ValueError("event_id must not be empty")

    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["event_id"] == clean_id:
            return record
    raise ValueError(f"USGS event not found in the classroom snapshot: {clean_id}")


if __name__ == "__main__":
    # With no transport argument, MCPServer uses stdio. Protocol messages use
    # stdout, so server diagnostics must be written to stderr rather than print().
    mcp.run()
