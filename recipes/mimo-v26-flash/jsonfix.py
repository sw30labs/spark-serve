"""Repair the one known invalid JSON file in the MiMo snapshot."""
from __future__ import annotations

import json
import re
from pathlib import Path

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def repair_json_text(text: str) -> str:
    """Return text unchanged when it already parses. Otherwise strip trailing commas."""
    try:
        json.loads(text)
    except json.JSONDecodeError:
        text = _TRAILING_COMMA.sub(r"\1", text)
        json.loads(text)
    return text


def repair_file(path: Path) -> bool:
    original = path.read_text()
    fixed = repair_json_text(original)
    if fixed == original:
        return False
    path.write_text(fixed)
    return True
