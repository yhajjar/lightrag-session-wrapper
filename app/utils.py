from __future__ import annotations

from pathlib import Path


def normalize_file_name(value: str | None) -> str:
    """Return a lower-case, sanitized file name suitable for comparisons."""
    if not value:
        return ""
    name = str(value).strip()
    if not name:
        return ""
    # Drop directory components
    name = Path(name).name
    # Remove parent directory traversal and control characters
    name = name.replace("..", "")
    name = "".join(c for c in name if 32 <= ord(c) < 127)
    name = name.strip().strip(".")
    if not name:
        return ""
    return name.lower()
