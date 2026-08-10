"""Conservative scalar dtype helpers shared by static collectors."""

from __future__ import annotations

import re


def dtype_bytes(dtype: str) -> int | None:
    """Return scalar storage width without multiplying a vector lane suffix."""
    normalized = dtype.strip().lower()
    if normalized == "bool":
        return 1
    match = re.fullmatch(r"(?:int|uint|float|bfloat)(\d+)(?:x\d+)?", normalized)
    return int(match.group(1)) // 8 if match and int(match.group(1)) % 8 == 0 else None
