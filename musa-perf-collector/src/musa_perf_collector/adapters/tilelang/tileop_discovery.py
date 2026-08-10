"""Lossless TileLang TileOp discovery.

TileLang marks supported high-level operations with the ``TLOpBuilder`` op
attribute.  This is intentionally used instead of a local name whitelist so
that a compiler upgrade cannot silently erase a new TileOp from analysis.
"""

from __future__ import annotations

from typing import Any


CONSTRUCTOR_OPS = {"tl.tileop.region"}


def op_name(call: Any) -> str:
    """Return the canonical TVM op name without relying on repr formatting."""
    name = getattr(call.op, "name", None)
    if name:
        return str(name)
    text = str(call.op)
    return text.split('name="')[-1].split('"')[0]


def has_tileop_builder(call: Any) -> bool:
    try:
        return call.op.get_attr("TLOpBuilder") is not None
    except (AttributeError, TypeError):
        return False


def classify_call(call: Any) -> str | None:
    """Classify a Call as action, constructor, or an unregistered TileOp.

    ``unregistered_action`` is retained for malformed/future IR with a
    ``tl.tileop.`` name but no builder.  Returning ``None`` means this Call is
    outside the TileOp system and should not affect TileOp coverage.
    """
    name = op_name(call)
    if not has_tileop_builder(call) and not name.startswith("tl.tileop."):
        return None
    if name in CONSTRUCTOR_OPS:
        return "constructor"
    return "action" if has_tileop_builder(call) else "unregistered_action"
