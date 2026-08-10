"""Correctness results without persisting full intermediate tensors."""

from __future__ import annotations

from typing import Any, Callable

from ...core.evidence import RUNTIME_DIRECT


def verify(run: Callable[[], Any], reference: Callable[[], Any], compare: Callable[[Any, Any], None]) -> dict[str, Any]:
    try:
        actual = run()
        expected = reference()
        compare(actual, expected)
        return {"status": "pass", "evidence_level": RUNTIME_DIRECT, "tensor_dump": False}
    except Exception as error:
        return {"status": "fail", "evidence_level": RUNTIME_DIRECT, "tensor_dump": False,
                "error_type": type(error).__name__, "message": str(error)}
