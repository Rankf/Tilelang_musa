"""Narrow, reviewable runtime workload contract; arbitrary benchmark scripts are not run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PreparedRuntimeCase:
    name: str
    kernel_id: str
    kernel_source: str
    run: Callable[[], Any]
    reference: Callable[[], Any]
    compare: Callable[[Any, Any], None]
    launch: dict[str, Any]
    instrumented: bool = False
    lineage: list[dict[str, Any]] | None = None
    tile_trace: Callable[[], list[dict[str, Any]]] | None = None
    # `run_timed` deliberately excludes trace readback and decoding.  It lets
    # the runtime compare callable submission overhead without accidentally
    # timing CPU synchronization or Python list construction.
    run_timed: Callable[[], Any] | None = None
    trace_metadata: dict[str, Any] | None = None
    enqueue_trace_readback: Callable[[], dict[str, Any]] | None = None
    finish_trace_readback: Callable[[], list[dict[str, Any]]] | None = None
    current_trace_generation: Callable[[], int] | None = None
    execution_stream: Any | None = None


class RuntimeWorkload:
    """A workload factory must compile before timing and expose a reference result."""

    def prepare(self, config: dict[str, Any], *, variant: str = "baseline") -> PreparedRuntimeCase:
        raise NotImplementedError

    def resolve_execution_stream(self, config: dict[str, Any]):
        """Return one shared stream for baseline and instrumented variants."""
        import torch

        policy = str(config.get("execution_stream", "current"))
        if policy not in {"current", "default", "custom"}:
            raise ValueError("execution_stream must be current, default, or custom")
        cached = getattr(self, "_execution_stream", None)
        cached_policy = getattr(self, "_execution_stream_policy", None)
        if cached is not None:
            if cached_policy != policy:
                raise RuntimeError("execution stream policy changed between workload variants")
            return cached
        if policy == "custom":
            stream = torch.musa.Stream()
        elif policy == "default":
            stream = torch.musa.default_stream()
        else:
            stream = torch.musa.current_stream()
        self._execution_stream = stream
        self._execution_stream_policy = policy
        return stream
