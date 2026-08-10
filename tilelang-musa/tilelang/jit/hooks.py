"""Stable, dependency-free hooks around TileLang JIT compilation.

The registry is intentionally empty by default.  Hooks may observe a user
compile and return a wrapper for the compiled baseline kernel, but they never
receive permission to mutate the baseline PrimFunc in place.  Internal variant
compiles use a context-local origin token and bypass hook dispatch.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import threading
from typing import Any, Literal
import uuid


logger = logging.getLogger(__name__)

CompileOrigin = Literal["user_baseline", "collector_instrumented_internal"]
HookAction = Literal["observe", "dual_variant_plan", "reject_instrumentation"]


@dataclass(frozen=True)
class CompileRequest:
    """Immutable description of one compile before baseline cache lookup."""

    prim_func: Any
    normalized_out_idx: tuple[int, ...]
    original_out_idx: tuple[int, ...]
    target: Any
    target_host: Any
    execution_backend: str | None
    pass_configs: dict[str, Any] | None
    instruments: tuple[Any, ...]
    compile_flags: tuple[str, ...]
    verbose: bool | None
    baseline_cache_identity: str
    prim_func_sha256: str
    compile_origin: CompileOrigin


InternalCompile = Callable[..., Any]
KernelWrapper = Callable[[Any, InternalCompile], Any]


@dataclass(frozen=True)
class CompileAugmentation:
    """A hook decision applied only after the baseline compiled successfully."""

    action: HookAction
    wrap_kernel: KernelWrapper | None = None
    reason: str | None = None
    owner: str = "unknown"


@dataclass(frozen=True)
class HookFailure:
    owner: str
    error_type: str
    message: str


_hooks: "OrderedDict[str, Callable[[CompileRequest], CompileAugmentation | None]]" = OrderedDict()
_hooks_lock = threading.RLock()
_compile_origin: ContextVar[CompileOrigin] = ContextVar("tilelang_compile_origin", default="user_baseline")
_instrumented_plan_hash: ContextVar[str | None] = ContextVar("tilelang_instrumented_plan_hash", default=None)


def register_compile_hook(hook: Callable[[CompileRequest], CompileAugmentation | None]) -> str:
    """Register *hook* and return an opaque token used for unregistration."""
    if not callable(hook):
        raise TypeError("compile hook must be callable")
    token = uuid.uuid4().hex
    with _hooks_lock:
        _hooks[token] = hook
    return token


def unregister_compile_hook(token: str) -> bool:
    """Remove one hook.  Unknown tokens are harmless and return ``False``."""
    with _hooks_lock:
        return _hooks.pop(token, None) is not None


def get_compile_hooks() -> tuple[Callable[[CompileRequest], CompileAugmentation | None], ...]:
    """Return a stable registry snapshot so callbacks run outside the lock."""
    with _hooks_lock:
        return tuple(_hooks.values())


def current_compile_origin() -> CompileOrigin:
    return _compile_origin.get()


def current_instrumented_plan_hash() -> str | None:
    return _instrumented_plan_hash.get()


@contextmanager
def compile_origin(origin: CompileOrigin) -> Iterator[None]:
    token = _compile_origin.set(origin)
    try:
        yield
    finally:
        _compile_origin.reset(token)


@contextmanager
def instrumented_plan_hash(plan_hash: str) -> Iterator[None]:
    if not plan_hash or "/" in plan_hash or ".." in plan_hash:
        raise ValueError("instrumented plan hash must be a non-empty safe path component")
    token = _instrumented_plan_hash.set(plan_hash)
    try:
        yield
    finally:
        _instrumented_plan_hash.reset(token)


def normalize_out_idx(out_idx: list[int] | tuple[int, ...] | int | None, parameter_count: int) -> tuple[int, ...]:
    """Resolve negative result indexes without changing the baseline ABI."""
    if parameter_count < 0:
        raise ValueError("parameter_count must be non-negative")
    if out_idx is None:
        values: list[int] = []
    elif isinstance(out_idx, int):
        values = [out_idx]
    else:
        values = [int(value) for value in out_idx]
    normalized = tuple(value + parameter_count if value < 0 else value for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"output indexes must be unique after normalization: {normalized}")
    invalid = [value for value in normalized if value < 0 or value >= parameter_count]
    if invalid:
        raise ValueError(f"output indexes outside [0, {parameter_count}): {invalid}")
    return normalized


def make_baseline_identity(
    prim_func: Any,
    *,
    out_idx: tuple[int, ...],
    target: Any,
    target_host: Any,
    execution_backend: str | None,
    pass_configs: dict[str, Any] | None,
    compile_flags: tuple[str, ...],
) -> str:
    """Create an observation identity without participating in cache lookup."""
    script = prim_func.script(show_meta=True)
    value = {
        "prim_func_sha256": sha256(script.encode("utf-8")).hexdigest(),
        "out_idx": out_idx,
        "target": str(target),
        "target_host": str(target_host) if target_host is not None else None,
        "execution_backend": execution_backend,
        "pass_configs": pass_configs,
        "compile_flags": compile_flags,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"


def invoke_compile_hooks(request: CompileRequest) -> tuple[list[CompileAugmentation], list[HookFailure]]:
    """Run hooks with exception isolation; internal compiles are never dispatched."""
    if request.compile_origin != "user_baseline":
        return [], []
    augmentations: list[CompileAugmentation] = []
    failures: list[HookFailure] = []
    for hook in get_compile_hooks():
        owner = f"{getattr(hook, '__module__', type(hook).__module__)}.{getattr(hook, '__qualname__', type(hook).__qualname__)}"
        try:
            result = hook(request)
            if result is not None:
                if not isinstance(result, CompileAugmentation):
                    raise TypeError("compile hook must return CompileAugmentation or None")
                augmentations.append(result)
        except Exception as error:  # hooks must never take down baseline compilation
            failure = HookFailure(owner=owner, error_type=type(error).__name__, message=str(error))
            failures.append(failure)
            logger.exception("TileLang compile hook %s failed; continuing with baseline", owner)
    return augmentations, failures
