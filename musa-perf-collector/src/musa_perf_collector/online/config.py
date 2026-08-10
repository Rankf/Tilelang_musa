"""Validated, immutable configuration for online collection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "musa-perf-collector/online-config-v1"
_CORRECTNESS = {"application_owned", "not_observed"}
_COMPILE_POLICIES = {"observe_only", "eager_dual_variant"}
_ARTIFACT_POLICIES = {"compiler_exact_only"}
_BUFFER_UNAVAILABLE_POLICIES = {"run_baseline_and_record_skip"}
_MAX_PENDING_READBACKS = 4096


@dataclass(frozen=True)
class OnlineBudget:
    max_instrumented_launches: int = 10
    max_trace_records: int = 128
    max_trace_bytes: int = 8192
    max_pending_readbacks: int = 2
    max_total_duration_s: float = 30.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "OnlineBudget":
        raw = dict(value or {})
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown online budget keys: {sorted(unknown)}")
        for name, item in raw.items():
            expected = (int, float) if name == "max_total_duration_s" else (int,)
            if isinstance(item, bool) or not isinstance(item, expected):
                kind = "numeric" if name == "max_total_duration_s" else "an integer"
                raise ValueError(f"budget.{name} must be {kind}")
        result = cls(**raw)
        for name in cls.__dataclass_fields__:
            if getattr(result, name) < 0:
                raise ValueError(f"budget.{name} must not be negative")
        if result.max_pending_readbacks <= 0:
            raise ValueError("budget.max_pending_readbacks must be positive")
        if result.max_pending_readbacks > _MAX_PENDING_READBACKS:
            raise ValueError(
                f"budget.max_pending_readbacks must not exceed {_MAX_PENDING_READBACKS}")
        if not math.isfinite(float(result.max_total_duration_s)):
            raise ValueError("budget.max_total_duration_s must be finite")
        return result


@dataclass(frozen=True)
class OnlineConfig:
    schema: str
    enabled: bool
    output: Path
    target: str = "musa"
    execution_backends: tuple[str, ...] = ("tvm_ffi",)
    artifact_policy: str = "compiler_exact_only"
    lineage_artifact: Path | None = None
    lowering_semantics: str = "single"
    kernel_selector: dict[str, Any] | None = None
    launch_candidate_count: int = 10000
    event_timing_every: int = 1
    launch_selector: dict[str, Any] | None = None
    block_selector: dict[str, Any] | None = None
    trace_op_ids: tuple[str, ...] = ()
    correctness_policy: str = "application_owned"
    compile_policy: str = "observe_only"
    buffer_unavailable_policy: str = "run_baseline_and_record_skip"
    telemetry: dict[str, Any] | None = None
    budget: OnlineBudget = OnlineBudget()
    config_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "enabled": self.enabled,
            "target": self.target,
            "execution_backends": list(self.execution_backends),
            "artifact_policy": self.artifact_policy,
            "lineage_artifact": str(self.lineage_artifact) if self.lineage_artifact else None,
            "lowering_semantics": self.lowering_semantics,
            "kernel_selector": self.kernel_selector,
            "launch_candidate_count": self.launch_candidate_count,
            "event_timing_every": self.event_timing_every,
            "launch_selector": self.launch_selector,
            "block_selector": self.block_selector,
            "trace_op_ids": list(self.trace_op_ids),
            "correctness_policy": self.correctness_policy,
            "compile_policy": self.compile_policy,
            "buffer_unavailable_policy": self.buffer_unavailable_policy,
            "telemetry": self.telemetry,
            "budget": {name: getattr(self.budget, name) for name in self.budget.__dataclass_fields__},
            "output": str(self.output),
            "config_sha256": self.config_sha256,
        }


def _selector(
    value: Any, name: str, *, allowed_kinds: set[str],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    kinds = [key for key in ("indices", "every", "probability") if key in value]
    if len(kinds) != 1:
        raise ValueError(f"{name} must contain exactly one selection kind")
    kind = kinds[0]
    if kind not in allowed_kinds:
        raise ValueError(f"{name} does not support {kind!r} selection")
    allowed_keys = {kind, "start"} if kind == "every" else {kind}
    unknown = set(value) - allowed_keys
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    if kind == "probability":
        probability = value[kind]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"{name}.probability must be numeric")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"{name}.probability must be in [0, 1]")
    elif kind == "every":
        every = value[kind]
        start = value.get("start", 0)
        if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
            raise ValueError(f"{name}.every must be a positive integer")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError(f"{name}.start must be a non-negative integer")
    else:
        indices = value[kind]
        if (
            not isinstance(indices, list) or not indices
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"{name}.indices must be non-empty, unique, non-negative integers")
    return dict(value)


def load_online_config(path: str | Path) -> OnlineConfig:
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("online config root must be an object")
    allowed = {
        "schema", "enabled", "target", "execution_backends", "artifact_policy", "lineage_artifact",
        "lowering_semantics", "event_timing_every",
        "kernel_selector", "launch_candidate_count", "launch_selector", "block_selector",
        "trace_op_ids", "correctness_policy", "compile_policy",
        "buffer_unavailable_policy", "budget", "output",
        "telemetry",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown online config keys: {sorted(unknown)}")
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"online config schema must be {SCHEMA!r}")
    if "enabled" in raw and not isinstance(raw["enabled"], bool):
        raise ValueError("online config enabled must be boolean")
    output_value = raw.get("output")
    if not output_value:
        raise ValueError("online config output is required")
    correctness = str(raw.get("correctness_policy", "application_owned"))
    if correctness not in _CORRECTNESS:
        raise ValueError(f"unsupported correctness_policy: {correctness}")
    compile_policy = str(raw.get("compile_policy", "observe_only"))
    if compile_policy not in _COMPILE_POLICIES:
        raise ValueError(f"unsupported compile_policy: {compile_policy}")
    lowering_semantics = str(raw.get("lowering_semantics", "single"))
    if lowering_semantics not in {"single", "source_envelope", "lowered_instance"}:
        raise ValueError("lowering_semantics must be single, source_envelope, or lowered_instance")
    candidate_value = raw.get("launch_candidate_count", 10000)
    if isinstance(candidate_value, bool) or not isinstance(candidate_value, int):
        raise ValueError("launch_candidate_count must be an integer")
    candidate_count = candidate_value
    if candidate_count < 0:
        raise ValueError("launch_candidate_count must not be negative")
    event_timing_every = raw.get("event_timing_every", 1)
    if (
        isinstance(event_timing_every, bool)
        or not isinstance(event_timing_every, int)
        or event_timing_every <= 0
    ):
        raise ValueError("event_timing_every must be a positive integer")
    backends = tuple(str(item) for item in raw.get("execution_backends", ["tvm_ffi"]))
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("execution_backends must be non-empty and unique")
    trace_op_ids = tuple(str(item) for item in raw.get("trace_op_ids", []))
    if len(set(trace_op_ids)) != len(trace_op_ids):
        raise ValueError("trace_op_ids must be unique")
    lineage_value = raw.get("lineage_artifact")
    lineage_artifact = Path(lineage_value).expanduser().resolve() if lineage_value else None
    if compile_policy == "eager_dual_variant" and lineage_artifact is None:
        raise ValueError("eager_dual_variant requires lineage_artifact")
    if lineage_artifact is not None and not lineage_artifact.exists():
        raise ValueError(f"lineage_artifact does not exist: {lineage_artifact}")
    if compile_policy == "eager_dual_variant" and not trace_op_ids:
        raise ValueError("eager_dual_variant requires non-empty trace_op_ids")
    artifact_policy = str(raw.get("artifact_policy", "compiler_exact_only"))
    if artifact_policy not in _ARTIFACT_POLICIES:
        raise ValueError(f"unsupported artifact_policy: {artifact_policy}")
    buffer_policy = str(
        raw.get("buffer_unavailable_policy", "run_baseline_and_record_skip"))
    if buffer_policy not in _BUFFER_UNAVAILABLE_POLICIES:
        raise ValueError(f"unsupported buffer_unavailable_policy: {buffer_policy}")
    telemetry_value = raw.get("telemetry", {"enabled": True, "device": 0})
    if not isinstance(telemetry_value, dict):
        raise ValueError("telemetry must be an object")
    telemetry = dict(telemetry_value)
    telemetry_unknown = set(telemetry) - {"enabled", "device"}
    if telemetry_unknown:
        raise ValueError(f"unknown telemetry keys: {sorted(telemetry_unknown)}")
    telemetry_enabled = telemetry.get("enabled", True)
    telemetry_device = telemetry.get("device", 0)
    if not isinstance(telemetry_enabled, bool):
        raise ValueError("telemetry.enabled must be boolean")
    if isinstance(telemetry_device, bool) or not isinstance(telemetry_device, int):
        raise ValueError("telemetry.device must be an integer")
    telemetry = {"enabled": telemetry_enabled, "device": telemetry_device}
    if telemetry["device"] < 0:
        raise ValueError("telemetry.device must not be negative")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"
    return OnlineConfig(
        schema=SCHEMA,
        enabled=bool(raw.get("enabled", False)),
        output=Path(output_value).expanduser().resolve(),
        target=str(raw.get("target", "musa")),
        execution_backends=backends,
        artifact_policy=artifact_policy,
        lineage_artifact=lineage_artifact,
        lowering_semantics=lowering_semantics,
        kernel_selector=_selector(
            raw.get("kernel_selector"), "kernel_selector", allowed_kinds={"probability"}),
        launch_candidate_count=candidate_count,
        event_timing_every=event_timing_every,
        launch_selector=_selector(
            raw.get("launch_selector", {"every": 64, "start": 0}), "launch_selector",
            allowed_kinds={"indices", "every", "probability"}),
        block_selector=_selector(
            raw.get("block_selector"), "block_selector", allowed_kinds={"indices"}),
        trace_op_ids=trace_op_ids,
        correctness_policy=correctness,
        compile_policy=compile_policy,
        buffer_unavailable_policy=buffer_policy,
        telemetry=telemetry,
        budget=OnlineBudget.from_dict(raw.get("budget")),
        config_sha256=digest,
    )
