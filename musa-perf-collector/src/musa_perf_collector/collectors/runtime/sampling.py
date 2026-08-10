"""Deterministic selection and non-negotiable collection budgets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable


SELECTION_ALGORITHM = "sha256_threshold_v1"


class BudgetExceeded(RuntimeError):
    """Raised when a collector would exceed its resolved collection budget."""


@dataclass(frozen=True)
class CollectionBudget:
    max_total_duration_s: float = 30.0
    max_profile_duration_ms: float = 500.0
    max_profiled_launches: int = 10
    max_instrumented_launches: int = 2
    max_sampled_blocks_per_launch: int = 1
    max_tileops_per_block: int = 3
    max_iterations_per_tileop: int = 4
    max_trace_records: int = 128
    max_trace_bytes: int = 8192
    max_counter_passes: int = 1
    max_dropped_records: int = 0
    on_budget_exceeded: str = "degrade"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CollectionBudget":
        values = {name: config.get(name, getattr(cls, name)) for name in cls.__dataclass_fields__}
        budget = cls(**values)
        if budget.on_budget_exceeded not in {"stop", "degrade", "fail"}:
            raise ValueError("on_budget_exceeded must be stop, degrade, or fail")
        for name, value in values.items():
            if name != "on_budget_exceeded" and value < 0:
                raise ValueError(f"{name} must not be negative")
        return budget


def stable_selected(identifier: str, *, seed: int, probability: float) -> bool:
    """Return a reproducible Bernoulli choice without process-local randomness."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < probability


def choose_indices(candidates: Iterable[int], selector: dict[str, Any] | None, *, seed: int) -> list[int]:
    """Select indices exactly, periodically, or by deterministic probability."""
    values = list(candidates)
    if not selector:
        return values
    if "indices" in selector:
        wanted = set(int(value) for value in selector["indices"])
        return [value for value in values if value in wanted]
    every = selector.get("every")
    if every is not None:
        if int(every) <= 0:
            raise ValueError("selector.every must be positive")
        start = int(selector.get("start", 0))
        return [value for value in values if value >= start and (value - start) % int(every) == 0]
    if "probability" in selector:
        return [value for value in values if stable_selected(str(value), seed=seed, probability=float(selector["probability"]))]
    raise ValueError("selector must contain indices, every, or probability")


def selection_evidence(
    candidates: Iterable[int], selector: dict[str, Any] | None, *, seed: int,
    dimension: str, budget_limit: int, policy: str,
) -> tuple[list[int], dict[str, Any], dict[str, Any] | None]:
    """Resolve one integer selector and preserve enough evidence to replay it.

    Selector precedence is exact ``indices``, then ``every``, then deterministic
    ``probability``.  ``choose_indices`` implements that precedence; this wrapper
    adds candidate-set identity and the hard-budget decision.
    """
    values = list(candidates)
    if len(set(values)) != len(values):
        raise ValueError(f"{dimension} candidates must be unique")
    if selector is not None and "indices" in selector:
        requested = [int(value) for value in selector["indices"]]
        if len(set(requested)) != len(requested):
            raise ValueError(f"{dimension} exact selector indices must be unique")
        missing = sorted(set(requested) - set(values))
        if missing:
            raise ValueError(f"{dimension} exact selector indices are outside the candidate set: {missing}")
    selected_before_budget = choose_indices(values, selector, seed=seed)
    selected, diagnostic = apply_budget(
        selected_before_budget, budget_limit, label=f"{dimension}_selection", policy=policy)
    canonical = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    resolved_selector = dict(selector or {"indices": values})
    if "indices" in resolved_selector:
        kind = "exact"
    elif "every" in resolved_selector:
        kind = "periodic"
    else:
        kind = "probability"
    evidence = {
        "dimension": dimension,
        "candidate_count": len(values),
        "candidates_hash": f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
        "selected_before_budget": selected_before_budget,
        "selected": selected,
        "selector": resolved_selector,
        "selector_kind": kind,
        "seed": seed,
        "algorithm": SELECTION_ALGORITHM if kind == "probability" else f"{kind}_v1",
        "budget_limit": budget_limit,
        "budget_truncated": diagnostic is not None,
    }
    return selected, evidence, diagnostic


def identity_selection_evidence(
    candidates: Iterable[str], selector: dict[str, Any] | None, *, seed: int, dimension: str,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve exact/periodic/probability selection for named runtime scopes."""
    values = [str(value) for value in candidates]
    if len(set(values)) != len(values):
        raise ValueError(f"{dimension} candidates must be unique")
    resolved_selector = dict(selector or {"values": values})
    if "values" in resolved_selector:
        wanted = {str(value) for value in resolved_selector["values"]}
        selected = [value for value in values if value in wanted]
        kind = "exact"
    elif "every" in resolved_selector:
        positions = choose_indices(range(len(values)), resolved_selector, seed=seed)
        selected = [values[index] for index in positions]
        kind = "periodic"
    elif "probability" in resolved_selector:
        probability = float(resolved_selector["probability"])
        selected = [value for value in values if stable_selected(value, seed=seed, probability=probability)]
        kind = "probability"
    else:
        raise ValueError("identity selector must contain values, every, or probability")
    canonical = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    evidence = {
        "dimension": dimension, "candidate_count": len(values),
        "candidates_hash": f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
        "selected": selected, "selector": resolved_selector, "selector_kind": kind, "seed": seed,
        "algorithm": SELECTION_ALGORITHM if kind == "probability" else f"{kind}_v1",
        "budget_truncated": False,
    }
    return selected, evidence


def linear_block_coordinates(grid: Iterable[int]) -> list[list[int]]:
    """Enumerate a static 3-D grid with x as the fastest-changing dimension."""
    extents = [int(value) for value in grid]
    if not 1 <= len(extents) <= 3 or any(value <= 0 for value in extents):
        raise ValueError("block selection requires one to three positive integer grid extents")
    extents += [1] * (3 - len(extents))
    return [[x, y, z] for z in range(extents[2]) for y in range(extents[1]) for x in range(extents[0])]


def apply_budget(values: list[Any], maximum: int, *, label: str, policy: str) -> tuple[list[Any], dict[str, Any] | None]:
    """Never silently expand a selection beyond the requested hard maximum."""
    if len(values) <= maximum:
        return values, None
    diagnostic = {"code": "BUDGET_EXCEEDED", "label": label, "requested": len(values), "allowed": maximum, "action": policy}
    if policy == "fail":
        raise BudgetExceeded(f"{label}: requested {len(values)}, budget {maximum}")
    return values[:maximum], diagnostic
