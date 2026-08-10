"""Context-preserving TileLang visitor for static action occurrences."""

from __future__ import annotations

from typing import Any, Callable

from .tileop_discovery import classify_call


def _expr(value: Any) -> str:
    return str(value)


def _integer(value: Any) -> int | None:
    try:
        return int(value.value)
    except (AttributeError, TypeError, ValueError):
        try:
            return int(str(value))
        except ValueError:
            return None


def collect_tileop_calls(prim_func: Any, on_action: Callable[[Any, dict[str, Any]], None]) -> dict[str, Any]:
    """Visit calls in lexical order while preserving enclosing loop predicates.

    ``on_action`` receives a TileOp Call and a JSON-serializable context.  The
    visitor is defined lazily because importing TileLang/TVM at package import
    time would load its native libraries for users that only inspect artifacts.
    """
    import tvm.tirx as tirx

    @tirx.functor.visitor
    class ContextVisitor(tirx.PyStmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.loop_stack: list[dict[str, Any]] = []
            self.condition_stack: list[str] = []
            self.action_ordinal = 0
            self.loop_ordinal = 0
            self.observed_tileop_calls = 0
            self.observed_action_calls = 0
            self.ignored_constructor_calls = 0
            self.unregistered_action_calls = 0

        def visit_for_(self, op):
            loop = {
                "loop_id": f"lexical_loop/{self.loop_ordinal}",
                "var": str(op.loop_var),
                "min": _expr(op.min),
                "extent": _expr(op.extent),
                "extent_value": _integer(op.extent),
                "kind": str(op.kind),
                "thread_binding": _expr(op.thread_binding) if op.thread_binding else None,
                "annotations": {str(key): _expr(value) for key, value in op.annotations.items()},
            }
            self.loop_ordinal += 1
            self.loop_stack.append(loop)
            try:
                super().visit_for_(op)
            finally:
                self.loop_stack.pop()

        def visit_if_then_else_(self, op):
            # The default visitor cannot distinguish then/else contexts.  Visit
            # manually so an action never receives an unconditional count when
            # it is guarded by a predicate.
            self.visit_expr(op.condition)
            condition = _expr(op.condition)
            self.condition_stack.append(condition)
            self.visit_stmt(op.then_case)
            self.condition_stack.pop()
            if op.else_case is not None:
                self.condition_stack.append(f"not ({condition})")
                self.visit_stmt(op.else_case)
                self.condition_stack.pop()

        def visit_call_(self, op):
            classification = classify_call(op)
            if classification is not None:
                self.observed_tileop_calls += 1
                if classification == "constructor":
                    self.ignored_constructor_calls += 1
                else:
                    self.observed_action_calls += 1
                    if classification == "unregistered_action":
                        self.unregistered_action_calls += 1
                    on_action(
                        op,
                        {
                            "ordinal": self.action_ordinal,
                            "statement_path": f"tileop/{self.action_ordinal}",
                            "loop_context": [dict(item) for item in self.loop_stack],
                            "condition_context": list(self.condition_stack),
                            "classification": classification,
                        },
                    )
                    self.action_ordinal += 1
            super().visit_call_(op)

    visitor = ContextVisitor()
    visitor.visit_stmt(prim_func.body)
    return {
        "observed_tileop_calls": visitor.observed_tileop_calls,
        "observed_action_calls": visitor.observed_action_calls,
        "ignored_constructor_calls": visitor.ignored_constructor_calls,
        "unregistered_action_calls": visitor.unregistered_action_calls,
    }
