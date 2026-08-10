"""Dependency-free contracts for collector-owned TileLang compile hooks."""

import pytest

from tilelang.jit.hooks import (
    CompileAugmentation,
    CompileRequest,
    compile_origin,
    current_compile_origin,
    current_instrumented_plan_hash,
    instrumented_plan_hash,
    invoke_compile_hooks,
    normalize_out_idx,
    register_compile_hook,
    unregister_compile_hook,
)


def _request():
    return CompileRequest(
        prim_func=object(), normalized_out_idx=(2,), original_out_idx=(-1,),
        target="musa", target_host=None, execution_backend="tvm_ffi",
        pass_configs=None, instruments=(), compile_flags=(), verbose=False,
        baseline_cache_identity="sha256:test", prim_func_sha256="test",
        compile_origin="user_baseline")


def test_hook_registry_isolates_failures_and_unregisters_exact_token():
    def failing(_request):
        raise RuntimeError("injected hook failure")

    def observing(_request):
        return CompileAugmentation(action="observe", owner="test")

    failing_token = register_compile_hook(failing)
    observing_token = register_compile_hook(observing)
    try:
        augmentations, failures = invoke_compile_hooks(_request())
        assert [item.action for item in augmentations] == ["observe"]
        assert [item.error_type for item in failures] == ["RuntimeError"]
        assert unregister_compile_hook(failing_token)
        assert not unregister_compile_hook(failing_token)
    finally:
        unregister_compile_hook(failing_token)
        unregister_compile_hook(observing_token)


def test_context_tokens_restore_after_exception_and_reject_unsafe_namespace():
    assert current_compile_origin() == "user_baseline"
    assert current_instrumented_plan_hash() is None
    with pytest.raises(LookupError):
        with compile_origin("collector_instrumented_internal"), instrumented_plan_hash("a" * 64):
            assert current_compile_origin() == "collector_instrumented_internal"
            assert current_instrumented_plan_hash() == "a" * 64
            raise LookupError("stop")
    assert current_compile_origin() == "user_baseline"
    assert current_instrumented_plan_hash() is None
    with pytest.raises(ValueError):
        with instrumented_plan_hash("../escape"):
            pass


def test_output_index_normalization_is_exact_and_fail_closed():
    assert normalize_out_idx([-1, 0], 3) == (2, 0)
    with pytest.raises(ValueError, match="unique"):
        normalize_out_idx([-1, 2], 3)
    with pytest.raises(ValueError, match="outside"):
        normalize_out_idx([-4], 3)
