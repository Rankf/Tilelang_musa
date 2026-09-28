"""Strict dtype-aware element-wise PrimFuncs for compiler-info integration."""

import tilelang.language as T


def make_add_prim_func(
    n=16384,
    threads=256,
    elements_per_thread=8,
    *,
    dtype="float32",
    use_write_through_store=False,
    output_store_policy=None,
):
    """Return C[i] = A[i] + B[i] with exact one-dimensional coverage.

    The optional Store-policy switches are S5000 measurement probes.  They
    replace only the final vector Store with one of MUSA's ``float4`` cache
    policy intrinsics.  The SDK headers do not document a Kernel-exit
    physical-DDR completion guarantee for any of them.  The legacy
    ``use_write_through_store`` flag remains as an alias for policy ``wt``.
    """
    if n <= 0 or threads <= 0 or elements_per_thread <= 0:
        raise ValueError("shape and launch parameters must be positive")
    elements_per_block = threads * elements_per_thread
    if n % elements_per_block:
        raise ValueError("n must be divisible by threads * elements_per_thread")
    if output_store_policy is None:
        output_store_policy = "wt" if use_write_through_store else "default"
    elif use_write_through_store:
        raise ValueError(
            "use_write_through_store and output_store_policy are mutually exclusive"
        )
    if output_store_policy not in {"default", "wb", "cg", "cs", "wt"}:
        raise ValueError(f"unsupported output_store_policy: {output_store_policy}")
    if output_store_policy != "default" and elements_per_thread != 4:
        raise ValueError("explicit vector Store policies require elements_per_thread=4")
    store_function = f"tl::st{output_store_policy}4"

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        C: T.Tensor((n,), dtype),
    ):
        with T.Kernel(n // elements_per_block, threads=threads) as bx:
            if output_store_policy != "default":
                tx = T.get_thread_binding(0)
                base = (bx * threads + tx) * elements_per_thread
                lanes = T.Ramp(base, 1, elements_per_thread)
                value = A[lanes] + B[lanes]
                T.evaluate(
                    T.call_extern(
                        "void", store_function, T.address_of(C[base]), value
                    )
                )
            else:
                for tx, local in T.Parallel(threads, elements_per_thread):
                    index = (bx * threads + tx) * elements_per_thread + local
                    value = A[index] + B[index]
                    C[index] = value

    return main


def make_relu_prim_func(
    n=16384,
    threads=256,
    elements_per_thread=4,
    *,
    dtype="float32",
    use_write_through_store=False,
    output_store_policy=None,
):
    """Return B[i] = max(A[i], 0) with an optional Store-policy probe."""
    if n <= 0 or threads <= 0 or elements_per_thread <= 0:
        raise ValueError("shape and launch parameters must be positive")
    elements_per_block = threads * elements_per_thread
    if n % elements_per_block:
        raise ValueError("n must be divisible by threads * elements_per_thread")
    if output_store_policy is None:
        output_store_policy = "wt" if use_write_through_store else "default"
    elif use_write_through_store:
        raise ValueError(
            "use_write_through_store and output_store_policy are mutually exclusive"
        )
    if output_store_policy not in {"default", "wb", "cg", "cs", "wt"}:
        raise ValueError(f"unsupported output_store_policy: {output_store_policy}")
    if output_store_policy != "default" and elements_per_thread != 4:
        raise ValueError("explicit vector Store policies require elements_per_thread=4")
    store_function = f"tl::st{output_store_policy}4"

    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
    ):
        with T.Kernel(n // elements_per_block, threads=threads) as bx:
            if output_store_policy != "default":
                tx = T.get_thread_binding(0)
                base = (bx * threads + tx) * elements_per_thread
                lanes = T.Ramp(base, 1, elements_per_thread)
                value = T.max(A[lanes], T.Broadcast(0.0, elements_per_thread))
                T.evaluate(
                    T.call_extern(
                        "void", store_function, T.address_of(B[base]), value
                    )
                )
            else:
                for tx, local in T.Parallel(threads, elements_per_thread):
                    index = (bx * threads + tx) * elements_per_thread + local
                    value = T.max(A[index], 0.0)
                    B[index] = value

    return main


def make_unary_math_prim_func(
    op="exp", n=16384, threads=256, elements_per_thread=8, *, dtype="float16"
):
    """Return an exact one-pass Exp/Log/Sqrt kernel for formal analysis."""
    if op not in {"exp", "log", "sqrt"}:
        raise ValueError(f"unsupported unary math op: {op}")
    elements_per_block = threads * elements_per_thread
    if n <= 0 or n % elements_per_block:
        raise ValueError("n must be positive and divisible by the block tile")

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(n // elements_per_block, threads=threads) as bx:
            for tx, local in T.Parallel(threads, elements_per_thread):
                index = (bx * threads + tx) * elements_per_thread + local
                value = T.cast(A[index], "float32")
                if op == "exp":
                    value = T.exp(value)
                elif op == "log":
                    value = T.log(value)
                else:
                    value = T.sqrt(value)
                B[index] = T.cast(value, dtype)

    return main
