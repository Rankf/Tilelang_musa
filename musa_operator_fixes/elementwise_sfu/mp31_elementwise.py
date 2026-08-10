"""MUSA MP31 elementwise kernel adapters (simple patch)."""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Patch 1: supported_archs ──
import tileops.kernels.elementwise as _elem
_patched = 0
for name in dir(_elem):
    cls = getattr(_elem, name)
    if isinstance(cls, type) and hasattr(cls, 'supported_archs'):
        if isinstance(cls.supported_archs, list) and 31 not in cls.supported_archs:
            cls.supported_archs = cls.supported_archs + [31]
            _patched += 1

# ── Patch 2: UnaryOp validation ──
import tileops.ops.elementwise._base as _base
def _patched_unary_validate(self, input):
    is_musa = hasattr(input, 'is_musa') and input.is_musa
    if not is_musa and not input.is_cuda:
        raise ValueError("Input must be a CUDA/MUSA tensor")
    if input.dtype != self.dtype:
        raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
    if input.numel() != self.N_total:
        raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
_base.UnaryOp._validate_input = _patched_unary_validate

print(f"[musa_operator_fixes.elementwise_sfu] patched {_patched} kernels + 1 validation for MP31")