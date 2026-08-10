import unittest

from musa_perf_collector.collectors.compiler.analysis import build_dag, reuse_distance, workload_domains
from musa_perf_collector.collectors.compiler.metadata import dtype_bytes


class AnalysisTest(unittest.TestCase):
    def test_dag_and_demand_are_conservative(self):
        ops = [
            {
                "op_id": "copy", "kind": "copy", "ordinal": 0,
                "reads": [{"buffer": "A", "logical_bytes": 16, "scope": "global"}],
                "writes": [{"buffer": "S", "logical_bytes": 16, "scope": "shared"}],
            },
            {
                "op_id": "gemm", "kind": "gemm", "ordinal": 1,
                "reads": [{"buffer": "S", "logical_bytes": 16, "scope": "shared"}],
                "writes": [{"buffer": "C", "logical_bytes": 32, "scope": "local"}],
                "gemm": {"flops": 128},
            },
        ]
        dag = build_dag(ops)
        demand = workload_domains(ops)["per_block"]
        self.assertEqual(len(dag["edges"]), 1)
        self.assertEqual(dag["edges"][0]["confidence"], "may_depend_buffer_match")
        self.assertEqual(demand["flops"], 128)
        self.assertEqual(demand["scope_transfer_bytes"], {"global->shared": 16})

    def test_dtype_bytes_does_not_corrupt_vector_suffixes(self):
        self.assertEqual(dtype_bytes("float16"), 2)
        self.assertEqual(dtype_bytes("float16x8"), 2)
        self.assertEqual(dtype_bytes("int32x4"), 4)
        self.assertIsNone(dtype_bytes("handle"))

    def test_conversion_copy_is_not_claimed_as_one_transfer_byte_count(self):
        ops = [{
            "op_id": "convert", "kind": "copy", "ordinal": 0,
            "reads": [{"buffer": "L", "logical_bytes": 16, "scope": "local"}],
            "writes": [{"buffer": "G", "logical_bytes": 8, "scope": "global"}],
        }]
        demand = workload_domains(ops)["per_block"]
        self.assertEqual(demand["scope_transfer_bytes"], {})
        self.assertEqual(demand["ambiguous_scope_transfers"][0]["source_logical_bytes"], 16)

    def test_same_operation_read_write_is_not_zero_distance_reuse(self):
        ops = [{
            "op_id": "accumulate", "kind": "gemm", "ordinal": 0,
            "reads": [{"buffer": "C", "indices": ["0"], "shape": ["16"], "logical_bytes": 64, "scope": "local"}],
            "writes": [{"buffer": "C", "indices": ["0"], "shape": ["16"], "logical_bytes": 64, "scope": "local"}],
        }]
        self.assertEqual(reuse_distance(ops, [1024])["reuse_events"], [])

    def test_symbolic_execution_is_emitted_as_an_expression_not_dropped(self):
        ops = [{
            "op_id": "symbolic", "kind": "gemm", "canonical_family": "gemm_dense", "ordinal": 0,
            "reads": [{"buffer": "A", "logical_bytes": 16, "scope": "global"}],
            "writes": [{"buffer": "C", "logical_bytes": 16, "scope": "global"}],
            "gemm": {"flops": 128}, "support_level": "structured",
            "execution": {"calls_per_block_value": None, "calls_per_block_expr": "K_tiles", "calls_per_kernel_value": None, "calls_per_kernel_expr": "K_tiles * blocks"},
        }]
        per_block = workload_domains(ops)["per_block"]
        self.assertEqual(per_block["flops"], "(128) * (K_tiles)")
        self.assertEqual(per_block["global_logical_bytes"], "(32) * (K_tiles)")
