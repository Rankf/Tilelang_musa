"""Real TileLang IR fixture tests; these do not invoke lowering or a device."""

from __future__ import annotations

import unittest

from musa_perf_collector.collectors.compiler.metadata import extract_metadata
from musa_perf_collector.examples.tileop_fixtures import build_atomic_prim_func, build_cumsum_prim_func, build_reduce_prim_func


class MetadataFixtureTest(unittest.TestCase):
    def test_cumsum_is_discovered_and_structurally_adapted(self):
        metadata = extract_metadata(build_cumsum_prim_func())
        cumsum = next(op for op in metadata["tile_ops"] if op["kind"] == "cumsum")
        self.assertEqual(cumsum["canonical_family"], "scan")
        self.assertEqual(cumsum["support_level"], "structural")
        self.assertEqual(cumsum["reads"][0]["scope"], "shared.dyn")
        self.assertEqual(cumsum["writes"][0]["scope"], "shared.dyn")
        self.assertTrue(metadata["tileop_coverage"]["lossless_export"])

    def test_atomic_add_is_discovered_as_read_modify_write(self):
        metadata = extract_metadata(build_atomic_prim_func())
        atomic = next(op for op in metadata["tile_ops"] if op["kind"] == "atomicadd")
        self.assertEqual(atomic["canonical_family"], "atomic")
        self.assertEqual(atomic["support_level"], "structured")
        self.assertEqual(atomic["reads"][-1]["buffer"], "B")
        self.assertEqual(atomic["writes"][0]["buffer"], "B")

    def test_reduce_is_discovered_as_structural(self):
        metadata = extract_metadata(build_reduce_prim_func())
        reduce = next(op for op in metadata["tile_ops"] if op["kind"] == "reduce")
        self.assertEqual(reduce["canonical_family"], "reduce")
        self.assertEqual(reduce["support_level"], "structural")
        self.assertEqual(reduce["reduction"]["kind"], '"sum"')
