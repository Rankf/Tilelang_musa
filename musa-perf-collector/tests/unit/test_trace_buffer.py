"""Trace buffer state transitions must fail closed."""

from __future__ import annotations

import unittest

from musa_perf_collector.collectors.runtime.trace_buffer import TraceBufferSlot, TraceBufferState


class TraceBufferSlotTest(unittest.TestCase):
    def test_single_window_lifecycle(self):
        slot = TraceBufferSlot(slot_id=0)
        slot.begin_write()
        slot.transition(TraceBufferState.DEVICE_READY)
        slot.transition(TraceBufferState.HOST_READY)
        slot.transition(TraceBufferState.FREE)
        self.assertEqual(slot.generation_id, 1)
        self.assertEqual(slot.state, TraceBufferState.FREE)

    def test_reuse_before_readback_is_rejected(self):
        slot = TraceBufferSlot(slot_id=0)
        slot.begin_write()
        with self.assertRaises(RuntimeError):
            slot.begin_write()

    def test_async_lifecycle_requires_copying_state(self):
        slot = TraceBufferSlot(slot_id=1)
        slot.begin_write()
        slot.transition(TraceBufferState.DEVICE_READY)
        slot.transition(TraceBufferState.D2H_COPYING)
        slot.transition(TraceBufferState.HOST_READY)
        slot.transition(TraceBufferState.FREE)
        self.assertEqual(slot.state, TraceBufferState.FREE)
