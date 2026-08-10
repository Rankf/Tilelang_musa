"""Bounded `mthreads-gmi` telemetry sampler."""

from __future__ import annotations

import subprocess
import time
import threading
from typing import Any

from ...core.evidence import RUNTIME_TELEMETRY


def snapshot(device: int = 0) -> dict[str, Any]:
    command = ["mthreads-gmi", "-q", "-i", str(device), "-d", "TEMPERATURE,CLOCK,POWER", "--json"]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"kind": "snapshot", "host_monotonic_s": time.monotonic(), "returncode": completed.returncode,
            "raw_json": completed.stdout.strip() if completed.returncode == 0 else None,
            "stderr": completed.stderr.strip(), "evidence_level": RUNTIME_TELEMETRY}


def periodic_window(device: int, interval_ms: int, duration_s: float) -> dict[str, Any]:
    """Capture a finite metrics window. The process is always terminated on return."""
    if interval_ms <= 0 or duration_s <= 0:
        raise ValueError("interval_ms and duration_s must be positive")
    command = ["mthreads-gmi", "--metrics-monitor=GpuId,Temp,Power,Freq", "-i", str(device), "-l", str(interval_ms)]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while time.monotonic() - started < duration_s:
            time.sleep(min(0.05, max(0.001, duration_s)))
    finally:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill(); stdout, stderr = process.communicate()
    rows = []
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit():
            try:
                rows.append({"gpu_id": int(fields[0]), "temperature_c": float(fields[1]), "power_w": float(fields[2]),
                             "frequency_mhz": float(fields[3]), "evidence_level": RUNTIME_TELEMETRY})
            except ValueError:
                pass
    return {"kind": "periodic_window", "interval_ms": interval_ms, "duration_s": time.monotonic() - started,
            "samples": rows, "stderr": stderr.strip(), "evidence_level": RUNTIME_TELEMETRY}


class TelemetrySampler:
    """A finite background sampler; it never outlives the collection session."""

    def __init__(self, device: int, interval_ms: int):
        self.device = device
        self.interval_ms = interval_ms
        self.process: subprocess.Popen[str] | None = None
        self.rows: list[dict[str, Any]] = []
        self.stderr = ""
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        command = ["mthreads-gmi", "--metrics-monitor=GpuId,Temp,Power,Freq", "-i", str(self.device), "-l", str(self.interval_ms)]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def consume() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 4 and fields[0].isdigit():
                    try:
                        self.rows.append({"host_monotonic_s": time.monotonic(), "gpu_id": int(fields[0]),
                                          "temperature_c": float(fields[1]), "power_w": float(fields[2]),
                                          "frequency_mhz": float(fields[3]), "evidence_level": RUNTIME_TELEMETRY})
                    except ValueError:
                        continue

        self._thread = threading.Thread(target=consume, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"status": "not_started", "samples": []}
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.stderr = self.process.stderr.read().strip() if self.process.stderr is not None else ""
        return {"status": "complete", "interval_ms": self.interval_ms, "samples": self.rows,
                "stderr": self.stderr, "evidence_level": RUNTIME_TELEMETRY}
