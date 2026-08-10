"""Read-only runtime capability probe for the current MUSA environment."""

from __future__ import annotations

import json
import ctypes
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _command_version(command: str) -> dict[str, Any]:
    path = shutil.which(command)
    if path is None:
        return {"status": "unavailable", "command": command}
    completed = subprocess.run([command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"status": "available", "command": command, "path": path, "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def probe(device: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"profile": "runtime_measurement_v1", "device": device, "collectors": {}, "diagnostics": []}
    try:
        import torch

        available = bool(hasattr(torch, "musa") and torch.musa.is_available())
        result["torch_version"] = torch.__version__
        result["musa_available"] = available
        result["musa_device_count"] = int(torch.musa.device_count()) if available else 0
        if available:
            properties = torch.musa.get_device_properties(device)
            result["device_properties"] = {key: getattr(properties, key, None) for key in ("name", "major", "minor", "total_memory", "multi_processor_count")}
            start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
            start.record(); end.record(); end.synchronize()
            result["collectors"]["musa_event"] = {"status": "available", "elapsed_ms": start.elapsed_time(end)}
            result["collectors"]["stream"] = {"status": "available", "stream_type": type(torch.musa.current_stream()).__name__}
            activities = [name for name in dir(torch.profiler.ProfilerActivity) if name.isupper()]
            result["collectors"]["torch_activity"] = {"status": "available" if "MUSA" in activities else "unavailable", "activities": activities}
        else:
            result["collectors"]["musa_event"] = {"status": "unavailable"}
            result["collectors"]["stream"] = {"status": "unavailable"}
            result["collectors"]["torch_activity"] = {"status": "unavailable"}
    except Exception as error:
        result["diagnostics"].append({"code": "TORCH_MUSA_PROBE_FAILED", "message": str(error)})
    for command, key in (("mthreads-gmi", "telemetry"), ("MUPTI_version", "mupti_version"), ("mcc_version", "mcc_version")):
        result["collectors"][key] = _command_version(command)
    libraries = sorted({str(path.resolve()) for path in Path("/usr/local/musa/lib").glob("libmupti.so*")})
    required_counters = {"muptiDeviceGetNumMetrics", "muptiDeviceGetNumEventDomains", "muptiEventGroupCreate", "muptiEventGroupReadEvent"}
    required_pc = {"muptiPCSamplingEnable", "muptiPCSamplingDisable", "muptiPCSamplingGetData", "muptiPCSamplingGetConfigurationAttribute"}
    exported: set[str] = set()
    load_error = None
    if libraries:
        try:
            library = ctypes.CDLL(libraries[0])
            exported = {symbol for symbol in required_counters | required_pc if hasattr(library, symbol)}
        except OSError as error:
            load_error = str(error)
    counter_missing = sorted(required_counters - exported)
    pc_missing = sorted(required_pc - exported)
    result["collectors"]["mupti"] = {"status": "available" if libraries and load_error is None else "unavailable",
                                      "libraries": libraries, "load_error": load_error,
                                      "exported_required_symbols": sorted(exported),
                                      "raw_counter_adapter": {"status": "available" if not counter_missing else "unavailable",
                                                              "missing_symbols": counter_missing},
                                      "pc_sampling": {"status": "available" if not pc_missing else "unavailable",
                                                      "missing_symbols": pc_missing}}
    return result


def probe_json(device: int = 0) -> str:
    return json.dumps(probe(device), sort_keys=True, default=str)
