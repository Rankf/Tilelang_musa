"""Environment-driven one-time installation for real TileLang applications."""

from __future__ import annotations

import os
from pathlib import Path
import threading

from .config import load_online_config
from .session import OnlineSession


_lock = threading.Lock()
_session: OnlineSession | None = None


def install_from_env() -> OnlineSession | None:
    global _session
    with _lock:
        if _session is not None:
            return _session
        path = os.environ.get("MUSA_PERF_COLLECTOR_ONLINE_CONFIG")
        if not path:
            return None
        config = load_online_config(Path(path))
        if not config.enabled:
            return None
        _session = OnlineSession(config).install()
        return _session


def current_session() -> OnlineSession | None:
    return _session


SESSION = install_from_env()

