from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


_RUNTIME_LOG_LOCK = Lock()


def runtime_log(event: str, **fields: Any) -> None:
    path = Path(
        os.environ.get("ARC_V9_TRACE_PATH")
        or os.environ.get("ARC_V8_TRACE_PATH")
        or "/tmp/arc_v9_agent_trace.jsonl"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _RUNTIME_LOG_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    except Exception:
        pass
