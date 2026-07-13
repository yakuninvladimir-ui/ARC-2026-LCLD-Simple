from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def runtime_log(event: str, **fields: Any) -> None:
    path = Path(os.environ.get("ARC_V8_TRACE_PATH", "/tmp/arc_v8_agent_trace.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    except Exception:
        pass
