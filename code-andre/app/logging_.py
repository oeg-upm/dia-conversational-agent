import json
import os
from datetime import datetime, timezone
from typing import Any, Dict
from .config import settings

def _today_path() -> str:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(settings.LOG_DIR, f"{day}.jsonl")

def log_event(event: Dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path = _today_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")