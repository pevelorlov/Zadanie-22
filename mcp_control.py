"""Атомарно сохраняемые общие выключатели возможностей приложения."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class PersistentFeatureControl:
    """Хранит постоянное разрешение на работу отдельной возможности."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def state(self) -> dict:
        with self._lock:
            if not self.path.exists():
                return {"enabled": True, "updated_at": None}
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"enabled": False, "updated_at": None, "error": "Файл состояния выключателя повреждён."}
            return {
                "enabled": value.get("enabled") is True,
                "updated_at": value.get("updated_at"),
            }

    def is_enabled(self) -> bool:
        return self.state()["enabled"]

    def set_enabled(self, enabled: bool) -> dict:
        with self._lock:
            state = {
                "enabled": bool(enabled),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return state


MCPControl = PersistentFeatureControl
