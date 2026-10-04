"""Отдельный локальный журнал сравнений ответов с RAG и без RAG."""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from storage import now_iso


class RagEvaluationStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._read()["items"])

    def add(self, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            document = self._read()
            item = {"id": uuid.uuid4().hex, "created_at": now_iso(), **deepcopy(value)}
            document["items"].append(item)
            self._write(document)
            return deepcopy(item)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "items": []}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("items"), list):
            raise ValueError("Журнал RAG-сравнений повреждён.")
        return {"schema_version": 1, "items": value["items"]}

    def _write(self, document: dict[str, Any]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
