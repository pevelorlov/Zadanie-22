"""Отдельное хранилище наборов инвариантов задачи (День 14)."""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class InvariantStore:
    """CRUD наборов инвариантов, физически отделённых от JSON диалогов."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "invariants.json"
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write({"schema_version": 1, "sets": []})

    def list(self, scope: str | None = None, owner_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            document = self._read()
        items = document["sets"]
        if scope is not None:
            items = [item for item in items if item.get("scope") == scope]
        if owner_id is not None:
            items = [item for item in items if item.get("owner_id") == owner_id]
        return deepcopy(items)

    def get(self, set_id: str | None) -> dict[str, Any] | None:
        if not set_id:
            return None
        return next((item for item in self.list() if item["id"] == set_id), None)

    def create(self, payload: Any) -> dict[str, Any]:
        name, description, rules, scope, owner_id, enabled = self._validate_payload(payload)
        timestamp = _now_iso()
        item = {
            "id": uuid.uuid4().hex,
            "name": name,
            "description": description,
            "rules": rules,
            "scope": scope,
            "owner_id": owner_id,
            "enabled": enabled,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock:
            document = self._read()
            document["sets"].append(item)
            self._write(document)
        return deepcopy(item)

    def update(self, set_id: str, payload: Any) -> dict[str, Any]:
        name, description, rules, scope, owner_id, enabled = self._validate_payload(payload)
        with self._lock:
            document = self._read()
            item = next((entry for entry in document["sets"] if entry["id"] == set_id), None)
            if item is None:
                raise FileNotFoundError(set_id)
            if item.get("scope") != scope or item.get("owner_id") != owner_id:
                raise ValueError("Область и владелец существующего набора не изменяются.")
            item.update({"name": name, "description": description, "rules": rules, "enabled": enabled, "updated_at": _now_iso()})
            self._write(document)
        return deepcopy(item)

    def delete(self, set_id: str) -> None:
        with self._lock:
            document = self._read()
            remaining = [item for item in document["sets"] if item["id"] != set_id]
            if len(remaining) == len(document["sets"]):
                raise FileNotFoundError(set_id)
            document["sets"] = remaining
            self._write(document)

    def bundle(
        self,
        *,
        profile_id: str,
        project_id: str | None,
        conversation_id: str,
        system_rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        layers = {
            "system": [
                {"id": f"system:{item.get('id')}", "text": str(item.get("text", "")), "layer": "system"}
                for item in system_rules or [] if isinstance(item, dict) and item.get("text")
            ],
            "user": self._layer_rules("user", profile_id),
            "project": self._layer_rules("project", project_id) if project_id else [],
            "task": self._layer_rules("task", conversation_id),
        }
        return {"layers": layers, "rules": [rule for name in ("system", "user", "project", "task") for rule in layers[name]]}

    def _layer_rules(self, scope: str, owner_id: str) -> list[dict[str, str]]:
        result = []
        for item in self.list(scope, owner_id):
            if item.get("enabled", True) is False:
                continue
            for rule in item.get("rules", []):
                if isinstance(rule, dict) and rule.get("enabled", True) is not False and rule.get("text"):
                    result.append({
                        "id": f"{scope}:{item['id']}:{rule.get('id')}",
                        "text": str(rule["text"]),
                        "layer": scope,
                        "set_id": item["id"],
                        "set_name": item["name"],
                    })
        return result

    @staticmethod
    def _validate_payload(payload: Any) -> tuple[str, str, list[dict[str, Any]], str, str, bool]:
        if not isinstance(payload, dict):
            raise ValueError("Ожидался JSON-объект набора инвариантов.")
        name = str(payload.get("name") or "").strip()
        description = str(payload.get("description") or "").strip()
        if not name or len(name) > 120:
            raise ValueError("Название набора должно содержать от 1 до 120 символов.")
        if len(description) > 1_000:
            raise ValueError("Описание набора не должно превышать 1000 символов.")
        scope = str(payload.get("scope") or "")
        if scope not in {"user", "project", "task"}:
            raise ValueError("Область набора должна быть user, project или task.")
        owner_id = str(payload.get("owner_id") or "").strip()
        if not owner_id or len(owner_id) > 100:
            raise ValueError("У набора должен быть корректный владелец области.")
        set_enabled = payload.get("enabled", True) is not False
        raw_rules = payload.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("Добавьте хотя бы один инвариант.")
        if len(raw_rules) > 50:
            raise ValueError("В одном наборе допускается не более 50 инвариантов.")
        rules: list[dict[str, Any]] = []
        for raw in raw_rules:
            if isinstance(raw, str):
                text, rule_enabled, rule_id = raw.strip(), True, uuid.uuid4().hex
            elif isinstance(raw, dict):
                text = str(raw.get("text") or "").strip()
                rule_enabled = raw.get("enabled", True) is not False
                rule_id = str(raw.get("id") or uuid.uuid4().hex)
            else:
                raise ValueError("Каждый инвариант должен быть строкой или объектом.")
            if not text or len(text) > 1_000:
                raise ValueError("Текст инварианта должен содержать от 1 до 1000 символов.")
            rules.append({"id": rule_id[:100], "text": text, "enabled": rule_enabled})
        return name, description, rules, scope, owner_id, set_enabled

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Не удалось прочитать хранилище инвариантов.") from error
        raw_sets = value.get("sets") if isinstance(value, dict) else None
        sets = []
        for item in raw_sets if isinstance(raw_sets, list) else []:
            if isinstance(item, dict) and item.get("scope") in {"user", "project", "task"} and item.get("owner_id"):
                sets.append(item)
        return {"schema_version": 1, "sets": sets}

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
