"""Иерархическая память проекта и пользователя (День 11)."""

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from typing import Any

from agent import normalize_token_usage
from storage import JsonStorage, now_iso


MEMORY_KINDS = {
    "goal", "context", "preference", "decision", "requirement", "constraint",
    "resource", "environment", "definition", "open_question", "result", "risk",
}
CONFIDENCE_VALUES = {"low", "medium", "high", "confirmed"}
MAX_MEMORY_ITEMS = 30
MAX_MEMORY_CHARS = 12_000
MAX_CANDIDATES = 12
PROFILE_TEXT_LIMIT = 2_000
SECRET_RE = re.compile(
    r"(?i)(api[_ -]?key|password|парол|секрет|token)\s*[:=]\s*\S+|sk-[A-Za-z0-9_-]{12,}"
)


def clean_memory_key(value: Any) -> str:
    key = re.sub(r"[^a-z0-9._-]+", "_", str(value).strip().lower()).strip("._-")
    if not key or len(key) > 120:
        raise ValueError("Ключ памяти должен содержать от 1 до 120 символов.")
    return key


def clean_memory_value(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Значение памяти не должно быть пустым.")
    text = value.strip()
    if len(text) > 2_000:
        raise ValueError("Значение памяти длиннее 2 000 символов.")
    if SECRET_RE.search(text):
        raise ValueError("Память не должна содержать секреты, ключи или пароли.")
    return text


def clean_kind(value: Any) -> str:
    kind = str(value or "context").strip().lower()
    if kind not in MEMORY_KINDS:
        raise ValueError("Выбран неподдерживаемый тип памяти.")
    return kind


def clean_profile_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name or len(name) > 80:
        raise ValueError("Имя профиля должно содержать от 1 до 80 символов.")
    return name


def normalize_preferences(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    defaults = {
        "language": "ru", "detail_level": "medium", "tone": "neutral",
        "response_structure": "result_then_explanation",
        "technical_level": "intermediate", "preferred_formats": [], "avoid_formats": [],
    }
    result = {}
    for key, default in defaults.items():
        raw = source.get(key, default)
        if isinstance(default, list):
            if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
                raise ValueError(f"Поле профиля {key} должно быть списком строк.")
            result[key] = [item.strip()[:80] for item in raw if item.strip()][:12]
        else:
            text = str(raw).strip()
            if not text or len(text) > 80:
                raise ValueError(f"Поле профиля {key} заполнено некорректно.")
            result[key] = text
    return result


def manual_entry(scope: str, kind: Any, key: Any, value: Any) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "id": uuid.uuid4().hex, "scope": scope, "kind": clean_kind(kind),
        "key": clean_memory_key(key), "value": clean_memory_value(value),
        "source": "manual", "locked": True, "confidence": "confirmed",
        "review_status": "confirmed", "created_at": timestamp, "updated_at": timestamp,
        "source_project_id": None, "source_conversation_id": None,
        "source_exchange_id": None, "status": "active",
    }


def normalize_entry(entry: Any, scope: str, *, automatic: bool = False) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Запись памяти должна быть JSON-объектом.")
    source = "automatic" if automatic else str(entry.get("source", "manual"))
    locked = False if automatic else bool(entry.get("locked", source == "manual"))
    timestamp = now_iso()
    supplied_confidence = str(entry.get("confidence", "confirmed"))
    if not automatic and supplied_confidence not in CONFIDENCE_VALUES:
        raise ValueError("Некорректная уверенность записи памяти.")
    confidence = "confirmed" if automatic else supplied_confidence
    status = str(entry.get("status", "active"))
    if status not in {"active", "inactive"}:
        raise ValueError("Статус памяти должен быть active или inactive.")
    return {
        "id": str(entry.get("id") or uuid.uuid4().hex), "scope": scope,
        "kind": clean_kind(entry.get("kind")), "key": clean_memory_key(entry.get("key")),
        "value": clean_memory_value(entry.get("value")), "source": source,
        "locked": locked, "confidence": confidence,
        "review_status": "confirmed" if automatic else str(entry.get("review_status", "confirmed")),
        "created_at": str(entry.get("created_at") or timestamp), "updated_at": timestamp,
        "source_project_id": entry.get("source_project_id"),
        "source_conversation_id": entry.get("source_conversation_id"),
        "source_exchange_id": entry.get("source_exchange_id"), "status": status,
    }


class MemoryManager:
    def __init__(self, storage: JsonStorage, policy_path) -> None:
        self.storage = storage
        self.policy_path = policy_path

    def policy(self) -> dict[str, Any]:
        return self.storage.read_external_json(self.policy_path)

    def create_project(self, name: Any, description: Any = "", owner_profile_id: str | None = None) -> dict[str, Any]:
        name = str(name).strip()
        description = str(description or "").strip()
        if not name or len(name) > 120:
            raise ValueError("Название проекта должно содержать от 1 до 120 символов.")
        if len(description) > 2_000:
            raise ValueError("Описание проекта длиннее 2 000 символов.")
        timestamp = now_iso()
        project = {
            "schema_version": 1, "id": uuid.uuid4().hex, "name": name,
            "description": description, "created_at": timestamp, "updated_at": timestamp,
            "settings": {"automatic_extraction": True}, "manual_memory": [],
            "learned_memory": [], "extraction_revisions": [],
            "owner_profile_id": owner_profile_id or self.storage.active_profile_id(),
            "participant_profile_ids": [],
        }
        return self.storage.save_project(project)

    def update_project(self, project_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project = self.storage.get_project(project_id)
        if "name" in values:
            name = str(values["name"]).strip()
            if not name or len(name) > 120:
                raise ValueError("Название проекта должно содержать от 1 до 120 символов.")
            project["name"] = name
        if "description" in values:
            description = str(values["description"] or "").strip()
            if len(description) > 2_000:
                raise ValueError("Описание проекта длиннее 2 000 символов.")
            project["description"] = description
        if "automatic_extraction" in values:
            if not isinstance(values["automatic_extraction"], bool):
                raise ValueError("Переключатель автоизвлечения должен быть логическим.")
            project.setdefault("settings", {})["automatic_extraction"] = values["automatic_extraction"]
        if "participant_profile_ids" in values:
            participants = values["participant_profile_ids"]
            if not isinstance(participants, list):
                raise ValueError("Участники проекта должны быть переданы списком.")
            known = {item["id"] for item in self.storage.list_profiles()}
            owner = project.get("owner_profile_id")
            cleaned = list(dict.fromkeys(str(item) for item in participants if str(item) != owner))
            if any(item not in known for item in cleaned):
                raise ValueError("Один из выбранных профилей не найден.")
            project["participant_profile_ids"] = cleaned
        return self.storage.save_project(project)

    def profiles_state(self) -> dict[str, Any]:
        return self.storage.load_profiles_document()

    def create_profile(self, values: dict[str, Any]) -> dict[str, Any]:
        document = self.profiles_state()
        name = clean_profile_name(values.get("name"))
        if any(item.get("name", "").casefold() == name.casefold() for item in document["profiles"]):
            raise ValueError("Профиль с таким именем уже существует.")
        timestamp = now_iso()
        custom = str(values.get("custom_instructions") or "").strip()
        if len(custom) > PROFILE_TEXT_LIMIT:
            raise ValueError("Дополнительные инструкции длиннее 2 000 символов.")
        profile = {
            "id": uuid.uuid4().hex, "name": name, "is_default": False,
            "preferences": normalize_preferences(values.get("preferences")),
            "custom_instructions": custom, "created_at": timestamp, "updated_at": timestamp,
            "settings": {"automatic_extraction": True}, "manual_memory": [],
            "learned_memory": [], "extraction_revisions": [],
        }
        document["profiles"].append(profile)
        self.storage.save_profiles_document(document)
        return deepcopy(profile)

    def update_profile(self, profile_id: str, values: dict[str, Any]) -> dict[str, Any]:
        document = self.profiles_state()
        profile = next((item for item in document["profiles"] if item.get("id") == profile_id), None)
        if profile is None:
            raise FileNotFoundError(profile_id)
        if "name" in values:
            name = clean_profile_name(values["name"])
            if any(item.get("id") != profile_id and item.get("name", "").casefold() == name.casefold() for item in document["profiles"]):
                raise ValueError("Профиль с таким именем уже существует.")
            profile["name"] = name
        if "preferences" in values:
            profile["preferences"] = normalize_preferences(values["preferences"])
        if "custom_instructions" in values:
            custom = str(values["custom_instructions"] or "").strip()
            if len(custom) > PROFILE_TEXT_LIMIT:
                raise ValueError("Дополнительные инструкции длиннее 2 000 символов.")
            profile["custom_instructions"] = custom
        profile["updated_at"] = now_iso()
        self.storage.save_profiles_document(document)
        return deepcopy(profile)

    def set_active_profile(self, profile_id: str) -> dict[str, Any]:
        return self.storage.set_active_profile(profile_id)

    def user_document(self, profile_id: str | None = None) -> dict[str, Any]:
        return self.storage.get_profile(profile_id or self.storage.active_profile_id())

    def set_user_extraction(self, enabled: Any, profile_id: str | None = None) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise ValueError("Переключатель автоизвлечения должен быть логическим.")
        profile_id = profile_id or self.storage.active_profile_id()
        document = self.user_document(profile_id)
        document.setdefault("settings", {})["automatic_extraction"] = enabled
        self._save_profile(document)
        return document

    def list_entries(self, scope: str, project_id: str | None = None, profile_id: str | None = None) -> dict[str, Any]:
        if scope == "project":
            document = self.storage.get_project(str(project_id))
        else:
            document = self.user_document(profile_id)
        return {
            "manual": deepcopy(document.get("manual_memory", [])),
            "learned": deepcopy(document.get("learned_memory", [])),
            "settings": deepcopy(document.get("settings", {})),
        }

    def add_manual(self, scope: str, values: dict[str, Any], project_id: str | None = None, profile_id: str | None = None) -> dict[str, Any]:
        document = self.storage.get_project(str(project_id)) if scope == "project" else self.user_document(profile_id)
        entry = manual_entry(scope, values.get("kind"), values.get("key"), values.get("value"))
        if any(item.get("key") == entry["key"] for group in ("manual_memory", "learned_memory") for item in document.get(group, [])):
            raise ValueError("Запись с таким ключом уже существует.")
        document.setdefault("manual_memory", []).append(entry)
        self._save(scope, document)
        return entry

    def update_entry(self, scope: str, memory_id: str, values: dict[str, Any], project_id: str | None = None, profile_id: str | None = None) -> dict[str, Any]:
        document = self.storage.get_project(str(project_id)) if scope == "project" else self.user_document(profile_id)
        group, entry = self._find(document, memory_id)
        if values.get("pin") is True and group == "learned_memory":
            document[group].remove(entry)
            entry.update({"source": "manual", "locked": True, "confidence": "confirmed", "review_status": "confirmed", "updated_at": now_iso()})
            document.setdefault("manual_memory", []).append(entry)
        else:
            entry["kind"] = clean_kind(values.get("kind", entry["kind"]))
            entry["key"] = clean_memory_key(values.get("key", entry["key"]))
            entry["value"] = clean_memory_value(values.get("value", entry["value"]))
            if "status" in values:
                if values["status"] not in {"active", "inactive"}:
                    raise ValueError("Статус памяти должен быть active или inactive.")
                entry["status"] = values["status"]
            entry["updated_at"] = now_iso()
        self._save(scope, document)
        return deepcopy(entry)

    def delete_entry(self, scope: str, memory_id: str, project_id: str | None = None, profile_id: str | None = None) -> None:
        document = self.storage.get_project(str(project_id)) if scope == "project" else self.user_document(profile_id)
        group, entry = self._find(document, memory_id)
        document[group].remove(entry)
        self._save(scope, document)

    def context_snapshot(self, conversation: dict[str, Any], context_mode: str, profile_id: str | None = None) -> dict[str, Any]:
        policy = self.policy()
        profile_id = profile_id or self.storage.active_profile_id()
        user = self.user_document(profile_id)
        project_id = conversation.get("project_id")
        project = self.storage.get_project(project_id) if project_id else None
        project_items = self._select(project or {})
        user_items = self._select(user)
        return {
            "project_id": project_id, "policy_rules": deepcopy(policy.get("rules", [])),
            "project_memories": project_items, "user_memories": user_items,
            "profile_snapshot": {
                "profile_id": profile_id, "profile_name": user.get("name", "Пользователь"),
                "preferences": deepcopy(user.get("preferences", {})),
                "custom_instructions": user.get("custom_instructions", ""),
            },
            "context_mode": context_mode, "created_at": now_iso(),
        }

    def route_candidates(
        self,
        result: Any,
        conversation: dict[str, Any],
        exchange_id: str,
        force_scopes: set[str] | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        import json
        project_id = conversation.get("project_id")
        project = self.storage.get_project(project_id) if project_id else None
        profile_id = profile_id or self.storage.active_profile_id()
        user = self.user_document(profile_id)
        revision = {
            "id": uuid.uuid4().hex, "created_at": now_iso(), "status": "completed",
            "conversation_id": conversation["id"], "exchange_id": exchange_id,
            "technical": {"usage": normalize_token_usage(result.technical.get("usage"))},
            "saved_count": 0, "rejected_count": 0,
        }
        try:
            parsed = json.loads(result.content)
            candidates = parsed.get("memories") if isinstance(parsed, dict) else None
            if not isinstance(candidates, list):
                raise ValueError("Модель не вернула массив memories.")
            enabled = {
                "project": bool(project and project.get("settings", {}).get("automatic_extraction", True)),
                "user": bool(user.get("settings", {}).get("automatic_extraction", False)),
            }
            if force_scopes is not None:
                enabled = {scope: scope in force_scopes for scope in enabled}
            for raw in candidates[:MAX_CANDIDATES]:
                scope = raw.get("scope") if isinstance(raw, dict) else None
                if scope not in enabled or not enabled[scope]:
                    continue
                try:
                    entry = normalize_entry(raw, scope, automatic=True)
                except ValueError:
                    revision["rejected_count"] += 1
                    continue
                entry.update({
                    "source_project_id": project_id, "source_conversation_id": conversation["id"],
                    "source_exchange_id": exchange_id,
                })
                document = project if scope == "project" else user
                if document is None or self._merge_automatic(document, entry) is False:
                    continue
                revision["saved_count"] += 1
            if project and enabled["project"]:
                project.setdefault("extraction_revisions", []).append(deepcopy(revision))
                self.storage.save_project(project)
            if enabled["user"]:
                user.setdefault("extraction_revisions", []).append(deepcopy(revision))
                self._save_profile(user)
        except Exception as error:
            revision.update({"status": "failed", "error": type(error).__name__})
            if project:
                project.setdefault("extraction_revisions", []).append(deepcopy(revision))
                self.storage.save_project(project)
            if user.get("settings", {}).get("automatic_extraction", False):
                user.setdefault("extraction_revisions", []).append(deepcopy(revision))
                self._save_profile(user)
        return revision

    @staticmethod
    def _merge_automatic(document: dict[str, Any], entry: dict[str, Any]) -> bool:
        manual = document.setdefault("manual_memory", [])
        learned = document.setdefault("learned_memory", [])
        if any(item.get("key") == entry["key"] and item.get("locked") for item in manual):
            return False
        exact = next((item for item in learned if item.get("key") == entry["key"] and item.get("value") == entry["value"]), None)
        if exact:
            return False
        previous = next((item for item in learned if item.get("key") == entry["key"]), None)
        if previous:
            origins = previous.setdefault("origin_history", [])
            origins.append({key: previous.get(key) for key in ("value", "source_project_id", "source_conversation_id", "source_exchange_id", "updated_at")})
            previous.update({key: value for key, value in entry.items() if key not in {"id", "created_at"}})
        else:
            learned.append(entry)
        return True

    @staticmethod
    def _select(document: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for group, priority in (("manual_memory", 0), ("learned_memory", 1)):
            for item in document.get(group, []):
                if isinstance(item, dict) and item.get("status", "active") == "active":
                    candidates.append((priority, str(item.get("created_at", "")), str(item.get("key", "")), item))
        selected, chars = [], 0
        for _, _, _, item in sorted(candidates, key=lambda row: row[:3]):
            compact = {key: deepcopy(item.get(key)) for key in ("id", "scope", "kind", "key", "value", "source", "locked", "review_status")}
            size = len(str(compact.get("key", ""))) + len(str(compact.get("value", "")))
            if len(selected) >= MAX_MEMORY_ITEMS or chars + size > MAX_MEMORY_CHARS:
                break
            selected.append(compact); chars += size
        return selected

    @staticmethod
    def _find(document: dict[str, Any], memory_id: str):
        for group in ("manual_memory", "learned_memory"):
            for entry in document.get(group, []):
                if entry.get("id") == memory_id:
                    return group, entry
        raise KeyError(memory_id)

    def _save(self, scope: str, document: dict[str, Any]) -> None:
        if scope == "project":
            self.storage.save_project(document)
        else:
            self._save_profile(document)

    def _save_profile(self, profile: dict[str, Any]) -> None:
        document = self.profiles_state()
        for index, item in enumerate(document["profiles"]):
            if item.get("id") == profile.get("id"):
                document["profiles"][index] = deepcopy(profile)
                self.storage.save_profiles_document(document)
                return
        raise FileNotFoundError(str(profile.get("id", "")))
