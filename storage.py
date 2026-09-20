"""Потокобезопасное JSON-хранилище диалогов и переносимых пакетов."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from task_state import ensure_task_state, initial_task_state


logger = logging.getLogger("deepseek_agent.storage")


ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JsonStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.conversations_dir = data_dir / "conversations"
        self.projects_dir = data_dir / "projects"
        self.memory_dir = data_dir / "memory"
        self.user_memory_path = self.memory_dir / "user_memory.json"
        self.profiles_path = data_dir / "profiles.json"
        self.presets_path = data_dir / "presets.json"
        self._lock = threading.RLock()
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.presets_path.exists():
            self._write_json(self.presets_path, {"schema_version": 1, "presets": []})
        if not self.user_memory_path.exists():
            self._write_json(self.user_memory_path, {
                "schema_version": 1, "settings": {"automatic_extraction": False},
                "manual_memory": [], "learned_memory": [], "extraction_revisions": [],
            })
        if not self.profiles_path.exists():
            legacy_memory = self._read_json(self.user_memory_path)
            timestamp = now_iso()
            profile_id = uuid.uuid4().hex
            self._write_json(self.profiles_path, {
                "schema_version": 2,
                "active_profile_id": profile_id,
                "profiles": [{
                    "id": profile_id,
                    "name": "Основной пользователь",
                    "is_default": True,
                    "preferences": {
                        "language": "ru",
                        "detail_level": "medium",
                        "tone": "neutral",
                        "response_structure": "result_then_explanation",
                        "technical_level": "intermediate",
                        "preferred_formats": [],
                        "avoid_formats": [],
                    },
                    "custom_instructions": "",
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "settings": deepcopy(legacy_memory.get("settings", {"automatic_extraction": False})),
                    "manual_memory": deepcopy(legacy_memory.get("manual_memory", [])),
                    "learned_memory": deepcopy(legacy_memory.get("learned_memory", [])),
                    "extraction_revisions": deepcopy(legacy_memory.get("extraction_revisions", [])),
                }],
            })
        else:
            self._migrate_profiles_document()

    def list_conversations(self, profile_id: str | None = None) -> list[dict[str, Any]]:
        accessible_projects = None
        if profile_id is not None:
            accessible_projects = {item["id"] for item in self.list_projects(profile_id)}
        with self._lock:
            result = []
            for path in self.conversations_dir.glob("*.json"):
                try:
                    item = self._read_json(path)
                    result.append(
                        {
                            "id": item["id"],
                            "title": item.get("title", "Новый диалог"),
                            "created_at": item.get("created_at", ""),
                            "updated_at": item.get("updated_at", ""),
                            "message_count": sum(
                                1 for message in item.get("messages", [])
                                if message.get("role") in {"user", "assistant"}
                            ),
                            "project_id": item.get("project_id"),
                            "owner_profile_id": item.get("owner_profile_id") or self.default_profile_id(),
                        }
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
            if profile_id is not None:
                result = [
                    item for item in result
                    if (
                        item.get("project_id") in accessible_projects
                        if item.get("project_id")
                        else item.get("owner_profile_id") == profile_id
                    )
                ]
            return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def create_conversation(
        self,
        title: str = "Новый диалог",
        project_id: str | None = None,
        owner_profile_id: str | None = None,
    ) -> dict[str, Any]:
        title = clean_title(title)
        owner_profile_id = owner_profile_id or self.active_profile_id()
        self.get_profile(owner_profile_id)
        if project_id is not None:
            self.get_project(project_id)
        timestamp = now_iso()
        conversation = {
            "schema_version": 3,
            "id": uuid.uuid4().hex,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "project_id": project_id,
            "owner_profile_id": owner_profile_id,
            "task_state": initial_task_state(timestamp),
            "task_handoffs": [],
            "artifacts": [],
            "messages": [],
            "context_management": {
                "mode": "full",
                "keep_recent_exchanges": 5,
                "summary_batch_exchanges": 5,
                "summary_model": "deepseek-v4-flash",
                "summary_reasoning_enabled": False,
                "active_summary_id": None,
                "sliding_window_exchanges": 5,
                "facts_window_exchanges": 5,
                "active_leaf_id": None,
                "pending_branch_from_id": None,
            },
            "summaries": [],
            "facts": [],
            "fact_revisions": [],
        }
        with self._lock:
            self._write_json(self._conversation_path(conversation["id"]), conversation)
        return deepcopy(conversation)

    def copy_conversation_to_project(self, conversation_id: str, project_id: str) -> dict[str, Any]:
        """Создаёт независимую копию диалога, не меняя оригинал и старый проект."""
        source = self.get_conversation(conversation_id)
        self.get_project(project_id)
        timestamp = now_iso()
        copied = deepcopy(source)
        copied["id"] = uuid.uuid4().hex
        copied["title"] = clean_title(f"{source.get('title', 'Новый диалог')} — копия")
        copied["project_id"] = project_id
        copied["owner_profile_id"] = self.active_profile_id()
        copied["created_at"] = timestamp
        copied["updated_at"] = timestamp
        copied["copied_from_conversation_id"] = source["id"]
        copied["copied_from_project_id"] = source.get("project_id")
        copied["copied_at"] = timestamp
        return self.save_conversation(copied)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            conversation = deepcopy(self._read_json(self._conversation_path(conversation_id)))
        conversation.setdefault("project_id", None)
        default_profile_id = self.default_profile_id()
        conversation.setdefault("owner_profile_id", default_profile_id)
        for message in conversation.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "user":
                message.setdefault("author_profile_id", default_profile_id)
        conversation.setdefault("memory_extraction_revisions", [])
        conversation["task_handoffs"] = conversation.get("task_handoffs") if isinstance(conversation.get("task_handoffs"), list) else []
        conversation["artifacts"] = conversation.get("artifacts") if isinstance(conversation.get("artifacts"), list) else []
        ensure_task_state(conversation)
        return conversation

    def save_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(conversation.get("id", ""))
        path = self._conversation_path(conversation_id)
        saved = deepcopy(conversation)
        saved["schema_version"] = 3
        saved.setdefault("owner_profile_id", self.default_profile_id())
        saved["task_handoffs"] = saved.get("task_handoffs") if isinstance(saved.get("task_handoffs"), list) else []
        saved["artifacts"] = saved.get("artifacts") if isinstance(saved.get("artifacts"), list) else []
        ensure_task_state(saved)
        saved["updated_at"] = now_iso()
        with self._lock:
            self._write_json(path, saved)
        return deepcopy(saved)

    def list_projects(self, profile_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            projects = []
            for path in self.projects_dir.glob("*.json"):
                try:
                    projects.append(self._normalize_project(self._read_json(path)))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        if profile_id is not None:
            projects = [item for item in projects if self.profile_can_access_project(item, profile_id)]
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return self._normalize_project(self._read_json(self._project_path(project_id)))

    def save_project(self, project: dict[str, Any]) -> dict[str, Any]:
        saved = deepcopy(project)
        saved["schema_version"] = 1
        saved.setdefault("owner_profile_id", self.default_profile_id())
        saved["participant_profile_ids"] = list(dict.fromkeys(saved.get("participant_profile_ids", [])))
        saved["updated_at"] = now_iso()
        with self._lock:
            self._write_json(self._project_path(str(saved.get("id", ""))), saved)
        return deepcopy(saved)

    def load_user_memory(self) -> dict[str, Any]:
        with self._lock:
            return self._confirmed_learned_memory(self._read_json(self.user_memory_path))

    def save_user_memory(self, document: dict[str, Any]) -> dict[str, Any]:
        saved = deepcopy(document)
        saved["schema_version"] = 1
        with self._lock:
            self._write_json(self.user_memory_path, saved)
        return deepcopy(saved)

    def load_profiles_document(self) -> dict[str, Any]:
        with self._lock:
            document = self._read_json(self.profiles_path)
        if not isinstance(document.get("profiles"), list) or not document["profiles"]:
            raise ValueError("Файл профилей повреждён.")
        return deepcopy(document)

    def save_profiles_document(self, document: dict[str, Any]) -> dict[str, Any]:
        saved = deepcopy(document)
        saved["schema_version"] = 2
        with self._lock:
            self._write_json(self.profiles_path, saved)
        return deepcopy(saved)

    def list_profiles(self) -> list[dict[str, Any]]:
        return deepcopy(self.load_profiles_document()["profiles"])

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        profile = next((item for item in self.list_profiles() if item.get("id") == profile_id), None)
        if profile is None:
            raise FileNotFoundError(profile_id)
        return self._confirmed_learned_memory(profile)

    def active_profile_id(self) -> str:
        document = self.load_profiles_document()
        profile_id = document.get("active_profile_id")
        if not any(item.get("id") == profile_id for item in document["profiles"]):
            profile_id = self.default_profile_id()
        return str(profile_id)

    def default_profile_id(self) -> str:
        profiles = self.load_profiles_document()["profiles"]
        profile = next((item for item in profiles if item.get("is_default")), profiles[0])
        return str(profile["id"])

    def set_active_profile(self, profile_id: str) -> dict[str, Any]:
        self.get_profile(profile_id)
        document = self.load_profiles_document()
        document["active_profile_id"] = profile_id
        self.save_profiles_document(document)
        return self.get_profile(profile_id)

    @staticmethod
    def profile_can_access_project(project: dict[str, Any], profile_id: str) -> bool:
        return profile_id == project.get("owner_profile_id") or profile_id in project.get("participant_profile_ids", [])

    def _normalize_project(self, project: dict[str, Any]) -> dict[str, Any]:
        normalized = self._confirmed_learned_memory(project)
        normalized.setdefault("owner_profile_id", self.default_profile_id())
        normalized.setdefault("participant_profile_ids", [])
        return normalized

    def _migrate_profiles_document(self) -> None:
        """Включает новый default автоизвлечения один раз, не трогая legacy-профиль."""
        document = self._read_json(self.profiles_path)
        if int(document.get("schema_version", 1)) >= 2:
            return
        for profile in document.get("profiles", []):
            if isinstance(profile, dict) and not profile.get("is_default"):
                profile.setdefault("settings", {})["automatic_extraction"] = True
        document["schema_version"] = 2
        self._write_json(self.profiles_path, document)

    @staticmethod
    def read_external_json(path: Path) -> dict[str, Any]:
        return JsonStorage._read_json(path)

    @staticmethod
    def _confirmed_learned_memory(document: dict[str, Any]) -> dict[str, Any]:
        """Treats existing and new automatic memories as confirmed but editable."""
        normalized = deepcopy(document)
        for entry in normalized.get("learned_memory", []):
            if isinstance(entry, dict) and entry.get("source") == "automatic":
                entry["confidence"] = "confirmed"
                entry["review_status"] = "confirmed"
        return normalized

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        conversation["title"] = clean_title(title)
        return self.save_conversation(conversation)

    def delete_conversation(self, conversation_id: str) -> None:
        path = self._conversation_path(conversation_id)
        with self._lock:
            path.unlink()

    def load_presets_document(self) -> dict[str, Any]:
        with self._lock:
            value = self._read_json(self.presets_path)
        if not isinstance(value.get("presets"), list):
            raise ValueError("Файл пресетов повреждён.")
        return deepcopy(value)

    def save_presets_document(self, document: dict[str, Any]) -> None:
        with self._lock:
            self._write_json(self.presets_path, document)

    def export_bundle(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        library = {
            preset.get("id"): preset for preset in self.load_presets_document()["presets"]
        }
        snapshots: dict[str, dict[str, Any]] = {}
        for message in conversation.get("messages", []):
            technical = message.get("technical", {})
            source = technical.get("configuration_source", {})
            preset_id = source.get("preset_id")
            if message.get("role") != "user" or not preset_id or preset_id in snapshots:
                continue
            snapshots[preset_id] = deepcopy(library.get(preset_id) or {
                "id": preset_id,
                "name": source.get("preset_name") or "Импортированный пресет",
                "description": "Восстановлен из снимка настроек диалога.",
                "created_at": message.get("created_at", now_iso()),
                "updated_at": message.get("created_at", now_iso()),
                "settings": technical.get("settings", {}),
            })
        bundle = {
            "format": "deepseek-agent-export",
            "schema_version": 1,
            "exported_at": now_iso(),
            "conversation": conversation,
            "presets": list(snapshots.values()),
        }
        project_id = conversation.get("project_id")
        if project_id:
            try:
                bundle["project"] = self.get_project(project_id)
            except FileNotFoundError:
                pass
        return bundle

    def import_bundle(self, bundle: Any, owner_profile_id: str | None = None) -> dict[str, Any]:
        if not isinstance(bundle, dict) or bundle.get("format") != "deepseek-agent-export":
            raise ValueError("Это не файл экспорта DeepSeek Agent.")
        source = bundle.get("conversation")
        if not isinstance(source, dict) or not isinstance(source.get("messages"), list):
            raise ValueError("В файле отсутствует корректный диалог.")

        imported = deepcopy(source)
        imported["id"] = uuid.uuid4().hex
        imported["title"] = clean_title(f"{source.get('title', 'Импортированный диалог')} — импорт")
        imported["created_at"] = now_iso()
        imported["updated_at"] = imported["created_at"]
        imported["schema_version"] = 3
        imported["project_id"] = None
        imported["owner_profile_id"] = owner_profile_id or self.active_profile_id()
        for message in imported.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "user":
                message.setdefault("author_profile_id", imported["owner_profile_id"])

        project = bundle.get("project")
        if isinstance(project, dict):
            project_copy = deepcopy(project)
            project_copy["id"] = uuid.uuid4().hex
            project_copy["name"] = clean_title(f"{project.get('name', 'Импортированный проект')} — импорт")
            project_copy["created_at"] = now_iso()
            project_copy["updated_at"] = project_copy["created_at"]
            project_copy["owner_profile_id"] = imported["owner_profile_id"]
            project_copy["participant_profile_ids"] = []
            self.save_project(project_copy)
            imported["project_id"] = project_copy["id"]

        from agent import AgentSettings
        for message in imported["messages"]:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "event"}:
                raise ValueError("В импортируемой истории найдено некорректное сообщение.")
            technical = message.get("technical", {})
            if message.get("role") in {"user", "assistant"}:
                AgentSettings.from_dict(technical.get("settings", {}))

        self.save_conversation(imported)

        raw_presets = bundle.get("presets", [])
        if isinstance(raw_presets, list):
            from presets import PresetManager
            PresetManager(self).merge_imported(raw_presets)
        return imported

    def _conversation_path(self, conversation_id: str) -> Path:
        if not ID_PATTERN.fullmatch(conversation_id):
            raise ValueError("Некорректный идентификатор диалога.")
        return self.conversations_dir / f"{conversation_id}.json"

    def _project_path(self, project_id: str) -> Path:
        if not ID_PATTERN.fullmatch(project_id):
            raise ValueError("Некорректный идентификатор проекта.")
        return self.projects_dir / f"{project_id}.json"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except Exception:
            logger.exception("Не удалось прочитать JSON | path=%s", path)
            raise
        if not isinstance(value, dict):
            raise ValueError("Ожидался JSON-объект.")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except Exception:
            logger.exception("Не удалось записать JSON | path=%s", path)
            raise
        finally:
            if temporary.exists():
                temporary.unlink()


def clean_title(value: Any) -> str:
    title = str(value).strip()
    if not title:
        return "Новый диалог"
    return title[:120]
