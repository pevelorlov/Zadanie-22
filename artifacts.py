"""Локальные файловые артефакты, привязанные к диалогу и сообщениям."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


MAX_ARTIFACTS_PER_RESPONSE = 20
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_EXPLICIT_FENCE = re.compile(
    r"^```artifact\s+path=(?P<quote>[\"']?)(?P<path>[^\s\"']+)(?P=quote)[ \t]*\r?\n"
    r"(?P<content>.*?)\r?\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_STANDARD_FENCE = re.compile(
    r"^```(?P<language>[A-Za-z0-9_+.-]*)[ \t]*\r?\n(?P<content>.*?)\r?\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


class ArtifactError(ValueError):
    """Артефакт нельзя безопасно сохранить или подтвердить."""


def ensure_artifacts(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    raw = conversation.get("artifacts")
    artifacts = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    conversation["artifacts"] = artifacts
    return artifacts


def normalize_artifact_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ArtifactError("Путь артефакта должен быть строкой.")
    text = value.strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or len(text) > 240
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ArtifactError(f"Небезопасный относительный путь артефакта: {text or 'пустой путь'}.")
    return path.as_posix()


def normalize_required_artifacts(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ArtifactError("required_artifacts должен быть массивом относительных путей.")
    return list(dict.fromkeys(normalize_artifact_path(item) for item in value))[:MAX_ARTIFACTS_PER_RESPONSE]


def active_artifacts(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    return [deepcopy(item) for item in ensure_artifacts(conversation) if item.get("active") is True]


def extract_artifact_payloads(content: str, required_paths: list[str] | None = None) -> list[dict[str, str]]:
    """Извлекает явные artifact-блоки и совместимые именованные Markdown-блоки."""
    if not isinstance(content, str):
        return []
    payloads: list[dict[str, str]] = []
    occupied_spans: list[tuple[int, int]] = []
    for match in _EXPLICIT_FENCE.finditer(content):
        payloads.append({"path": normalize_artifact_path(match.group("path")), "content": match.group("content")})
        occupied_spans.append(match.span())

    expected = normalize_required_artifacts(required_paths or [])
    found_paths = {item["path"].casefold() for item in payloads}
    for path in expected:
        if path.casefold() in found_paths:
            continue
        candidates = []
        for match in _STANDARD_FENCE.finditer(content):
            if any(start <= match.start() < end for start, end in occupied_spans):
                continue
            prefix = content[max(0, match.start() - 240):match.start()]
            if re.search(rf"(?<![\w./-]){re.escape(path)}(?![\w./-])", prefix, re.IGNORECASE):
                candidates.append(match)
        if len(candidates) == 1:
            payloads.append({"path": path, "content": candidates[0].group("content")})
            found_paths.add(path.casefold())

    unique: dict[str, dict[str, str]] = {}
    for item in payloads:
        unique[item["path"].casefold()] = item
    if len(unique) > MAX_ARTIFACTS_PER_RESPONSE:
        raise ArtifactError(f"За один ответ разрешено не более {MAX_ARTIFACTS_PER_RESPONSE} артефактов.")
    return list(unique.values())


def save_response_artifacts(
    data_dir: Path,
    conversation: dict[str, Any],
    content: str,
    *,
    source_message_id: str,
    stage_run_id: str,
) -> list[dict[str, Any]]:
    required = conversation.get("task_state", {}).get("required_artifacts", [])
    payloads = extract_artifact_payloads(content, required)
    records = ensure_artifacts(conversation)
    created: list[dict[str, Any]] = []
    for payload in payloads:
        body = payload["content"].encode("utf-8")
        if len(body) > MAX_ARTIFACT_BYTES:
            raise ArtifactError(f"Артефакт {payload['path']} превышает лимит 2 МБ.")
        artifact_id = uuid.uuid4().hex
        logical_path = payload["path"]
        versions = [
            int(item.get("version", 0)) for item in records
            if str(item.get("path", "")).casefold() == logical_path.casefold()
        ]
        for item in records:
            if str(item.get("path", "")).casefold() == logical_path.casefold() and item.get("active") is True:
                item["active"] = False
                item["superseded_by"] = artifact_id
        filename = PurePosixPath(logical_path).name
        relative = Path("artifacts") / str(conversation["id"]) / artifact_id / filename
        target = data_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(body)
        os.replace(temporary, target)
        mime_type = mimetypes.guess_type(filename)[0] or "text/plain"
        record = {
            "id": artifact_id,
            "path": logical_path,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "version": max(versions, default=0) + 1,
            "active": True,
            "created_at": _now_iso(),
            "source_message_id": source_message_id,
            "stage_run_id": stage_run_id,
            "storage_path": relative.as_posix(),
        }
        records.append(record)
        created.append(deepcopy(record))
    return created


def missing_required_artifacts(conversation: dict[str, Any], data_dir: Path) -> list[str]:
    required = normalize_required_artifacts(conversation.get("task_state", {}).get("required_artifacts", []))
    current = {str(item.get("path", "")).casefold(): item for item in active_artifacts(conversation)}
    missing = []
    for logical_path in required:
        record = current.get(logical_path.casefold())
        if not record or not artifact_file_path(data_dir, record).is_file():
            missing.append(logical_path)
    return missing


def verify_required_artifacts(
    conversation: dict[str, Any], data_dir: Path, *, validation_stage_run_id: str
) -> list[dict[str, Any]]:
    missing = missing_required_artifacts(conversation, data_dir)
    if missing:
        raise ArtifactError("Отсутствуют обязательные артефакты: " + ", ".join(missing) + ".")
    required = {
        item.casefold() for item in normalize_required_artifacts(
            conversation.get("task_state", {}).get("required_artifacts", [])
        )
    }
    verified = []
    for record in ensure_artifacts(conversation):
        if record.get("active") is not True or str(record.get("path", "")).casefold() not in required:
            continue
        path = artifact_file_path(data_dir, record)
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest != record.get("sha256") or len(body) != record.get("size_bytes"):
            raise ArtifactError(f"Артефакт {record.get('path')} изменён вне системы после регистрации.")
        record["validated_at"] = _now_iso()
        record["validated_stage_run_id"] = validation_stage_run_id
        verified.append(deepcopy(record))
    return verified


def required_artifacts_are_validated(conversation: dict[str, Any], data_dir: Path) -> bool:
    state = conversation.get("task_state", {})
    required = normalize_required_artifacts(state.get("required_artifacts", []))
    if not required:
        return True
    if missing_required_artifacts(conversation, data_dir):
        return False
    run_id = state.get("stage_run_id")
    current = {str(item.get("path", "")).casefold(): item for item in active_artifacts(conversation)}
    return all(current[path.casefold()].get("validated_stage_run_id") == run_id for path in required)


def artifact_file_path(data_dir: Path, record: dict[str, Any]) -> Path:
    root = (data_dir / "artifacts").resolve()
    candidate = (data_dir / str(record.get("storage_path", ""))).resolve()
    if not candidate.is_relative_to(root):
        raise ArtifactError("Повреждён путь хранения артефакта.")
    return candidate


def copy_artifact_files(data_dir: Path, source: dict[str, Any], target: dict[str, Any]) -> None:
    source_id = str(source.get("id", ""))
    target_id = str(target.get("id", ""))
    for record in ensure_artifacts(target):
        old_path = artifact_file_path(data_dir, record)
        relative = Path("artifacts") / target_id / str(record["id"]) / str(record["filename"])
        new_path = data_dir / relative
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
        record["storage_path"] = relative.as_posix()
    if source_id == target_id:
        raise ArtifactError("Нельзя копировать артефакты диалога в тот же диалог.")


def delete_artifact_files(data_dir: Path, conversation_id: str) -> None:
    target = (data_dir / "artifacts" / conversation_id).resolve()
    root = (data_dir / "artifacts").resolve()
    if target.parent != root:
        raise ArtifactError("Некорректный идентификатор каталога артефактов.")
    if target.exists():
        shutil.rmtree(target)


def public_artifact(record: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in record.items() if key != "storage_path"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
