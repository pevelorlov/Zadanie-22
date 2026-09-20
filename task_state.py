"""Формализованное состояние задачи и контролируемые переходы её этапов."""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any

from artifacts import normalize_required_artifacts


TASK_STAGES = ("planning", "execution", "validation", "done")
TASK_ACTIVITIES = ("active", "paused")
TRANSITION_MODES = ("manual", "automatic")
TASK_EVENTS = (
    "approve_plan",
    "complete_execution",
    "pass_validation",
    "return_to_planning",
    "validation_failed",
    "pause",
    "resume",
)
TEXT_LIMITS = {
    "description": 2_000,
    "current_step": 1_000,
    "expected_action": 1_000,
    "plan": 4_000,
}
EVENT_LABELS = {
    "approve_plan": "Утвердить план и перейти к выполнению",
    "complete_execution": "Завершить выполнение и перейти к проверке",
    "pass_validation": "Подтвердить валидацию и завершить задачу",
    "return_to_planning": "Вернуть задачу в планирование",
    "validation_failed": "Вернуть задачу на выполнение",
    "pause": "Приостановить задачу",
    "resume": "Продолжить задачу",
    "edit": "Изменить данные задачи",
    "set_transition_mode": "Изменить управление переходами",
}
STAGE_DEFAULTS = {
    "planning": ("Подготовить и утвердить план", "Утвердить план"),
    "execution": ("Выполнить утверждённый план", "Завершить реализацию"),
    "validation": ("Проверить результат", "Подтвердить результат валидации"),
    "done": ("Задача завершена", "Дополнительных действий не требуется"),
}
STAGE_TRANSITIONS = {
    ("planning", "approve_plan"): "execution",
    ("execution", "complete_execution"): "validation",
    ("execution", "return_to_planning"): "planning",
    ("validation", "pass_validation"): "done",
    ("validation", "validation_failed"): "execution",
}


class TaskTransitionError(ValueError):
    """Событие известно, но недопустимо для текущего состояния задачи."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def initial_task_state(timestamp: str = "") -> dict[str, Any]:
    """Возвращает безопасное начальное состояние нового или старого диалога."""
    step, action = STAGE_DEFAULTS["planning"]
    return {
        "description": "",
        "stage": "planning",
        "current_step": step,
        "expected_action": action,
        "plan": "",
        "activity": "active",
        "transition_mode": "manual",
        "stage_run_id": uuid.uuid4().hex,
        "stage_started_at": timestamp,
        "active_handoff_ids": [],
        "required_artifacts": [],
        "updated_at": timestamp,
        "transition_history": [],
    }


def ensure_task_state(conversation: dict[str, Any]) -> dict[str, Any]:
    """Совместимо нормализует сохранённое состояние задачи schema v3."""
    raw = conversation.get("task_state")
    raw = raw if isinstance(raw, dict) else {}
    stage = raw.get("stage") if raw.get("stage") in TASK_STAGES else "planning"
    activity = raw.get("activity") if raw.get("activity") in TASK_ACTIVITIES else "active"
    transition_mode = raw.get("transition_mode") if raw.get("transition_mode") in TRANSITION_MODES else "manual"
    default_step, default_action = STAGE_DEFAULTS[stage]
    history = raw.get("transition_history")
    history = deepcopy([item for item in history if isinstance(item, dict)]) if isinstance(history, list) else []
    normalized = {
        "description": _clean_stored_text(raw.get("description"), TEXT_LIMITS["description"]),
        "stage": stage,
        "current_step": _clean_stored_text(raw.get("current_step"), TEXT_LIMITS["current_step"]) or default_step,
        "expected_action": _clean_stored_text(raw.get("expected_action"), TEXT_LIMITS["expected_action"]) or default_action,
        "plan": _clean_stored_text(raw.get("plan"), TEXT_LIMITS["plan"]),
        "activity": activity,
        "transition_mode": transition_mode,
        "stage_run_id": _stage_run_id(raw.get("stage_run_id"), conversation, stage),
        "stage_started_at": str(raw.get("stage_started_at") or raw.get("updated_at") or conversation.get("created_at") or ""),
        "active_handoff_ids": _handoff_ids(raw.get("active_handoff_ids")),
        "required_artifacts": normalize_required_artifacts(raw.get("required_artifacts", [])),
        "updated_at": str(raw.get("updated_at") or conversation.get("updated_at") or conversation.get("created_at") or ""),
        "transition_history": history,
    }
    if isinstance(raw, dict):
        raw.clear()
        raw.update(normalized)
        normalized = raw
    conversation["task_state"] = normalized
    return normalized


def allowed_task_events(state_or_conversation: dict[str, Any]) -> list[str]:
    """Возвращает события, которые сервер примет в текущем состоянии."""
    if "task_state" in state_or_conversation:
        state = ensure_task_state(state_or_conversation)
    else:
        holder = {"task_state": deepcopy(state_or_conversation)}
        state = ensure_task_state(holder)
    if state["activity"] == "paused":
        return ["resume"]
    events = {
        "planning": ["approve_plan"],
        "execution": ["complete_execution", "return_to_planning"],
        "validation": ["pass_validation", "validation_failed"],
        "done": [],
    }[state["stage"]]
    return [*events, "pause"]


def update_task_state(conversation: dict[str, Any], changes: Any, *, source: str = "ui") -> dict[str, Any]:
    """Редактирует содержимое задачи, но не её этап или активность."""
    if not isinstance(changes, dict):
        raise ValueError("Ожидался JSON-объект состояния задачи.")
    if "activity" in changes:
        raise ValueError(
            "Пауза изменяется только явными событиями жизненного цикла задачи."
        )
    state = ensure_task_state(conversation)
    before = _state_snapshot(state)
    changed_fields: list[str] = []
    if "transition_mode" in changes:
        mode = changes["transition_mode"]
        if mode not in TRANSITION_MODES:
            raise ValueError("Неизвестный режим управления переходами.")
        if state["transition_mode"] != mode:
            state["transition_mode"] = mode
            changed_fields.append("transition_mode")
    if "lifecycle_mode" in changes:
        raise ValueError("Поле lifecycle_mode больше не поддерживается. Используйте transition_mode.")
    if "stage" in changes:
        raise ValueError("Этап изменяется только разрешённым событием автомата Дня 15.")
    for key, limit in TEXT_LIMITS.items():
        if key in changes:
            value = _require_text(changes[key], limit, key)
            if state[key] != value:
                state[key] = value
                changed_fields.append(key)
    if changed_fields:
        state["updated_at"] = _now_iso()
        _append_history(
            state,
            event="edit",
            source=source,
            reason="Изменены поля: " + ", ".join(changed_fields),
            before=before,
        )
    conversation["task_state"] = state
    return state


def apply_task_event(
    conversation: dict[str, Any],
    event: Any,
    *,
    source: str = "ui",
    reason: Any = "",
) -> dict[str, Any]:
    """Применяет одно разрешённое событие или отклоняет переход без изменений."""
    if not isinstance(event, str) or event not in TASK_EVENTS:
        raise ValueError("Неизвестное событие жизненного цикла задачи.")
    clean_reason = _require_text(reason, 1_000, "reason") if reason is not None else ""
    state = ensure_task_state(conversation)
    before = _state_snapshot(state)

    if event == "pause":
        if state["activity"] == "paused":
            raise TaskTransitionError("Задача уже приостановлена.")
        state["activity"] = "paused"
    elif event == "resume":
        if state["activity"] != "paused":
            raise TaskTransitionError("Задача не находится на паузе.")
        state["activity"] = "active"
    else:
        if state["activity"] == "paused":
            raise TaskTransitionError("Задача приостановлена. Сначала продолжите её.")
        target = STAGE_TRANSITIONS.get((state["stage"], event))
        if target is None:
            raise TaskTransitionError(_invalid_transition_message(state["stage"], event))
        state["stage"] = target
        state["current_step"], state["expected_action"] = STAGE_DEFAULTS[target]
        state["stage_run_id"] = uuid.uuid4().hex
        state["stage_started_at"] = _now_iso()

    state["updated_at"] = _now_iso()
    _append_history(
        state,
        event=event,
        source=source,
        reason=clean_reason or EVENT_LABELS[event],
        before=before,
    )
    conversation["task_state"] = state
    return state


def task_state_report(state_or_conversation: dict[str, Any]) -> str:
    """Формирует детерминированный отчёт без обращения к LLM."""
    if "task_state" in state_or_conversation:
        state = ensure_task_state(state_or_conversation)
    else:
        holder = {"task_state": deepcopy(state_or_conversation)}
        state = ensure_task_state(holder)
    stage_labels = {
        "planning": "Планирование",
        "execution": "Выполнение",
        "validation": "Проверка",
        "done": "Завершено",
    }
    activity = "приостановлена" if state["activity"] == "paused" else "активна"
    events = allowed_task_events(state)
    next_actions = ", ".join(EVENT_LABELS[item] for item in events) if events else "нет"
    return (
        f"Этап: {stage_labels[state['stage']]} ({state['stage']})\n"
        f"Текущий шаг: {state['current_step']}\n"
        f"Ожидаемое действие: {state['expected_action']}\n"
        f"Активность: {activity}\n"
        f"Запуск этапа: {state['stage_run_id']}\n"
        f"Активных handoff: {len(state.get('active_handoff_ids', []))}\n"
        f"Обязательные артефакты: {', '.join(state.get('required_artifacts', [])) or 'нет'}\n"
        f"Переходы: {'автопилот' if state['transition_mode'] == 'automatic' else 'вручную по кнопкам'}\n"
        f"Разрешённые действия: {next_actions}"
    )


def task_is_paused(conversation: dict[str, Any]) -> bool:
    return ensure_task_state(conversation)["activity"] == "paused"


def _state_snapshot(state: dict[str, Any]) -> dict[str, str]:
    return {
        "stage": state["stage"],
        "activity": state["activity"],
        "transition_mode": state["transition_mode"],
        "current_step": state["current_step"],
        "expected_action": state["expected_action"],
        "stage_run_id": state["stage_run_id"],
    }


def _stage_run_id(value: Any, conversation: dict[str, Any], stage: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:100]
    conversation_id = str(conversation.get("id") or "legacy")
    return f"legacy:{conversation_id}:{stage}"[:100]


def _handoff_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item)[:100] for item in value if isinstance(item, str) and item))[:20]


def _append_history(
    state: dict[str, Any],
    *,
    event: str,
    source: str,
    reason: str,
    before: dict[str, str],
) -> None:
    state.setdefault("transition_history", []).append({
        "id": uuid.uuid4().hex,
        "created_at": state["updated_at"],
        "event": event,
        "event_label": EVENT_LABELS[event],
        "source": str(source or "unknown")[:100],
        "reason": reason,
        "from": before,
        "to": _state_snapshot(state),
    })


def _invalid_transition_message(stage: str, event: str) -> str:
    if event == "complete_execution" and stage != "execution":
        return "Нельзя завершить реализацию: задача не находится на этапе выполнения."
    if event == "pass_validation" and stage != "validation":
        return "Нельзя завершить задачу: сначала требуется этап валидации."
    if event == "approve_plan" and stage != "planning":
        return "План можно утвердить только на этапе планирования."
    return f"Событие «{EVENT_LABELS[event]}» недопустимо на этапе {stage}."


def _require_text(value: Any, limit: int, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Поле {field} должно быть строкой.")
    text = value.strip()
    if len(text) > limit:
        raise ValueError(f"Поле {field} не должно превышать {limit} символов.")
    return text


def _clean_stored_text(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""
