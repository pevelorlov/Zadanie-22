"""Стратегии управления контекстом и дерево веток отдельного диалога."""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from typing import Any

from agent import Agent, normalize_token_usage
from storage import now_iso


logger = logging.getLogger("deepseek_agent.context")


FULL_MODE = "full"
SUMMARY_MODE = "summary"
SLIDING_MODE = "sliding"
FACTS_MODE = "facts"
LEGACY_BRANCHING_MODE = "branching"
CONTEXT_MODES = {FULL_MODE, SUMMARY_MODE, SLIDING_MODE, FACTS_MODE}
KEEP_RECENT_EXCHANGES = 5
SUMMARY_BATCH_EXCHANGES = 5
SUMMARY_MODEL = "deepseek-v4-flash"
MIN_WINDOW_EXCHANGES = 1
MAX_WINDOW_EXCHANGES = 50
FACT_KEY_PREFIXES = {
    "goal", "context", "entities", "requirements", "constraints", "preferences",
    "decisions", "agreements", "resources", "environment", "numbers", "deadlines",
    "definitions", "output", "open_questions", "next_steps", "risks",
}


def ensure_context_management(conversation: dict[str, Any]) -> dict[str, Any]:
    """Мигрирует старый диалог к безопасной текущей схеме без потери данных."""
    raw = conversation.get("context_management")
    value = raw if isinstance(raw, dict) else {}
    mode = value.get("mode")
    if mode == LEGACY_BRANCHING_MODE or mode not in CONTEXT_MODES:
        mode = FULL_MODE
    normalized = {
        "mode": mode,
        "keep_recent_exchanges": KEEP_RECENT_EXCHANGES,
        "summary_batch_exchanges": SUMMARY_BATCH_EXCHANGES,
        "summary_model": SUMMARY_MODEL,
        "summary_reasoning_enabled": False,
        "active_summary_id": value.get("active_summary_id"),
        "sliding_window_exchanges": _window(value.get("sliding_window_exchanges", KEEP_RECENT_EXCHANGES)),
        "facts_window_exchanges": _window(value.get("facts_window_exchanges", KEEP_RECENT_EXCHANGES)),
        "active_leaf_id": value.get("active_leaf_id"),
        "pending_branch_from_id": value.get("pending_branch_from_id"),
    }
    if isinstance(raw, dict):
        raw.clear()
        raw.update(normalized)
        normalized = raw
    conversation["context_management"] = normalized
    conversation["summaries"] = conversation.get("summaries") if isinstance(conversation.get("summaries"), list) else []
    conversation["facts"] = conversation.get("facts") if isinstance(conversation.get("facts"), list) else []
    conversation["fact_revisions"] = conversation.get("fact_revisions") if isinstance(conversation.get("fact_revisions"), list) else []
    ensure_message_tree(conversation)
    if not any(item.get("id") == normalized["active_summary_id"] for item in conversation["summaries"] if isinstance(item, dict)):
        normalized["active_summary_id"] = None
    return normalized


def ensure_message_tree(conversation: dict[str, Any]) -> None:
    """Добавляет parent_id линейной истории старого формата."""
    messages = conversation.get("messages")
    if not isinstance(messages, list):
        conversation["messages"] = []
        messages = conversation["messages"]
    previous_id: str | None = None
    valid_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id:
            message_id = uuid.uuid4().hex
            message["id"] = message_id
        if message.get("role") in {"user", "assistant"}:
            parent_id = message.get("parent_id")
            if parent_id not in valid_ids:
                message["parent_id"] = previous_id
            previous_id = message_id
            valid_ids.add(message_id)
        elif message.get("role") == "event" and "parent_id" not in message:
            message["parent_id"] = previous_id
    context = conversation.get("context_management", {})
    if context.get("active_leaf_id") not in valid_ids:
        context["active_leaf_id"] = previous_id
    if context.get("pending_branch_from_id") not in valid_ids:
        context["pending_branch_from_id"] = None


def set_context_mode(conversation: dict[str, Any], mode: Any) -> None:
    if mode not in CONTEXT_MODES:
        raise ValueError("Неизвестная стратегия управления контекстом.")
    ensure_context_management(conversation)["mode"] = mode


def set_window_sizes(conversation: dict[str, Any], sliding: Any = None, facts: Any = None) -> None:
    context = ensure_context_management(conversation)
    if sliding is not None:
        context["sliding_window_exchanges"] = _window(sliding, strict=True)
    if facts is not None:
        context["facts_window_exchanges"] = _window(facts, strict=True)


def active_path_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Возвращает выбранный путь дерева плюс привязанные к нему события."""
    context = ensure_context_management(conversation)
    messages = conversation.get("messages", [])
    by_id = {item.get("id"): item for item in messages if isinstance(item, dict) and item.get("role") in {"user", "assistant"}}
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = context.get("active_leaf_id")
    while current in by_id and current not in seen:
        seen.add(current)
        item = by_id[current]
        path.append(item)
        current = item.get("parent_id")
    path.reverse()
    path_ids = {item.get("id") for item in path}
    events_by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for item in messages:
        if isinstance(item, dict) and item.get("role") == "event" and item.get("parent_id") in path_ids | {None}:
            events_by_parent.setdefault(item.get("parent_id"), []).append(item)
    result: list[dict[str, Any]] = []
    result.extend(events_by_parent.get(None, []))
    for item in path:
        result.append(item)
        result.extend(events_by_parent.get(item.get("id"), []))
    return result


def branch_points(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Описывает развилки выбранного пути для переключателей 1/N."""
    path = active_path_messages(conversation)
    path_ids = {item.get("id") for item in path if item.get("role") in {"user", "assistant"}}
    messages = [item for item in conversation.get("messages", []) if isinstance(item, dict) and item.get("role") in {"user", "assistant"}]
    children: dict[str | None, list[dict[str, Any]]] = {}
    for item in messages:
        children.setdefault(item.get("parent_id"), []).append(item)
    result = []
    for checkpoint in path:
        if checkpoint.get("role") not in {"user", "assistant"}:
            continue
        options = children.get(checkpoint.get("id"), [])
        if len(options) < 2:
            continue
        active_index = next((index for index, child in enumerate(options) if child.get("id") in path_ids), 0)
        result.append({
            "checkpoint_id": checkpoint.get("id"),
            "active_index": active_index,
            "options": [{"number": index + 1, "child_id": child.get("id")} for index, child in enumerate(options)],
        })
    return result


def select_branch(conversation: dict[str, Any], checkpoint_id: Any, child_id: Any) -> None:
    ensure_context_management(conversation)
    messages = [item for item in conversation.get("messages", []) if isinstance(item, dict) and item.get("role") in {"user", "assistant"}]
    child = next((item for item in messages if item.get("id") == child_id and item.get("parent_id") == checkpoint_id), None)
    if child is None:
        raise ValueError("Выбранная ветка не найдена.")
    descendants = {child.get("id")}
    changed = True
    while changed:
        changed = False
        for item in messages:
            if item.get("parent_id") in descendants and item.get("id") not in descendants:
                descendants.add(item.get("id"))
                changed = True
    leaf = child.get("id")
    for item in messages:
        if item.get("id") in descendants:
            leaf = item.get("id")
    context = conversation["context_management"]
    context["active_leaf_id"] = leaf
    context["pending_branch_from_id"] = None
    context["active_summary_id"] = None


def begin_branch(conversation: dict[str, Any], checkpoint_id: Any) -> dict[str, Any]:
    context = ensure_context_management(conversation)
    checkpoint = next((item for item in conversation.get("messages", []) if isinstance(item, dict) and item.get("id") == checkpoint_id and item.get("role") in {"user", "assistant"}), None)
    if checkpoint is None:
        raise ValueError("Сообщение для ветвления не найдено.")
    context["active_leaf_id"] = checkpoint["id"]
    context["pending_branch_from_id"] = checkpoint["id"]
    context["active_summary_id"] = None
    return checkpoint


def append_tree_message(conversation: dict[str, Any], message: dict[str, Any], parent_id: str | None = None) -> None:
    context = ensure_context_management(conversation)
    if parent_id is None:
        parent_id = context.get("active_leaf_id")
    message["parent_id"] = parent_id
    conversation["messages"].append(message)
    if message.get("role") in {"user", "assistant"}:
        context["active_leaf_id"] = message.get("id")
        context["pending_branch_from_id"] = None


def completed_exchanges(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Возвращает полные успешные пары user/assistant в порядке выбранного пути."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for message in messages:
        exchange_id = message.get("exchange_id")
        role = message.get("role")
        technical = message.get("technical")
        if not exchange_id or role not in {"user", "assistant"} or not isinstance(technical, dict):
            continue
        if technical.get("request_status") != "completed":
            continue
        if exchange_id not in grouped:
            grouped[exchange_id] = {"exchange_id": exchange_id, "messages": {}}
            order.append(exchange_id)
        grouped[exchange_id]["messages"][role] = message
    return [
        {"exchange_id": exchange_id, "messages": [grouped[exchange_id]["messages"]["user"], grouped[exchange_id]["messages"]["assistant"]]}
        for exchange_id in order if set(grouped[exchange_id]["messages"]) == {"user", "assistant"}
    ]


def current_stage_exchanges(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Ограничивает успешную историю текущим посещением этапа после первого handoff."""
    exchanges = completed_exchanges(active_path_messages(conversation))
    if not conversation.get("task_handoffs"):
        current_stage = conversation.get("task_state", {}).get("stage")
        selected_reversed = []
        for exchange in reversed(exchanges):
            assistant = next((item for item in exchange.get("messages", []) if item.get("role") == "assistant"), None)
            snapshot_stage = (assistant or {}).get("technical", {}).get("task_state", {}).get("stage")
            if snapshot_stage == current_stage:
                selected_reversed.append(exchange)
            elif selected_reversed and snapshot_stage:
                break
        return list(reversed(selected_reversed)) if selected_reversed else exchanges
    state = conversation.get("task_state") if isinstance(conversation.get("task_state"), dict) else {}
    run_id = state.get("stage_run_id")
    result = []
    for exchange in exchanges:
        assistant = next((item for item in exchange.get("messages", []) if item.get("role") == "assistant"), None)
        snapshot = (assistant or {}).get("technical", {}).get("task_state", {})
        if snapshot.get("stage_run_id") == run_id:
            result.append(exchange)
    return result


def current_stage_facts(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    facts = [item for item in conversation.get("facts", []) if isinstance(item, dict)]
    if not conversation.get("task_handoffs"):
        return facts
    state = conversation.get("task_state") if isinstance(conversation.get("task_state"), dict) else {}
    run_id = state.get("stage_run_id")
    return [item for item in facts if item.get("stage_run_id") == run_id]


def active_summary(conversation: dict[str, Any]) -> dict[str, Any] | None:
    context = ensure_context_management(conversation)
    for summary in reversed(conversation["summaries"]):
        if isinstance(summary, dict) and summary.get("id") == context["active_summary_id"]:
            return summary
    return None


def update_summaries(conversation: dict[str, Any], agent: Agent) -> list[dict[str, Any]]:
    context = ensure_context_management(conversation)
    if context["mode"] != SUMMARY_MODE:
        return []
    exchanges = current_stage_exchanges(conversation)
    current = active_summary(conversation)
    covered_ids = list(current.get("covered_exchange_ids", [])) if current else []
    completed_ids = [item["exchange_id"] for item in exchanges]
    if completed_ids[:len(covered_ids)] != covered_ids:
        context["active_summary_id"] = None
        current = None
        covered_ids = []
    older_count = max(0, len(exchanges) - KEEP_RECENT_EXCHANGES)
    created: list[dict[str, Any]] = []
    while older_count - len(covered_ids) >= SUMMARY_BATCH_EXCHANGES:
        batch = exchanges[len(covered_ids):len(covered_ids) + SUMMARY_BATCH_EXCHANGES]
        logger.info(
            "Обновление summary начато | conversation_id=%s | exchanges=%s | previous=%s",
            conversation.get("id", "-"), len(batch), current.get("id") if current else "-",
        )
        result = agent.summarize(current.get("content", "") if current else "", batch)
        source_messages = [message for exchange in batch for message in exchange["messages"]]
        new_covered_ids = covered_ids + [exchange["exchange_id"] for exchange in batch]
        revision = {
            "id": uuid.uuid4().hex, "created_at": now_iso(), "content": result.content,
            "supersedes_summary_id": current.get("id") if current else None,
            "source_exchange_ids": [exchange["exchange_id"] for exchange in batch],
            "source_message_ids": [message.get("id") for message in source_messages],
            "covered_exchange_ids": new_covered_ids, "covered_exchange_count": len(new_covered_ids),
            "technical": {**result.technical, "usage": normalize_token_usage(result.technical.get("usage")), "purpose": "history_summary"},
        }
        conversation["summaries"].append(revision)
        context["active_summary_id"] = revision["id"]
        current, covered_ids = revision, new_covered_ids
        created.append(revision)
        logger.info(
            "Summary обновлён | conversation_id=%s | summary_id=%s | covered=%s",
            conversation.get("id", "-"), revision["id"], len(new_covered_ids),
        )
    return created


def request_history(conversation: dict[str, Any]) -> tuple[list[dict[str, Any]], str, list[dict[str, str]], dict[str, Any]]:
    """Собирает историю, summary/facts и снимок реально отправленной стратегии."""
    context = ensure_context_management(conversation)
    path = active_path_messages(conversation)
    exchanges = current_stage_exchanges(conversation)
    mode = context["mode"]
    if mode == SUMMARY_MODE:
        current = active_summary(conversation)
        covered_ids = current.get("covered_exchange_ids", []) if current else []
        raw_exchanges = exchanges[len(covered_ids):]
        history = [deepcopy(message) for exchange in raw_exchanges for message in exchange["messages"]]
        return history, current.get("content", "") if current else "", [], {
            "mode": mode, "summary_id": current.get("id") if current else None,
            "summary_exchange_count": len(covered_ids), "verbatim_exchange_count": len(raw_exchanges),
            "keep_recent_exchanges": KEEP_RECENT_EXCHANGES,
            "stage_run_id": conversation.get("task_state", {}).get("stage_run_id"),
        }
    if mode == SLIDING_MODE:
        limit = context["sliding_window_exchanges"]
        selected = exchanges[-max(0, limit - 1):] if limit > 1 else []
    elif mode == FACTS_MODE:
        limit = context["facts_window_exchanges"]
        selected = exchanges[-max(0, limit - 1):] if limit > 1 else []
    else:
        limit = None
        selected = exchanges
    history = [deepcopy(message) for exchange in selected for message in exchange["messages"]]
    facts = [{"key": item["key"], "value": item["value"]} for item in current_stage_facts(conversation) if item.get("key") and item.get("value")] if mode == FACTS_MODE else []
    return history, "", facts, {
        "mode": mode, "summary_id": None, "summary_exchange_count": 0,
        "verbatim_exchange_count": len(selected), "window_exchanges": limit,
        "fact_count": len(facts), "active_leaf_id": context.get("active_leaf_id"),
        "stage_run_id": conversation.get("task_state", {}).get("stage_run_id"),
    }


def update_sticky_facts(conversation: dict[str, Any], agent: Agent, user_text: str) -> dict[str, Any] | None:
    context = ensure_context_management(conversation)
    if context["mode"] != FACTS_MODE:
        return None
    stage_run_id = conversation.get("task_state", {}).get("stage_run_id")
    stage_items = current_stage_facts(conversation)
    current = [{"key": item.get("key"), "value": item.get("value"), "locked": bool(item.get("locked"))} for item in stage_items]
    revision: dict[str, Any] = {"id": uuid.uuid4().hex, "created_at": now_iso(), "purpose": "facts_update"}
    try:
        logger.info(
            "Обновление Sticky Facts начато | conversation_id=%s | current_facts=%s",
            conversation.get("id", "-"), len(current),
        )
        result = agent.extract_facts(current, user_text)
        parsed = json.loads(result.content)
        raw_facts = parsed.get("facts") if isinstance(parsed, dict) else None
        if not isinstance(raw_facts, list):
            raise ValueError("DeepSeek не вернул массив facts.")
        locked = {item["key"]: item for item in stage_items if item.get("locked") and item.get("key")}
        updated: list[dict[str, Any]] = list(locked.values())
        used_keys = set(locked)
        for raw in raw_facts:
            if not isinstance(raw, dict):
                continue
            key = clean_fact_key(raw.get("key"))
            value = clean_fact_value(raw.get("value"))
            if not key or not value or key in used_keys:
                continue
            previous = next((item for item in stage_items if item.get("key") == key), None)
            updated.append({
                "id": previous.get("id") if previous else uuid.uuid4().hex,
                "key": key, "value": value, "locked": False, "source": "auto",
                "created_at": previous.get("created_at") if previous else now_iso(), "updated_at": now_iso(),
                "stage_run_id": stage_run_id,
            })
            used_keys.add(key)
        old_stage_ids = {id(item) for item in stage_items}
        preserved = [item for item in conversation["facts"] if id(item) not in old_stage_ids]
        conversation["facts"] = preserved + updated
        revision.update({
            "status": "completed", "fact_count": len(updated),
            "technical": {**result.technical, "usage": normalize_token_usage(result.technical.get("usage"))},
        })
        logger.info(
            "Sticky Facts обновлены | conversation_id=%s | facts=%s",
            conversation.get("id", "-"), len(updated),
        )
    except Exception as error:
        logger.warning(
            "Не удалось обновить Sticky Facts | conversation_id=%s",
            conversation.get("id", "-"), exc_info=True,
        )
        revision.update({"status": "failed", "error": str(error), "technical": {"usage": normalize_token_usage({})}})
    conversation["fact_revisions"].append(revision)
    return revision


def add_fact(conversation: dict[str, Any], key: Any, value: Any) -> dict[str, Any]:
    ensure_context_management(conversation)
    clean_key, clean_value = clean_fact_key(key), clean_fact_value(value)
    if not clean_key or not clean_value:
        raise ValueError("Укажите ключ вида goal.name и непустое значение.")
    if any(item.get("key") == clean_key for item in current_stage_facts(conversation)):
        raise ValueError("Факт с таким ключом уже существует.")
    fact = {"id": uuid.uuid4().hex, "key": clean_key, "value": clean_value, "locked": True, "source": "manual", "created_at": now_iso(), "updated_at": now_iso(), "stage_run_id": conversation.get("task_state", {}).get("stage_run_id")}
    conversation["facts"].append(fact)
    return fact


def edit_fact(conversation: dict[str, Any], fact_id: str, key: Any, value: Any, locked: Any = True) -> dict[str, Any]:
    ensure_context_management(conversation)
    fact = next((item for item in conversation["facts"] if isinstance(item, dict) and item.get("id") == fact_id), None)
    if fact is None:
        raise KeyError(fact_id)
    clean_key, clean_value = clean_fact_key(key), clean_fact_value(value)
    if not clean_key or not clean_value:
        raise ValueError("Укажите ключ вида goal.name и непустое значение.")
    if any(item.get("id") != fact_id and item.get("key") == clean_key for item in conversation["facts"] if isinstance(item, dict)):
        raise ValueError("Факт с таким ключом уже существует.")
    fact.update({"key": clean_key, "value": clean_value, "locked": bool(locked), "source": "manual" if locked else "auto", "updated_at": now_iso()})
    return fact


def delete_fact(conversation: dict[str, Any], fact_id: str) -> None:
    ensure_context_management(conversation)
    before = len(conversation["facts"])
    conversation["facts"] = [item for item in conversation["facts"] if not isinstance(item, dict) or item.get("id") != fact_id]
    if len(conversation["facts"]) == before:
        raise KeyError(fact_id)


def summary_token_totals(summaries: list[dict[str, Any]]) -> dict[str, int]:
    return _auxiliary_token_totals(summaries)


def facts_token_totals(revisions: list[dict[str, Any]]) -> dict[str, int]:
    return _auxiliary_token_totals(revisions)


def _auxiliary_token_totals(items: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "request_count": 0}
    for item in items:
        technical = item.get("technical") if isinstance(item, dict) else None
        if not isinstance(technical, dict) or not isinstance(technical.get("usage"), dict):
            continue
        usage = normalize_token_usage(technical["usage"])
        if usage["total_tokens"] <= 0:
            continue
        totals["request_count"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            totals[field] += usage[field]
    return totals


def clean_fact_key(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_")[:120]
    if "." not in key or key.split(".", 1)[0] not in FACT_KEY_PREFIXES:
        return ""
    return key


def clean_fact_value(value: Any) -> str:
    return str(value or "").strip()[:2000]


def _window(value: Any, strict: bool = False) -> int:
    if isinstance(value, bool):
        if strict:
            raise ValueError("Размер окна должен быть целым числом от 1 до 50.")
        return KEEP_RECENT_EXCHANGES
    try:
        result = int(value)
    except (TypeError, ValueError):
        if strict:
            raise ValueError("Размер окна должен быть целым числом от 1 до 50.")
        return KEEP_RECENT_EXCHANGES
    if not MIN_WINDOW_EXCHANGES <= result <= MAX_WINDOW_EXCHANGES:
        if strict:
            raise ValueError("Размер окна должен быть целым числом от 1 до 50.")
        return KEEP_RECENT_EXCHANGES
    return result
