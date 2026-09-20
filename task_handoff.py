"""Структурированные снимки передачи контекста между этапами задачи."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any

from agent import Agent, normalize_token_usage
from context_manager import current_stage_exchanges
from storage import now_iso
from task_state import ensure_task_state


HANDOFF_ARRAY_FIELDS = (
    "approved_plan",
    "decisions",
    "constraints",
    "acceptance_criteria",
    "completed_work",
    "validation_findings",
    "open_questions",
)
MAX_TRANSCRIPT_CHARS = 200_000


class TaskHandoffError(ValueError):
    """Handoff нельзя безопасно сформировать или разобрать."""


def ensure_task_handoffs(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    raw = conversation.get("task_handoffs")
    handoffs = [item for item in raw if isinstance(item, dict) and item.get("id")] if isinstance(raw, list) else []
    conversation["task_handoffs"] = handoffs
    state = ensure_task_state(conversation)
    valid_ids = {str(item["id"]) for item in handoffs}
    state["active_handoff_ids"] = [
        item for item in state.get("active_handoff_ids", []) if item in valid_ids
    ]
    return handoffs


def active_task_handoffs(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    handoffs = ensure_task_handoffs(conversation)
    active_ids = set(ensure_task_state(conversation).get("active_handoff_ids", []))
    return [deepcopy(item) for item in handoffs if item.get("id") in active_ids]


def task_handoff_context(conversation: dict[str, Any]) -> dict[str, Any]:
    state = ensure_task_state(conversation)
    active = active_task_handoffs(conversation)
    return {
        "stage": state["stage"],
        "stage_run_id": state["stage_run_id"],
        "handoffs": active,
        "handoff_count": len(active),
    }


def stage_exchanges(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Возвращает успешные обмены текущего посещения этапа, сохраняя legacy-совместимость."""
    return current_stage_exchanges(conversation)


def build_stage_handoff(
    conversation: dict[str, Any],
    agent: Agent,
    *,
    target_stage: str,
    event: str,
) -> dict[str, Any]:
    """Создаёт неизменяемый handoff до мутации этапа."""
    state = deepcopy(ensure_task_state(conversation))
    exchanges = stage_exchanges(conversation)
    source_ids = [str(item.get("exchange_id")) for item in exchanges if item.get("exchange_id")]
    active_before = active_task_handoffs(conversation)
    if exchanges:
        result = agent.extract_stage_handoff(
            source_stage=state["stage"],
            target_stage=target_stage,
            event=event,
            task_state=state,
            previous_handoffs=active_before,
            exchanges=exchanges,
            max_transcript_chars=MAX_TRANSCRIPT_CHARS,
        )
        payload = _parse_payload(result.content)
        technical = {
            **result.technical,
            "usage": normalize_token_usage(result.technical.get("usage")),
            "purpose": "task_stage_handoff",
        }
    else:
        payload = _deterministic_payload(state, target_stage)
        technical = {"usage": normalize_token_usage({}), "purpose": "task_stage_handoff", "source": "deterministic"}
    payload = _merge_authoritative_task_fields(payload, state)
    return {
        "id": uuid.uuid4().hex,
        "created_at": now_iso(),
        "source_stage": state["stage"],
        "target_stage": target_stage,
        "source_stage_run_id": state["stage_run_id"],
        "target_stage_run_id": None,
        "event": event,
        "summary": payload["summary"],
        **{field: payload[field] for field in HANDOFF_ARRAY_FIELDS},
        "source_exchange_ids": source_ids,
        "previous_handoff_ids": [str(item["id"]) for item in active_before],
        "technical": technical,
    }


def activate_stage_handoff(conversation: dict[str, Any], handoff: dict[str, Any]) -> dict[str, Any]:
    """Сохраняет handoff после успешного перехода и выбирает контекст нового этапа."""
    handoffs = ensure_task_handoffs(conversation)
    state = ensure_task_state(conversation)
    saved = deepcopy(handoff)
    saved["target_stage_run_id"] = state["stage_run_id"]
    handoffs.append(saved)
    previous_ids = [item for item in saved.get("previous_handoff_ids", []) if isinstance(item, str)]
    source_stage, target_stage = saved.get("source_stage"), saved.get("target_stage")
    if (source_stage, target_stage) in {("planning", "execution"), ("execution", "planning")}:
        active_ids = [saved["id"]]
    elif (source_stage, target_stage) == ("validation", "execution"):
        previous = {item.get("id"): item for item in handoffs}
        active_ids = [
            item_id for item_id in previous_ids
            if previous.get(item_id, {}).get("source_stage") == "planning"
            and previous.get(item_id, {}).get("target_stage") == "execution"
        ] + [saved["id"]]
    else:
        active_ids = previous_ids + [saved["id"]]
    state["active_handoff_ids"] = list(dict.fromkeys(active_ids))[-10:]
    if target_stage == "execution" and not state.get("plan") and saved.get("approved_plan"):
        state["plan"] = "\n".join(f"{index}. {item}" for index, item in enumerate(saved["approved_plan"], start=1))[:4_000]
    conversation["task_handoffs"] = handoffs
    conversation["task_state"] = state
    conversation.get("context_management", {})["active_summary_id"] = None
    return saved


def handoff_token_totals(handoffs: list[dict[str, Any]]) -> dict[str, int]:
    fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "uncached_input_tokens", "reasoning_tokens")
    totals = {field: 0 for field in fields}
    totals["request_count"] = 0
    for item in handoffs:
        usage = item.get("technical", {}).get("usage") if isinstance(item, dict) else None
        if not isinstance(usage, dict) or not usage.get("total_tokens"):
            continue
        normalized = normalize_token_usage(usage)
        totals["request_count"] += 1
        for field in fields:
            totals[field] += normalized[field]
    return totals


def _parse_payload(content: str) -> dict[str, Any]:
    value = _json_object(content)
    summary = _text(value.get("summary"), 4_000)
    if not summary:
        raise TaskHandoffError("Модель не вернула итог этапа для handoff.")
    payload = {"summary": summary}
    for field in HANDOFF_ARRAY_FIELDS:
        payload[field] = _text_list(value.get(field))
    return payload


def _deterministic_payload(state: dict[str, Any], target_stage: str) -> dict[str, Any]:
    description = str(state.get("description") or "Задача без отдельного описания.").strip()
    plan = str(state.get("plan") or "").strip()
    return {
        "summary": f"Этап {state['stage']} завершён без успешных диалоговых обменов. Следующий этап: {target_stage}. {description}"[:4_000],
        "approved_plan": [plan] if state["stage"] == "planning" and plan else [],
        "decisions": [],
        "constraints": [],
        "acceptance_criteria": [str(state.get("expected_action"))] if state.get("expected_action") else [],
        "completed_work": [],
        "validation_findings": [],
        "open_questions": [],
    }


def _merge_authoritative_task_fields(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    plan = str(state.get("plan") or "").strip()
    if state["stage"] == "planning" and plan and plan not in payload["approved_plan"]:
        payload["approved_plan"].insert(0, plan)
    return payload


def _json_object(content: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise TaskHandoffError("Модель не вернула JSON handoff.")
    text = content.strip().lstrip("\ufeff")
    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        newline = text.find("\n")
        if newline >= 0:
            candidates.append(text[newline + 1:-3].strip())
    opening, closing = text.find("{"), text.rfind("}")
    if opening >= 0 and closing > opening:
        candidates.append(text[opening:closing + 1])
    for candidate in dict.fromkeys(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise TaskHandoffError("Модель не вернула корректный JSON handoff.")


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:30]:
        text = _text(item, 2_000)
        if text and text not in result:
            result.append(text)
    return result
