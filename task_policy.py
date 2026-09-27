"""Политика допустимого поведения агента на этапах задачи."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from artifacts import ArtifactError, normalize_required_artifacts
from task_state import EVENT_LABELS, allowed_task_events


STAGE_ACTIONS = {
    "planning": ("clarification", "analysis", "planning", "request_plan_approval", "schedule_management", "file_export", "refusal"),
    "execution": ("clarification", "implementation", "progress_report", "request_validation", "request_rollback", "schedule_management", "file_export", "refusal"),
    "validation": ("clarification", "validation", "defect_report", "validation_result", "schedule_management", "file_export", "refusal"),
    "done": ("clarification", "final_summary", "result_explanation", "schedule_management", "file_export", "refusal"),
}

ACTION_LABELS = {
    "clarification": "уточнение",
    "analysis": "анализ",
    "planning": "подготовка плана",
    "request_plan_approval": "запрос утверждения плана",
    "implementation": "реализация",
    "progress_report": "отчёт о выполнении",
    "request_validation": "запрос проверки",
    "request_rollback": "запрос возврата",
    "validation": "проверка",
    "defect_report": "отчёт о дефектах",
    "validation_result": "результат проверки",
    "final_summary": "итог",
    "result_explanation": "объяснение результата",
    "schedule_management": "управление расписанием",
    "file_export": "экспорт результата в локальный файл",
    "refusal": "контролируемый отказ",
}


class PolicyValidationError(ValueError):
    """Проверяющая модель вернула результат, который нельзя безопасно принять."""


def active_invariants(invariant_bundle: dict[str, Any] | None) -> list[dict[str, str]]:
    if not invariant_bundle:
        return []
    return [deepcopy(item) for item in invariant_bundle.get("rules", []) if isinstance(item, dict) and item.get("text")]


def policy_context(task_state: dict[str, Any], invariant_set: dict[str, Any] | None) -> dict[str, Any]:
    stage = str(task_state.get("stage") or "planning")
    events = [event for event in allowed_task_events(task_state) if event not in {"pause", "resume"}]
    invariants = []
    for index, item in enumerate(active_invariants(invariant_set), start=1):
        invariants.append({**item, "ref": f"I{index}"})
    return {
        "stage": stage,
        "transition_mode": task_state.get("transition_mode", "manual"),
        "allowed_action_types": list(STAGE_ACTIONS.get(stage, ("refusal",))),
        "allowed_events": events,
        "required_artifacts": normalize_required_artifacts(task_state.get("required_artifacts", [])),
        "invariants": invariants,
    }


def generation_guidance(task_state: dict[str, Any], invariant_set: dict[str, Any] | None) -> str:
    context = policy_context(task_state, invariant_set)
    invariant_lines = "\n".join(
        f"- {item['ref']}: {item['text']}" for item in context["invariants"]
    ) or "- Активных инвариантов нет."
    return (
        "\n\nОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА:\n"
        "Верни обычный ответ пользователю. Не добавляй служебный JSON, самоотчёт, action_type, "
        "checked_invariant_ids или другие поля внутреннего контроля.\n"
        f"Разрешённые action_type на текущем этапе: {', '.join(context['allowed_action_types'])}.\n"
        "Учитывай каждый активный инвариант ниже. Если запрос требует запрещённого этапом действия или нарушает "
        "инвариант, откажись выполнять запрещённую часть и понятно объясни причину. Не выполняй запрещённое действие "
        "внутри текста отказа. Переход состояния выполняет только сервер.\n"
        + (
            "Работай автономно: выполни весь доступный объём текущего этапа. Не проси формального подтверждения, "
            "если нет критического выбора или недостающих данных; завершённость этапа отдельно определит контроллер.\n"
            if context["transition_mode"] == "automatic" else ""
        )
        + (
            "Если создаёшь или изменяешь файл, выведи его полное содержимое в отдельном блоке строго такого вида:\n"
            "```artifact path=относительный/путь.ext\n"
            "полное содержимое файла\n"
            "```\n"
            "Не утверждай, что файл создан, если такого блока нет. Обязательные пути: "
            + (", ".join(context["required_artifacts"]) or "не заявлены") + ".\n"
            if context["stage"] == "execution" else ""
        )
        + "Активные инварианты:\n" + invariant_lines
    )


def validation_prompt(
    user_text: str,
    draft: str,
    task_state: dict[str, Any],
    invariant_set: dict[str, Any] | None,
    performed_actions: list[str] | None = None,
) -> str:
    context = policy_context(task_state, invariant_set)
    return (
        "Независимо проверь черновик ответа перед показом пользователю. Не продолжай задачу и не улучшай ответ. "
        "Определи фактический тип действия по содержанию, а не по заявленной метке. Проверь этап и каждый инвариант. "
        "Для ссылок на инварианты используй только короткие значения ref (I1, I2, ...) из политики. "
        "Верни только JSON вида "
        '{"allowed":true,"detected_action_type":"planning","checked_invariant_ids":["I1"],'
        '"violated_invariant_ids":[],"stage_complete":true,"recommended_event":"approve_plan",'
        '"required_artifacts":["index.html"],"explanation":""}. '
        "checked_invariant_ids должен содержать ref каждого переданного активного инварианта. "
        "Поле allowed оценивает только допустимость фактического действия на текущем этапе и соблюдение инвариантов. "
        "Оно не зависит от завершённости этапа: если действие разрешено и нарушений нет, верни allowed=true, "
        "даже когда stage_complete=false и recommended_event=null. "
        "stage_complete=true ставь только когда ответ действительно завершает текущий этап, не оставляет критических "
        "вопросов и позволяет немедленно продолжать работу. Тогда recommended_event должен быть одним из allowed_events. "
        "Если этап не завершён, верни stage_complete=false и recommended_event=null. В done всегда верни false/null.\n\n"
        "На planning поле required_artifacts должно содержать точные относительные пути файлов, которые обязаны быть "
        "созданы для результата; если файловый результат не требуется — пустой массив. На остальных этапах повтори "
        "required_artifacts из политики без добавления новых путей.\n\n"
        f"Запрос пользователя:\n{user_text}\n\n"
        f"Черновик ответа:\n{draft}\n\n"
        + (
            "Подтверждённые сервером MCP-действия в этом ответе: "
            + ", ".join(performed_actions)
            + ". Если ответ только сообщает результат Scheduler MCP, классифицируй его как schedule_management. "
            "Если write_file успешно сохранил подготовленный по запросу отчёт, классифицируй ответ как file_export; "
            "это самостоятельный экспорт, а не артефакт жизненного цикла задачи, поэтому на planning оставь "
            "required_artifacts пустым, stage_complete=false и recommended_event=null. "
            "Если в тексте есть отдельное действие другого типа, классифицируй по фактическому содержанию.\n\n"
            if performed_actions else ""
        )
        + f"Политика:\n{json.dumps(context, ensure_ascii=False)}"
    )


def parse_validation_result(content: str, task_state: dict[str, Any], invariant_set: dict[str, Any] | None) -> dict[str, Any]:
    value = _json_object(content, "Проверяющая модель не вернула структурированный результат.")
    context = policy_context(task_state, invariant_set)
    allowed = value.get("allowed")
    action = value.get("detected_action_type")
    checked_refs = _string_list(value.get("checked_invariant_ids"))
    checked, unknown_checked = _resolve_invariant_refs(checked_refs, context)
    required_ids = [item["id"] for item in context["invariants"]]
    if set(checked) != set(required_ids):
        raise PolicyValidationError("Проверяющая модель не подтвердила проверку всех активных инвариантов.")
    violation_refs = _string_list(value.get("violated_invariant_ids"))
    violations, unknown_violations = _resolve_invariant_refs(violation_refs, context)
    explanation = str(value.get("explanation") or "").strip()[:2_000]
    if not isinstance(allowed, bool) or action not in ACTION_LABELS:
        raise PolicyValidationError("Проверяющая модель вернула неполный результат.")
    stage_complete = value.get("stage_complete", False)
    recommended_event = value.get("recommended_event")
    if not isinstance(stage_complete, bool):
        raise PolicyValidationError("Проверяющая модель вернула неверный признак завершения этапа.")
    if recommended_event is not None and not isinstance(recommended_event, str):
        raise PolicyValidationError("Проверяющая модель вернула неверное событие перехода.")
    if context["stage"] == "done":
        # done — терминальный этап. Финальный текст ещё проходит смысловую
        # проверку и инварианты, но ошибочная рекомендация LLM не должна
        # создавать несуществующий переход или блокировать выдачу результата.
        stage_complete = False
        recommended_event = None
    elif stage_complete and recommended_event not in context["allowed_events"]:
        raise PolicyValidationError("Проверяющая модель предложила недопустимый переход этапа.")
    if not stage_complete:
        recommended_event = None
    try:
        required_artifacts = normalize_required_artifacts(value.get("required_artifacts", []))
    except ArtifactError as error:
        raise PolicyValidationError(str(error)) from error
    if context["stage"] != "planning":
        expected = normalize_required_artifacts(task_state.get("required_artifacts", []))
        if required_artifacts != expected:
            raise PolicyValidationError("Проверяющая модель самовольно изменила список обязательных артефактов.")
    # Сервер сам принимает окончательное решение по нормализованному типу
    # действия и ссылкам на инварианты. Флаг allowed от вероятностного
    # валидатора сохраняется для аудита, но не может запретить разрешённый
    # промежуточный ответ только потому, что stage_complete=false.
    server_allowed = action in context["allowed_action_types"] and not violations and not unknown_violations
    return {
        "allowed": server_allowed,
        "validator_allowed": allowed,
        "detected_action_type": action,
        "checked_invariant_ids": checked,
        "violated_invariant_ids": violations,
        "unknown_invariant_refs": list(dict.fromkeys(unknown_checked + unknown_violations)),
        "stage_complete": stage_complete,
        "recommended_event": recommended_event,
        "required_artifacts": required_artifacts,
        "explanation": explanation or (
            "Проверка вернула неизвестную ссылку на инвариант." if unknown_violations else ""
        ),
    }


def blocked_content(reason: str, task_state: dict[str, Any], invariant_set: dict[str, Any] | None, violations: list[str] | None = None) -> str:
    stage = str(task_state.get("stage") or "planning")
    rules = {item["id"]: item["text"] for item in active_invariants(invariant_set)}
    lines = ["Ответ заблокирован контроллером состояния задачи.", f"Текущий этап: {stage}.", reason]
    for rule_id in violations or []:
        if rule_id in rules:
            lines.append(f"Нарушенный инвариант: {rules[rule_id]}")
    lines.append("Измените запрос либо выполните разрешённый переход состояния.")
    return "\n".join(lines)


def policy_audit(
    validation: dict[str, Any] | None,
    task_state: dict[str, Any],
    invariant_set: dict[str, Any] | None,
    *,
    accepted: bool,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "stage": task_state.get("stage"),
        "transition_mode": task_state.get("transition_mode"),
        "detected_action_type": (validation or {}).get("detected_action_type"),
        "validator_allowed": (validation or {}).get("validator_allowed"),
        "stage_complete": bool((validation or {}).get("stage_complete")),
        "recommended_event": (validation or {}).get("recommended_event"),
        "required_artifacts": list((validation or {}).get("required_artifacts", [])),
        "checked_invariants": deepcopy(active_invariants(invariant_set)),
        "checked_invariant_ids": list((validation or {}).get("checked_invariant_ids", [])),
        "violated_invariant_ids": list((validation or {}).get("violated_invariant_ids", [])),
        "unknown_invariant_refs": list((validation or {}).get("unknown_invariant_refs", [])),
        "reason": reason,
    }


def _json_object(content: str, message: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise PolicyValidationError(message)
    text = content.strip().lstrip("\ufeff")
    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            candidates.append(text[first_newline + 1:-3].strip())
    opening = text.find("{")
    closing = text.rfind("}")
    if opening >= 0 and closing > opening:
        candidates.append(text[opening:closing + 1])
    for candidate in dict.fromkeys(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PolicyValidationError(message)


def _resolve_invariant_refs(values: list[str], context: dict[str, Any]) -> tuple[list[str], list[str]]:
    aliases: dict[str, str] = {}
    for item in context.get("invariants", []):
        invariant_id = str(item.get("id") or "")
        reference = str(item.get("ref") or "")
        if invariant_id:
            aliases[invariant_id] = invariant_id
        if reference:
            aliases[reference] = invariant_id
    resolved = [aliases[value] for value in values if value in aliases]
    unknown = [value for value in values if value not in aliases]
    return list(dict.fromkeys(resolved)), list(dict.fromkeys(unknown))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PolicyValidationError("Ожидался список строк в результате проверки политики.")
    return list(dict.fromkeys(value))
