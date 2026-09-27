"""Инкапсулированный агент и провайдер DeepSeek."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Protocol


logger = logging.getLogger("deepseek_agent.provider")

from openai import OpenAI

from task_state import EVENT_LABELS, allowed_task_events


DEEPSEEK_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)

DEEPSEEK_CONTEXT_WINDOWS = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
}


PROVIDER_CAPABILITIES: dict[str, Any] = {
    "provider": "deepseek",
    "models": list(DEEPSEEK_MODELS),
    "context_windows": DEEPSEEK_CONTEXT_WINDOWS,
    "controls": {
        "system_prompt": {"type": "text", "max_length": 20_000},
        "temperature": {"type": "number", "min": 0, "max": 2, "step": 0.01},
        "top_p": {"type": "number", "min": 0, "max": 1, "step": 0.01},
        "reasoning_enabled": {"type": "boolean"},
        "reasoning_effort": {"type": "select", "options": ["low", "high", "max"]},
        "max_tokens": {"type": "optional_integer", "min": 1, "max": 65_536},
        "stop": {"type": "optional_string_list", "max_items": 16},
        "response_format": {"type": "select", "options": ["text", "json_object"]},
        "logprobs": {"type": "boolean"},
        "top_logprobs": {"type": "optional_integer", "min": 0, "max": 20},
    },
    "unsupported": ["seed", "frequency_penalty", "presence_penalty"],
}


@dataclass(frozen=True)
class AgentSettings:
    model: str = "deepseek-v4-flash"
    system_prompt: str = "Ты полезный и внимательный ассистент."
    temperature: float = 1.0
    top_p: float = 1.0
    reasoning_enabled: bool = False
    reasoning_effort: str = "high"
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    response_format: str = "text"
    logprobs: bool = False
    top_logprobs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "AgentSettings":
        if not isinstance(value, dict):
            raise ValueError("Настройки агента должны быть JSON-объектом.")

        model = str(value.get("model", cls.model)).strip()
        if model not in DEEPSEEK_MODELS:
            raise ValueError("Выбрана неподдерживаемая модель DeepSeek.")

        system_prompt = str(value.get("system_prompt", cls.system_prompt)).strip()
        if not system_prompt:
            raise ValueError("Системный промпт не должен быть пустым.")
        if len(system_prompt) > 20_000:
            raise ValueError("Системный промпт длиннее 20 000 символов.")

        temperature = _number(value.get("temperature", 1), "Температура", 0, 2)
        top_p = _number(value.get("top_p", 1), "top_p", 0, 1)
        reasoning_enabled = _boolean(value.get("reasoning_enabled", False), "Reasoning")
        reasoning_effort = str(value.get("reasoning_effort", "high"))
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort должен быть low, high или max.")

        max_tokens = _optional_integer(value.get("max_tokens"), "Лимит токенов", 1, 65_536)
        response_format = str(value.get("response_format", "text"))
        if response_format not in {"text", "json_object"}:
            raise ValueError("Формат ответа должен быть text или json_object.")

        logprobs = _boolean(value.get("logprobs", False), "logprobs")
        top_logprobs = _optional_integer(value.get("top_logprobs"), "top_logprobs", 0, 20)
        if top_logprobs is not None and not logprobs:
            raise ValueError("top_logprobs можно задать только при включённом logprobs.")

        raw_stop = value.get("stop", [])
        if raw_stop in (None, ""):
            stop: list[str] = []
        elif isinstance(raw_stop, list):
            stop = [str(item) for item in raw_stop if str(item)]
        else:
            raise ValueError("Стоп-последовательности должны быть списком строк.")
        if len(stop) > 16 or any(len(item) > 500 for item in stop):
            raise ValueError("Разрешено до 16 стоп-последовательностей длиной до 500 символов.")

        return cls(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )


@dataclass(frozen=True)
class AgentResult:
    content: str
    reasoning_content: str
    technical: dict[str, Any]


class ChatProvider(Protocol):
    def complete(self, messages: list[dict[str, str]], settings: AgentSettings) -> AgentResult: ...


class DeepSeekProvider:
    """Тонкий адаптер DeepSeek. Его можно заменить другим провайдером."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Не найден DEEPSEEK_API_KEY. Добавьте ключ в файл .env и перезапустите приложение."
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=180.0,
            max_retries=1,
        )
        return self._client

    def complete(self, messages: list[dict[str, str]], settings: AgentSettings) -> AgentResult:
        request_args: dict[str, Any] = {
            "model": settings.model,
            "messages": messages,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "stream": False,
            "response_format": {"type": settings.response_format},
            "logprobs": settings.logprobs,
            "extra_body": {
                "thinking": {"type": "enabled" if settings.reasoning_enabled else "disabled"},
                "reasoning_effort": settings.reasoning_effort,
            },
        }
        if settings.max_tokens is not None:
            request_args["max_tokens"] = settings.max_tokens
        if settings.stop:
            request_args["stop"] = settings.stop
        if settings.top_logprobs is not None:
            request_args["top_logprobs"] = settings.top_logprobs

        started = time.perf_counter()
        logger.info(
            "DeepSeek-запрос начат | model=%s | messages=%s | chars=%s | reasoning=%s | format=%s",
            settings.model, len(messages),
            sum(len(str(item.get("content", ""))) for item in messages),
            settings.reasoning_enabled, settings.response_format,
        )
        try:
            response = self._get_client().chat.completions.create(**request_args)
        except Exception:
            logger.exception(
                "DeepSeek-запрос завершился ошибкой | model=%s | elapsed=%.3f",
                settings.model, time.perf_counter() - started,
            )
            raise
        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        technical = {
            "provider": "deepseek",
            "request_id": getattr(response, "id", None),
            "model": getattr(response, "model", None) or settings.model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "elapsed_seconds": round(elapsed, 3),
            "usage": {
                "input_tokens": _usage_value(usage, "prompt_tokens"),
                "output_tokens": _usage_value(usage, "completion_tokens"),
                "total_tokens": _usage_value(usage, "total_tokens"),
                "cached_input_tokens": _usage_value(usage, "prompt_cache_hit_tokens"),
                "uncached_input_tokens": _usage_value(usage, "prompt_cache_miss_tokens"),
                "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
            },
            "settings": settings.to_dict(),
        }
        logger.info(
            "DeepSeek-запрос завершён | request_id=%s | model=%s | finish=%s | elapsed=%.3f | input=%s | output=%s | total=%s",
            technical["request_id"], technical["model"], technical["finish_reason"], elapsed,
            technical["usage"]["input_tokens"], technical["usage"]["output_tokens"],
            technical["usage"]["total_tokens"],
        )
        return AgentResult(
            content=getattr(message, "content", None) or "(Модель не вернула текст ответа)",
            reasoning_content=getattr(message, "reasoning_content", None) or "",
            technical=technical,
        )

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        settings: AgentSettings,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_rounds: int = 6,
    ) -> AgentResult:
        """Выполняет цикл DeepSeek function calling, а сами функции вызывает через MCP."""
        api_tools = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        } for tool in tools]
        allowed_names = {tool["name"] for tool in tools}
        working_messages = list(messages)
        tool_audit: list[dict[str, Any]] = []
        request_ids: list[str] = []
        total_usage = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_input_tokens": 0, "uncached_input_tokens": 0, "reasoning_tokens": 0,
        }
        started = time.perf_counter()
        final_message = None
        final_choice = None
        final_response = None

        for round_number in range(1, max_rounds + 1):
            request_args: dict[str, Any] = {
                "model": settings.model,
                "messages": working_messages,
                "tools": api_tools,
                "tool_choice": "auto",
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "stream": False,
                "response_format": {"type": settings.response_format},
                "logprobs": settings.logprobs,
                "extra_body": {
                    "thinking": {"type": "enabled" if settings.reasoning_enabled else "disabled"},
                    "reasoning_effort": settings.reasoning_effort,
                },
            }
            if settings.max_tokens is not None:
                request_args["max_tokens"] = settings.max_tokens
            if settings.stop:
                request_args["stop"] = settings.stop
            if settings.top_logprobs is not None:
                request_args["top_logprobs"] = settings.top_logprobs

            response = self._get_client().chat.completions.create(**request_args)
            choice = response.choices[0]
            message = choice.message
            usage = getattr(response, "usage", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            values = {
                "input_tokens": _usage_value(usage, "prompt_tokens"),
                "output_tokens": _usage_value(usage, "completion_tokens"),
                "total_tokens": _usage_value(usage, "total_tokens"),
                "cached_input_tokens": _usage_value(usage, "prompt_cache_hit_tokens"),
                "uncached_input_tokens": _usage_value(usage, "prompt_cache_miss_tokens"),
                "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
            }
            for key, value in values.items():
                total_usage[key] += value
            if getattr(response, "id", None):
                request_ids.append(response.id)
            final_message, final_choice, final_response = message, choice, response
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": getattr(message, "content", None),
                "tool_calls": [call.model_dump(exclude_none=True) for call in tool_calls],
            }
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            working_messages.append(assistant_message)

            for call in tool_calls:
                name = str(call.function.name)
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Аргументы инструмента должны быть JSON-объектом.")
                    if name not in allowed_names:
                        raise ValueError(f"Инструмент {name!r} не разрешён.")
                    output = tool_executor(name, arguments)
                    audit = {
                        "round": round_number,
                        "server": output.get("mcp_server"),
                        "name": name,
                        "arguments": arguments,
                        "is_error": bool(output.get("is_error")),
                    }
                except Exception as error:
                    output = {"tool": name, "is_error": True, "error": str(error)}
                    audit = {"round": round_number, "name": name, "arguments": {}, "is_error": True, "error": str(error)}
                tool_audit.append(audit)
                working_messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(output, ensure_ascii=False),
                })
        else:
            raise RuntimeError(f"DeepSeek не завершил цепочку инструментов за {max_rounds} раундов.")

        if final_message is None or final_choice is None or final_response is None:
            raise RuntimeError("DeepSeek не вернул ответ в цикле инструментов.")
        elapsed = time.perf_counter() - started
        technical = {
            "provider": "deepseek",
            "request_id": request_ids[-1] if request_ids else None,
            "request_ids": request_ids,
            "model": getattr(final_response, "model", None) or settings.model,
            "system_fingerprint": getattr(final_response, "system_fingerprint", None),
            "finish_reason": getattr(final_choice, "finish_reason", None),
            "elapsed_seconds": round(elapsed, 3),
            "usage": total_usage,
            "settings": settings.to_dict(),
            "mcp_tool_calls": tool_audit,
        }
        logger.info("DeepSeek MCP-цикл завершён | rounds=%s | tool_calls=%s | elapsed=%.3f",
                    len(request_ids), len(tool_audit), elapsed)
        return AgentResult(
            content=getattr(final_message, "content", None) or "(Модель не вернула текст ответа)",
            reasoning_content=getattr(final_message, "reasoning_content", None) or "",
            technical=technical,
        )


class Agent:
    """Самостоятельная сущность, управляющая контекстом запроса и ответом LLM."""

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def reply(
        self,
        history: list[dict[str, Any]],
        user_text: str,
        settings: AgentSettings,
        summary: str = "",
        facts: list[dict[str, str]] | None = None,
        memory_context: dict[str, Any] | None = None,
        task_handoff_context: dict[str, Any] | None = None,
        task_state: dict[str, Any] | None = None,
        policy_guidance: str = "",
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> AgentResult:
        system_prompt = settings.system_prompt
        if memory_context:
            policy_lines = "\n".join(
                f"- {item.get('text', '')}" for item in memory_context.get("policy_rules", [])
                if isinstance(item, dict) and item.get("text")
            )
            if policy_lines:
                system_prompt += f"\n\nПОСТОЯННАЯ ПОЛИТИКА АГЕНТА:\n{policy_lines}"
            profile = memory_context.get("profile_snapshot") or {}
            preferences = profile.get("preferences") or {}
            if profile:
                profile_lines = [
                    f"Имя: {profile.get('profile_name', 'Пользователь')}",
                    f"Язык: {preferences.get('language', 'auto')}",
                    f"Подробность: {preferences.get('detail_level', 'medium')}",
                    f"Тон: {preferences.get('tone', 'neutral')}",
                    f"Структура: {preferences.get('response_structure', 'free')}",
                    f"Технический уровень: {preferences.get('technical_level', 'intermediate')}",
                ]
                if preferences.get("preferred_formats"):
                    profile_lines.append("Предпочтительные форматы: " + ", ".join(preferences["preferred_formats"]))
                if preferences.get("avoid_formats"):
                    profile_lines.append("Нежелательные форматы: " + ", ".join(preferences["avoid_formats"]))
                if profile.get("custom_instructions"):
                    profile_lines.append("Дополнительные инструкции: " + str(profile["custom_instructions"]))
                system_prompt += "\n\nАКТИВНЫЙ ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:\n" + "\n".join(profile_lines)
            # Порядок блоков соответствует архитектуре: пользователь, затем более
            # конкретная память проекта. Внутри списков manual уже стоит раньше learned.
            for title, key in (
                ("ДОЛГОВРЕМЕННАЯ ПАМЯТЬ ПОЛЬЗОВАТЕЛЯ", "user_memories"),
                ("РАБОЧАЯ ПАМЯТЬ ПРОЕКТА", "project_memories"),
            ):
                lines = []
                for item in memory_context.get(key, []):
                    marker = "подтверждено" if item.get("review_status") == "confirmed" else "не подтверждено"
                    lines.append(f"- [{marker}] {item.get('kind')}.{item.get('key')}: {item.get('value')}")
                if lines:
                    system_prompt += f"\n\n{title}:\n" + "\n".join(lines)
        if summary:
            system_prompt += (
                "\n\nСЖАТАЯ ПАМЯТЬ ПРЕДЫДУЩЕГО ДИАЛОГА:\n"
                f"{summary}\n\n"
                "Используй эту память как контекст разговора. Более новые сообщения ниже имеют приоритет."
            )
        if facts:
            fact_lines = "\n".join(f"- {item['key']} = {item['value']}" for item in facts)
            system_prompt += (
                "\n\nSTICKY FACTS — ВАЖНЫЕ ДАННЫЕ ДИАЛОГА:\n"
                f"{fact_lines}\n\n"
                "Считай эти пары ключ-значение устойчивой памятью. "
                "Более новые сообщения пользователя имеют приоритет при явном противоречии."
            )
        if task_handoff_context and task_handoff_context.get("handoffs"):
            handoff_blocks = []
            labels = {
                "approved_plan": "Утверждённый план",
                "decisions": "Решения",
                "constraints": "Ограничения",
                "acceptance_criteria": "Критерии готовности",
                "completed_work": "Выполненная работа",
                "validation_findings": "Результаты проверки",
                "open_questions": "Открытые вопросы",
            }
            for handoff in task_handoff_context["handoffs"]:
                lines = [
                    f"Передача {handoff.get('source_stage', '?')} → {handoff.get('target_stage', '?')}",
                    f"Итог: {handoff.get('summary', '')}",
                ]
                for key, label in labels.items():
                    values = handoff.get(key, [])
                    if values:
                        lines.append(f"{label}: " + " | ".join(str(item) for item in values))
                handoff_blocks.append("\n".join(lines))
            system_prompt += (
                "\n\nАВТОРИТЕТНЫЙ HANDOFF ТЕКУЩЕГО ЭТАПА:\n"
                + "\n\n".join(handoff_blocks)
                + "\n\nЭто структурированный результат предыдущих этапов. Используй его вместо "
                "восстановления плана или решений из старой переписки."
            )
        # Task State добавляется последним: это текущее явно заданное состояние,
        # а упоминания этапов в памяти, summary, facts и истории могут быть устаревшими.
        if task_state:
            stage_labels = {
                "planning": "Планирование",
                "execution": "Выполнение",
                "validation": "Проверка",
                "done": "Завершено",
            }
            stage = str(task_state.get("stage") or "planning")
            allowed_events = allowed_task_events(task_state)
            allowed_text = ", ".join(
                f"{event} ({EVENT_LABELS[event]})" for event in allowed_events
            ) or "нет — жизненный цикл завершён"
            system_prompt += (
                "\n\nАВТОРИТЕТНОЕ СОСТОЯНИЕ ТЕКУЩЕЙ ЗАДАЧИ:\n"
                f"Описание: {task_state.get('description') or 'не задано'}\n"
                f"Код этапа: {stage}\n"
                f"Название этапа: {stage_labels.get(stage, stage)}\n"
                f"Текущий шаг: {task_state.get('current_step') or 'не задан'}\n"
                f"Ожидаемое действие: {task_state.get('expected_action') or 'не задано'}\n"
                f"Краткий план: {task_state.get('plan') or 'не задан'}\n"
                f"Обязательные артефакты: {', '.join(task_state.get('required_artifacts', [])) or 'нет'}\n"
                f"Зарегистрированные артефакты: {json.dumps(task_state.get('registered_artifacts', []), ensure_ascii=False)}\n"
                f"Активность: {task_state.get('activity', 'active')}\n"
                f"Управление переходами: {task_state.get('transition_mode', 'manual')}\n\n"
                f"Разрешённые сервером события: {allowed_text}\n\n"
                "ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА СОСТОЯНИЯ:\n"
                "- Эти значения явно установлены пользователем и являются единственным источником истины о текущем состоянии.\n"
                "- Не определяй и не переопределяй текущий этап по смыслу истории, плану, выполненной работе или собственным выводам.\n"
                "- Упоминания других этапов в памяти, summary, facts и истории считай устаревшими историческими сведениями.\n"
                f"- Если пользователь спрашивает текущий этап, отвечай точным значением: {stage}.\n"
                "- Не утверждай, что этап изменился, пока серверное событие перехода не выполнено.\n"
                "- Если пользователь просит перепрыгнуть этап, объясни, какое разрешённое действие требуется сначала.\n"
                "- Ты читаешь состояние, но не изменяешь его: переходы выполняет только серверный автомат."
            )
        if policy_guidance:
            system_prompt += policy_guidance
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in history:
            role = item.get("role")
            content = item.get("content")
            technical = item.get("technical", {})
            if technical.get("request_status") == "failed":
                continue
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        request_settings = replace(settings, response_format="text") if policy_guidance else settings
        complete_with_tools = getattr(self.provider, "complete_with_tools", None)
        if tools and tool_executor and callable(complete_with_tools):
            return complete_with_tools(messages, request_settings, tools, tool_executor)
        return self.provider.complete(messages, request_settings)

    def validate_task_response(self, prompt: str, settings: AgentSettings) -> AgentResult:
        """Вторым независимым JSON-вызовом проверяет фактическое действие черновика."""
        validator_settings = AgentSettings(
            model=settings.model,
            system_prompt="Ты строгий контроллер этапов задачи и инвариантов. Проверяй содержание, а не метки автора.",
            temperature=0,
            top_p=1,
            reasoning_enabled=False,
            reasoning_effort="low",
            max_tokens=1_500,
            response_format="json_object",
        )
        return self.provider.complete([
            {"role": "system", "content": validator_settings.system_prompt},
            {"role": "user", "content": prompt},
        ], validator_settings)

    def extract_stage_handoff(
        self,
        *,
        source_stage: str,
        target_stage: str,
        event: str,
        task_state: dict[str, Any],
        previous_handoffs: list[dict[str, Any]],
        exchanges: list[dict[str, Any]],
        max_transcript_chars: int,
    ) -> AgentResult:
        """Сворачивает завершённое посещение этапа в структурированный контракт следующего."""
        transcript = []
        for exchange in exchanges:
            for message in exchange.get("messages", []):
                role = "Пользователь" if message.get("role") == "user" else "Ассистент"
                transcript.append(f"{role}: {message.get('content', '')}")
        joined = "\n\n".join(transcript)
        if len(joined) > max_transcript_chars:
            half = max_transcript_chars // 2
            joined = joined[:half] + "\n\n[середина длинного этапа опущена]\n\n" + joined[-half:]
        previous = [
            {
                key: item.get(key)
                for key in ("source_stage", "target_stage", "summary", "approved_plan", "decisions", "constraints", "acceptance_criteria", "completed_work", "validation_findings", "open_questions")
            }
            for item in previous_handoffs
        ]
        state_payload = {
            key: task_state.get(key)
            for key in ("description", "stage", "current_step", "expected_action", "plan", "stage_run_id", "required_artifacts")
        }
        prompt = (
            f"Сверни завершённое посещение этапа {source_stage} перед переходом в {target_stage}. "
            "Это контракт для следующего этапа, а не пересказ беседы. Не добавляй домыслов. "
            "Сохрани принятый план, решения, ограничения, критерии готовности, фактически выполненную работу, "
            "результаты проверки и открытые вопросы. Верни только JSON с полями: summary (строка), "
            "approved_plan, decisions, constraints, acceptance_criteria, completed_work, validation_findings, "
            "open_questions (массивы строк).\n\n"
            f"Событие перехода: {event}\n"
            f"Авторитетное состояние задачи:\n{json.dumps(state_payload, ensure_ascii=False)}\n\n"
            f"Активные handoff предыдущих этапов:\n{json.dumps(previous, ensure_ascii=False)}\n\n"
            f"Успешная история завершаемого этапа:\n{joined}"
        )
        settings = AgentSettings(
            model="deepseek-v4-flash",
            system_prompt="Ты формируешь структурированный handoff между этапами задачи.",
            temperature=0.1,
            top_p=1.0,
            reasoning_enabled=False,
            reasoning_effort="low",
            max_tokens=3_000,
            response_format="json_object",
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)

    def extract_memories(
        self, user_text: str, assistant_text: str, *, allow_project: bool, allow_user: bool
    ) -> AgentResult:
        """Одним строгим JSON-вызовом извлекает кандидатов обоих разрешённых scope."""
        scopes = []
        if allow_project:
            scopes.append("project")
        if allow_user:
            scopes.append("user")
        prompt = (
            "Извлеки до 12 устойчивых записей памяти из успешного обмена. "
            f"Разрешённые scope: {', '.join(scopes)}. "
            "Для user сохраняй только долговременные сведения о пользователе, полезные вне проекта. "
            "Не сохраняй секреты, пароли, ключи, временные задачи, tool chatter и случайные имена файлов. "
            "Для project сохраняй цели, решения, требования, ограничения, окружение, ресурсы, результаты и риски. "
            "Тип kind выбирай только из: goal, context, preference, decision, requirement, constraint, "
            "resource, environment, definition, open_question, result, risk. "
            "Верни только JSON вида {\"memories\":[{\"scope\":\"project\",\"kind\":\"decision\","
            "\"key\":\"storage.format\",\"value\":\"Локальные JSON-файлы\",\"confidence\":\"high\"}]}.\n\n"
            f"Пользователь:\n{user_text}\n\nАссистент:\n{assistant_text}"
        )
        settings = AgentSettings(
            model="deepseek-v4-flash", system_prompt="Ты извлекаешь безопасную структурированную память.",
            temperature=0.1, reasoning_enabled=False, reasoning_effort="low",
            max_tokens=2_000, response_format="json_object",
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)

    def extract_project_memories_from_history(self, messages: list[dict[str, Any]]) -> AgentResult:
        """Извлекает проектную память из успешной истории по явному действию пользователя."""
        transcript = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
                continue
            label = "Пользователь" if role == "user" else "Ассистент"
            transcript.append(f"{label}: {content}")
        if not transcript:
            raise ValueError("В диалоге нет успешной истории для извлечения памяти.")
        prompt = (
            "Извлеки до 12 актуальных записей рабочей памяти проекта из истории диалога. "
            "Сохраняй цели, решения, требования, ограничения, окружение, ресурсы, результаты, "
            "термины, открытые вопросы и риски. Не сохраняй секреты, пароли, API-ключи, "
            "tool chatter и устаревшие промежуточные детали. Тип kind выбирай только из: goal, context, "
            "preference, decision, requirement, constraint, resource, environment, definition, open_question, result, risk. "
            "Верни только JSON вида "
            '{"memories":[{"scope":"project","kind":"decision","key":"storage.format",'
            '"value":"Локальные JSON-файлы","confidence":"high"}]}.\n\n'
            "Успешная история активной ветки:\n" + "\n\n".join(transcript)
        )
        settings = AgentSettings(
            model="deepseek-v4-flash", system_prompt="Ты извлекаешь безопасную рабочую память проекта.",
            temperature=0.1, reasoning_enabled=False, reasoning_effort="low",
            max_tokens=2_000, response_format="json_object",
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)

    def extract_facts(self, current_facts: list[dict[str, Any]], user_text: str) -> AgentResult:
        """Возвращает обновлённую key-value память в строгом JSON."""
        facts_json = json.dumps(current_facts, ensure_ascii=False, indent=2)
        prompt = (
            "Текущие facts:\n"
            f"{facts_json}\n\n"
            "Новое сообщение пользователя:\n"
            f"{user_text}\n\n"
            "Верни JSON-объект строго вида "
            '{"facts":[{"key":"category.name","value":"точное значение"}]}. '
            "Это должен быть полный актуальный набор, а не только изменения. "
            "Разрешённые категории ключей: goal, context, entities, requirements, constraints, "
            "preferences, decisions, agreements, resources, environment, numbers, deadlines, "
            "definitions, output, open_questions, next_steps, risks. "
            "Ключи с locked=true сохрани дословно. Удаляй только явно устаревшие незакреплённые данные. "
            "Не выдумывай фактов и не добавляй пояснений вне JSON."
        )
        settings = AgentSettings(
            model="deepseek-v4-flash",
            system_prompt="Ты обновляешь точную структурированную память диалога.",
            temperature=0.1,
            top_p=1.0,
            reasoning_enabled=False,
            reasoning_effort="low",
            max_tokens=2_000,
            response_format="json_object",
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)

    def summarize(self, previous_summary: str, exchanges: list[dict[str, Any]]) -> AgentResult:
        """Обновляет накопительную память дешёвой моделью без reasoning."""
        transcript: list[str] = []
        for exchange in exchanges:
            for message in exchange.get("messages", []):
                label = "Пользователь" if message.get("role") == "user" else "Ассистент"
                transcript.append(f"{label}: {message.get('content', '')}")
        previous = previous_summary or "(предыдущего summary ещё нет)"
        joined_transcript = "\n\n".join(transcript)
        prompt = (
            "Предыдущий накопительный summary:\n"
            f"{previous}\n\n"
            "Новые обмены для добавления:\n"
            f"{joined_transcript}\n\n"
            "Верни новый цельный summary на русском языке. Сохрани факты, имена, числа, решения, "
            "предпочтения пользователя, незавершённые задачи и важные ограничения. Не добавляй домыслов. "
            "Не описывай сам процесс суммаризации и не используй JSON."
        )
        settings = AgentSettings(
            model="deepseek-v4-flash",
            system_prompt="Ты создаёшь точную компактную память длительного диалога.",
            temperature=0.2,
            top_p=1.0,
            reasoning_enabled=False,
            reasoning_effort="low",
            max_tokens=2_000,
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)


TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "reasoning_tokens",
)


def normalize_token_usage(value: Any) -> dict[str, int]:
    """Возвращает безопасный набор целочисленных счётчиков одного API-вызова."""
    source = value if isinstance(value, dict) else {}
    result: dict[str, int] = {}
    for field_name in TOKEN_USAGE_FIELDS:
        raw_value = source.get(field_name, 0)
        try:
            result[field_name] = max(0, int(raw_value or 0))
        except (TypeError, ValueError):
            result[field_name] = 0
    if result["total_tokens"] == 0:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def dialogue_token_totals(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Суммирует фактический usage всех успешно завершённых вызовов диалога."""
    totals = {field_name: 0 for field_name in TOKEN_USAGE_FIELDS}
    totals["request_count"] = 0
    for message in messages:
        technical = message.get("technical")
        if message.get("role") != "assistant" or not isinstance(technical, dict):
            continue
        if technical.get("request_status") == "failed" or not isinstance(technical.get("usage"), dict):
            continue
        usage = normalize_token_usage(technical["usage"])
        totals["request_count"] += 1
        for field_name in TOKEN_USAGE_FIELDS:
            totals[field_name] += usage[field_name]
    return totals


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} должна быть числом от {minimum:g} до {maximum:g}.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} должна быть от {minimum:g} до {maximum:g}.")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} должен быть логическим значением.")
    return value


def _optional_integer(value: Any, label: str, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} должен быть целым числом от {minimum} до {maximum}.")
    return value


def _usage_value(source: Any, name: str) -> int:
    value = getattr(source, name, 0) if source is not None else 0
    return int(value or 0)
