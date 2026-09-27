"""Локальные тесты без реальных запросов и расходов DeepSeek API."""

import json
import logging
import os
import socket
import tempfile
import unittest
from copy import deepcopy
from unittest import mock
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from agent import (
    DEEPSEEK_CONTEXT_WINDOWS,
    Agent,
    AgentResult,
    AgentSettings,
    DeepSeekProvider,
    dialogue_token_totals,
)
from artifacts import ArtifactError, extract_artifact_payloads, normalize_artifact_path
import app as app_module
from app import create_app, friendly_api_error
from context_manager import completed_exchanges, ensure_context_management, summary_token_totals
from main import find_available_port
from logging_setup import BACKUP_COUNT, MAX_LOG_BYTES, configure_logging
from presets import PresetManager
from storage import JsonStorage
from task_policy import PolicyValidationError, generation_guidance, parse_validation_result, policy_context
from task_state import (
    TaskTransitionError,
    allowed_task_events,
    apply_task_event,
    ensure_task_state,
    task_state_report,
    update_task_state,
)
from voice import WhisperService


def policy_generation_calls(provider):
    return [call for call in provider.calls if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in call[0][0]["content"]]


class FakeProvider:
    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def complete(self, messages, settings):
        self.calls.append((messages, settings))
        if self.fail:
            raise RuntimeError("Connection error")
        system = messages[0]["content"]
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in system:
            content = "Тестовый ответ"
        elif "строгий контроллер этапов" in system:
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            actions = {"planning": "analysis", "execution": "progress_report", "validation": "validation", "done": "final_summary"}
            content = json.dumps({
                "allowed": True, "detected_action_type": actions[context["stage"]],
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [], "explanation": "",
            }, ensure_ascii=False)
        elif "структурированный handoff между этапами" in system:
            content = json.dumps({
                "summary": "Этап завершён; зафиксирован результат для продолжения.",
                "approved_plan": ["Выполнить согласованный план"],
                "decisions": ["Использовать утверждённую архитектуру"],
                "constraints": ["Не выходить за рамки задачи"],
                "acceptance_criteria": ["Результат проверен"],
                "completed_work": ["Работа текущего этапа завершена"],
                "validation_findings": [],
                "open_questions": [],
            }, ensure_ascii=False)
        elif "обновляешь точную структурированную память" in system:
            content = '{"facts":[{"key":"preferences.language","value":"Русский"}]}'
        else:
            content = '{"facts":[]}' if settings.response_format == "json_object" else "Тестовый ответ"
        return AgentResult(
            content=content,
            reasoning_content="Скрытое рассуждение",
            technical={
                "provider": "deepseek",
                "request_id": "req-test",
                "model": settings.model,
                "finish_reason": "stop",
                "elapsed_seconds": 0.321,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "cached_input_tokens": 7,
                    "uncached_input_tokens": 3,
                    "reasoning_tokens": 4,
                },
                "settings": settings.to_dict(),
            },
        )


class ToolAwareFakeProvider(FakeProvider):
    def complete_with_tools(self, messages, settings, tools, tool_executor):
        base = super().complete(messages, settings)
        arguments = {"latitude": 55.03, "longitude": 82.92}
        output = tool_executor("get_current_weather", arguments)
        return AgentResult(
            content="Сейчас 12 °C. Подойдёт лёгкая куртка.",
            reasoning_content="",
            technical={**base.technical, "mcp_tool_calls": [{
                "round": 1, "name": "get_current_weather", "arguments": arguments,
                "is_error": output["is_error"],
            }]},
        )


class WeatherReportPipelineProvider(FakeProvider):
    """Имитирует выбор погоды, обработку моделью и последующую запись отчёта."""

    def complete(self, messages, settings):
        system = messages[0]["content"]
        if "строгий контроллер этапов" in system and "write_file" in messages[-1]["content"]:
            self.calls.append((messages, settings))
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            return AgentResult(json.dumps({
                "allowed": True, "detected_action_type": "file_export",
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [], "stage_complete": False,
                "recommended_event": None, "required_artifacts": [], "explanation": "",
            }, ensure_ascii=False), "", super().complete(messages, settings).technical)
        return super().complete(messages, settings)

    def complete_with_tools(self, messages, settings, tools, tool_executor):
        base = super().complete(messages, settings)
        names = {tool["name"] for tool in tools}
        if not {"get_current_weather", "write_file"}.issubset(names):
            raise AssertionError(f"Не получены инструменты композиции: {sorted(names)}")
        weather_arguments = {"latitude": 55.03, "longitude": 82.92}
        weather = tool_executor("get_current_weather", weather_arguments)
        temperature = weather["result"]["current"]["temperature_2m"]
        report = (
            "# Погодный отчёт для Новосибирска\n\n"
            f"Температура: {temperature} °C. Для прогулки подойдёт лёгкая куртка.\n"
        )
        file_arguments = {"path": "../../чужая-папка/weather.md", "content": report}
        saved = tool_executor("write_file", file_arguments)
        relative_path = saved["result"]["relative_path"]
        return AgentResult(
            content=f"Прогноз обработан, отчёт сохранён: {relative_path}",
            reasoning_content="",
            technical={**base.technical, "mcp_tool_calls": [
                {"round": 1, "name": "get_current_weather", "arguments": weather_arguments,
                 "is_error": weather["is_error"]},
                {"round": 2, "name": "write_file", "arguments": file_arguments,
                 "is_error": saved["is_error"]},
            ]},
        )


class BlockedWeatherReportPipelineProvider(WeatherReportPipelineProvider):
    def complete(self, messages, settings):
        system = messages[0]["content"]
        if "строгий контроллер этапов" in system and "write_file" in messages[-1]["content"]:
            self.calls.append((messages, settings))
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            return AgentResult(json.dumps({
                "allowed": False, "detected_action_type": "implementation",
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [], "stage_complete": False,
                "recommended_event": None, "required_artifacts": [],
                "explanation": "Экспорт ошибочно классифицирован как реализация.",
            }, ensure_ascii=False), "", FakeProvider.complete(self, messages, settings).technical)
        return FakeProvider.complete(self, messages, settings)


class OrchestrationFakeProvider(FakeProvider):
    """Имитирует зависимую цепочку MediaWiki -> World Bank."""

    def complete_with_tools(self, messages, settings, tools, tool_executor):
        base = super().complete(messages, settings)
        names = {tool["name"] for tool in tools}
        if not {"search-page", "worldbank_get_data"}.issubset(names):
            raise AssertionError(f"Не получены инструменты оркестрации: {sorted(names)}")
        if "update-page" in names:
            raise AssertionError("MediaWiki write-инструмент не должен передаваться модели")
        wiki_args = {"query": "largest countries South America", "limit": 5}
        wiki = tool_executor("search-page", wiki_args)
        country_codes = wiki["result"]["country_codes"]
        bank_args = {"indicator_id": "SP.POP.TOTL", "countries": country_codes, "mrnev": 1}
        bank = tool_executor("worldbank_get_data", bank_args)
        return AgentResult(
            content="MediaWiki определил страны, World Bank вернул показатели населения.",
            reasoning_content="",
            technical={**base.technical, "mcp_tool_calls": [
                {"round": 1, "server": wiki["mcp_server"], "name": "search-page", "arguments": wiki_args, "is_error": False},
                {"round": 2, "server": bank["mcp_server"], "name": "worldbank_get_data", "arguments": bank_args, "is_error": False},
            ]},
        )


class MemoryProvider(FakeProvider):
    def complete(self, messages, settings):
        result = super().complete(messages, settings)
        if settings.response_format == "json_object" and "извлекаешь безопасную" in messages[0]["content"]:
            result = AgentResult(
                content=json.dumps({"memories": [
                    {"scope": "project", "kind": "decision", "key": "storage.format", "value": "Локальные JSON-файлы", "confidence": "high"},
                    {"scope": "user", "kind": "preference", "key": "response.language", "value": "Русский язык", "confidence": "high"},
                ]}, ensure_ascii=False),
                reasoning_content="", technical=result.technical,
            )
        return result


class ExtractionFailProvider(FakeProvider):
    def complete(self, messages, settings):
        if settings.response_format == "json_object" and "извлекаешь безопасную" in messages[0]["content"]:
            self.calls.append((messages, settings))
            raise RuntimeError("memory extraction failed")
        return super().complete(messages, settings)


class ProfileAwareProvider(FakeProvider):
    def complete(self, messages, settings):
        result = super().complete(messages, settings)
        system = messages[0]["content"]
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in system:
            return AgentResult(
                content="Краткий ответ" if "Подробность: low" in system else "Подробный пошаговый ответ",
                reasoning_content=result.reasoning_content,
                technical=result.technical,
            )
        if settings.response_format != "json_object":
            return AgentResult(
                content="Краткий ответ" if "Подробность: low" in system else "Подробный пошаговый ответ",
                reasoning_content=result.reasoning_content,
                technical=result.technical,
            )
        return result


class MislabeledImplementationProvider(FakeProvider):
    """Имитирует модель, маскирующую реализацию под разрешённый анализ."""

    def complete(self, messages, settings):
        result = super().complete(messages, settings)
        system = messages[0]["content"]
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in system:
            return AgentResult("```python\nprint('реализация')\n```", result.reasoning_content, result.technical)
        if "строгий контроллер этапов" in system:
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            verdict = {
                "allowed": False, "detected_action_type": "implementation",
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [], "explanation": "Черновик фактически содержит реализацию до утверждения плана.",
            }
            return AgentResult(json.dumps(verdict, ensure_ascii=False), "", result.technical)
        return result


class IncompleteAnalysisProvider(FakeProvider):
    """Имитирует ошибку валидатора: разрешённый анализ принят за незавершённый план."""

    def complete(self, messages, settings):
        result = super().complete(messages, settings)
        system = messages[0]["content"]
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in system:
            return AgentResult("Сейчас +12 °C, подойдёт лёгкая куртка.", "", result.technical)
        if "строгий контроллер этапов" in system:
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            verdict = {
                "allowed": False,
                "detected_action_type": "analysis",
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [],
                "stage_complete": False,
                "recommended_event": None,
                "required_artifacts": [],
                "explanation": "Ответ не завершает этап planning.",
            }
            return AgentResult(json.dumps(verdict, ensure_ascii=False), "", result.technical)
        return result


class InvariantRefusalProvider(FakeProvider):
    def complete(self, messages, settings):
        result = super().complete(messages, settings)
        system = messages[0]["content"]
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in system:
            return AgentResult(
                "Запрос отклонён: он нарушает инвариант текущей задачи.",
                result.reasoning_content, result.technical,
            )
        if "строгий контроллер этапов" in system:
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            verdict = {
                "allowed": True, "detected_action_type": "refusal",
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [], "explanation": "",
            }
            return AgentResult(json.dumps(verdict, ensure_ascii=False), "", result.technical)
        return result


class HandoffFailProvider(FakeProvider):
    def complete(self, messages, settings):
        if "структурированный handoff между этапами" in messages[0]["content"]:
            self.calls.append((messages, settings))
            raise RuntimeError("handoff failed")
        return super().complete(messages, settings)


class AutopilotProvider(FakeProvider):
    def complete(self, messages, settings):
        if "строгий контроллер этапов" in messages[0]["content"]:
            self.calls.append((messages, settings))
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            actions = {"planning": "planning", "execution": "implementation", "validation": "validation_result", "done": "final_summary"}
            events = {"planning": "approve_plan", "execution": "complete_execution", "validation": "pass_validation", "done": None}
            event = events[context["stage"]]
            return AgentResult(json.dumps({
                "allowed": True,
                "detected_action_type": actions[context["stage"]],
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [],
                "stage_complete": event is not None,
                "recommended_event": event,
                "explanation": "",
            }, ensure_ascii=False), "", super().complete(messages, settings).technical)
        return super().complete(messages, settings)


class ArtifactLifecycleProvider(FakeProvider):
    def complete(self, messages, settings):
        system = messages[0]["content"]
        if "строгий контроллер этапов" in system:
            self.calls.append((messages, settings))
            context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
            events = {"planning": "approve_plan", "execution": "complete_execution", "validation": "pass_validation", "done": None}
            actions = {"planning": "planning", "execution": "implementation", "validation": "validation_result", "done": "final_summary"}
            required = ["index.html"] if context["stage"] == "planning" else context["required_artifacts"]
            return AgentResult(json.dumps({
                "allowed": True,
                "detected_action_type": actions[context["stage"]],
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [],
                "stage_complete": True,
                "recommended_event": "pass_validation" if context["stage"] == "done" else events[context["stage"]],
                "required_artifacts": required,
                "explanation": "",
            }, ensure_ascii=False), "", super().complete(messages, settings).technical)
        if "структурированный handoff между этапами" in system:
            return super().complete(messages, settings)
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in system:
            if "Код этапа: execution" in system:
                content = "Готовый сайт:\n```artifact path=index.html\n<!doctype html><html lang=\"ru\"><title>Визитка</title></html>\n```"
            elif "Код этапа: validation" in system:
                content = "Артефакт index.html проверен, обязательные требования выполнены."
            elif "Код этапа: done" in system:
                content = "Сайт-визитка готов и прошёл проверку. Создан итоговый файл index.html."
            else:
                content = "План: создать один самодостаточный файл index.html и проверить его."
            result = super().complete(messages, settings)
            return AgentResult(content, result.reasoning_content, result.technical)
        return super().complete(messages, settings)


class FinalCompletionFailProvider(ArtifactLifecycleProvider):
    def complete(self, messages, settings):
        if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in messages[0]["content"] and "Код этапа: done" in messages[0]["content"]:
            self.calls.append((messages, settings))
            raise RuntimeError("final completion failed")
        return super().complete(messages, settings)


class FakeVoiceService:
    def __init__(self):
        self.started = False
        self.received = None

    def status(self):
        return {
            "phase": "ready", "ready": True, "message": "Whisper готов · тест",
            "model": "large-v3-turbo", "backend": "Vulkan", "language": "ru",
            "pid": 1234, "port": 8091, "reused": False,
        }

    def start(self):
        self.started = True

    def transcribe(self, audio, filename, content_type):
        self.received = (audio, filename, content_type)
        return {"text": "Распознанный текст", "language": "ru"}

    def stop(self):
        pass


class FakeMCPManager:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0
        self.tools_calls = 0
        self.running = False

    def status(self):
        return {
            "phase": "running" if self.running else "stopped",
            "connected": self.running,
            "message": "MCP-соединение установлено." if self.running else "MCP-сервер не запущен.",
            "error": None,
            "server": {"name": "filesystem-test", "version": "1.0.0"} if self.running else {"name": None, "version": None},
            "protocol_version": "2025-11-25" if self.running else None,
            "workspace": "C:\\test\\workspace",
            "tool_count": 1 if self.tools_calls else 0,
        }

    def start(self):
        if not self.running:
            self.start_calls += 1
            self.running = True
        return self.status()

    def list_tools(self):
        if not self.running:
            raise RuntimeError("MCP-соединение не запущено.")
        self.tools_calls += 1
        return {
            **self.status(),
            "tool_count": 1,
            "tools": [{
                "name": "read_text_file",
                "title": None,
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }],
        }

    def stop(self):
        if self.running:
            self.stop_calls += 1
            self.running = False
        return self.status()


class FakeWeatherMCPManager(FakeMCPManager):
    def __init__(self, running=False):
        super().__init__()
        self.running = running
        self.called = []

    def status(self):
        status = super().status()
        status["server"] = {"name": "open-meteo-test", "version": "1.0.0"} if self.running else {"name": None, "version": None}
        return status

    def list_tools(self):
        if not self.running:
            raise RuntimeError("MCP-соединение не запущено.")
        self.tools_calls += 1
        return {**self.status(), "tool_count": 1, "tools": self.tools_for_model()}

    def tools_for_model(self):
        return [{
            "name": "get_current_weather", "title": None, "description": "Текущая погода",
            "input_schema": {"type": "object", "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}}},
        }]

    def call_tool(self, name, arguments):
        self.called.append((name, arguments))
        return {"tool": name, "arguments": arguments, "is_error": False, "result": {"current": {"temperature_2m": 12}}}


class FakeOrchestrationMCPManager(FakeMCPManager):
    def __init__(self, server_name, tools, results, running=True):
        super().__init__()
        self.server_name = server_name
        self._tools = tools
        self.results = results
        self.running = running
        self.called = []

    def status(self):
        status = super().status()
        status["server"] = {"name": self.server_name, "version": "1.0.0"} if self.running else {"name": None, "version": None}
        status["tool_count"] = len(self._tools) if self.running else 0
        return status

    def list_tools(self):
        if not self.running:
            raise RuntimeError("MCP-соединение не запущено.")
        return {**self.status(), "tools": self.tools_for_model()}

    def tools_for_model(self):
        return [{
            "name": name, "title": None, "description": name,
            "input_schema": {"type": "object", "properties": {}},
        } for name in self._tools]

    def call_tool(self, name, arguments):
        self.called.append((name, arguments))
        return {"tool": name, "arguments": arguments, "is_error": False, "result": self.results[name]}


class FakeFilesystemWriteMCPManager(FakeMCPManager):
    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = workspace.resolve()
        self.called = []

    def status(self):
        status = super().status()
        status["workspace"] = str(self.workspace)
        return status

    def list_tools(self):
        if not self.running:
            raise RuntimeError("MCP-соединение не запущено.")
        self.tools_calls += 1
        return {**self.status(), "tool_count": 1, "tools": [{
            "name": "write_file", "title": "Write File", "description": "Write a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        }]}

    def call_tool(self, name, arguments):
        self.called.append((name, arguments))
        path = Path(arguments["path"])
        path.write_text(arguments["content"], encoding="utf-8")
        return {"tool": name, "arguments": arguments, "is_error": False,
                "result": {"content": f"Successfully wrote to {path}"}}


class AgentTests(unittest.TestCase):
    def test_agent_owns_context_and_uses_current_system_prompt(self):
        provider = FakeProvider()
        agent = Agent(provider)
        settings = AgentSettings(system_prompt="Ты технический специалист", temperature=0.2)
        history = [
            {"role": "user", "content": "Меня зовут Павел", "technical": {"request_status": "completed"}},
            {"role": "assistant", "content": "Запомнил", "technical": {"request_status": "completed"}},
            {"role": "event", "content": "Смена настроек"},
        ]
        agent.reply(history, "Как меня зовут?", settings)
        messages = provider.calls[0][0]
        self.assertEqual(messages[0], {"role": "system", "content": "Ты технический специалист"})
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Как меня зовут?")

    def test_failed_message_is_not_returned_to_context(self):
        provider = FakeProvider()
        history = [{"role": "user", "content": "Не доставлено", "technical": {"request_status": "failed"}}]
        Agent(provider).reply(history, "Новый запрос", AgentSettings())
        self.assertEqual(len(provider.calls[0][0]), 2)

    def test_deepseek_provider_collects_full_metadata(self):
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            prompt_cache_hit_tokens=9,
            prompt_cache_miss_tokens=3,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
        message = SimpleNamespace(content="Ответ", reasoning_content="Мысли")
        response = SimpleNamespace(
            id="abc", model="deepseek-v4-pro", system_fingerprint="fp-test",
            usage=usage, choices=[SimpleNamespace(message=message, finish_reason="stop")],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
        result = DeepSeekProvider(client).complete([], AgentSettings(model="deepseek-v4-pro", reasoning_enabled=True))
        self.assertEqual(result.reasoning_content, "Мысли")
        self.assertEqual(result.technical["usage"]["cached_input_tokens"], 9)
        self.assertEqual(result.technical["usage"]["reasoning_tokens"], 5)
        self.assertEqual(result.technical["usage"]["total_tokens"], 20)
        self.assertIn("elapsed_seconds", result.technical)

    def test_deepseek_provider_calls_mcp_tool_and_uses_result(self):
        class ToolCall:
            id = "call-weather"
            function = SimpleNamespace(name="get_current_weather", arguments='{"latitude":55,"longitude":83}')

            def model_dump(self, exclude_none=True):
                return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}

        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8,
                                prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=5,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
        first = SimpleNamespace(
            id="req-tool", model="deepseek-v4-flash", system_fingerprint=None, usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, reasoning_content="", tool_calls=[ToolCall()]), finish_reason="tool_calls")],
        )
        second = SimpleNamespace(
            id="req-final", model="deepseek-v4-flash", system_fingerprint=None, usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content="Наденьте куртку.", reasoning_content="", tool_calls=[]), finish_reason="stop")],
        )
        requests = []
        responses = iter([first, second])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: requests.append(kwargs) or next(responses)
        )))
        calls = []
        result = DeepSeekProvider(client).complete_with_tools(
            [{"role": "user", "content": "Что надеть?"}], AgentSettings(),
            [{"name": "get_current_weather", "description": "Погода", "input_schema": {"type": "object"}}],
            lambda name, arguments: calls.append((name, arguments)) or {"is_error": False, "result": {"temperature": 12}},
        )
        self.assertEqual(result.content, "Наденьте куртку.")
        self.assertEqual(calls[0][0], "get_current_weather")
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertIn('"temperature": 12', requests[1]["messages"][-1]["content"])
        self.assertEqual(result.technical["usage"]["total_tokens"], 16)
        self.assertEqual(result.technical["mcp_tool_calls"][0]["name"], "get_current_weather")

    def test_deepseek_provider_composes_weather_then_write_file(self):
        class ToolCall:
            def __init__(self, call_id, name, arguments):
                self.id = call_id
                self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False))

            def model_dump(self, exclude_none=True):
                return {"id": self.id, "type": "function", "function": {
                    "name": self.function.name, "arguments": self.function.arguments,
                }}

        usage = SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6,
                                prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=4,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=0))

        def response(request_id, content, tool_calls, finish_reason):
            return SimpleNamespace(
                id=request_id, model="deepseek-v4-flash", system_fingerprint=None, usage=usage,
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=content, reasoning_content="", tool_calls=tool_calls,
                ), finish_reason=finish_reason)],
            )

        first = response("req-weather", None, [ToolCall(
            "call-weather", "get_current_weather", {"latitude": 55.03, "longitude": 82.92},
        )], "tool_calls")
        second = response("req-write", None, [ToolCall(
            "call-write", "write_file", {
                "path": "weather.md", "content": "# Отчёт\n\nТемпература: 12 °C. Лёгкая куртка.\n",
            },
        )], "tool_calls")
        third = response("req-final", "Отчёт сохранён.", [], "stop")
        requests = []
        responses = iter([first, second, third])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: requests.append(deepcopy(kwargs)) or next(responses)
        )))
        calls = []

        def execute(name, arguments):
            calls.append((name, arguments))
            if name == "get_current_weather":
                return {"is_error": False, "result": {"current": {"temperature_2m": 12}}}
            return {"is_error": False, "result": {"relative_path": "reports/weather.md"}}

        result = DeepSeekProvider(client).complete_with_tools(
            [{"role": "user", "content": "Узнай погоду и сохрани отчёт"}], AgentSettings(),
            [
                {"name": "get_current_weather", "description": "Погода", "input_schema": {"type": "object"}},
                {"name": "write_file", "description": "Запись", "input_schema": {"type": "object"}},
            ], execute,
        )

        self.assertEqual([item[0] for item in calls], ["get_current_weather", "write_file"])
        self.assertIn("Температура: 12 °C", calls[1][1]["content"])
        self.assertIn('"temperature_2m": 12', requests[1]["messages"][-1]["content"])
        self.assertEqual(requests[2]["messages"][-1]["role"], "tool")
        self.assertEqual(result.content, "Отчёт сохранён.")
        self.assertEqual([item["name"] for item in result.technical["mcp_tool_calls"]],
                         ["get_current_weather", "write_file"])

    def test_settings_reject_unsupported_seed_and_model_is_whitelisted(self):
        settings = AgentSettings.from_dict(AgentSettings().to_dict() | {"seed": 42})
        self.assertFalse(hasattr(settings, "seed"))
        with self.assertRaises(ValueError):
            AgentSettings.from_dict(AgentSettings().to_dict() | {"model": "unknown"})

    def test_dialogue_totals_sum_only_assistant_usage(self):
        messages = [
            {"role": "user", "technical": {"usage": {"total_tokens": 999}}},
            {"role": "assistant", "technical": {"request_status": "completed", "usage": {
                "input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                "cached_input_tokens": 7, "uncached_input_tokens": 3, "reasoning_tokens": 4,
            }}},
            {"role": "assistant", "technical": {"request_status": "failed", "usage": {"total_tokens": 500}}},
        ]
        totals = dialogue_token_totals(messages)
        self.assertEqual(totals["request_count"], 1)
        self.assertEqual(totals["total_tokens"], 30)
        self.assertEqual(totals["cached_input_tokens"], 7)
        self.assertEqual(totals["reasoning_tokens"], 4)

    def test_supported_models_publish_one_million_token_context_windows(self):
        self.assertEqual(DEEPSEEK_CONTEXT_WINDOWS["deepseek-v4-flash"], 1_000_000)
        self.assertEqual(DEEPSEEK_CONTEXT_WINDOWS["deepseek-v4-pro"], 1_000_000)

    def test_summary_is_injected_into_system_prompt(self):
        provider = FakeProvider()
        Agent(provider).reply([], "Продолжай", AgentSettings(), summary="Пользователя зовут Павел.")
        sent = provider.calls[0][0]
        self.assertIn("СЖАТАЯ ПАМЯТЬ", sent[0]["content"])
        self.assertIn("Пользователя зовут Павел", sent[0]["content"])
        self.assertEqual(sent[-1], {"role": "user", "content": "Продолжай"})

    def test_task_state_is_injected_into_system_prompt(self):
        provider = FakeProvider()
        Agent(provider).reply([], "Продолжай", AgentSettings(), task_state={
            "description": "Сделать форму регистрации",
            "stage": "execution",
            "current_step": "Проверка пароля",
            "expected_action": "Изменить серверную валидацию",
            "plan": "Сервер, затем интерфейс",
            "activity": "active",
        })
        system = provider.calls[0][0][0]["content"]
        self.assertIn("АВТОРИТЕТНОЕ СОСТОЯНИЕ ТЕКУЩЕЙ ЗАДАЧИ", system)
        self.assertIn("Код этапа: execution", system)
        self.assertIn("Проверка пароля", system)

    def test_every_task_stage_overrides_conflicting_context_in_the_final_system_block(self):
        conflicts = {
            "planning": "done",
            "execution": "planning",
            "validation": "execution",
            "done": "validation",
        }
        for stage, old_stage in conflicts.items():
            with self.subTest(stage=stage):
                provider = FakeProvider()
                Agent(provider).reply(
                    [{
                        "role": "assistant",
                        "content": f"Старый этап был {old_stage}",
                        "technical": {"request_status": "completed"},
                    }],
                    "Какой сейчас этап?",
                    AgentSettings(),
                    summary=f"Ранее этап считался {old_stage}",
                    facts=[{"key": "goal.stage", "value": old_stage}],
                    task_state={
                        "description": "Проверка приоритета",
                        "stage": stage,
                        "current_step": "Текущий шаг",
                        "expected_action": "Следующее действие",
                        "plan": "Краткий план",
                        "activity": "active",
                    },
                )
                system = provider.calls[0][0][0]["content"]
                self.assertIn(f"Код этапа: {stage}", system)
                self.assertIn(f"отвечай точным значением: {stage}", system)
                self.assertGreater(
                    system.rfind("АВТОРИТЕТНОЕ СОСТОЯНИЕ ТЕКУЩЕЙ ЗАДАЧИ"),
                    system.rfind("STICKY FACTS"),
                )
                self.assertIn("Разрешённые сервером события", system)
                self.assertTrue(system.endswith("переходы выполняет только серверный автомат."))


class TaskStateTests(unittest.TestCase):
    def setUp(self):
        self.conversation = {"created_at": "2026-09-19T00:00:00+07:00"}
        ensure_task_state(self.conversation)

    def test_happy_path_and_rollbacks_are_controlled_by_events(self):
        self.assertEqual(allowed_task_events(self.conversation), ["approve_plan", "pause"])
        apply_task_event(self.conversation, "approve_plan")
        self.assertEqual(self.conversation["task_state"]["stage"], "execution")
        apply_task_event(self.conversation, "return_to_planning")
        self.assertEqual(self.conversation["task_state"]["stage"], "planning")
        apply_task_event(self.conversation, "approve_plan")
        apply_task_event(self.conversation, "complete_execution")
        self.assertEqual(self.conversation["task_state"]["stage"], "validation")
        apply_task_event(self.conversation, "validation_failed")
        self.assertEqual(self.conversation["task_state"]["stage"], "execution")
        apply_task_event(self.conversation, "complete_execution")
        apply_task_event(self.conversation, "pass_validation")
        self.assertEqual(self.conversation["task_state"]["stage"], "done")
        self.assertEqual(allowed_task_events(self.conversation), ["pause"])

    def test_invalid_jump_does_not_change_state(self):
        with self.assertRaises(TaskTransitionError):
            apply_task_event(self.conversation, "pass_validation")
        self.assertEqual(self.conversation["task_state"]["stage"], "planning")
        self.assertEqual(self.conversation["task_state"]["transition_history"], [])

    def test_pause_resume_preserves_exact_stage_and_step(self):
        apply_task_event(self.conversation, "approve_plan")
        update_task_state(self.conversation, {
            "current_step": "Реализовать обработчик",
            "expected_action": "Запустить тесты",
        })
        apply_task_event(self.conversation, "pause")
        with self.assertRaises(TaskTransitionError):
            apply_task_event(self.conversation, "complete_execution")
        apply_task_event(self.conversation, "resume")
        state = self.conversation["task_state"]
        self.assertEqual((state["stage"], state["current_step"]), ("execution", "Реализовать обработчик"))
        self.assertIn("complete_execution", allowed_task_events(state))

    def test_stage_cannot_be_patched_and_transition_mode_does_not_bypass_graph(self):
        with self.assertRaisesRegex(ValueError, "только разрешённым событием"):
            update_task_state(self.conversation, {"stage": "done"})
        update_task_state(self.conversation, {"transition_mode": "automatic"})
        self.assertEqual(self.conversation["task_state"]["transition_mode"], "automatic")
        with self.assertRaises(TaskTransitionError):
            apply_task_event(self.conversation, "pass_validation")
        self.assertEqual(self.conversation["task_state"]["stage"], "planning")
        with self.assertRaisesRegex(ValueError, "только явными событиями"):
            update_task_state(self.conversation, {"activity": "paused"})

    def test_report_reads_authoritative_state(self):
        apply_task_event(self.conversation, "approve_plan")
        report = task_state_report(self.conversation)
        self.assertIn("Выполнение (execution)", report)
        self.assertIn("Завершить выполнение", report)


class TaskPolicyTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "stage": "planning", "activity": "active", "transition_mode": "manual",
        }
        self.invariants = {"rules": [
            {"id": "system:protect-secrets", "text": "Не раскрывать секреты", "layer": "system"},
            {"id": "project:long-id:rule-id", "text": "Не использовать MySQL", "layer": "project"},
        ]}

    def test_generation_guidance_requests_plain_answer_and_contains_all_invariants(self):
        guidance = generation_guidance(self.state, self.invariants)
        self.assertIn("Верни обычный ответ пользователю", guidance)
        self.assertNotIn('"action_type"', guidance)
        self.assertIn("I1: Не раскрывать секреты", guidance)
        self.assertIn("I2: Не использовать MySQL", guidance)

    def test_validator_accepts_short_refs_and_markdown_fence(self):
        payload = json.dumps({
            "allowed": True, "detected_action_type": "planning",
            "checked_invariant_ids": ["I1", "I2"],
            "violated_invariant_ids": [], "explanation": "",
        }, ensure_ascii=False)
        parsed = parse_validation_result(f"```json\n{payload}\n```", self.state, self.invariants)
        self.assertTrue(parsed["allowed"])
        self.assertEqual(parsed["checked_invariant_ids"], [
            "system:protect-secrets", "project:long-id:rule-id",
        ])

    def test_incomplete_allowed_analysis_is_not_blocked_by_validator_flag(self):
        verdict = json.dumps({
            "allowed": False,
            "detected_action_type": "analysis",
            "checked_invariant_ids": ["I1", "I2"],
            "violated_invariant_ids": [],
            "stage_complete": False,
            "recommended_event": None,
            "required_artifacts": [],
            "explanation": "Ответ не завершает этап planning.",
        }, ensure_ascii=False)
        parsed = parse_validation_result(verdict, self.state, self.invariants)
        self.assertTrue(parsed["allowed"])
        self.assertFalse(parsed["validator_allowed"])
        self.assertFalse(parsed["stage_complete"])

    def test_action_matrix_keeps_stage_boundaries_for_ordinary_tasks(self):
        allowed_actions = {
            "planning": "analysis",
            "execution": "implementation",
            "validation": "validation_result",
            "done": "result_explanation",
        }
        blocked_actions = {
            "planning": "implementation",
            "execution": "planning",
            "validation": "implementation",
            "done": "implementation",
        }
        for stage, action in allowed_actions.items():
            with self.subTest(stage=stage, action=action, expected="allowed"):
                state = {**self.state, "stage": stage}
                verdict = json.dumps({
                    "allowed": False,
                    "detected_action_type": action,
                    "checked_invariant_ids": ["I1", "I2"],
                    "violated_invariant_ids": [],
                    "stage_complete": False,
                    "recommended_event": None,
                    "required_artifacts": [],
                    "explanation": "Промежуточный ответ.",
                }, ensure_ascii=False)
                parsed = parse_validation_result(verdict, state, self.invariants)
                self.assertTrue(parsed["allowed"])
                self.assertFalse(parsed["validator_allowed"])

        for stage, action in blocked_actions.items():
            with self.subTest(stage=stage, action=action, expected="blocked"):
                state = {**self.state, "stage": stage}
                verdict = json.dumps({
                    "allowed": True,
                    "detected_action_type": action,
                    "checked_invariant_ids": ["I1", "I2"],
                    "violated_invariant_ids": [],
                    "stage_complete": False,
                    "recommended_event": None,
                    "required_artifacts": [],
                    "explanation": "",
                }, ensure_ascii=False)
                parsed = parse_validation_result(verdict, state, self.invariants)
                self.assertFalse(parsed["allowed"])
                self.assertTrue(parsed["validator_allowed"])

    def test_known_invariant_violation_still_blocks_allowed_action(self):
        verdict = json.dumps({
            "allowed": True,
            "detected_action_type": "analysis",
            "checked_invariant_ids": ["I1", "I2"],
            "violated_invariant_ids": ["I1"],
            "stage_complete": False,
            "recommended_event": None,
            "required_artifacts": [],
            "explanation": "Черновик раскрывает секрет.",
        }, ensure_ascii=False)
        parsed = parse_validation_result(verdict, self.state, self.invariants)
        self.assertFalse(parsed["allowed"])
        self.assertEqual(parsed["violated_invariant_ids"], ["system:protect-secrets"])

    def test_unknown_validator_reference_blocks_without_parser_failure(self):
        verdict = json.dumps({
            "allowed": True, "detected_action_type": "planning",
            "checked_invariant_ids": ["I1", "I2"],
            "violated_invariant_ids": ["invented-rule"], "explanation": "",
        }, ensure_ascii=False)
        parsed = parse_validation_result(verdict, self.state, self.invariants)
        self.assertFalse(parsed["allowed"])
        self.assertEqual(parsed["unknown_invariant_refs"], ["invented-rule"])

    def test_validator_must_confirm_every_active_invariant(self):
        verdict = json.dumps({
            "allowed": True, "detected_action_type": "planning",
            "checked_invariant_ids": ["I1"], "violated_invariant_ids": [], "explanation": "",
        }, ensure_ascii=False)
        with self.assertRaisesRegex(PolicyValidationError, "всех активных инвариантов"):
            parse_validation_result(verdict, self.state, self.invariants)

    def test_policy_context_assigns_stable_short_refs(self):
        context = policy_context(self.state, self.invariants)
        self.assertEqual([item["ref"] for item in context["invariants"]], ["I1", "I2"])


class WhisperLifecycleTests(unittest.TestCase):
    def make_service(self, root: Path) -> WhisperService:
        server = root / "tools" / "whisper" / "bin" / "whisper-server.exe"
        model = root / "models" / "ggml-large-v3-turbo.bin"
        server.parent.mkdir(parents=True)
        model.parent.mkdir(parents=True)
        server.touch()
        model.touch()
        with mock.patch.dict(os.environ, {
            "WHISPER_SERVER_PATH": str(server),
            "WHISPER_MODEL_PATH": str(model),
            "WHISPER_PORT": "8091",
        }):
            return WhisperService(root, autostart=False)

    def test_imported_flask_app_does_not_autostart_real_whisper(self):
        status = app_module.app.extensions["whisper_service"].status()
        self.assertEqual(status["phase"], "stopped")
        self.assertIsNone(status["pid"])

    def test_status_exposes_server_pid_port_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory))
            service._server_pid = 4321
            service._phase = "ready"
            service._reused = True
            with mock.patch.object(service, "_pid_running", return_value=True):
                status = service.status()
            self.assertEqual((status["pid"], status["port"], status["reused"]), (4321, 8091, True))
            service._server_pid = None

    def test_runtime_registry_reuses_matching_project_server(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory))
            service._write_runtime(4321, [os.getpid()])
            with (
                mock.patch.object(service, "_pid_running", return_value=True),
                mock.patch.object(service, "_process_path", return_value=str(service.server_path)),
            ):
                self.assertEqual(service._find_reusable_server(), 4321)

    def test_foreign_listener_blocks_second_whisper(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory))
            with (
                mock.patch.object(service, "_find_reusable_server", return_value=None),
                mock.patch.object(service, "_listener_pid", return_value=9876),
            ):
                service._start_worker()
            self.assertEqual(service.status()["phase"], "error")
            self.assertIn("второй Whisper не запущен", service.status()["message"])

    def test_last_client_stops_adopted_server_and_removes_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(Path(directory))
            service._server_pid = 4321
            service._write_runtime(4321, [os.getpid()])
            with (
                mock.patch.object(service, "_pid_running", return_value=True),
                mock.patch.object(service, "_terminate_server_pid") as terminate,
            ):
                service.stop()
            terminate.assert_called_once_with(4321)
            self.assertFalse(service.runtime_path.exists())


class StorageTests(unittest.TestCase):
    def test_conversation_survives_new_storage_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = JsonStorage(root)
            conversation = first.create_conversation("Проверка перезапуска")
            conversation["messages"].append({"role": "user", "content": "Запомни 17"})
            first.save_conversation(conversation)
            restored = JsonStorage(root).get_conversation(conversation["id"])
            self.assertEqual(restored["messages"][0]["content"], "Запомни 17")
            self.assertEqual(restored["context_management"]["mode"], "full")
        self.assertEqual(restored["summaries"], [])

    def test_task_state_survives_restart_and_legacy_dialog_gets_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = JsonStorage(root)
            conversation = storage.create_conversation("Состояние")
            update_task_state(conversation, {
                "description": "Добавить профиль",
                "current_step": "Изменить API",
                "expected_action": "Закончить серверную часть",
                "plan": "API, затем UI",
            })
            apply_task_event(conversation, "approve_plan")
            update_task_state(conversation, {
                "current_step": "Изменить API",
                "expected_action": "Закончить серверную часть",
            })
            apply_task_event(conversation, "pause")
            storage.save_conversation(conversation)
            restored = JsonStorage(root).get_conversation(conversation["id"])
            self.assertEqual(restored["task_state"]["stage"], "execution")
            self.assertEqual(restored["task_state"]["current_step"], "Изменить API")
            self.assertEqual(restored["task_state"]["activity"], "paused")
            restored.pop("task_state")
            self.assertEqual(ensure_task_state(restored)["stage"], "planning")
            self.assertEqual(restored["task_state"]["activity"], "active")

    def test_legacy_branching_mode_migrates_to_full_without_losing_tree_state(self):
        conversation = {
            "messages": [{"id": "root", "role": "user", "content": "Тест", "parent_id": None}],
            "context_management": {"mode": "branching", "active_leaf_id": "root"},
        }
        context = ensure_context_management(conversation)
        self.assertEqual(context["mode"], "full")
        self.assertEqual(context["active_leaf_id"], "root")
        self.assertEqual(conversation["messages"][0]["id"], "root")

    def test_preset_keeps_valid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonStorage(Path(directory))
            preset = PresetManager(storage).create({
                "name": "Технический специалист",
                "description": "Точный ответ",
                "settings": AgentSettings(temperature=0.1).to_dict(),
            })
            self.assertEqual(preset["settings"]["temperature"], 0.1)
            self.assertEqual(JsonStorage(Path(directory)).load_presets_document()["presets"][0]["name"], "Технический специалист")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = FakeProvider()
        self.voice = FakeVoiceService()
        self.mcp = FakeMCPManager()
        self.app = create_app(Path(self.temp.name), Agent(self.provider), self.voice, self.mcp)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def create_conversation(self):
        return self.client.post("/api/conversations", json={"title": "Новый диалог"}).get_json()["conversation"]

    def test_index_is_chat_not_temperature_lab(self):
        text = self.client.get("/").get_data(as_text=True)
        self.assertIn("DeepSeek Agent", text)
        self.assertIn("Новый диалог", text)
        self.assertIn("Нет — выключен", text)
        self.assertIn("Да — включён", text)
        self.assertIn("Запрос отправлен · ожидаем DeepSeek", text)
        self.assertIn("Надиктовать сообщение", text)
        self.assertNotIn('<option value="branching">', text)
        self.assertIn("Ветвление доступно во всех режимах", text)
        self.assertIn('id="facts-help-button"', text)
        self.assertIn('id="facts-help-dialog"', text)
        self.assertIn('id="task-state-title"', text)
        self.assertIn('id="task-activity-toggle"', text)
        self.assertIn('id="task-transition-actions"', text)
        self.assertIn('id="task-state-show"', text)
        self.assertIn('id="task-transition-mode"', text)
        self.assertIn('id="task-autopilot-stop"', text)
        self.assertNotIn("Свободные этапы · День 13", text)
        self.assertIn('id="invariants-title"', text)
        self.assertIn('id="invariant-scope"', text)
        self.assertIn('id="mcp-start"', text)
        self.assertIn('id="mcp-tools"', text)
        self.assertIn('id="mcp-stop"', text)
        self.assertIn('id="weather-mcp-start"', text)
        self.assertIn("Open‑Meteo MCP", text)
        self.assertIn("только папку проекта <code>workspace</code>", text)
        self.assertIn("Разрешённые категории", text)
        self.assertIn("preferences.language", text)
        self.assertNotIn("Запустить 3 температуры", text)
        self.assertNotIn('id="message-input" maxlength="50000" rows="1" placeholder="Напишите сообщение…" required', text)

    def test_state_exposes_model_context_windows(self):
        provider = self.client.get("/api/state").get_json()["provider"]
        self.assertEqual(provider["context_windows"]["deepseek-v4-flash"], 1_000_000)
        self.assertEqual(provider["context_windows"]["deepseek-v4-pro"], 1_000_000)

    def test_mcp_long_lived_session_lists_tools_and_prevents_duplicate_start(self):
        initial = self.client.get("/api/mcp/status")
        self.assertEqual(initial.status_code, 200)
        self.assertFalse(initial.get_json()["mcp"]["connected"])

        started = self.client.post("/api/mcp/start")
        repeated = self.client.post("/api/mcp/start")
        self.assertEqual(started.status_code, 200)
        self.assertEqual(repeated.status_code, 200)
        self.assertTrue(started.get_json()["mcp"]["connected"])
        self.assertEqual(self.mcp.start_calls, 1)

        listed = self.client.get("/api/mcp/tools")
        payload = listed.get_json()["mcp"]
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(payload["server"]["name"], "filesystem-test")
        self.assertEqual(payload["protocol_version"], "2025-11-25")
        self.assertEqual(payload["tool_count"], 1)
        self.assertEqual(payload["tools"][0]["name"], "read_text_file")
        self.assertEqual(payload["tools"][0]["input_schema"]["type"], "object")

        stopped = self.client.post("/api/mcp/stop")
        self.assertEqual(stopped.status_code, 200)
        self.assertFalse(stopped.get_json()["mcp"]["connected"])
        self.assertEqual(self.mcp.stop_calls, 1)

        unavailable = self.client.get("/api/mcp/tools")
        self.assertEqual(unavailable.status_code, 409)
        self.assertFalse(unavailable.get_json()["mcp"]["connected"])

    def test_weather_mcp_has_separate_lifecycle_routes(self):
        weather = FakeWeatherMCPManager()
        app = create_app(Path(self.temp.name), Agent(self.provider), self.voice, self.mcp, weather)
        app.config.update(TESTING=True)
        client = app.test_client()
        started = client.post("/api/weather-mcp/start")
        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.get_json()["mcp"]["server"]["name"], "open-meteo-test")
        self.assertEqual(started.get_json()["mcp"]["tools"][0]["name"], "get_current_weather")
        self.assertEqual(client.post("/api/weather-mcp/stop").status_code, 200)

    def test_chat_uses_connected_weather_mcp_and_saves_audit(self):
        provider = ToolAwareFakeProvider()
        weather = FakeWeatherMCPManager(running=True)
        app = create_app(Path(self.temp.name), Agent(provider), self.voice, self.mcp, weather)
        app.config.update(TESTING=True)
        client = app.test_client()
        conversation = client.post("/api/conversations", json={"title": "Погода"}).get_json()["conversation"]
        response = client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Что надеть в Новосибирске?", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(weather.called[0][0], "get_current_weather")
        assistant = response.get_json()["conversation"]["visible_messages"][-1]
        self.assertIn("лёгкая куртка", assistant["content"])
        self.assertEqual(assistant["technical"]["mcp_tool_calls"][0]["name"], "get_current_weather")
        self.assertEqual(self.mcp.start_calls, 0)

    def test_day20_routes_dependent_calls_across_two_mcp_servers_and_tracks_activity(self):
        mediawiki = FakeOrchestrationMCPManager(
            "mediawiki-test", ["search-page", "update-page"],
            {"search-page": {"country_codes": ["BRA", "ARG"]}},
        )
        worldbank = FakeOrchestrationMCPManager(
            "worldbank-test", ["worldbank_get_data"],
            {"worldbank_get_data": {"observations": [{"country": "BRA", "value": 1}]}},
        )
        app = create_app(
            Path(self.temp.name), Agent(OrchestrationFakeProvider()), self.voice, self.mcp,
            mediawiki_mcp_manager=mediawiki, worldbank_mcp_manager=worldbank,
        )
        app.config.update(TESTING=True)
        client = app.test_client()
        conversation = client.post("/api/conversations", json={"title": "Оркестрация"}).get_json()["conversation"]
        activity_id = "a" * 32

        response = client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Найди крупнейшие страны Южной Америки и сравни население.",
            "mcp_activity_id": activity_id,
            "settings": AgentSettings().to_dict(),
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item[0] for item in mediawiki.called], ["search-page"])
        self.assertEqual(worldbank.called[0][1]["countries"], ["BRA", "ARG"])
        assistant = response.get_json()["conversation"]["visible_messages"][-1]
        calls = assistant["technical"]["mcp_tool_calls"]
        self.assertEqual([(item["server"], item["name"]) for item in calls], [
            ("mediawiki", "search-page"), ("worldbank", "worldbank_get_data"),
        ])
        activity = client.get(f"/api/mcp/activity/{activity_id}").get_json()["activity"]
        self.assertEqual(activity["phase"], "completed")
        self.assertEqual([(item["server"], item["name"], item["status"]) for item in activity["calls"]], [
            ("mediawiki", "search-page", "completed"),
            ("worldbank", "worldbank_get_data", "completed"),
        ])

    def test_day20_mcp_lifecycle_routes_are_independent(self):
        mediawiki = FakeOrchestrationMCPManager("mediawiki-test", ["search-page"], {}, running=False)
        worldbank = FakeOrchestrationMCPManager("worldbank-test", ["worldbank_get_data"], {}, running=False)
        app = create_app(
            Path(self.temp.name), Agent(self.provider), self.voice, self.mcp,
            mediawiki_mcp_manager=mediawiki, worldbank_mcp_manager=worldbank,
        )
        app.config.update(TESTING=True)
        client = app.test_client()
        self.assertEqual(client.post("/api/orchestration-mcp/mediawiki/start").status_code, 200)
        self.assertEqual(client.post("/api/orchestration-mcp/worldbank/start").status_code, 200)
        self.assertEqual(client.get("/api/orchestration-mcp/mediawiki/tools").get_json()["mcp"]["tools"][0]["name"], "search-page")
        self.assertEqual(client.get("/api/orchestration-mcp/worldbank/tools").get_json()["mcp"]["tools"][0]["name"], "worldbank_get_data")

    def test_file_export_intent_requires_explicit_positive_command(self):
        self.assertTrue(app_module.requests_file_export("Узнай погоду и сохрани отчёт в файл"))
        self.assertTrue(app_module.requests_file_export("Создай документ с прогнозом"))
        self.assertFalse(app_module.requests_file_export("Что надеть сегодня?"))
        self.assertFalse(app_module.requests_file_export("Покажи прогноз, но не сохраняй его"))

    def test_weather_report_pipeline_starts_filesystem_and_writes_unique_report(self):
        provider = WeatherReportPipelineProvider()
        weather = FakeWeatherMCPManager(running=True)
        workspace = Path(self.temp.name) / "mcp-workspace"
        filesystem = FakeFilesystemWriteMCPManager(workspace)
        app = create_app(Path(self.temp.name), Agent(provider), self.voice, filesystem, weather)
        app.config.update(TESTING=True)
        client = app.test_client()
        conversation = client.post("/api/conversations", json={"title": "Погодный отчёт"}).get_json()["conversation"]

        response = client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Узнай погоду в Новосибирске, посоветуй одежду и сохрани отчёт в файл.",
            "settings": AgentSettings().to_dict(),
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(filesystem.start_calls, 1)
        self.assertEqual([item[0] for item in weather.called], ["get_current_weather"])
        self.assertEqual([item[0] for item in filesystem.called], ["write_file"])
        written_path = Path(filesystem.called[0][1]["path"])
        self.assertEqual(written_path.parent, workspace / "reports")
        self.assertRegex(written_path.name, r"^weather-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.md$")
        self.assertTrue(written_path.is_file())
        self.assertIn("Температура: 12 °C", written_path.read_text(encoding="utf-8"))
        assistant = response.get_json()["conversation"]["visible_messages"][-1]
        self.assertEqual(
            [item["name"] for item in assistant["technical"]["mcp_tool_calls"]],
            ["get_current_weather", "write_file"],
        )
        self.assertEqual(assistant["technical"]["policy_audit"]["detected_action_type"], "file_export")

    def test_blocked_file_export_removes_report_created_before_postvalidation(self):
        provider = BlockedWeatherReportPipelineProvider()
        weather = FakeWeatherMCPManager(running=True)
        workspace = Path(self.temp.name) / "blocked-workspace"
        filesystem = FakeFilesystemWriteMCPManager(workspace)
        app = create_app(Path(self.temp.name), Agent(provider), self.voice, filesystem, weather)
        app.config.update(TESTING=True)
        client = app.test_client()
        conversation = client.post("/api/conversations", json={"title": "Заблокированный отчёт"}).get_json()["conversation"]

        response = client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Сохрани погодный отчёт в файл.", "settings": AgentSettings().to_dict(),
        })

        self.assertEqual(response.status_code, 200)
        assistant = response.get_json()["conversation"]["visible_messages"][-1]
        self.assertEqual(assistant["technical"]["request_status"], "blocked")
        self.assertIn("Ответ заблокирован", assistant["content"])
        self.assertEqual(list((workspace / "reports").glob("*.md")), [])

    def test_task_lifecycle_controls_transitions_pause_and_resume(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        changed = self.client.patch(f"{endpoint}/task-state", json={
            "description": "Сделать форму регистрации",
            "current_step": "Согласовать план",
            "expected_action": "Утвердить план",
            "plan": "Backend, frontend, проверка",
        })
        self.assertEqual(changed.status_code, 200)
        state = changed.get_json()["conversation"]["task_state"]
        self.assertEqual(state["stage"], "planning")
        self.assertEqual(state["allowed_events"], ["approve_plan", "pause"])

        invalid = self.client.post(f"{endpoint}/task-state/events", json={"event": "pass_validation"})
        self.assertEqual(invalid.status_code, 409)
        self.assertEqual(invalid.get_json()["conversation"]["task_state"]["stage"], "planning")

        approved = self.client.post(f"{endpoint}/task-state/events", json={"event": "approve_plan"})
        self.assertEqual(approved.status_code, 200)
        state = approved.get_json()["conversation"]["task_state"]
        self.assertEqual(state["stage"], "execution")
        self.assertEqual(state["current_step"], "Выполнить утверждённый план")

        paused_response = self.client.post(f"{endpoint}/task-state/events", json={"event": "pause"})
        self.assertEqual(paused_response.status_code, 200)
        paused_state = paused_response.get_json()["conversation"]["task_state"]
        self.assertEqual(paused_state["activity"], "paused")
        self.assertEqual(paused_state["allowed_events"], ["resume"])

        blocked = self.client.post(f"{endpoint}/messages", json={
            "content": "Продолжай", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.get_json()["conversation"]["messages"], [])
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(
            self.client.post(f"{endpoint}/branches", json={"checkpoint_id": "missing"}).status_code,
            409,
        )
        self.assertEqual(
            self.client.post(f"{endpoint}/extract-project-memory", json={}).status_code,
            409,
        )

        resumed = self.client.post(f"{endpoint}/task-state/events", json={"event": "resume"})
        self.assertEqual(resumed.get_json()["conversation"]["task_state"]["current_step"], "Выполнить утверждённый план")
        sent = self.client.post(f"{endpoint}/messages", json={
            "content": "Продолжай", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(sent.status_code, 200)
        self.assertIn("Выполнить утверждённый план", self.provider.calls[0][0][0]["content"])
        self.assertEqual(
            sent.get_json()["conversation"]["messages"][-1]["technical"]["task_state"]["stage"],
            "execution",
        )
        self.assertGreaterEqual(len(resumed.get_json()["conversation"]["task_state"]["transition_history"]), 4)

    def test_stage_transition_creates_handoff_and_excludes_previous_stage_history(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        planning = self.client.post(f"{endpoint}/messages", json={
            "content": "Составь план формы", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(planning.status_code, 200)
        planning_user = planning.get_json()["conversation"]["visible_messages"][0]
        planning_run_id = planning_user["technical"]["task_state"]["stage_run_id"]

        transitioned = self.client.post(f"{endpoint}/task-state/events", json={"event": "approve_plan"})
        self.assertEqual(transitioned.status_code, 200)
        transitioned_conversation = transitioned.get_json()["conversation"]
        state = transitioned_conversation["task_state"]
        self.assertEqual(state["stage"], "execution")
        self.assertNotEqual(state["stage_run_id"], planning_run_id)
        self.assertEqual(len(transitioned_conversation["task_handoffs"]), 1)
        self.assertEqual(len(transitioned_conversation["active_task_handoffs"]), 1)
        handoff = transitioned_conversation["active_task_handoffs"][0]
        self.assertEqual((handoff["source_stage"], handoff["target_stage"]), ("planning", "execution"))
        self.assertIn("Выполнить согласованный план", handoff["approved_plan"])
        self.assertTrue(handoff["source_exchange_ids"])
        self.assertIn("Выполнить согласованный план", state["plan"])
        stale_branch = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": planning_user["id"]})
        self.assertEqual(stale_branch.status_code, 409)
        self.assertIn("предыдущего этапа", stale_branch.get_json()["error"])

        executed = self.client.post(f"{endpoint}/messages", json={
            "content": "Выполняй первый пункт", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(executed.status_code, 200)
        sent = policy_generation_calls(self.provider)[-1][0]
        self.assertIn("АВТОРИТЕТНЫЙ HANDOFF ТЕКУЩЕГО ЭТАПА", sent[0]["content"])
        self.assertIn("Выполнить согласованный план", sent[0]["content"])
        self.assertNotIn("Составь план формы", [item["content"] for item in sent[1:]])
        result = executed.get_json()["conversation"]
        self.assertEqual(result["current_stage_exchange_count"], 1)
        self.assertEqual(result["visible_messages"][-1]["technical"]["task_handoff_context"]["handoff_count"], 1)

    def test_failed_handoff_does_not_change_stage(self):
        provider = HandoffFailProvider()
        self.app.extensions["chat_agent"].provider = provider
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.post(f"{endpoint}/messages", json={
            "content": "Обсудим план", "settings": AgentSettings().to_dict(),
        })
        failed = self.client.post(f"{endpoint}/task-state/events", json={"event": "approve_plan"})
        self.assertEqual(failed.status_code, 502)
        stored = self.client.get(endpoint).get_json()["conversation"]
        self.assertEqual(stored["task_state"]["stage"], "planning")
        self.assertEqual(stored["task_handoffs"], [])

    def test_validation_rollback_keeps_plan_and_feedback_but_not_old_execution_dialogue(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.post(f"{endpoint}/messages", json={
            "content": "Согласуем исходный план", "settings": AgentSettings().to_dict(),
        })
        self.client.post(f"{endpoint}/task-state/events", json={"event": "approve_plan"})
        self.client.post(f"{endpoint}/messages", json={
            "content": "Первая реализация", "settings": AgentSettings().to_dict(),
        })
        self.client.post(f"{endpoint}/task-state/events", json={"event": "complete_execution"})
        self.client.post(f"{endpoint}/messages", json={
            "content": "Проверь первую реализацию", "settings": AgentSettings().to_dict(),
        })
        rolled_back = self.client.post(f"{endpoint}/task-state/events", json={"event": "validation_failed"})
        self.assertEqual(rolled_back.status_code, 200)
        state = rolled_back.get_json()["conversation"]
        self.assertEqual(state["task_state"]["stage"], "execution")
        self.assertEqual(len(state["active_task_handoffs"]), 2)
        self.assertEqual(
            [(item["source_stage"], item["target_stage"]) for item in state["active_task_handoffs"]],
            [("planning", "execution"), ("validation", "execution")],
        )

        continued = self.client.post(f"{endpoint}/messages", json={
            "content": "Исправляй замечания", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(continued.status_code, 200)
        sent = policy_generation_calls(self.provider)[-1][0]
        raw_history = [item["content"] for item in sent[1:-1]]
        self.assertNotIn("Первая реализация", raw_history)
        self.assertNotIn("Проверь первую реализацию", raw_history)
        self.assertIn("planning → execution", sent[0]["content"])
        self.assertIn("validation → execution", sent[0]["content"])

    def test_task_state_rejects_direct_stage_patch_and_unknown_event(self):
        conversation = self.create_conversation()
        response = self.client.patch(
            f"/api/conversations/{conversation['id']}/task-state",
            json={"stage": "done"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("только разрешённым событием", response.get_json()["error"])
        unknown = self.client.post(
            f"/api/conversations/{conversation['id']}/task-state/events",
            json={"event": "skip_everything"},
        )
        self.assertEqual(unknown.status_code, 400)
        self.assertIn("Неизвестное событие", unknown.get_json()["error"])

    def test_task_state_report_and_local_state_command_do_not_call_provider(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.post(f"{endpoint}/task-state/events", json={"event": "approve_plan"})
        report = self.client.get(f"{endpoint}/task-state")
        self.assertEqual(report.status_code, 200)
        self.assertIn("Выполнение (execution)", report.get_json()["report"])

        command = self.client.post(f"{endpoint}/messages", json={
            "content": "/state", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(command.status_code, 200)
        messages = command.get_json()["conversation"]["messages"]
        self.assertEqual(messages[-1]["technical"]["local_command"], "task_state")
        self.assertIn("Выполнение (execution)", messages[-1]["content"])
        self.assertEqual(self.provider.calls, [])
        branch = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": messages[-1]["id"]})
        self.assertEqual(branch.status_code, 400)
        self.assertIn("нельзя разветвить", branch.get_json()["error"])

    def test_day_13_stage_assignment_is_removed(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        changed = self.client.patch(f"{endpoint}/task-state", json={
            "stage": "validation",
        })
        self.assertEqual(changed.status_code, 400)
        self.assertIn("только разрешённым событием", changed.get_json()["error"])

    def test_automatic_transition_requires_latest_audit_and_automatic_message_is_server_owned(self):
        provider = AutopilotProvider()
        self.app.extensions["chat_agent"].provider = provider
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        mode = self.client.patch(f"{endpoint}/task-state", json={"transition_mode": "automatic"})
        self.assertEqual(mode.status_code, 200)

        response = self.client.post(f"{endpoint}/messages", json={
            "content": "Составь полный план", "settings": AgentSettings().to_dict(),
        })
        audit = response.get_json()["conversation"]["messages"][-1]["technical"]["policy_audit"]
        self.assertTrue(audit["stage_complete"])
        self.assertEqual(audit["recommended_event"], "approve_plan")

        forged = self.client.post(f"{endpoint}/task-state/events", json={
            "event": "complete_execution", "automatic": True,
        })
        self.assertEqual(forged.status_code, 409)
        advanced = self.client.post(f"{endpoint}/task-state/events", json={
            "event": "approve_plan", "automatic": True,
        })
        self.assertEqual(advanced.status_code, 200)
        self.assertEqual(advanced.get_json()["conversation"]["task_state"]["stage"], "execution")

        continued = self.client.post(f"{endpoint}/messages", json={
            "content": "подменённый текст", "automatic": True, "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(continued.status_code, 200)
        messages = continued.get_json()["conversation"]["messages"]
        auto_user = messages[-2]
        self.assertTrue(auto_user["technical"]["automatic_continuation"])
        self.assertTrue(auto_user["content"].startswith("[Автопилот]"))
        self.assertNotIn("подменённый текст", auto_user["content"])

    def test_semantic_validator_blocks_implementation_during_planning(self):
        provider = MislabeledImplementationProvider()
        self.app.extensions["chat_agent"].provider = provider
        conversation = self.create_conversation()
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Сразу напиши код", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["conversation"]
        assistant = result["messages"][-1]
        self.assertEqual(assistant["technical"]["request_status"], "blocked")
        self.assertFalse(assistant["technical"]["policy_audit"]["accepted"])
        self.assertIn("Ответ заблокирован", assistant["content"])
        self.assertNotIn("print('реализация')", assistant["content"])
        self.assertEqual(completed_exchanges(result["messages"]), [])

    def test_informational_answer_does_not_need_to_complete_planning(self):
        provider = IncompleteAnalysisProvider()
        self.app.extensions["chat_agent"].provider = provider
        conversation = self.create_conversation()
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Что надеть при такой погоде?", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["conversation"]
        assistant = result["messages"][-1]
        self.assertEqual(assistant["technical"]["request_status"], "completed")
        self.assertIn("лёгкая куртка", assistant["content"])
        audit = assistant["technical"]["policy_audit"]
        self.assertTrue(audit["accepted"])
        self.assertFalse(audit["validator_allowed"])
        self.assertFalse(audit["stage_complete"])
        self.assertEqual(result["task_state"]["stage"], "planning")

    def test_blocked_answer_can_be_regenerated_without_duplicate_user_message(self):
        provider = MislabeledImplementationProvider()
        self.app.extensions["chat_agent"].provider = provider
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        blocked = self.client.post(f"{endpoint}/messages", json={
            "content": "Подготовь допустимый план", "settings": AgentSettings().to_dict(),
        }).get_json()["conversation"]
        user = next(item for item in blocked["visible_messages"] if item["role"] == "user")
        blocked_assistant = blocked["visible_messages"][-1]
        self.assertEqual(blocked_assistant["technical"]["request_status"], "blocked")

        self.app.extensions["chat_agent"].provider = FakeProvider()
        retried_response = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": user["id"]})
        self.assertEqual(retried_response.status_code, 201)
        retried = retried_response.get_json()["conversation"]
        visible_users = [item for item in retried["visible_messages"] if item["role"] == "user"]
        retried_assistant = retried["visible_messages"][-1]
        self.assertEqual(len(visible_users), 1)
        self.assertEqual(visible_users[0]["technical"]["request_status"], "completed")
        self.assertEqual(retried_assistant["technical"]["request_status"], "completed")
        self.assertTrue(retried_assistant["technical"]["retry_after_policy_block"])
        self.assertEqual(len(completed_exchanges(retried["visible_messages"])), 1)
        point = next(item for item in retried["branch_points"] if item["checkpoint_id"] == user["id"])
        self.assertEqual(len(point["options"]), 2)

    def test_policy_retry_button_targets_blocked_answer_parent(self):
        script = (Path(__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
        template = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('message.technical?.request_status === "blocked"', script)
        self.assertIn("createBranch(message.parent_id)", script)
        self.assertIn('class="policy-retry-button"', template)
        self.assertIn("Сгенерировать заново", template)

    def test_generator_returns_plain_text_and_only_validator_uses_json(self):
        conversation = self.create_conversation()
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Подготовь план", "settings": AgentSettings(response_format="json_object").to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        generation_messages, generation_settings = policy_generation_calls(self.provider)[-1]
        self.assertEqual(generation_settings.response_format, "text")
        self.assertIn("Верни обычный ответ пользователю", generation_messages[0]["content"])
        validator_messages, validator_settings = next(
            call for call in reversed(self.provider.calls)
            if "строгий контроллер этапов" in call[0][0]["content"]
        )
        self.assertEqual(validator_settings.response_format, "json_object")
        self.assertIn("Черновик ответа:\nТестовый ответ", validator_messages[-1]["content"])
        assistant = response.get_json()["conversation"]["messages"][-1]
        self.assertEqual(assistant["content"], "Тестовый ответ")
        self.assertNotIn("action_type", assistant["technical"]["policy_audit"])

    def test_artifact_lifecycle_blocks_missing_files_and_emits_final_result(self):
        provider = ArtifactLifecycleProvider()
        self.app.extensions["chat_agent"].provider = provider
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"

        planned = self.client.post(f"{endpoint}/messages", json={
            "content": "Сделай сайт-визитку одним файлом", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(planned.status_code, 200)
        self.assertEqual(planned.get_json()["conversation"]["task_state"]["required_artifacts"], ["index.html"])
        self.assertEqual(
            self.client.post(f"{endpoint}/task-state/events", json={"event": "approve_plan"}).status_code,
            200,
        )

        missing = self.client.post(f"{endpoint}/task-state/events", json={"event": "complete_execution"})
        self.assertEqual(missing.status_code, 409)
        self.assertIn("index.html", missing.get_json()["error"])

        executed = self.client.post(f"{endpoint}/messages", json={
            "content": "Выполни план", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(executed.status_code, 200)
        execution_conversation = executed.get_json()["conversation"]
        self.assertEqual(len(execution_conversation["active_artifacts"]), 1)
        artifact = execution_conversation["active_artifacts"][0]
        self.assertEqual(artifact["path"], "index.html")
        self.assertNotIn("storage_path", artifact)
        self.assertEqual(
            self.client.post(f"{endpoint}/task-state/events", json={"event": "complete_execution"}).status_code,
            200,
        )

        unvalidated = self.client.post(f"{endpoint}/task-state/events", json={"event": "pass_validation"})
        self.assertEqual(unvalidated.status_code, 409)
        validated = self.client.post(f"{endpoint}/messages", json={
            "content": "Проверь результат", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(validated.status_code, 200)
        completed = self.client.post(f"{endpoint}/task-state/events", json={"event": "pass_validation"})
        self.assertEqual(completed.status_code, 200)
        result = completed.get_json()["conversation"]
        self.assertEqual(result["task_state"]["stage"], "done")
        final = result["visible_messages"][-1]
        self.assertTrue(final["technical"]["final_completion"])
        self.assertEqual(final["technical"]["policy_audit"]["stage"], "done")
        self.assertIn("Сайт-визитка готов", final["content"])
        self.assertIn("index.html", final["content"])
        self.assertEqual(final["technical"]["result_manifest"]["artifacts"][0]["path"], "index.html")
        self.assertTrue(any(
            "Код этапа: done" in call[0][0]["content"]
            for call in provider.calls
            if "ОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА" in call[0][0]["content"]
        ))

        downloaded = self.client.get(f"{endpoint}/artifacts/{artifact['id']}")
        self.assertEqual(downloaded.status_code, 200)
        self.assertIn(b"<title>\xd0\x92\xd0\xb8\xd0\xb7\xd0\xb8\xd1\x82\xd0\xba\xd0\xb0</title>", downloaded.data)
        downloaded.close()
        preview = self.client.get(f"{endpoint}/artifacts/{artifact['id']}?disposition=inline")
        self.assertIn("sandbox", preview.headers["Content-Security-Policy"])
        preview.close()

    def test_artifact_paths_reject_traversal_and_named_fence_is_extracted(self):
        with self.assertRaises(ArtifactError):
            normalize_artifact_path("../secret.txt")
        payloads = extract_artifact_payloads(
            "**index.html**\n```html\n<h1>Готово</h1>\n```",
            ["index.html"],
        )
        self.assertEqual(payloads, [{"path": "index.html", "content": "<h1>Готово</h1>"}])

    def test_non_file_task_reaches_done_without_artifact_gate(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        for event in ("approve_plan", "complete_execution", "pass_validation"):
            response = self.client.post(f"{endpoint}/task-state/events", json={"event": event})
            self.assertEqual(response.status_code, 200)
        result = response.get_json()["conversation"]
        self.assertEqual(result["task_state"]["stage"], "done")
        self.assertEqual(result["task_state"]["required_artifacts"], [])
        final = result["visible_messages"][-1]
        self.assertEqual(final["technical"]["result_manifest"]["artifacts"], [])
        self.assertTrue(final["technical"]["final_completion"])
        self.assertIn("Тестовый ответ", final["content"])
        self.assertIn("файловых артефактов нет", final["content"])

    def test_failed_final_llm_response_keeps_task_in_validation(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        for event in ("approve_plan", "complete_execution"):
            self.assertEqual(
                self.client.post(f"{endpoint}/task-state/events", json={"event": event}).status_code,
                200,
            )
        self.app.extensions["chat_agent"].provider = FinalCompletionFailProvider()
        failed = self.client.post(f"{endpoint}/task-state/events", json={"event": "pass_validation"})
        self.assertEqual(failed.status_code, 502)
        persisted = self.client.get(endpoint).get_json()["conversation"]
        self.assertEqual(persisted["task_state"]["stage"], "validation")
        self.assertFalse(any(
            item.get("technical", {}).get("final_completion") is True
            for item in persisted["messages"]
        ))

    def test_artifact_result_ui_has_open_and_download_actions(self):
        script = (Path(__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
        style = (Path(__file__).parent / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("renderMessageArtifacts", script)
        self.assertIn('open.textContent = "Открыть"', script)
        self.assertIn('download.textContent = "Скачать"', script)
        self.assertIn(".artifact-panel.final-result", style)

    def test_invariants_have_user_project_task_layers_and_conflict_refusal(self):
        project = self.client.post("/api/projects", json={
            "name": "Проект правил", "description": "", "create_dialog": True,
        }).get_json()
        conversation = project["conversation"]
        profile_id = self.client.get("/api/state").get_json()["active_profile_id"]
        layer_specs = (
            ("user", profile_id, "Пользователь", "Всегда отвечать по-русски"),
            ("project", project["project"]["id"], "Проект", "Использовать только Flask"),
            ("task", conversation["id"], "Задача", "Не менять утверждённую архитектуру"),
        )
        for scope, owner_id, name, rule in layer_specs:
            created = self.client.post("/api/invariants", json={
                "name": name, "scope": scope, "owner_id": owner_id, "rules": [rule], "enabled": True,
            })
            self.assertEqual(created.status_code, 201)
        context = self.client.get(f"/api/invariants/context/{conversation['id']}").get_json()
        self.assertEqual([len(context["bundle"]["layers"][key]) for key in ("user", "project", "task")], [1, 1, 1])
        self.assertTrue(context["bundle"]["layers"]["system"])
        stored = self.app.extensions["json_storage"].get_conversation(conversation["id"])
        self.assertNotIn("invariants", stored)

        provider = InvariantRefusalProvider()
        self.app.extensions["chat_agent"].provider = provider
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Нарушь архитектуру", "settings": AgentSettings().to_dict(),
        })
        assistant = response.get_json()["conversation"]["messages"][-1]
        self.assertEqual(assistant["technical"]["request_status"], "completed")
        self.assertEqual(assistant["technical"]["policy_audit"]["detected_action_type"], "refusal")
        self.assertIn("нарушает инвариант", assistant["content"])
        self.assertEqual(len(assistant["technical"]["policy_audit"]["checked_invariants"]), 7)

    def test_local_voice_status_and_transcription(self):
        status = self.client.get("/api/voice/status").get_json()["voice"]
        self.assertTrue(status["ready"])
        self.assertEqual((status["pid"], status["port"], status["reused"]), (1234, 8091, False))
        started = self.client.post("/api/voice/start")
        self.assertEqual(started.status_code, 202)
        self.assertTrue(self.voice.started)
        response = self.client.post(
            "/api/voice/transcribe",
            data={"audio": (BytesIO(b"audio-bytes"), "recording.webm")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "Распознанный текст")
        self.assertEqual(self.voice.received[0], b"audio-bytes")

    def test_voice_send_button_requests_immediate_transcription_delivery(self):
        script = (Path(__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('if (mediaRecorder?.state === "recording")', script)
        self.assertIn("voiceSubmitAfterTranscription = true", script)
        self.assertIn("if (directText) {", script)
        self.assertIn("await sendMessageContent(directText)", script)

    def test_message_saves_reasoning_time_cache_and_settings(self):
        conversation = self.create_conversation()
        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "Тест", "settings": AgentSettings(reasoning_enabled=True).to_dict()},
        )
        self.assertEqual(response.status_code, 200)
        messages = response.get_json()["conversation"]["messages"]
        assistant = messages[-1]
        self.assertEqual(assistant["reasoning_content"], "Скрытое рассуждение")
        self.assertEqual(assistant["technical"]["elapsed_seconds"], 0.321)
        self.assertEqual(assistant["technical"]["usage"]["cached_input_tokens"], 14)
        self.assertEqual(assistant["technical"]["policy_generation_usage"]["cached_input_tokens"], 7)
        self.assertTrue(assistant["technical"]["policy_audit"]["accepted"])
        self.assertTrue(assistant["technical"]["settings"]["reasoning_enabled"])
        user = messages[-2]
        self.assertEqual(user["technical"]["token_usage"]["context_tokens"], 20)
        self.assertEqual(response.get_json()["conversation"]["token_totals"]["total_tokens"], 60)

    def test_second_request_receives_previous_dialogue(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        body = {"content": "Первый", "settings": AgentSettings().to_dict()}
        self.client.post(endpoint, json=body)
        body["content"] = "Второй"
        self.client.post(endpoint, json=body)
        sent = policy_generation_calls(self.provider)[-1][0]
        self.assertEqual([item["content"] for item in sent[-3:]], ["Первый", "Тестовый ответ", "Второй"])
        result = self.client.get(f"/api/conversations/{conversation['id']}").get_json()["conversation"]
        self.assertEqual(result["token_totals"]["request_count"], 2)
        self.assertEqual(result["token_totals"]["input_tokens"], 40)
        self.assertEqual(result["token_totals"]["output_tokens"], 80)
        self.assertEqual(result["token_totals"]["total_tokens"], 120)

    def test_summary_mode_is_per_conversation_and_rolls_every_five_old_exchanges(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        changed = self.client.patch(f"{endpoint}/context", json={"mode": "summary"})
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.get_json()["conversation"]["context_management"]["mode"], "summary")

        for index in range(1, 12):
            response = self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Вопрос {index}", "settings": AgentSettings().to_dict()},
            )
            self.assertEqual(response.status_code, 200)

        result = response.get_json()["conversation"]
        self.assertEqual(len(result["summaries"]), 1)
        summary = result["summaries"][0]
        self.assertEqual(summary["covered_exchange_count"], 5)
        self.assertEqual(len(summary["source_exchange_ids"]), 5)
        self.assertEqual(len(summary["source_message_ids"]), 10)
        self.assertEqual(summary["technical"]["settings"]["model"], "deepseek-v4-flash")
        self.assertFalse(summary["technical"]["settings"]["reasoning_enabled"])
        self.assertEqual(result["summary_token_totals"]["total_tokens"], 30)

        main_call_messages, _ = policy_generation_calls(self.provider)[-1]
        sent_text = "\n".join(item["content"] for item in main_call_messages)
        sent_items = [item["content"] for item in main_call_messages]
        self.assertIn("СЖАТАЯ ПАМЯТЬ", main_call_messages[0]["content"])
        self.assertNotIn("Вопрос 1", sent_items)
        self.assertIn("Вопрос 6", sent_text)
        self.assertIn("Вопрос 11", sent_text)
        self.assertEqual(result["messages"][-1]["technical"]["context_management"]["summary_exchange_count"], 5)
        self.assertEqual(result["messages"][-1]["technical"]["context_management"]["verbatim_exchange_count"], 5)

        exported = json.loads(self.client.get(f"{endpoint}/export").data)
        self.assertEqual(len(exported["conversation"]["summaries"]), 1)
        imported = self.client.post("/api/import", json=exported).get_json()["conversation"]
        self.assertEqual(len(imported["summaries"]), 1)
        self.assertEqual(imported["context_management"]["active_summary_id"], imported["summaries"][0]["id"])

        other = self.create_conversation()
        self.assertEqual(ensure_context_management(other)["mode"], "full")

    def test_switching_back_to_full_keeps_summary_history(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.patch(f"{endpoint}/context", json={"mode": "summary"})
        for index in range(11):
            self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Сообщение {index}", "settings": AgentSettings().to_dict()},
            )
        changed = self.client.patch(f"{endpoint}/context", json={"mode": "full"}).get_json()["conversation"]
        self.assertEqual(changed["context_management"]["mode"], "full")
        self.assertEqual(len(changed["summaries"]), 1)
        self.assertEqual(summary_token_totals(changed["summaries"])["request_count"], 1)
        self.assertEqual(len(completed_exchanges(changed["messages"])), 11)

    def test_sliding_window_sends_only_last_n_complete_exchanges(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        changed = self.client.patch(
            f"{endpoint}/context",
            json={"mode": "sliding", "sliding_window_exchanges": 2},
        )
        self.assertEqual(changed.status_code, 200)
        for index in range(1, 5):
            self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Окно {index}", "settings": AgentSettings().to_dict()},
            )
        sent = [item["content"] for item in policy_generation_calls(self.provider)[-1][0]]
        self.assertNotIn("Окно 1", sent)
        self.assertNotIn("Окно 2", sent)
        self.assertIn("Окно 3", sent)
        self.assertEqual(sent[-1], "Окно 4")
        result = self.client.get(endpoint).get_json()["conversation"]
        self.assertEqual(result["context_management"]["sliding_window_exchanges"], 2)
        self.assertEqual(result["visible_messages"][-1]["technical"]["context_management"]["window_exchanges"], 2)

    def test_sticky_facts_auto_manual_lock_edit_and_delete(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.patch(f"{endpoint}/context", json={"mode": "facts", "facts_window_exchanges": 3})
        response = self.client.post(
            f"{endpoint}/messages",
            json={"content": "Отвечай по-русски", "settings": AgentSettings().to_dict()},
        )
        result = response.get_json()["conversation"]
        self.assertEqual(result["facts"][0]["key"], "preferences.language")
        self.assertIn("STICKY FACTS", policy_generation_calls(self.provider)[-1][0][0]["content"])
        self.assertEqual(result["facts_token_totals"]["request_count"], 1)

        added = self.client.post(
            f"{endpoint}/facts",
            json={"key": "goal.primary", "value": "Собрать ТЗ"},
        ).get_json()["conversation"]
        manual = next(item for item in added["facts"] if item["key"] == "goal.primary")
        self.assertTrue(manual["locked"])
        edited = self.client.put(
            f"{endpoint}/facts/{manual['id']}",
            json={"key": "goal.primary", "value": "Подготовить ТЗ", "locked": True},
        ).get_json()["conversation"]
        self.assertEqual(next(item for item in edited["facts"] if item["id"] == manual["id"])["value"], "Подготовить ТЗ")
        removed = self.client.delete(f"{endpoint}/facts/{manual['id']}")
        self.assertEqual(removed.status_code, 200)
        self.assertFalse(any(item["id"] == manual["id"] for item in removed.get_json()["conversation"]["facts"]))

    def test_branching_is_available_in_full_and_switches_numbered_siblings(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        for text in ("Первый вопрос", "Исходное продолжение"):
            self.client.post(
                f"{endpoint}/messages",
                json={"content": text, "settings": AgentSettings().to_dict()},
            )
        state = self.client.get(endpoint).get_json()["conversation"]
        first_assistant = next(item for item in state["visible_messages"] if item["role"] == "assistant")
        started = self.client.post(
            f"{endpoint}/branches", json={"checkpoint_id": first_assistant["id"]},
        )
        self.assertEqual(started.status_code, 201)
        alternative = self.client.post(
            f"{endpoint}/messages",
            json={"content": "Альтернативное продолжение", "settings": AgentSettings().to_dict()},
        ).get_json()["conversation"]
        visible_text = [item["content"] for item in alternative["visible_messages"]]
        self.assertIn("Альтернативное продолжение", visible_text)
        self.assertNotIn("Исходное продолжение", visible_text)
        point = next(item for item in alternative["branch_points"] if item["checkpoint_id"] == first_assistant["id"])
        self.assertEqual([item["number"] for item in point["options"]], [1, 2])

        original_child = point["options"][0]["child_id"]
        switched = self.client.patch(
            f"{endpoint}/branches/active",
            json={"checkpoint_id": first_assistant["id"], "child_id": original_child},
        ).get_json()["conversation"]
        switched_text = [item["content"] for item in switched["visible_messages"]]
        self.assertIn("Исходное продолжение", switched_text)
        self.assertNotIn("Альтернативное продолжение", switched_text)

    def test_branch_after_user_message_generates_new_assistant_sibling(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        sent = self.client.post(
            f"{endpoint}/messages",
            json={"content": "Дай вариант", "settings": AgentSettings().to_dict()},
        ).get_json()["conversation"]
        user = next(item for item in sent["visible_messages"] if item["role"] == "user")
        forked = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": user["id"]})
        self.assertEqual(forked.status_code, 201)
        result = forked.get_json()["conversation"]
        point = next(item for item in result["branch_points"] if item["checkpoint_id"] == user["id"])
        self.assertEqual(len(point["options"]), 2)

    def test_branching_is_available_in_summary_sliding_and_facts_modes(self):
        for mode in ("summary", "sliding", "facts"):
            with self.subTest(mode=mode):
                conversation = self.create_conversation()
                endpoint = f"/api/conversations/{conversation['id']}"
                changed = self.client.patch(f"{endpoint}/context", json={"mode": mode})
                self.assertEqual(changed.status_code, 200)
                sent = self.client.post(
                    f"{endpoint}/messages",
                    json={"content": f"Вопрос для {mode}", "settings": AgentSettings().to_dict()},
                ).get_json()["conversation"]
                user = next(item for item in sent["visible_messages"] if item["role"] == "user")
                forked = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": user["id"]})
                self.assertEqual(forked.status_code, 201)
                result = forked.get_json()["conversation"]
                self.assertEqual(result["context_management"]["mode"], mode)
                point = next(item for item in result["branch_points"] if item["checkpoint_id"] == user["id"])
                self.assertEqual(len(point["options"]), 2)

    def test_branch_from_user_rebuilds_summary_for_selected_path(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.patch(f"{endpoint}/context", json={"mode": "summary"})
        for index in range(1, 12):
            sent = self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Summary-ветка {index}", "settings": AgentSettings().to_dict()},
            ).get_json()["conversation"]
        latest_user = [item for item in sent["visible_messages"] if item["role"] == "user"][-1]
        forked = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": latest_user["id"]})
        self.assertEqual(forked.status_code, 201)
        result = forked.get_json()["conversation"]
        self.assertEqual(result["context_management"]["mode"], "summary")
        self.assertIsNotNone(result["context_management"]["active_summary_id"])
        alternative = result["visible_messages"][-1]
        self.assertEqual(alternative["role"], "assistant")
        self.assertEqual(alternative["technical"]["context_management"]["summary_exchange_count"], 5)

    def test_preset_change_creates_history_event_and_snapshot(self):
        preset_response = self.client.post("/api/presets", json={
            "name": "Специалист", "description": "", "settings": AgentSettings(temperature=0.2).to_dict(),
        })
        preset = preset_response.get_json()["preset"]
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        self.client.post(endpoint, json={"content": "Один", "preset_id": preset["id"], "settings": preset["settings"]})
        custom = AgentSettings(temperature=1.4).to_dict()
        response = self.client.post(endpoint, json={"content": "Два", "settings": custom})
        messages = response.get_json()["conversation"]["messages"]
        self.assertTrue(any(item["role"] == "event" for item in messages))
        self.assertEqual(messages[-2]["technical"]["settings"]["temperature"], 1.4)
        self.assertEqual(messages[-2]["technical"]["configuration_source"]["type"], "custom")

    def test_export_and_import_copy_dialogue_and_used_preset(self):
        preset = self.client.post("/api/presets", json={
            "name": "Переносимый", "description": "", "settings": AgentSettings().to_dict(),
        }).get_json()["preset"]
        conversation = self.create_conversation()
        self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "Экспорт", "preset_id": preset["id"], "settings": preset["settings"]},
        )
        exported = json.loads(self.client.get(f"/api/conversations/{conversation['id']}/export").data)
        imported = self.client.post("/api/import", json=exported)
        self.assertEqual(imported.status_code, 201)
        self.assertNotEqual(imported.get_json()["conversation"]["id"], conversation["id"])
        self.assertEqual(len(imported.get_json()["conversation"]["messages"]), 2)

    def test_api_failure_is_saved_but_not_fed_back(self):
        self.provider.fail = True
        conversation = self.create_conversation()
        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "Ошибка", "settings": AgentSettings().to_dict()},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["conversation"]["messages"][-1]["technical"]["request_status"], "failed")
        self.assertIn("соединиться", friendly_api_error(RuntimeError("Connection error")))

    def test_context_overflow_has_clear_message(self):
        message = friendly_api_error(RuntimeError("maximum context length exceeded: too many tokens"))
        self.assertIn("превысил лимит модели", message)
        self.assertIn("сократите историю", message)

    def test_launcher_skips_occupied_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            port = occupied.getsockname()[1]
            self.assertEqual(find_available_port(port, port + 1), port + 1)


class MemoryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = MemoryProvider()
        self.app = create_app(Path(self.temp.name), Agent(self.provider), FakeVoiceService())
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def create_conversation(self):
        return self.client.post("/api/conversations", json={"title": "Диалог"}).get_json()["conversation"]

    def create_project(self, name="Проект"):
        return self.client.post("/api/projects", json={"name": name, "description": "Описание"}).get_json()["project"]

    def test_legacy_conversation_and_project_links(self):
        conversation = self.create_conversation()
        path = Path(self.temp.name) / "conversations" / f"{conversation['id']}.json"
        stored = json.loads(path.read_text(encoding="utf-8")); stored.pop("project_id", None)
        path.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
        self.assertIsNone(self.client.get(f"/api/conversations/{conversation['id']}").get_json()["conversation"]["project_id"])
        project = self.create_project()
        second = self.create_conversation()
        for item in (conversation, second):
            response = self.client.patch(f"/api/conversations/{item['id']}/project", json={"project_id": project["id"]})
            self.assertEqual(response.status_code, 200)
        linked = self.client.get(f"/api/projects/{project['id']}").get_json()["project"]["conversations"]
        self.assertEqual(len(linked), 2)
        self.assertTrue(project["settings"]["automatic_extraction"])

    def test_sidebar_project_creation_also_creates_empty_linked_dialogue(self):
        response = self.client.post("/api/projects", json={
            "name": "Проект из сайдбара", "description": "", "create_dialog": True,
        })
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["conversation"]["project_id"], data["project"]["id"])
        self.assertEqual(data["conversation"]["messages"], [])

    def test_copy_between_projects_preserves_source_and_creates_independent_dialogue(self):
        source_project, target_project = self.create_project("Старый"), self.create_project("Новый")
        source = self.create_conversation()
        self.client.patch(f"/api/conversations/{source['id']}/project", json={"project_id": source_project["id"]})
        self.client.post(f"/api/conversations/{source['id']}/messages", json={
            "content": "История для копии", "settings": AgentSettings().to_dict(),
        })
        before = self.app.extensions["json_storage"].get_conversation(source["id"])
        old_project_before = self.app.extensions["json_storage"].get_project(source_project["id"])
        copied_response = self.client.post(f"/api/conversations/{source['id']}/copy-to-project", json={
            "project_id": target_project["id"],
        })
        self.assertEqual(copied_response.status_code, 201)
        copied = copied_response.get_json()["conversation"]
        original = self.app.extensions["json_storage"].get_conversation(source["id"])
        old_project_after = self.app.extensions["json_storage"].get_project(source_project["id"])
        self.assertEqual(original["project_id"], source_project["id"])
        self.assertEqual(original["messages"], before["messages"])
        self.assertEqual(old_project_after, old_project_before)
        self.assertNotEqual(copied["id"], original["id"])
        self.assertEqual(copied["project_id"], target_project["id"])
        self.assertEqual(copied["messages"], original["messages"])
        self.assertTrue(copied["title"].endswith("— копия"))
        self.assertEqual(copied["copied_from_conversation_id"], original["id"])
        copied_document = self.app.extensions["json_storage"].get_conversation(copied["id"])
        copied_document["title"] = "Независимая копия"
        self.app.extensions["json_storage"].save_conversation(copied_document)
        self.assertNotEqual(
            self.app.extensions["json_storage"].get_conversation(copied["id"])["title"],
            self.app.extensions["json_storage"].get_conversation(original["id"])["title"],
        )

    def test_explicit_history_extraction_is_separate_and_project_only(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Используем локальные JSON-файлы", "settings": AgentSettings().to_dict(),
        })
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        calls_before = len(self.provider.calls)
        response = self.client.post(f"/api/conversations/{conversation['id']}/extract-project-memory", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.provider.calls), calls_before + 1)
        result = response.get_json()
        self.assertEqual(result["revision"]["purpose"], "project_history_extraction")
        self.assertEqual(result["revision"]["exchange_count"], 1)
        self.assertEqual(len(result["project"]["learned_memory"]), 1)
        self.assertEqual(self.client.get("/api/memory/user").get_json()["learned"], [])
        self.assertIn("Используем локальные JSON-файлы", self.provider.calls[-1][0][-1]["content"])

    def test_manual_learned_routing_lock_dedup_and_snapshot(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        manual = self.client.post(f"/api/projects/{project['id']}/memory", json={
            "kind": "decision", "key": "storage.format", "value": "  Сохранять дословно  ",
        }).get_json()["memory"]
        self.assertEqual(manual["value"], "Сохранять дословно")
        self.assertTrue(manual["locked"])
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Запомни решение и язык", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["conversation"]
        self.assertEqual(result["memory_token_totals"]["request_count"], 1)
        assistant = [item for item in result["messages"] if item["role"] == "assistant"][-1]
        snapshot = assistant["technical"]["memory_context"]
        self.assertTrue(snapshot["policy_rules"])
        self.assertEqual(snapshot["project_memories"][0]["value"], "Сохранять дословно")
        project_after = self.client.get(f"/api/projects/{project['id']}").get_json()["project"]
        self.assertEqual(project_after["manual_memory"][0]["value"], "Сохранять дословно")
        self.assertEqual(project_after["learned_memory"], [])  # automatic не перезаписывает locked key
        self.assertEqual(len(self.provider.calls), 3)  # генерация + проверка политики + extraction
        self.client.delete(f"/api/projects/{project['id']}/memory/{manual['id']}")
        saved = self.client.get(f"/api/conversations/{conversation['id']}/memory-context").get_json()["snapshots"]
        self.assertEqual(saved[0]["memory_context"]["project_memories"][0]["value"], "Сохранять дословно")

    def test_user_memory_is_global_but_project_memory_is_isolated(self):
        first_project, second_project = self.create_project("Первый"), self.create_project("Второй")
        first, second = self.create_conversation(), self.create_conversation()
        self.client.patch(f"/api/conversations/{first['id']}/project", json={"project_id": first_project["id"]})
        self.client.patch(f"/api/conversations/{second['id']}/project", json={"project_id": second_project["id"]})
        self.client.post(f"/api/projects/{first_project['id']}/memory", json={"kind": "goal", "key": "goal.primary", "value": "Только первый"})
        self.client.post("/api/memory/user", json={"kind": "preference", "key": "response.style", "value": "Кратко"})
        first_snapshot = self.app.extensions["memory_manager"].context_snapshot(self.app.extensions["json_storage"].get_conversation(first["id"]), "full")
        second_snapshot = self.app.extensions["memory_manager"].context_snapshot(self.app.extensions["json_storage"].get_conversation(second["id"]), "full")
        self.assertEqual(len(first_snapshot["project_memories"]), 1)
        self.assertEqual(second_snapshot["project_memories"], [])
        self.assertEqual(second_snapshot["user_memories"][0]["value"], "Кратко")
        self.assertFalse(self.client.get("/api/memory/user").get_json()["settings"]["automatic_extraction"])

    def test_one_extraction_routes_both_scopes_with_origin_and_no_duplicates(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        self.client.post("/api/memory/user", json={"automatic_extraction": True})
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        for text in ("Первый", "Второй"):
            self.assertEqual(self.client.post(endpoint, json={"content": text, "settings": AgentSettings().to_dict()}).status_code, 200)
        stored_project = self.client.get(f"/api/projects/{project['id']}").get_json()["project"]
        stored_user = self.client.get("/api/memory/user").get_json()
        self.assertEqual(len(stored_project["learned_memory"]), 1)
        self.assertEqual(len(stored_user["learned"]), 1)
        learned = stored_project["learned_memory"][0]
        self.assertEqual(learned["source"], "automatic")
        self.assertEqual(learned["confidence"], "confirmed")
        self.assertEqual(learned["review_status"], "confirmed")
        self.assertFalse(learned["locked"])
        self.assertEqual(learned["source_conversation_id"], conversation["id"])
        self.assertTrue(learned["source_exchange_id"])
        pinned = self.client.put(f"/api/projects/{project['id']}/memory/{learned['id']}", json={"pin": True}).get_json()["memory"]
        self.assertTrue(pinned["locked"]); self.assertEqual(pinned["review_status"], "confirmed")
        self.assertEqual(len(self.provider.calls), 6)  # две генерации + две проверки политики + два extraction

    def test_automatic_memory_is_confirmed_but_removable(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Запомни решение", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        learned = self.client.get(f"/api/projects/{project['id']}").get_json()["project"]["learned_memory"][0]
        self.assertEqual(learned["review_status"], "confirmed")
        self.assertEqual(learned["confidence"], "confirmed")
        self.assertFalse(learned["locked"])
        deleted = self.client.delete(f"/api/projects/{project['id']}/memory/{learned['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.get_json()["ok"])
        self.assertEqual(self.client.get(f"/api/projects/{project['id']}").get_json()["project"]["learned_memory"], [])

    def test_memory_routing_accepts_model_confidence_and_skips_only_invalid_candidate(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        conversation = self.app.extensions["json_storage"].get_conversation(conversation["id"])
        result = AgentResult(
            content=json.dumps({"memories": [
                {"scope": "project", "kind": "decision", "key": "backend.framework", "value": "Flask", "confidence": 0.98},
                {"scope": "project", "kind": "unsupported", "key": "invalid.one", "value": "Не сохранять"},
            ]}, ensure_ascii=False),
            reasoning_content="",
            technical={"usage": {}},
        )
        revision = self.app.extensions["memory_manager"].route_candidates(result, conversation, "exchange-test")
        self.assertEqual(revision["status"], "completed")
        self.assertEqual(revision["saved_count"], 1)
        self.assertEqual(revision["rejected_count"], 1)
        learned = self.client.get(f"/api/projects/{project['id']}").get_json()["project"]["learned_memory"]
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0]["confidence"], "confirmed")

    def test_failed_or_pending_exchange_does_not_extract_and_extraction_failure_is_nonfatal(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        self.provider.fail = True
        failed = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={"content": "Сбой", "settings": AgentSettings().to_dict()})
        self.assertEqual(failed.status_code, 502)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.app.extensions["json_storage"].get_conversation(conversation["id"])["memory_extraction_revisions"], [])

        extraction_provider = ExtractionFailProvider()
        self.app.extensions["chat_agent"].provider = extraction_provider
        completed = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={"content": "Основной ответ успешен", "settings": AgentSettings().to_dict()})
        self.assertEqual(completed.status_code, 200)
        revisions = completed.get_json()["conversation"]["memory_extraction_revisions"]
        self.assertEqual(revisions[-1]["status"], "failed")
        self.assertNotIn("Основной ответ успешен", json.dumps(revisions, ensure_ascii=False))

    def test_inactive_memory_excluded_and_export_import_copies_project(self):
        project = self.create_project()
        conversation = self.create_conversation()
        self.client.patch(f"/api/conversations/{conversation['id']}/project", json={"project_id": project["id"]})
        memory = self.client.post(f"/api/projects/{project['id']}/memory", json={"kind": "risk", "key": "risk.one", "value": "Не включать"}).get_json()["memory"]
        self.client.put(f"/api/projects/{project['id']}/memory/{memory['id']}", json={"status": "inactive"})
        snapshot = self.app.extensions["memory_manager"].context_snapshot(self.app.extensions["json_storage"].get_conversation(conversation["id"]), "full")
        self.assertEqual(snapshot["project_memories"], [])
        exported = json.loads(self.client.get(f"/api/conversations/{conversation['id']}/export").data)
        imported = self.client.post("/api/import", json=exported).get_json()["conversation"]
        self.assertNotEqual(imported["project_id"], project["id"])
        copied = self.client.get(f"/api/projects/{imported['project_id']}").get_json()["project"]
        self.assertTrue(copied["name"].endswith("— импорт"))

    def test_ui_exposes_memory_tabs_policy_and_snapshot_action(self):
        html = self.client.get("/").get_data(as_text=True)
        for label in ("Память", "Текущий диалог", "Проект", "Пользователь", "Политика", "Какая память использована", "＋ Проект"):
            self.assertIn(label, html)
        self.assertEqual(self.client.patch("/api/memory/policy", json={"rules": []}).status_code, 405)


class PersonalizationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = FakeProvider()
        self.app = create_app(Path(self.temp.name), Agent(self.provider), FakeVoiceService())
        self.client = self.app.test_client()
        state = self.client.get("/api/state").get_json()
        self.default_profile = state["profiles"][0]

    def tearDown(self):
        self.temp.cleanup()

    def create_profile(self, name, detail="high", level="beginner"):
        response = self.client.post("/api/profiles", json={
            "name": name,
            "preferences": {
                "language": "ru", "detail_level": detail, "tone": "friendly",
                "response_structure": "step_by_step", "technical_level": level,
                "preferred_formats": ["steps", "examples"], "avoid_formats": ["tables"],
            },
            "custom_instructions": "Объяснять термины простыми словами.",
        })
        self.assertEqual(response.status_code, 201)
        return response.get_json()["profile"]

    def switch_profile(self, profile_id):
        response = self.client.patch("/api/profiles/active", json={"profile_id": profile_id})
        self.assertEqual(response.status_code, 200)

    def test_new_profile_starts_empty_and_personal_memory_is_isolated(self):
        owned = self.client.post("/api/conversations", json={"title": "Личный"}).get_json()["conversation"]
        self.client.post("/api/memory/user", json={
            "kind": "preference", "key": "response.style", "value": "Кратко",
        })
        second = self.create_profile("Новичок")
        self.switch_profile(second["id"])
        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["conversations"], [])
        self.assertEqual(state["projects"], [])
        self.assertEqual(self.client.get("/api/memory/user").get_json()["manual"], [])
        self.assertEqual(self.client.get(f"/api/conversations/{owned['id']}").status_code, 404)

    def test_new_profile_automatically_extracts_only_its_own_user_memory(self):
        provider = MemoryProvider()
        self.app.extensions["chat_agent"].provider = provider
        second = self.create_profile("Автопамять")
        self.assertTrue(second["settings"]["automatic_extraction"])
        self.switch_profile(second["id"])
        conversation = self.client.post("/api/conversations", json={"title": "Личный"}).get_json()["conversation"]
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Предпочитаю русский язык", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        learned = self.client.get("/api/memory/user").get_json()["learned"]
        self.assertEqual([item["value"] for item in learned], ["Русский язык"])
        self.switch_profile(self.default_profile["id"])
        self.assertEqual(self.client.get("/api/memory/user").get_json()["learned"], [])

    def test_owner_shares_whole_project_and_member_can_chat_but_not_administer(self):
        project = self.client.post("/api/projects", json={
            "name": "Общий проект", "description": "", "create_dialog": True,
        }).get_json()
        conversation = project["conversation"]
        second = self.create_profile("Участник", detail="high", level="beginner")
        shared = self.client.patch(f"/api/projects/{project['project']['id']}", json={
            "participant_profile_ids": [second["id"]], "automatic_extraction": False,
        })
        self.assertEqual(shared.status_code, 200)

        self.switch_profile(second["id"])
        state = self.client.get("/api/state").get_json()
        self.assertEqual([item["id"] for item in state["projects"]], [project["project"]["id"]])
        self.assertEqual([item["id"] for item in state["conversations"]], [conversation["id"]])
        self.assertEqual(
            self.client.patch(f"/api/projects/{project['project']['id']}", json={"name": "Чужое имя"}).status_code,
            404,
        )
        created = self.client.post("/api/conversations", json={
            "title": "Диалог участника", "project_id": project["project"]["id"],
        })
        self.assertEqual(created.status_code, 201)

        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Объясни генераторы Python", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        messages = response.get_json()["conversation"]["messages"]
        user = [item for item in messages if item["role"] == "user"][-1]
        assistant = [item for item in messages if item["role"] == "assistant"][-1]
        self.assertEqual(user["author_profile_id"], second["id"])
        self.assertEqual(assistant["technical"]["profile_snapshot"]["profile_id"], second["id"])
        main_call = policy_generation_calls(self.provider)[-1]
        self.assertIn("Имя: Участник", main_call[0][0]["content"])
        self.assertIn("Технический уровень: beginner", main_call[0][0]["content"])

    def test_automatic_project_memory_is_shared_with_other_profile(self):
        provider = MemoryProvider()
        self.app.extensions["chat_agent"].provider = provider
        project_data = self.client.post("/api/projects", json={
            "name": "Общая память", "description": "", "create_dialog": True,
        }).get_json()
        project = project_data["project"]
        conversation = project_data["conversation"]
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        first = self.client.post(endpoint, json={
            "content": "Храним данные в локальных JSON-файлах", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(first.status_code, 200)
        stored = self.client.get(f"/api/projects/{project['id']}").get_json()["project"]
        self.assertEqual([item["value"] for item in stored["learned_memory"]], ["Локальные JSON-файлы"])

        second = self.create_profile("Соавтор")
        self.client.patch(f"/api/projects/{project['id']}", json={
            "participant_profile_ids": [second["id"]],
        })
        self.switch_profile(second["id"])
        response = self.client.post(endpoint, json={
            "content": "Какой формат хранения используем?", "settings": AgentSettings().to_dict(),
        })
        self.assertEqual(response.status_code, 200)
        assistant = [
            item for item in response.get_json()["conversation"]["messages"]
            if item["role"] == "assistant"
        ][-1]
        snapshot = assistant["technical"]["memory_context"]
        self.assertEqual([item["value"] for item in snapshot["project_memories"]], ["Локальные JSON-файлы"])
        self.assertEqual(snapshot["user_memories"], [])
        self.assertIn("Локальные JSON-файлы", policy_generation_calls(provider)[-1][0][0]["content"])

    def test_legacy_user_memory_becomes_default_profile_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            memory.mkdir(parents=True)
            (memory / "user_memory.json").write_text(json.dumps({
                "schema_version": 1, "settings": {"automatic_extraction": True},
                "manual_memory": [{"id": "legacy", "source": "manual", "value": "Старое значение"}],
                "learned_memory": [], "extraction_revisions": [],
            }, ensure_ascii=False), encoding="utf-8")
            storage = JsonStorage(root)
            profile = storage.get_profile(storage.default_profile_id())
            self.assertEqual(profile["name"], "Основной пользователь")
            self.assertEqual(profile["manual_memory"][0]["value"], "Старое значение")
            self.assertTrue(profile["settings"]["automatic_extraction"])

    def test_existing_non_default_profiles_receive_new_auto_extraction_default_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profiles.json").write_text(json.dumps({
                "schema_version": 1,
                "active_profile_id": "main",
                "profiles": [
                    {"id": "main", "name": "Основной пользователь", "is_default": True, "settings": {"automatic_extraction": False}},
                    {"id": "second", "name": "Второй", "is_default": False, "settings": {"automatic_extraction": False}},
                ],
            }, ensure_ascii=False), encoding="utf-8")
            storage = JsonStorage(root)
            self.assertFalse(storage.get_profile("main")["settings"]["automatic_extraction"])
            self.assertTrue(storage.get_profile("second")["settings"]["automatic_extraction"])
            self.assertEqual(storage.load_profiles_document()["schema_version"], 2)

    def test_ui_exposes_profile_controls_and_project_sharing(self):
        html = self.client.get("/").get_data(as_text=True)
        for value in ("Активный профиль", 'id="profile-create"', "Профиль пользователя", "Участники проекта"):
            self.assertIn(value, html)

    def test_same_question_uses_each_authors_profile_automatically(self):
        provider = ProfileAwareProvider()
        self.app.extensions["chat_agent"].provider = provider
        default_preferences = {
            "language": "ru", "detail_level": "low", "tone": "neutral",
            "response_structure": "result_then_explanation", "technical_level": "advanced",
            "preferred_formats": [], "avoid_formats": [],
        }
        self.client.patch(f"/api/profiles/{self.default_profile['id']}", json={
            "preferences": default_preferences,
        })
        project_data = self.client.post("/api/projects", json={
            "name": "Сравнение", "description": "", "create_dialog": True,
        }).get_json()
        project = project_data["project"]
        conversation = project_data["conversation"]
        second = self.create_profile("Начинающий", detail="high", level="beginner")
        self.client.patch(f"/api/projects/{project['id']}", json={
            "participant_profile_ids": [second["id"]], "automatic_extraction": False,
        })
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        first = self.client.post(endpoint, json={
            "content": "Что делает генератор?", "settings": AgentSettings().to_dict(),
        }).get_json()["conversation"]
        first_answer = [item for item in first["messages"] if item["role"] == "assistant"][-1]
        self.switch_profile(second["id"])
        second_result = self.client.post(endpoint, json={
            "content": "Что делает генератор?", "settings": AgentSettings().to_dict(),
        }).get_json()["conversation"]
        second_answer = [item for item in second_result["messages"] if item["role"] == "assistant"][-1]
        self.assertEqual(first_answer["content"], "Краткий ответ")
        self.assertEqual(second_answer["content"], "Подробный пошаговый ответ")
        self.assertEqual(first_answer["technical"]["profile_snapshot"]["profile_id"], self.default_profile["id"])
        self.assertEqual(second_answer["technical"]["profile_snapshot"]["profile_id"], second["id"])


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.previous_level = logging.getLogger().level
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_deepseek_agent_handler", False):
                root.removeHandler(handler)
                handler.close()
        root.setLevel(self.previous_level)
        self.temp.cleanup()

    def test_rotating_logs_separate_errors_and_redact_api_keys(self):
        paths = configure_logging(Path(self.temp.name))
        logger = logging.getLogger("deepseek_agent.test")
        logger.info("Приложение запущено")
        logger.error("DEEPSEEK_API_KEY=sk-secret123456789")

        app = create_app(Path(self.temp.name) / "data", Agent(FakeProvider()), FakeVoiceService())
        app.config.update(TESTING=True)
        client = app.test_client()
        conversation = client.post("/api/conversations", json={"title": "Новый диалог"}).get_json()["conversation"]
        client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "СЕКРЕТНЫЙ ТЕКСТ ПОЛЬЗОВАТЕЛЯ", "settings": AgentSettings().to_dict()},
        )
        for handler in logging.getLogger().handlers:
            handler.flush()

        app_text = paths["app"].read_text(encoding="utf-8")
        error_text = paths["errors"].read_text(encoding="utf-8")
        self.assertIn("Приложение запущено", app_text)
        self.assertIn("<скрыто>", app_text)
        self.assertNotIn("sk-secret123456789", app_text + error_text)
        self.assertNotIn("СЕКРЕТНЫЙ ТЕКСТ ПОЛЬЗОВАТЕЛЯ", app_text + error_text)
        self.assertNotIn("Приложение запущено", error_text)
        self.assertIn("ERROR", error_text)

        rotating = [
            handler for handler in logging.getLogger().handlers
            if hasattr(handler, "maxBytes")
        ]
        self.assertTrue(rotating)
        self.assertTrue(all(handler.maxBytes == MAX_LOG_BYTES for handler in rotating))
        self.assertTrue(all(handler.backupCount == BACKUP_COUNT for handler in rotating))


if __name__ == "__main__":
    unittest.main()
