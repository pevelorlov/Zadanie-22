"""Flask API для многодиалогового DeepSeek Agent."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request, send_file

from artifacts import (
    ArtifactError,
    active_artifacts,
    artifact_file_path,
    copy_artifact_files,
    delete_artifact_files,
    ensure_artifacts,
    missing_required_artifacts,
    public_artifact,
    required_artifacts_are_validated,
    save_response_artifacts,
    verify_required_artifacts,
)

from agent import (
    DEEPSEEK_MODELS,
    PROVIDER_CAPABILITIES,
    Agent,
    AgentResult,
    AgentSettings,
    DeepSeekProvider,
    dialogue_token_totals,
    normalize_token_usage,
)
from context_manager import (
    active_path_messages,
    add_fact,
    append_tree_message,
    begin_branch,
    branch_points,
    completed_exchanges,
    current_stage_exchanges,
    current_stage_facts,
    delete_fact,
    edit_fact,
    ensure_context_management,
    facts_token_totals,
    request_history,
    select_branch,
    set_context_mode,
    set_window_sizes,
    summary_token_totals,
    update_sticky_facts,
    update_summaries,
)
from logging_setup import reset_request_id, set_request_id
from invariants import InvariantStore
from memory_manager import MemoryManager
from document_index import RAG_STRATEGIES, RagIndexService, build_rag_context
from document_index.evaluations import RagEvaluationStore
from mcp_control import PersistentFeatureControl
from mcp_manager import (
    MCPManager,
    MediaWikiMCPManager,
    OpenMeteoMCPManager,
    SchedulerMCPManager,
    StdioMCPManager,
    WorldBankMCPManager,
)
from presets import PresetManager
from scheduler import SchedulerRepository, SchedulerService, TIMEZONE_NAME, scheduler_now
from storage import JsonStorage, now_iso
from task_state import (
    TaskTransitionError,
    allowed_task_events,
    apply_task_event,
    ensure_task_state,
    task_is_paused,
    task_state_report,
    update_task_state,
)
from task_policy import (
    PolicyValidationError,
    blocked_content,
    generation_guidance,
    parse_validation_result,
    policy_audit,
    validation_prompt,
)
from task_handoff import (
    activate_stage_handoff,
    active_task_handoffs,
    build_stage_handoff,
    ensure_task_handoffs,
    handoff_token_totals,
    task_handoff_context,
)
from voice import WhisperService


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MAX_MESSAGE_LENGTH = 50_000
logger = logging.getLogger("deepseek_agent.app")
_profile_override: ContextVar[str | None] = ContextVar("scheduler_profile_override", default=None)
_automation_run_override: ContextVar[dict[str, Any] | None] = ContextVar("automation_run_override", default=None)
_scheduled_tool_allowlist: ContextVar[set[str] | None] = ContextVar("scheduled_tool_allowlist", default=None)
SCHEDULER_MODEL_TOOLS = {
    "create_scheduled_task", "list_scheduled_tasks", "get_scheduled_task",
    "get_scheduled_task_runs", "get_scheduler_summary", "get_scheduler_capabilities",
}
FILESYSTEM_MODEL_TOOLS = {"write_file"}
MEDIAWIKI_MODEL_TOOLS = {
    "search-page", "search-page-by-prefix", "get-page", "get-pages",
    "get-category-members", "get-site-info",
}
WORLDBANK_MODEL_TOOLS = {
    "worldbank_list_topics", "worldbank_list_sources", "worldbank_list_countries",
    "worldbank_get_country", "worldbank_search_indicators", "worldbank_get_indicator",
    "worldbank_get_data", "worldbank_get_poverty", "worldbank_search_projects",
}
_FILE_EXPORT_NEGATIONS = (
    "не сохраняй", "не сохранять", "без сохранения", "не записывай",
    "не записывать", "не создавай файл", "не создавай отчёт", "не создавай отчет",
)
_FILE_EXPORT_PATTERNS = (
    r"\bсохран\w*\b", r"\bвыгруз\w*\b", r"\bэкспорт\w*\b",
    r"\bзапиш\w*\b.{0,30}\b(?:файл|документ|отч[её]т)\b",
    r"\bсозда\w*\b.{0,30}\b(?:файл|документ|отч[её]т)\b",
    r"\b(?:файл|документ|отч[её]т)\b.{0,20}\b(?:на диск|в файл|в документ)\b",
    r"\b(?:save|write|export)\b.{0,30}\b(?:file|document|report)\b",
)


def requests_file_export(text: str) -> bool:
    """Распознаёт только явную просьбу создать локальный файловый результат."""
    lowered = str(text or "").lower().replace("ё", "е")
    if any(marker.replace("ё", "е") in lowered for marker in _FILE_EXPORT_NEGATIONS):
        return False
    return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in _FILE_EXPORT_PATTERNS)


def normalize_rag_options(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    enabled = raw.get("enabled") is True
    strategy = str(raw.get("strategy") or "structural").strip().lower()
    if strategy not in RAG_STRATEGIES:
        raise ValueError("Стратегия RAG должна быть fixed, structural или combined.")
    try:
        top_k = int(raw.get("top_k", 5))
    except (TypeError, ValueError) as error:
        raise ValueError("Top-K для RAG должен быть целым числом.") from error
    if not 1 <= top_k <= 20:
        raise ValueError("Top-K для RAG должен быть от 1 до 20.")
    return {"enabled": enabled, "strategy": strategy, "top_k": top_k}


def rag_snapshot(options: dict[str, Any], retrieval: dict | None = None) -> dict[str, Any]:
    snapshot = deepcopy(options)
    if retrieval:
        snapshot.update({
            "run_id": retrieval.get("run_id"),
            "query": retrieval.get("query"),
            "chunks": deepcopy(retrieval.get("chunks") or []),
        })
    else:
        snapshot["chunks"] = []
    return snapshot


def unique_report_path(workspace: Path, requested_path: Any) -> Path:
    """Создаёт уникальный Markdown-путь строго внутри workspace/reports."""
    reports_dir = (workspace.resolve() / "reports").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(str(requested_path or "weather-report").replace("\\", "/")).name
    stem = Path(raw_name).stem.strip() or "weather-report"
    safe_stem = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "-", stem).strip(" .-_")[:80]
    safe_stem = safe_stem or "weather-report"
    timestamp = scheduler_now().strftime("%Y-%m-%d-%H-%M-%S")
    candidate = reports_dir / f"{safe_stem}-{timestamp}.md"
    if candidate.exists():
        candidate = reports_dir / f"{safe_stem}-{timestamp}-{uuid.uuid4().hex[:8]}.md"
    return candidate


def create_app(
    data_dir: Path | None = None,
    agent_instance: Agent | None = None,
    voice_service: WhisperService | None = None,
    mcp_manager: MCPManager | None = None,
    weather_mcp_manager: StdioMCPManager | None = None,
    scheduler_mcp_manager: StdioMCPManager | None = None,
    scheduler_service: SchedulerService | None = None,
    scheduler_autostart: bool = False,
    mediawiki_mcp_manager: StdioMCPManager | None = None,
    worldbank_mcp_manager: StdioMCPManager | None = None,
    orchestration_autostart: bool = False,
    rag_index_service: RagIndexService | None = None,
) -> Flask:
    flask_app = Flask(__name__)
    flask_app.json.ensure_ascii = False
    flask_app.config["MAX_CONTENT_LENGTH"] = 26 * 1024 * 1024
    storage = JsonStorage(data_dir or BASE_DIR / "data")
    preset_manager = PresetManager(storage)
    chat_agent = agent_instance or Agent(DeepSeekProvider())
    memory_manager = MemoryManager(storage, BASE_DIR / "config" / "agent_policy.json")
    invariant_store = InvariantStore(storage.data_dir)
    whisper = voice_service or WhisperService(BASE_DIR, autostart=data_dir is None)
    mcp = mcp_manager or MCPManager(BASE_DIR)
    weather_mcp = weather_mcp_manager or OpenMeteoMCPManager(BASE_DIR)
    scheduler_repository = (
        scheduler_service.repository if scheduler_service is not None
        else SchedulerRepository(storage.data_dir / "scheduler.sqlite3")
    )
    scheduler_mcp = scheduler_mcp_manager or SchedulerMCPManager(BASE_DIR, scheduler_repository.database_path)
    mediawiki_mcp = mediawiki_mcp_manager or MediaWikiMCPManager(BASE_DIR)
    worldbank_mcp = worldbank_mcp_manager or WorldBankMCPManager(BASE_DIR)
    scheduler = scheduler_service or SchedulerService(scheduler_repository)
    rag_index = rag_index_service or RagIndexService(
        (BASE_DIR / "rag_documents") if data_dir is None else (storage.data_dir / "rag_documents"),
        storage.data_dir / "rag_index.sqlite3",
    )
    rag_evaluations = RagEvaluationStore(storage.data_dir / "rag_evaluations.json")
    mcp_control = PersistentFeatureControl(storage.data_dir / "mcp_control.json")
    task_control = PersistentFeatureControl(storage.data_dir / "task_control.json")
    mcp_activity_lock = threading.RLock()
    mcp_activities: dict[str, dict[str, Any]] = {}

    flask_app.extensions["json_storage"] = storage
    flask_app.extensions["preset_manager"] = preset_manager
    flask_app.extensions["chat_agent"] = chat_agent
    flask_app.extensions["memory_manager"] = memory_manager
    flask_app.extensions["invariant_store"] = invariant_store
    flask_app.extensions["whisper_service"] = whisper
    flask_app.extensions["mcp_manager"] = mcp
    flask_app.extensions["weather_mcp_manager"] = weather_mcp
    flask_app.extensions["scheduler_mcp_manager"] = scheduler_mcp
    flask_app.extensions["mediawiki_mcp_manager"] = mediawiki_mcp
    flask_app.extensions["worldbank_mcp_manager"] = worldbank_mcp
    flask_app.extensions["scheduler_service"] = scheduler
    flask_app.extensions["rag_index_service"] = rag_index
    flask_app.extensions["rag_evaluation_store"] = rag_evaluations
    flask_app.extensions["mcp_control"] = mcp_control
    flask_app.extensions["task_control"] = task_control
    logger.info("Flask-приложение создано | data_dir=%s", storage.data_dir)

    def current_profile_id() -> str:
        return _profile_override.get() or storage.active_profile_id()

    def mcp_blocked_response():
        return api_error(
            "Все MCP принудительно отключены. Сначала нажмите «Разрешить MCP».",
            409,
        )

    def task_control_blocked_response():
        return api_error(
            "Машина задач принудительно отключена. Сначала нажмите «Включить машину задач».",
            409,
        )

    def begin_mcp_activity(activity_id: str | None) -> str | None:
        if not activity_id or not re.fullmatch(r"[0-9a-f]{32}", activity_id):
            return None
        with mcp_activity_lock:
            if len(mcp_activities) >= 100:
                oldest = min(mcp_activities, key=lambda key: mcp_activities[key].get("created_at", ""))
                mcp_activities.pop(oldest, None)
            mcp_activities[activity_id] = {
                "id": activity_id, "phase": "thinking", "created_at": now_iso(), "calls": [],
            }
        return activity_id

    def public_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("content", "password", "secret", "token", "key")):
                safe[key] = f"<{len(str(value))} символов>" if value is not None else None
                continue
            encoded = json.dumps(value, ensure_ascii=False)
            safe[key] = value if len(encoded) <= 500 else encoded[:497] + "…"
        return safe

    def set_mcp_activity_phase(activity_id: str | None, phase: str, error: str | None = None) -> None:
        if not activity_id:
            return
        with mcp_activity_lock:
            activity = mcp_activities.get(activity_id)
            if activity:
                activity["phase"] = phase
                activity["updated_at"] = now_iso()
                if error:
                    activity["error"] = error

    def begin_tool_activity(activity_id: str | None, server: str | None, name: str,
                            arguments: dict[str, Any]) -> int | None:
        if not activity_id:
            return None
        with mcp_activity_lock:
            activity = mcp_activities.get(activity_id)
            if not activity:
                return None
            call_id = len(activity["calls"]) + 1
            activity["phase"] = "tools"
            activity["calls"].append({
                "id": call_id, "server": server, "name": name,
                "arguments": public_tool_arguments(arguments), "status": "running",
                "started_at": now_iso(),
            })
            activity["updated_at"] = now_iso()
            return call_id

    def finish_tool_activity(activity_id: str | None, call_id: int | None, *, is_error: bool,
                             error: str | None = None) -> None:
        if not activity_id or call_id is None:
            return
        with mcp_activity_lock:
            activity = mcp_activities.get(activity_id)
            if not activity:
                return
            call = next((item for item in activity["calls"] if item["id"] == call_id), None)
            if call:
                call["status"] = "error" if is_error else "completed"
                call["finished_at"] = now_iso()
                if error:
                    call["error"] = error
            activity["updated_at"] = now_iso()

    def invariant_context(conversation: dict[str, Any], profile_id: str) -> dict[str, Any]:
        policy = memory_manager.policy()
        return invariant_store.bundle(
            profile_id=profile_id,
            project_id=conversation.get("project_id"),
            conversation_id=conversation["id"],
            system_rules=policy.get("rules", []),
        )

    def controlled_agent_reply(
        *,
        history: list[dict[str, Any]],
        user_text: str,
        settings: AgentSettings,
        summary: str,
        facts: list[dict[str, str]],
        memory_context: dict[str, Any],
        task_state: dict[str, Any],
        handoff_context: dict[str, Any],
        invariants: dict[str, Any],
        rag_context: str = "",
        tool_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Генерирует черновик и не выпускает его без независимой проверки политики."""
        available_tools: list[dict[str, Any]] = []
        tool_owners: dict[str, str] = {}
        created_scheduler_tasks: list[tuple[str, str]] = []
        created_report_paths: list[Path] = []
        mcp_enabled = mcp_control.is_enabled()
        task_control_enabled = task_control.is_enabled()
        if mcp_enabled and weather_mcp.status().get("connected"):
            for tool in weather_mcp.tools_for_model():
                available_tools.append(tool)
                tool_owners[tool["name"]] = "open-meteo"
        if mcp_enabled and mediawiki_mcp.status().get("connected"):
            for tool in mediawiki_mcp.tools_for_model():
                if tool["name"] in MEDIAWIKI_MODEL_TOOLS:
                    available_tools.append(tool)
                    tool_owners[tool["name"]] = "mediawiki"
        if mcp_enabled and worldbank_mcp.status().get("connected"):
            for tool in worldbank_mcp.tools_for_model():
                if tool["name"] in WORLDBANK_MODEL_TOOLS:
                    available_tools.append(tool)
                    tool_owners[tool["name"]] = "worldbank"
        if mcp_enabled and scheduler_mcp.status().get("connected") and _scheduled_tool_allowlist.get() is None:
            for tool in scheduler_mcp.tools_for_model():
                if tool["name"] in SCHEDULER_MODEL_TOOLS:
                    available_tools.append(tool)
                    tool_owners[tool["name"]] = "scheduler"
        allowlist = _scheduled_tool_allowlist.get()
        file_export_enabled = (
            mcp_enabled
            and bool((tool_context or {}).get("enable_file_export"))
            and allowlist is None
        )
        if file_export_enabled:
            if not mcp.status().get("connected"):
                mcp.start()
            filesystem_tools = mcp.list_tools().get("tools") or []
            for tool in filesystem_tools:
                if tool.get("name") not in FILESYSTEM_MODEL_TOOLS:
                    continue
                safe_tool = deepcopy(tool)
                safe_tool["description"] = (
                    "Сохранить подготовленный Markdown-отчёт. Инструмент доступен только при явной просьбе "
                    "пользователя сохранить файл. Передайте полное содержимое отчёта и осмысленное базовое имя; "
                    "приложение принудительно сохранит новый .md-файл в workspace/reports и добавит дату и время."
                )
                schema = deepcopy(safe_tool.get("input_schema") or {})
                properties = schema.setdefault("properties", {})
                properties.setdefault("path", {})["description"] = (
                    "Базовое имя отчёта, например weather-novosibirsk.md. Каталог будет заменён на workspace/reports."
                )
                properties.setdefault("content", {})["description"] = "Полное содержимое Markdown-отчёта."
                schema["required"] = ["path", "content"]
                safe_tool["input_schema"] = schema
                available_tools.append(safe_tool)
                tool_owners[safe_tool["name"]] = "filesystem"
        if allowlist is not None:
            available_tools = [tool for tool in available_tools if tool["name"] in allowlist]

        def execute_owned_tool(name: str, arguments: dict[str, Any], owner: str | None) -> dict[str, Any]:
            if not mcp_control.is_enabled():
                raise PermissionError("Все MCP принудительно отключены пользователем.")
            if owner == "open-meteo":
                return weather_mcp.call_tool(name, arguments)
            if owner == "mediawiki":
                return mediawiki_mcp.call_tool(name, arguments)
            if owner == "worldbank":
                return worldbank_mcp.call_tool(name, arguments)
            if owner == "filesystem":
                if name != "write_file" or not file_export_enabled:
                    raise PermissionError("Запись файлов не разрешена для этого запроса.")
                content = arguments.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Для отчёта требуется непустое текстовое содержимое.")
                if len(content) > 500_000:
                    raise ValueError("Отчёт превышает допустимый размер 500 000 символов.")
                workspace_value = mcp.status().get("workspace")
                if not workspace_value:
                    raise RuntimeError("Filesystem MCP не сообщил разрешённую рабочую папку.")
                workspace_path = Path(str(workspace_value)).resolve()
                report_path = unique_report_path(workspace_path, arguments.get("path"))
                output = mcp.call_tool(name, {"path": str(report_path), "content": content})
                if not output.get("is_error"):
                    created_report_paths.append(report_path)
                    output["result"] = {
                        "mcp_result": output.get("result"),
                        "relative_path": report_path.relative_to(workspace_path).as_posix(),
                        "absolute_path": str(report_path),
                    }
                return output
            if owner != "scheduler":
                raise RuntimeError(f"MCP-инструмент {name!r} не принадлежит подключённому серверу.")
            profile_id = str((tool_context or {}).get("profile_id") or current_profile_id())
            task_id = str(arguments.get("task_id") or "")
            if task_id:
                existing = scheduler_repository.get_task(task_id)
                if existing.get("owner_profile_id") not in {None, profile_id}:
                    raise PermissionError("Запланированное задание принадлежит другому профилю.")
            output = scheduler_mcp.call_tool(name, arguments)
            if output.get("is_error"):
                return output
            if name == "create_scheduled_task":
                created = output.get("result")
                if not isinstance(created, dict) or not created.get("id"):
                    raise RuntimeError("Scheduler MCP не вернул созданное задание.")
                try:
                    conversation = storage.create_conversation(
                        f"Автоматизация · {created.get('title', 'Задание')}",
                        (tool_context or {}).get("project_id"), profile_id,
                    )
                    attached = scheduler_repository.attach_context(
                        str(created["id"]), owner_profile_id=profile_id,
                        conversation_id=conversation["id"],
                        source_conversation_id=(tool_context or {}).get("source_conversation_id"),
                        project_id=(tool_context or {}).get("project_id"),
                        settings=(tool_context or {}).get("settings") or {},
                    )
                except Exception:
                    scheduler_repository.delete_task(str(created["id"]))
                    raise
                created_scheduler_tasks.append((str(created["id"]), conversation["id"]))
                output["result"] = attached
            elif name == "list_scheduled_tasks":
                tasks = scheduler_repository.list_tasks(profile_id)
                output["result"] = {"count": len(tasks), "timezone": TIMEZONE_NAME, "tasks": tasks}
            elif name == "get_scheduler_summary":
                output["result"] = scheduler_repository.summary(profile_id)
            return output

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            owner = tool_owners.get(name)
            activity_id = str((tool_context or {}).get("mcp_activity_id") or "") or None
            call_id = begin_tool_activity(activity_id, owner, name, arguments)
            try:
                output = execute_owned_tool(name, arguments, owner)
                output["mcp_server"] = owner
                finish_tool_activity(activity_id, call_id, is_error=bool(output.get("is_error")))
                return output
            except Exception as error:
                finish_tool_activity(activity_id, call_id, is_error=True, error=str(error))
                raise

        def rollback_created_schedules() -> None:
            for task_id, automation_conversation_id in created_scheduler_tasks:
                try:
                    scheduler_repository.delete_task(task_id)
                    storage.delete_conversation(automation_conversation_id)
                except (FileNotFoundError, ValueError):
                    logger.warning("Не удалось откатить Scheduler MCP | task_id=%s", task_id)

        def rollback_created_reports() -> None:
            for report_path in created_report_paths:
                try:
                    report_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Не удалось откатить файловый отчёт | path=%s", report_path)

        try:
            draft_result = chat_agent.reply(
                history,
                user_text,
                settings,
                summary=summary,
                facts=facts,
                memory_context=memory_context,
                task_handoff_context=handoff_context if task_control_enabled else {},
                task_state=task_state if task_control_enabled else {},
                policy_guidance=(generation_guidance(task_state, invariants) if task_control_enabled else "") + rag_context + (
                    "\n\nКОМПОЗИЦИЯ ОТЧЁТА:\n"
                    "Пользователь явно попросил сохранить результат. Сначала получи необходимые исходные данные "
                    "через доступные MCP-инструменты, затем самостоятельно проанализируй их строго по запросу "
                    "пользователя и вызови write_file ровно с готовым полным Markdown-отчётом. Не утверждай, что "
                    "файл сохранён, пока write_file не вернул успешный результат.\n"
                    if file_export_enabled else ""
                ) + (
                    "\n\nОРКЕСТРАЦИЯ MCP:\n"
                    "Выбирай только необходимые инструменты. Если результат одного MCP-сервера определяет "
                    "аргументы следующего, сначала получи первый результат, затем вызови следующий инструмент "
                    "с фактически найденными значениями. Не подменяй вызов догадкой.\n"
                ),
                tools=available_tools or None,
                tool_executor=execute_tool if available_tools else None,
            )
        except Exception:
            set_mcp_activity_phase(
                str((tool_context or {}).get("mcp_activity_id") or "") or None,
                "error",
            )
            rollback_created_schedules()
            rollback_created_reports()
            raise
        if not task_control_enabled:
            set_mcp_activity_phase(
                str((tool_context or {}).get("mcp_activity_id") or "") or None,
                "completed",
            )
            return AgentResult(
                content=draft_result.content,
                reasoning_content=draft_result.reasoning_content,
                technical={
                    **draft_result.technical,
                    "request_status": "completed",
                    "task_control": {"enabled": False},
                },
            )
        validation = None
        validation_result = None
        accepted = False
        reason = ""
        try:
            performed_actions = [
                str(item.get("name")) for item in draft_result.technical.get("mcp_tool_calls", [])
                if isinstance(item, dict) and (
                    str(item.get("name", "")).startswith(("create_scheduled_", "list_scheduled_", "get_scheduled_", "get_scheduler_"))
                    or str(item.get("name", "")) == "write_file"
                )
            ]
            validation_result = chat_agent.validate_task_response(
                validation_prompt(
                    user_text, draft_result.content, task_state, invariants,
                    performed_actions=performed_actions,
                ), settings
            )
            validation = parse_validation_result(validation_result.content, task_state, invariants)
            accepted = validation["allowed"]
            if not accepted:
                reason = validation.get("explanation") or "Фактическое действие ответа запрещено текущей политикой."
        except PolicyValidationError as error:
            reason = str(error)

        if not accepted:
            rollback_created_schedules()
            rollback_created_reports()

        set_mcp_activity_phase(
            str((tool_context or {}).get("mcp_activity_id") or "") or None,
            "completed" if accepted else "blocked",
        )

        content = draft_result.content if accepted else blocked_content(
            reason or "Ответ не прошёл обязательную проверку.",
            task_state,
            invariants,
            (validation or {}).get("violated_invariant_ids", []),
        )
        generation_usage = normalize_token_usage(draft_result.technical.get("usage"))
        validation_usage = normalize_token_usage(
            validation_result.technical.get("usage") if validation_result else {}
        )
        combined_usage = {key: generation_usage[key] + validation_usage[key] for key in generation_usage}
        audit = policy_audit(
            validation, task_state, invariants,
            accepted=accepted,
            reason=reason,
        )
        return AgentResult(
            content=content,
            reasoning_content=draft_result.reasoning_content if accepted else "",
            technical={
                **draft_result.technical,
                "usage": combined_usage,
                "policy_audit": audit,
                "policy_generation_usage": generation_usage,
                "policy_validation_usage": validation_usage,
                "request_status": "completed" if accepted else "blocked",
            },
        )

    def automatic_event_is_authorized(conversation: dict[str, Any], event: str) -> bool:
        """Принимает автопереход только из последнего проверенного ответа текущего запуска этапа."""
        if not task_control.is_enabled():
            return False
        state = ensure_task_state(conversation)
        if state.get("transition_mode") != "automatic" or event not in allowed_task_events(state):
            return False
        exchanges = current_stage_exchanges(conversation)
        if not exchanges:
            return False
        assistant = next(
            (item for item in reversed(exchanges[-1].get("messages", [])) if item.get("role") == "assistant"),
            None,
        )
        technical = (assistant or {}).get("technical", {})
        audit = technical.get("policy_audit", {}) if isinstance(technical, dict) else {}
        snapshot = technical.get("task_state", {}) if isinstance(technical, dict) else {}
        return bool(
            technical.get("request_status") == "completed"
            and audit.get("accepted") is True
            and audit.get("stage_complete") is True
            and audit.get("recommended_event") == event
            and snapshot.get("stage_run_id") == state.get("stage_run_id")
        )

    def automatic_continuation_text(state: dict[str, Any]) -> str:
        """Формирует видимое служебное сообщение автопилота без пользовательского ввода."""
        return (
            "[Автопилот] Продолжи задачу на текущем этапе. Используй активный handoff, состояние задачи, "
            "память и инварианты. Выполни весь доступный объём этапа. Если для безопасного продолжения нужен "
            "критический выбор пользователя или отсутствуют обязательные данные, явно запроси их и не считай "
            f"этап завершённым. Текущий этап: {state['stage']}; шаг: {state['current_step']}."
        )

    def assert_artifact_transition(conversation: dict[str, Any], event: str) -> None:
        if event == "complete_execution":
            missing = missing_required_artifacts(conversation, storage.data_dir)
            if missing:
                raise TaskTransitionError(
                    "Нельзя завершить выполнение: отсутствуют обязательные артефакты: " + ", ".join(missing) + "."
                )
        if event == "pass_validation" and not required_artifacts_are_validated(conversation, storage.data_dir):
            raise TaskTransitionError(
                "Нельзя завершить задачу: обязательные артефакты не подтверждены текущим этапом валидации."
            )

    def append_completion_result(conversation: dict[str, Any]) -> None:
        """Запрашивает у LLM итог этапа done и прикладывает серверный реестр файлов."""
        artifacts = [public_artifact(item) for item in active_artifacts(conversation)]
        final_handoff = next(
            (
                item for item in reversed(conversation.get("task_handoffs", []))
                if item.get("target_stage") == "done"
            ),
            {},
        )
        summary = str(final_handoff.get("summary") or "").strip()
        completed_work = [str(item) for item in final_handoff.get("completed_work", []) if str(item).strip()]
        artifact_lines = [
            f"- {item['path']} · версия {item['version']} · {item['size_bytes']} байт · SHA-256 {item['sha256'][:12]}…"
            for item in artifacts
        ]
        prompt = (
            "[Завершение задачи] Сформируй последнее итоговое сообщение пользователю. "
            "Задача уже прошла валидацию и находится на этапе done. Используй только авторитетный handoff, "
            "состояние задачи, память, инварианты и приведённый ниже точный реестр артефактов. "
            "Кратко сообщи конечный результат, перечисли фактически выполненную работу и обязательно перечисли "
            "каждый созданный артефакт по точному пути. Не придумывай отсутствующие файлы, проверки или ссылки; "
            "полное содержимое файлов повторять не нужно — интерфейс приложит кнопки открытия и скачивания.\n\n"
            f"Итог handoff: {summary or 'не указан'}\n"
            "Выполнено по handoff:\n"
            + ("\n".join(f"- {item}" for item in completed_work) or "- не указано")
            + "\nТочный реестр артефактов:\n"
            + ("\n".join(artifact_lines) or "- файловых артефактов нет")
        )
        settings_source = last_configuration(active_path_messages(conversation)) or {}
        settings = AgentSettings.from_dict(settings_source.get("settings", AgentSettings().to_dict()))
        configuration = deepcopy(settings_source.get("configuration_source")) or {
            "type": "custom", "preset_id": None, "preset_name": None,
        }
        history, summary_text, facts, context_snapshot = request_history(conversation)
        profile_id = current_profile_id()
        memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
        state_snapshot = deepcopy(ensure_task_state(conversation))
        state_snapshot["registered_artifacts"] = artifacts
        handoff_snapshot = task_handoff_context(conversation)
        invariant_snapshot = invariant_context(conversation, profile_id)
        result = controlled_agent_reply(
            history=history,
            user_text=prompt,
            settings=settings,
            summary=summary_text,
            facts=facts,
            memory_context=memory_snapshot,
            task_state=state_snapshot,
            handoff_context=handoff_snapshot,
            invariants=invariant_snapshot,
        )
        if result.technical.get("request_status") != "completed":
            reason = result.technical.get("policy_audit", {}).get("reason") or "финальный ответ заблокирован контроллером"
            raise TaskTransitionError(f"Переход в done не сохранён: {reason}.")
        exact_manifest = "\n".join([
            "",
            "Созданные артефакты:",
            *(artifact_lines or ["- файловых артефактов нет"]),
        ])
        usage = normalize_token_usage(result.technical.get("usage"))
        if artifacts:
            exact_manifest += "\nОткрыть или скачать файлы можно кнопками в карточке итогового результата."
        message = {
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "content": result.content.rstrip() + "\n" + exact_manifest,
            "reasoning_content": result.reasoning_content,
            "created_at": now_iso(),
            "technical": {
                **result.technical,
                "request_status": "completed",
                "final_completion": True,
                "configuration_source": configuration,
                "settings": settings.to_dict(),
                "context_management": context_snapshot,
                "memory_context": memory_snapshot,
                "profile_snapshot": deepcopy(memory_snapshot.get("profile_snapshot", {})),
                "task_state": state_snapshot,
                "task_handoff_context": handoff_snapshot,
                "invariants": invariant_snapshot,
                "result_manifest": {
                    "status": "done",
                    "summary": summary,
                    "completed_work": completed_work,
                    "artifacts": artifacts,
                },
            },
        }
        append_tree_message(conversation, message)
        totals = dialogue_token_totals(conversation.get("messages", []))
        conversation["token_totals"] = totals
        message["technical"]["dialogue_totals"] = totals

    def accessible_project(project_id: str, *, owner_only: bool = False) -> dict[str, Any]:
        project = storage.get_project(project_id)
        profile_id = current_profile_id()
        allowed = profile_id == project.get("owner_profile_id") if owner_only else storage.profile_can_access_project(project, profile_id)
        if not allowed:
            raise PermissionError(project_id)
        return project

    def accessible_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = storage.get_conversation(conversation_id)
        if conversation.get("project_id"):
            accessible_project(str(conversation["project_id"]))
        elif conversation.get("owner_profile_id") != current_profile_id():
            raise PermissionError(conversation_id)
        return conversation

    def paused_task_response(conversation: dict[str, Any]):
        return jsonify({
            "ok": False,
            "error": "Задача приостановлена. Нажмите «Продолжить», чтобы снова выполнять действия агента.",
            "conversation": with_token_totals(conversation),
        }), 409

    def append_task_state_report(conversation: dict[str, Any], content: str) -> dict[str, Any]:
        """Сохраняет локальную пару /state, не вызывая провайдера и не меняя контекст LLM."""
        ensure_context_management(conversation)
        exchange_id = uuid.uuid4().hex
        timestamp = now_iso()
        snapshot = deepcopy(ensure_task_state(conversation))
        user_message = {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "user",
            "content": content,
            "created_at": timestamp,
            "author_profile_id": current_profile_id(),
            "technical": {"request_status": "local", "local_command": "task_state"},
        }
        append_tree_message(conversation, user_message)
        append_tree_message(conversation, {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "assistant",
            "content": task_state_report(snapshot),
            "reasoning_content": "",
            "created_at": timestamp,
            "technical": {
                "request_status": "local",
                "local_command": "task_state",
                "task_state": snapshot,
            },
        }, parent_id=user_message["id"])
        if conversation.get("title") == "Новый диалог":
            conversation["title"] = "Состояние задачи"
        return storage.save_conversation(conversation)

    @flask_app.before_request
    def begin_request_log() -> None:
        g.request_started = time.perf_counter()
        g.request_id = uuid.uuid4().hex[:12]
        g.logging_request_token = set_request_id(g.request_id)

    @flask_app.after_request
    def finish_request_log(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        path = request.path
        level = logging.DEBUG if path == "/api/voice/status" else (
            logging.WARNING if response.status_code >= 400 else logging.INFO
        )
        if path.startswith("/api/"):
            logger.log(
                level,
                "HTTP | request_id=%s | method=%s | path=%s | status=%s | elapsed_ms=%.1f",
                g.get("request_id", "-"), request.method, path, response.status_code,
                (time.perf_counter() - g.get("request_started", time.perf_counter())) * 1000,
            )
        return response

    @flask_app.teardown_request
    def log_unhandled_request_error(error: BaseException | None) -> None:
        if error is not None:
            logger.error(
                "Необработанная ошибка HTTP | request_id=%s | method=%s | path=%s",
                g.get("request_id", "-"), request.method, request.path,
                exc_info=(type(error), error, error.__traceback__),
            )
        token = g.get("logging_request_token")
        if token is not None:
            reset_request_id(token)

    @flask_app.get("/")
    def index():
        return render_template("index.html")

    @flask_app.get("/api/state")
    def state():
        default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if default_model not in DEEPSEEK_MODELS:
            default_model = "deepseek-v4-flash"
        profile_id = current_profile_id()
        return jsonify({
            "ok": True,
            "provider": PROVIDER_CAPABILITIES,
            "default_settings": AgentSettings(model=default_model).to_dict(),
            "conversations": storage.list_conversations(profile_id),
            "projects": storage.list_projects(profile_id),
            "profiles": storage.list_profiles(),
            "active_profile_id": profile_id,
            "presets": preset_manager.list(),
        })

    @flask_app.get("/api/rag/state")
    def rag_state():
        return jsonify({"ok": True, **rag_index.state()})

    @flask_app.post("/api/rag/index")
    def rag_build_index():
        try:
            return jsonify({"ok": True, "job": rag_index.start_build(json_body())}), 202
        except RuntimeError as error:
            return api_error(str(error), 409)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/rag/chunks")
    def rag_chunks():
        try:
            strategy = str(request.args.get("strategy", "fixed"))
            page = int(request.args.get("page", 1))
            page_size = int(request.args.get("page_size", 20))
            source = str(request.args.get("source", ""))
            return jsonify({"ok": True, **rag_index.list_chunks(strategy, page, page_size, source)})
        except (TypeError, ValueError) as error:
            return api_error(str(error), 400)

    @flask_app.post("/api/rag/search")
    def rag_search():
        try:
            data = json_body()
            return jsonify({"ok": True, **rag_index.search(data.get("query", ""), data.get("top_k", 5))})
        except ValueError as error:
            return api_error(str(error), 400)
        except RuntimeError as error:
            return api_error(str(error), 409)

    @flask_app.get("/api/rag/evaluations")
    def rag_evaluation_state():
        questions_path = BASE_DIR / "docs" / "day22_control_questions.json"
        questions = []
        if questions_path.exists():
            value = json.loads(questions_path.read_text(encoding="utf-8"))
            questions = value.get("questions", []) if isinstance(value, dict) else []
        return jsonify({"ok": True, "questions": questions, "items": rag_evaluations.list()})

    @flask_app.post("/api/conversations/<conversation_id>/rag-compare")
    def compare_rag_answers(conversation_id: str):
        """Два изолированных LLM-вызова с одинаковыми настройками; историю не изменяет."""
        try:
            data = json_body()
            conversation = accessible_conversation(conversation_id)
            question = require_message(data.get("question"))
            settings = AgentSettings.from_dict(data.get("settings"))
            options = normalize_rag_options({
                "enabled": True,
                "strategy": data.get("strategy", "structural"),
                "top_k": data.get("top_k", 5),
            })
            retrieval = rag_index.retrieve(question, options["strategy"], options["top_k"])
            history, summary, facts, context_snapshot = request_history(conversation)
            profile_id = current_profile_id()
            memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
            state_snapshot = deepcopy(ensure_task_state(conversation))
            state_snapshot["registered_artifacts"] = [
                public_artifact(item) for item in active_artifacts(conversation)
            ]
            handoff_snapshot = task_handoff_context(conversation)
            invariant_snapshot = invariant_context(conversation, profile_id)
            common = {
                "history": history,
                "user_text": question,
                "settings": settings,
                "summary": summary,
                "facts": facts,
                "memory_context": memory_snapshot,
                "task_handoff_context": handoff_snapshot,
                "task_state": state_snapshot,
            }
            guidance = generation_guidance(state_snapshot, invariant_snapshot)
            without_rag = chat_agent.reply(**common, policy_guidance=guidance)
            with_rag = chat_agent.reply(
                **common,
                policy_guidance=guidance + build_rag_context(retrieval),
            )
            known_questions = {}
            questions_path = BASE_DIR / "docs" / "day22_control_questions.json"
            if questions_path.exists():
                value = json.loads(questions_path.read_text(encoding="utf-8"))
                known_questions = {
                    str(item.get("id")): item for item in value.get("questions", [])
                    if isinstance(item, dict) and item.get("id")
                }
            control = known_questions.get(str(data.get("question_id") or ""))
            item = rag_evaluations.add({
                "conversation_id": conversation_id,
                "question_id": (control or {}).get("id"),
                "question": question,
                "expectation": deepcopy((control or {}).get("expectation")),
                "expected_sources": deepcopy((control or {}).get("expected_sources", [])),
                "settings": settings.to_dict(),
                "retrieval": rag_snapshot(options, retrieval),
                "without_rag": {
                    "content": without_rag.content,
                    "reasoning_content": without_rag.reasoning_content,
                    "technical": without_rag.technical,
                },
                "with_rag": {
                    "content": with_rag.content,
                    "reasoning_content": with_rag.reasoning_content,
                    "technical": with_rag.technical,
                },
            })
            return jsonify({"ok": True, "evaluation": item})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except (ValueError, RuntimeError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            logger.exception("Ошибка сравнения RAG | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.get("/api/profiles")
    def list_profiles():
        return jsonify({
            "ok": True,
            "profiles": storage.list_profiles(),
            "active_profile_id": current_profile_id(),
        })

    @flask_app.post("/api/profiles")
    def create_profile():
        try:
            profile = memory_manager.create_profile(json_body())
            return jsonify({"ok": True, "profile": profile}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/profiles/<profile_id>")
    def update_profile(profile_id: str):
        try:
            return jsonify({"ok": True, "profile": memory_manager.update_profile(profile_id, json_body())})
        except (FileNotFoundError, PermissionError):
            return api_error("Профиль не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/profiles/active")
    def change_active_profile():
        try:
            profile = memory_manager.set_active_profile(str(json_body().get("profile_id", "")))
            return jsonify({"ok": True, "profile": profile})
        except FileNotFoundError:
            return api_error("Профиль не найден.", 404)

    @flask_app.get("/api/voice/status")
    def voice_status():
        return jsonify({"ok": True, "voice": whisper.status()})

    @flask_app.post("/api/voice/start")
    def voice_start():
        whisper.start()
        return jsonify({"ok": True, "voice": whisper.status()}), 202

    @flask_app.post("/api/voice/transcribe")
    def voice_transcribe():
        uploaded = request.files.get("audio")
        if uploaded is None:
            return api_error("Аудиозапись не передана.", 400)
        try:
            result = whisper.transcribe(
                uploaded.read(),
                uploaded.filename or "recording.webm",
                uploaded.mimetype or "application/octet-stream",
            )
            return jsonify({"ok": True, **result})
        except ValueError as error:
            return api_error(str(error), 400)
        except RuntimeError as error:
            return api_error(str(error), 503)

    @flask_app.get("/api/mcp/status")
    def mcp_status():
        return jsonify({"ok": True, "mcp": mcp.status()})

    @flask_app.get("/api/task-control")
    def task_control_status():
        return jsonify({"ok": True, "control": task_control.state()})

    @flask_app.post("/api/task-control/disable")
    def task_control_disable():
        control = task_control.set_enabled(False)
        logger.info("Машина задач принудительно отключена")
        return jsonify({"ok": True, "control": control})

    @flask_app.post("/api/task-control/enable")
    def task_control_enable():
        control = task_control.set_enabled(True)
        logger.info("Машина задач снова включена")
        return jsonify({"ok": True, "control": control})

    @flask_app.get("/api/mcp-control")
    def mcp_control_status():
        return jsonify({"ok": True, "control": mcp_control.state()})

    @flask_app.post("/api/mcp-control/disable")
    def mcp_control_disable():
        control = mcp_control.set_enabled(False)
        managers = {
            "filesystem": mcp,
            "open-meteo": weather_mcp,
            "scheduler": scheduler_mcp,
            "mediawiki": mediawiki_mcp,
            "worldbank": worldbank_mcp,
        }
        stopped: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for name, manager in managers.items():
            try:
                stopped[name] = manager.stop()
            except RuntimeError as error:
                errors[name] = str(error)
                stopped[name] = manager.status()
        logger.info("Все MCP принудительно отключены | errors=%s", sorted(errors))
        return jsonify({
            "ok": True,
            "control": control,
            "servers": stopped,
            "errors": errors,
        })

    @flask_app.post("/api/mcp-control/enable")
    def mcp_control_enable():
        control = mcp_control.set_enabled(True)
        logger.info("Запуск MCP снова разрешён; серверы остаются остановленными")
        return jsonify({"ok": True, "control": control})

    @flask_app.post("/api/mcp/start")
    def mcp_start():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": mcp.start()})
        except RuntimeError as error:
            logger.warning("MCP не запущен | reason=%s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": mcp.status()}), 503

    @flask_app.get("/api/mcp/tools")
    def mcp_tools():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": mcp.list_tools()})
        except RuntimeError as error:
            logger.warning("Не удалось получить MCP-инструменты | reason=%s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": mcp.status()}), 409

    @flask_app.post("/api/mcp/stop")
    def mcp_stop():
        try:
            return jsonify({"ok": True, "mcp": mcp.stop()})
        except RuntimeError as error:
            logger.warning("MCP не остановлен штатно | reason=%s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": mcp.status()}), 503

    @flask_app.get("/api/weather-mcp/status")
    def weather_mcp_status():
        return jsonify({"ok": True, "mcp": weather_mcp.status()})

    @flask_app.post("/api/weather-mcp/start")
    def weather_mcp_start():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            status = weather_mcp.start()
            tools = weather_mcp.list_tools()
            return jsonify({"ok": True, "mcp": {**status, **tools}})
        except RuntimeError as error:
            logger.warning("Open-Meteo MCP не запущен: %s", error)
            return jsonify({"ok": False, "error": str(error), "mcp": weather_mcp.status()}), 503

    @flask_app.get("/api/weather-mcp/tools")
    def weather_mcp_tools():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": weather_mcp.list_tools()})
        except RuntimeError as error:
            return jsonify({"ok": False, "error": str(error), "mcp": weather_mcp.status()}), 409

    @flask_app.post("/api/weather-mcp/stop")
    def weather_mcp_stop():
        try:
            return jsonify({"ok": True, "mcp": weather_mcp.stop()})
        except RuntimeError as error:
            return jsonify({"ok": False, "error": str(error), "mcp": weather_mcp.status()}), 503

    orchestration_managers = {
        "mediawiki": mediawiki_mcp,
        "worldbank": worldbank_mcp,
    }

    @flask_app.get("/api/orchestration-mcp/<server_name>/status")
    def orchestration_mcp_status(server_name: str):
        manager = orchestration_managers.get(server_name)
        if manager is None:
            return api_error("Неизвестный MCP-сервер.", 404)
        return jsonify({"ok": True, "mcp": manager.status()})

    @flask_app.route("/api/orchestration-mcp/<server_name>/<action>", methods=["GET", "POST"])
    def orchestration_mcp_action(server_name: str, action: str):
        manager = orchestration_managers.get(server_name)
        if manager is None or action not in {"start", "tools", "stop"}:
            return api_error("Неизвестная операция MCP.", 404)
        try:
            if action == "start":
                if not mcp_control.is_enabled():
                    return mcp_blocked_response()
                status = manager.start()
                tools = manager.list_tools()
                return jsonify({"ok": True, "mcp": {**status, **tools}})
            if action == "tools":
                if not mcp_control.is_enabled():
                    return mcp_blocked_response()
                return jsonify({"ok": True, "mcp": manager.list_tools()})
            return jsonify({"ok": True, "mcp": manager.stop()})
        except RuntimeError as error:
            status_code = 409 if action == "tools" else 503
            return jsonify({"ok": False, "error": str(error), "mcp": manager.status()}), status_code

    @flask_app.get("/api/mcp/activity/<activity_id>")
    def mcp_activity(activity_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", activity_id):
            return api_error("Некорректный идентификатор выполнения.", 400)
        with mcp_activity_lock:
            activity = deepcopy(mcp_activities.get(activity_id))
        if activity is None:
            return api_error("Выполнение ещё не зарегистрировано.", 404)
        return jsonify({"ok": True, "activity": activity})

    def owned_scheduled_task(task_id: str) -> dict[str, Any]:
        task = scheduler_repository.get_task(task_id)
        if task.get("owner_profile_id") != current_profile_id():
            raise PermissionError("Запланированное задание принадлежит другому профилю.")
        return task

    @flask_app.get("/api/scheduler/state")
    def scheduler_state():
        profile_id = current_profile_id()
        return jsonify({
            "ok": True,
            "scheduler": {
                "running": scheduler.status()["running"],
                "timezone": TIMEZONE_NAME,
            },
            "mcp": scheduler_mcp.status(),
            "tasks": scheduler_repository.list_tasks(profile_id),
            "runs": [
                run for run in scheduler_repository.list_runs(limit=100)
                if scheduler_repository.get_task(run["task_id"]).get("owner_profile_id") == profile_id
            ],
            "summary": scheduler_repository.summary(profile_id),
        })

    @flask_app.get("/api/scheduler/tools")
    def scheduler_tools():
        if not mcp_control.is_enabled():
            return mcp_blocked_response()
        try:
            return jsonify({"ok": True, "mcp": scheduler_mcp.list_tools()})
        except RuntimeError as error:
            return jsonify({"ok": False, "error": str(error), "mcp": scheduler_mcp.status()}), 409

    @flask_app.post("/api/scheduler/tasks")
    def create_scheduled_task_api():
        data = json_body()
        project_id = str(data.get("project_id") or "") or None
        try:
            if project_id:
                accessible_project(project_id)
            settings = AgentSettings.from_dict(data.get("settings")).to_dict()
            task = scheduler_repository.create_task(
                title=data.get("title", "Автоматизация"),
                prompt=data.get("prompt", ""),
                schedule_type=data.get("schedule_type", "once"),
                run_at=data.get("run_at"),
                delay_minutes=data.get("delay_minutes"),
                interval_minutes=data.get("interval_minutes"),
                mcp_servers=data.get("mcp_servers") or [],
                allowed_tools=data.get("allowed_tools") or [],
                owner_profile_id=current_profile_id(),
                source_conversation_id=data.get("source_conversation_id"),
                project_id=project_id,
                settings=settings,
            )
            try:
                conversation = storage.create_conversation(
                    f"Автоматизация · {task['title']}", project_id, current_profile_id()
                )
                task = scheduler_repository.attach_context(
                    task["id"], owner_profile_id=current_profile_id(),
                    conversation_id=conversation["id"],
                    source_conversation_id=data.get("source_conversation_id"),
                    project_id=project_id, settings=settings,
                )
            except Exception:
                scheduler_repository.delete_task(task["id"])
                raise
            return jsonify({"ok": True, "task": task, "conversation": with_token_totals(conversation)}), 201
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден или недоступен.", 404)

    @flask_app.patch("/api/scheduler/tasks/<task_id>")
    def update_scheduled_task_api(task_id: str):
        try:
            current = owned_scheduled_task(task_id)
            data = json_body()
            task = scheduler_repository.update_task(task_id, **data)
            if data.get("title") and current.get("conversation_id"):
                storage.rename_conversation(current["conversation_id"], f"Автоматизация · {task['title']}")
            return jsonify({"ok": True, "task": task})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)

    @flask_app.post("/api/scheduler/tasks/<task_id>/<action>")
    def scheduled_task_action(task_id: str, action: str):
        try:
            owned_scheduled_task(task_id)
            if action == "pause":
                task = scheduler_repository.set_status(task_id, "paused")
            elif action == "resume":
                task = scheduler_repository.set_status(task_id, "enabled")
            elif action == "run-now":
                task = scheduler_repository.request_run_now(task_id)
            else:
                return api_error("Неизвестное действие расписания.", 404)
            return jsonify({"ok": True, "task": task})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)
        except ValueError as error:
            return api_error(str(error), 409)

    @flask_app.delete("/api/scheduler/tasks/<task_id>")
    def delete_scheduled_task_api(task_id: str):
        try:
            owned_scheduled_task(task_id)
            scheduler_repository.delete_task(task_id)
            return jsonify({"ok": True})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)
        except ValueError as error:
            return api_error(str(error), 409)

    @flask_app.get("/api/scheduler/tasks/<task_id>/runs")
    def scheduled_task_runs(task_id: str):
        try:
            owned_scheduled_task(task_id)
            return jsonify({"ok": True, "runs": scheduler_repository.list_runs(task_id, 100)})
        except FileNotFoundError:
            return api_error("Запланированное задание не найдено.", 404)
        except PermissionError:
            return api_error("Нет доступа к запланированному заданию.", 403)

    @flask_app.post("/api/scheduler/notifications/read")
    def read_scheduler_notifications():
        count = scheduler_repository.mark_notifications_read(current_profile_id())
        return jsonify({"ok": True, "read_count": count, "summary": scheduler_repository.summary(current_profile_id())})

    @flask_app.post("/api/conversations")
    def create_conversation():
        try:
            data = json_body()
            project_id = data.get("project_id")
            if project_id:
                accessible_project(str(project_id))
            conversation = storage.create_conversation(
                data.get("title", "Новый диалог"), str(project_id) if project_id else None,
                current_profile_id(),
            )
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/conversations/<conversation_id>")
    def get_conversation(conversation_id: str):
        try:
            return jsonify({"ok": True, "conversation": with_token_totals(accessible_conversation(conversation_id))})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.patch("/api/conversations/<conversation_id>")
    def rename_conversation(conversation_id: str):
        try:
            accessible_conversation(conversation_id)
            conversation = storage.rename_conversation(conversation_id, json_body().get("title", ""))
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/conversations/<conversation_id>/task-state")
    def update_conversation_task_state(conversation_id: str):
        if not task_control.is_enabled():
            return task_control_blocked_response()
        try:
            conversation = accessible_conversation(conversation_id)
            changes = json_body()
            update_task_state(conversation, changes)
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)
        except Exception as error:
            logger.exception("Ошибка формирования handoff при ручной смене этапа | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.get("/api/conversations/<conversation_id>/task-state")
    def get_conversation_task_state(conversation_id: str):
        """Читает авторитетное состояние без обращения к модели."""
        try:
            conversation = accessible_conversation(conversation_id)
            state = ensure_task_state(conversation)
            return jsonify({
                "ok": True,
                "task_state": {**deepcopy(state), "allowed_events": allowed_task_events(state)},
                "report": task_state_report(state),
            })
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/task-state/events")
    def apply_conversation_task_event(conversation_id: str):
        """Применяет событие только через серверный граф переходов."""
        if not task_control.is_enabled():
            return task_control_blocked_response()
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            event = data.get("event")
            automatic = data.get("automatic") is True
            if automatic and not automatic_event_is_authorized(conversation, str(event)):
                raise TaskTransitionError("Автопереход не подтверждён последним проверенным ответом текущего этапа.")
            assert_artifact_transition(conversation, str(event))
            preview = deepcopy(conversation)
            before = deepcopy(ensure_task_state(conversation))
            apply_task_event(
                preview,
                event,
                source="autopilot" if automatic else "ui",
                reason=data.get("reason", ""),
            )
            target = ensure_task_state(preview)
            handoff = None
            if before["stage"] != target["stage"]:
                handoff = build_stage_handoff(
                    conversation, chat_agent, target_stage=target["stage"], event=str(event),
                )
            if handoff:
                activate_stage_handoff(preview, handoff)
            if ensure_task_state(preview)["stage"] == "done":
                append_completion_result(preview)
            conversation = storage.save_conversation(preview)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except TaskTransitionError as error:
            payload = {"ok": False, "error": str(error)}
            if "conversation" in locals():
                payload["conversation"] = with_token_totals(conversation)
            return jsonify(payload), 409
        except ValueError as error:
            return api_error(str(error), 400)
        except Exception as error:
            logger.exception("Ошибка handoff или финального ответа при переходе | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.get("/api/projects")
    def list_projects():
        return jsonify({"ok": True, "projects": storage.list_projects(current_profile_id())})

    @flask_app.post("/api/projects")
    def create_project():
        try:
            data = json_body()
            project = memory_manager.create_project(data.get("name"), data.get("description", ""), current_profile_id())
            payload: dict[str, Any] = {"ok": True, "project": project}
            if data.get("create_dialog") is True:
                payload["conversation"] = with_token_totals(storage.create_conversation("Новый диалог", project["id"], current_profile_id()))
            return jsonify(payload), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/projects/<project_id>")
    def get_project(project_id: str):
        try:
            project = accessible_project(project_id)
            project["conversations"] = [item for item in storage.list_conversations(current_profile_id()) if item.get("project_id") == project_id]
            return jsonify({"ok": True, "project": project})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)

    @flask_app.patch("/api/projects/<project_id>")
    def update_project(project_id: str):
        try:
            accessible_project(project_id, owner_only=True)
            return jsonify({"ok": True, "project": memory_manager.update_project(project_id, json_body())})
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/conversations/<conversation_id>/project")
    def set_conversation_project(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            project_id = json_body().get("project_id")
            if project_id is not None:
                accessible_project(str(project_id))
            conversation["project_id"] = project_id
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог или проект не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/copy-to-project")
    def copy_conversation_to_project(conversation_id: str):
        try:
            source = accessible_conversation(conversation_id)
            project_id = str(json_body().get("project_id", ""))
            accessible_project(project_id)
            copied = storage.copy_conversation_to_project(conversation_id, project_id)
            copy_artifact_files(storage.data_dir, source, copied)
            copied = storage.save_conversation(copied)
            return jsonify({"ok": True, "conversation": with_token_totals(copied)}), 201
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог или проект не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/extract-project-memory")
    def extract_project_memory_from_history(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            if task_is_paused(conversation):
                return paused_task_response(conversation)
            if not conversation.get("project_id"):
                raise ValueError("Сначала подключите диалог к проекту.")
            path = active_path_messages(conversation)
            exchanges = completed_exchanges(path)
            successful_messages = [
                deepcopy(message)
                for exchange in exchanges
                for message in exchange.get("messages", [])
            ]
            extraction_id = uuid.uuid4().hex
            result = chat_agent.extract_project_memories_from_history(successful_messages)
            revision = memory_manager.route_candidates(
                result, conversation, extraction_id, force_scopes={"project"},
            )
            revision["purpose"] = "project_history_extraction"
            revision["exchange_count"] = len(exchanges)
            conversation.setdefault("memory_extraction_revisions", []).append(revision)
            conversation = storage.save_conversation(conversation)
            if revision.get("status") != "completed":
                return api_error("Не удалось извлечь память из истории диалога.", 502)
            return jsonify({
                "ok": True, "revision": revision,
                "conversation": with_token_totals(conversation),
                "project": accessible_project(conversation["project_id"]),
            })
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог или проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)
        except Exception as error:
            logger.warning(
                "Ошибка явного извлечения памяти из истории | conversation_id=%s | error=%s",
                conversation_id, type(error).__name__, exc_info=True,
            )
            return api_error("Не удалось извлечь память из истории диалога.", 502)

    @flask_app.get("/api/projects/<project_id>/memory")
    def get_project_memory(project_id: str):
        try:
            accessible_project(project_id)
            return jsonify({"ok": True, **memory_manager.list_entries("project", project_id)})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)

    @flask_app.post("/api/projects/<project_id>/memory")
    def create_project_memory(project_id: str):
        try:
            accessible_project(project_id)
            entry = memory_manager.add_manual("project", json_body(), project_id)
            return jsonify({"ok": True, "memory": entry}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Проект не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/projects/<project_id>/memory/<memory_id>")
    def update_project_memory(project_id: str, memory_id: str):
        try:
            accessible_project(project_id)
            return jsonify({"ok": True, "memory": memory_manager.update_entry("project", memory_id, json_body(), project_id)})
        except (FileNotFoundError, KeyError, PermissionError):
            return api_error("Проект или запись памяти не найдены.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/projects/<project_id>/memory/<memory_id>")
    def delete_project_memory(project_id: str, memory_id: str):
        try:
            accessible_project(project_id)
            memory_manager.delete_entry("project", memory_id, project_id)
            return jsonify({"ok": True})
        except (FileNotFoundError, KeyError, PermissionError):
            return api_error("Проект или запись памяти не найдены.", 404)

    @flask_app.get("/api/memory/user")
    def get_user_memory():
        return jsonify({"ok": True, **memory_manager.list_entries("user", profile_id=current_profile_id())})

    @flask_app.post("/api/memory/user")
    def create_user_memory():
        try:
            data = json_body()
            if set(data) == {"automatic_extraction"}:
                return jsonify({"ok": True, "document": memory_manager.set_user_extraction(data["automatic_extraction"], current_profile_id())})
            return jsonify({"ok": True, "memory": memory_manager.add_manual("user", data, profile_id=current_profile_id())}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/memory/user/<memory_id>")
    def update_user_memory(memory_id: str):
        try:
            return jsonify({"ok": True, "memory": memory_manager.update_entry("user", memory_id, json_body(), profile_id=current_profile_id())})
        except KeyError:
            return api_error("Запись памяти не найдена.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/memory/user/<memory_id>")
    def delete_user_memory(memory_id: str):
        try:
            memory_manager.delete_entry("user", memory_id, profile_id=current_profile_id())
            return jsonify({"ok": True})
        except KeyError:
            return api_error("Запись памяти не найдена.", 404)

    @flask_app.get("/api/memory/policy")
    def get_memory_policy():
        return jsonify({"ok": True, "policy": memory_manager.policy()})

    def authorize_invariant_owner(scope: str, owner_id: str) -> None:
        if scope == "user":
            if owner_id != current_profile_id():
                raise PermissionError(owner_id)
        elif scope == "project":
            accessible_project(owner_id, owner_only=True)
        elif scope == "task":
            accessible_conversation(owner_id)
        else:
            raise ValueError("Область инвариантов должна быть user, project или task.")

    @flask_app.get("/api/invariants/context/<conversation_id>")
    def get_invariant_context(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            profile_id = current_profile_id()
            bundle = invariant_context(conversation, profile_id)
            editable_sets = {
                "user": invariant_store.list("user", profile_id),
                "project": invariant_store.list("project", conversation.get("project_id")) if conversation.get("project_id") else [],
                "task": invariant_store.list("task", conversation_id),
            }
            return jsonify({"ok": True, "bundle": bundle, "sets": editable_sets})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.post("/api/invariants")
    def create_invariant_set():
        try:
            data = json_body()
            authorize_invariant_owner(str(data.get("scope", "")), str(data.get("owner_id", "")))
            return jsonify({"ok": True, "invariant_set": invariant_store.create(data)}), 201
        except PermissionError:
            return api_error("Недостаточно прав для этой области инвариантов.", 403)
        except (ValueError, FileNotFoundError) as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/invariants/<set_id>")
    def update_invariant_set(set_id: str):
        try:
            existing = invariant_store.get(set_id)
            if existing is None:
                raise FileNotFoundError(set_id)
            authorize_invariant_owner(existing["scope"], existing["owner_id"])
            data = json_body()
            data.update({"scope": existing["scope"], "owner_id": existing["owner_id"]})
            return jsonify({"ok": True, "invariant_set": invariant_store.update(set_id, data)})
        except PermissionError:
            return api_error("Недостаточно прав для этой области инвариантов.", 403)
        except FileNotFoundError:
            return api_error("Набор инвариантов не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/invariants/<set_id>")
    def delete_invariant_set(set_id: str):
        try:
            existing = invariant_store.get(set_id)
            if existing is None:
                raise FileNotFoundError(set_id)
            authorize_invariant_owner(existing["scope"], existing["owner_id"])
            invariant_store.delete(set_id)
            return jsonify({"ok": True})
        except PermissionError:
            return api_error("Недостаточно прав для этой области инвариантов.", 403)
        except FileNotFoundError:
            return api_error("Набор инвариантов не найден.", 404)

    @flask_app.get("/api/conversations/<conversation_id>/memory-context")
    def get_memory_context(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            snapshots = [
                {"message_id": item.get("id"), "created_at": item.get("created_at"), "memory_context": item.get("technical", {}).get("memory_context")}
                for item in conversation.get("messages", [])
                if item.get("role") == "assistant" and item.get("technical", {}).get("memory_context")
            ]
            return jsonify({"ok": True, "snapshots": snapshots})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.patch("/api/conversations/<conversation_id>/context")
    def update_conversation_context(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            previous_mode = ensure_context_management(conversation)["mode"]
            data = json_body()
            mode = data.get("mode")
            set_context_mode(conversation, mode)
            set_window_sizes(
                conversation,
                sliding=data.get("sliding_window_exchanges"),
                facts=data.get("facts_window_exchanges"),
            )
            if previous_mode != mode:
                labels = {
                    "full": "Полная история",
                    "summary": "Summary + последние обмены",
                    "sliding": "Sliding Window",
                    "facts": "Sticky Facts",
                }
                conversation["messages"].append({
                    "id": uuid.uuid4().hex,
                    "role": "event",
                    "parent_id": conversation["context_management"].get("active_leaf_id"),
                    "content": f"Режим контекста: {labels[mode]}.",
                    "created_at": now_iso(),
                })
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id: str):
        try:
            accessible_conversation(conversation_id)
            delete_artifact_files(storage.data_dir, conversation_id)
            storage.delete_conversation(conversation_id)
            return jsonify({"ok": True})
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/messages")
    def send_message(conversation_id: str):
        try:
            data = json_body()
            mcp_activity_id = begin_mcp_activity(str(data.get("mcp_activity_id") or "") or None)
            settings = AgentSettings.from_dict(data.get("settings"))
            rag_options = normalize_rag_options(data.get("rag"))
            source = configuration_source(data.get("preset_id"), settings, preset_manager)
            conversation = accessible_conversation(conversation_id)
            automatic = data.get("automatic") is True
            task_control_enabled = task_control.is_enabled()
            state = ensure_task_state(conversation)
            if automatic:
                if not task_control_enabled:
                    raise TaskTransitionError("Машина задач отключена; автоматическое продолжение запрещено.")
                if state.get("transition_mode") != "automatic":
                    raise TaskTransitionError("Автопилот остановлен или переключён в ручной режим.")
                content = automatic_continuation_text(state)
            else:
                content = require_message(data.get("content"))
            if content.strip().lower() in {"/state", "/task-state"}:
                conversation = append_task_state_report(conversation, content)
                return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
            if task_control_enabled and task_is_paused(conversation):
                return paused_task_response(conversation)
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except TaskTransitionError as error:
            return api_error(str(error), 409)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)

        exchange_id = uuid.uuid4().hex
        created_at = now_iso()
        settings_snapshot = settings.to_dict()
        ensure_context_management(conversation)
        previous_source = last_configuration(active_path_messages(conversation))
        if previous_source is not None and (
            previous_source.get("settings") != settings_snapshot
            or previous_source.get("configuration_source") != source
        ):
            conversation["messages"].append({
                "id": uuid.uuid4().hex,
                "role": "event",
                "parent_id": conversation["context_management"].get("active_leaf_id"),
                "content": configuration_event_text(source),
                "created_at": created_at,
            })

        user_message = {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "user",
            "content": content,
            "created_at": created_at,
            "author_profile_id": current_profile_id(),
            "technical": {
                "request_status": "pending",
                "configuration_source": source,
                "settings": settings_snapshot,
                "rag": rag_snapshot(rag_options),
                "automatic_continuation": automatic,
                "scheduled_automation": deepcopy(_automation_run_override.get()),
                "task_control": {"enabled": task_control_enabled},
            },
        }
        append_tree_message(conversation, user_message)
        if conversation.get("title") == "Новый диалог":
            conversation["title"] = content.replace("\n", " ")[:60]
        conversation = storage.save_conversation(conversation)
        logger.info(
            "Сообщение принято | conversation_id=%s | exchange_id=%s | model=%s | mode=%s | chars=%s",
            conversation_id, exchange_id, settings.model,
            conversation["context_management"]["mode"], len(content),
        )

        try:
            retrieval = (
                rag_index.retrieve(content, rag_options["strategy"], rag_options["top_k"])
                if rag_options["enabled"] else None
            )
            request_rag_snapshot = rag_snapshot(rag_options, retrieval)
            update_sticky_facts(conversation, chat_agent, content, stage_scoped=task_control_enabled)
            update_summaries(conversation, chat_agent, stage_scoped=task_control_enabled)
            history_for_request, active_summary_text, active_facts, context_snapshot = request_history(
                conversation, stage_scoped=task_control_enabled,
            )
            profile_id = user_message["author_profile_id"]
            memory_snapshot = memory_manager.context_snapshot(conversation, context_snapshot["mode"], profile_id)
            task_state_snapshot = deepcopy(ensure_task_state(conversation))
            task_state_snapshot["registered_artifacts"] = [
                public_artifact(item) for item in active_artifacts(conversation)
            ]
            handoff_snapshot = task_handoff_context(conversation)
            invariant_snapshot = invariant_context(conversation, profile_id)
            conversation = storage.save_conversation(conversation)
            result = controlled_agent_reply(
                history=history_for_request,
                user_text=content,
                settings=settings,
                summary=active_summary_text,
                facts=active_facts,
                memory_context=memory_snapshot,
                task_state=task_state_snapshot,
                handoff_context=handoff_snapshot,
                invariants=invariant_snapshot,
                rag_context=build_rag_context(retrieval or {}),
                tool_context={
                    "profile_id": profile_id,
                    "source_conversation_id": conversation_id,
                    "project_id": conversation.get("project_id"),
                    "settings": settings_snapshot,
                    "enable_file_export": requests_file_export(content),
                    "mcp_activity_id": mcp_activity_id,
                },
            )
            usage = normalize_token_usage(result.technical.get("usage"))
            request_status = result.technical.get("request_status", "completed")
            assistant_message_id = uuid.uuid4().hex
            audit = result.technical.get("policy_audit", {})
            if task_control_enabled and request_status == "completed" and task_state_snapshot.get("stage") == "planning" and audit.get("stage_complete"):
                ensure_task_state(conversation)["required_artifacts"] = list(audit.get("required_artifacts", []))
                task_state_snapshot["required_artifacts"] = list(audit.get("required_artifacts", []))
            if task_control_enabled and request_status == "completed" and task_state_snapshot.get("stage") == "execution":
                try:
                    created_artifacts = save_response_artifacts(
                        storage.data_dir,
                        conversation,
                        result.content,
                        source_message_id=assistant_message_id,
                        stage_run_id=str(task_state_snapshot.get("stage_run_id", "")),
                    )
                    result.technical["artifacts"] = [public_artifact(item) for item in created_artifacts]
                    missing = missing_required_artifacts(conversation, storage.data_dir)
                    if audit.get("stage_complete") and missing:
                        audit.update({
                            "stage_complete": False,
                            "recommended_event": None,
                            "reason": "Ответ не создал обязательные артефакты: " + ", ".join(missing) + ".",
                        })
                except ArtifactError as artifact_error:
                    audit.update({
                        "stage_complete": False,
                        "recommended_event": None,
                        "reason": str(artifact_error),
                    })
            if (
                task_control_enabled
                and request_status == "completed"
                and task_state_snapshot.get("stage") == "validation"
                and audit.get("stage_complete")
                and audit.get("recommended_event") == "pass_validation"
            ):
                try:
                    verified = verify_required_artifacts(
                        conversation,
                        storage.data_dir,
                        validation_stage_run_id=str(task_state_snapshot.get("stage_run_id", "")),
                    )
                    result.technical["verified_artifacts"] = [public_artifact(item) for item in verified]
                except ArtifactError as artifact_error:
                    audit.update({
                        "stage_complete": False,
                        "recommended_event": None,
                        "reason": str(artifact_error),
                    })
            for message in conversation["messages"]:
                if message.get("id") == user_message["id"]:
                    message["technical"].update({
                        "request_status": request_status,
                        "token_usage": {
                            "context_tokens": usage["input_tokens"],
                            "cached_context_tokens": usage["cached_input_tokens"],
                            "uncached_context_tokens": usage["uncached_input_tokens"],
                        },
                        "context_management": context_snapshot,
                        "task_state": task_state_snapshot,
                        "task_handoff_context": handoff_snapshot,
                        "invariants": invariant_snapshot,
                        "rag": request_rag_snapshot,
                        "task_control": {"enabled": task_control_enabled},
                    })
                    break
            assistant_message = {
                "id": assistant_message_id,
                "exchange_id": exchange_id,
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "created_at": now_iso(),
                "technical": {
                    **result.technical,
                    "request_status": request_status,
                    "configuration_source": source,
                    "settings": settings_snapshot,
                    "context_management": context_snapshot,
                    "memory_context": memory_snapshot,
                    "profile_snapshot": deepcopy(memory_snapshot.get("profile_snapshot", {})),
                    "task_state": task_state_snapshot,
                    "task_handoff_context": handoff_snapshot,
                    "invariants": invariant_snapshot,
                    "rag": request_rag_snapshot,
                    "scheduled_automation": deepcopy(_automation_run_override.get()),
                    "task_control": {"enabled": task_control_enabled},
                },
            }
            append_tree_message(conversation, assistant_message, parent_id=user_message["id"])
            totals = dialogue_token_totals(conversation["messages"])
            conversation["token_totals"] = totals
            assistant_message["technical"]["dialogue_totals"] = totals
            conversation = storage.save_conversation(conversation)
            project = None
            if conversation.get("project_id"):
                project = storage.get_project(conversation["project_id"])
            allow_project = bool(project and project.get("settings", {}).get("automatic_extraction", True))
            allow_user = bool(memory_manager.user_document(profile_id).get("settings", {}).get("automatic_extraction", False))
            if request_status == "completed" and (allow_project or allow_user):
                try:
                    extraction_result = chat_agent.extract_memories(
                        content, result.content, allow_project=allow_project, allow_user=allow_user,
                    )
                    revision = memory_manager.route_candidates(extraction_result, conversation, exchange_id, profile_id=profile_id)
                except Exception as extraction_error:
                    logger.warning("Ошибка извлечения памяти | conversation_id=%s | error=%s", conversation_id, type(extraction_error).__name__)
                    revision = {
                        "id": uuid.uuid4().hex, "created_at": now_iso(), "status": "failed",
                        "error": type(extraction_error).__name__, "technical": {"usage": normalize_token_usage({})},
                    }
                conversation.setdefault("memory_extraction_revisions", []).append(revision)
                conversation = storage.save_conversation(conversation)
            logger.info(
                "Ответ сохранён | conversation_id=%s | exchange_id=%s | request_id=%s | total_tokens=%s",
                conversation_id, exchange_id, result.technical.get("request_id"), usage["total_tokens"],
            )
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except Exception as error:
            set_mcp_activity_phase(mcp_activity_id, "error", friendly_api_error(error))
            logger.exception(
                "Ошибка обработки сообщения | conversation_id=%s | exchange_id=%s | model=%s",
                conversation_id, exchange_id, settings.model,
            )
            message = friendly_api_error(error)
            for item in conversation["messages"]:
                if item.get("id") == user_message["id"]:
                    item["technical"].update({"request_status": "failed", "error": message})
                    break
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": False, "error": message, "conversation": with_token_totals(conversation)}), 502

    @flask_app.get("/api/conversations/<conversation_id>/export")
    def export_conversation(conversation_id: str):
        try:
            accessible_conversation(conversation_id)
            bundle = storage.export_bundle(conversation_id)
            bundle["conversation"] = with_token_totals(bundle["conversation"])
        except (ValueError, FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        payload = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(
            BytesIO(payload),
            mimetype="application/json; charset=utf-8",
            as_attachment=True,
            download_name=f"deepseek-dialog-{conversation_id[:8]}.json",
        )

    @flask_app.get("/api/conversations/<conversation_id>/artifacts/<artifact_id>")
    def download_artifact(conversation_id: str, artifact_id: str):
        """Отдаёт только зарегистрированный артефакт доступного пользователю диалога."""
        try:
            conversation = accessible_conversation(conversation_id)
            record = next(
                (item for item in conversation.get("artifacts", []) if item.get("id") == artifact_id),
                None,
            )
            if record is None:
                raise FileNotFoundError
            path = artifact_file_path(storage.data_dir, record)
            if not path.is_file():
                raise FileNotFoundError
            inline = request.args.get("disposition") == "inline"
            response = send_file(
                path,
                mimetype=record.get("mime_type") or "application/octet-stream",
                as_attachment=not inline,
                download_name=record.get("filename") or path.name,
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            if inline:
                response.headers["Content-Security-Policy"] = (
                    "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:"
                )
            return response
        except (ArtifactError, FileNotFoundError, PermissionError):
            return api_error("Артефакт не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/branches")
    def create_branch(conversation_id: str):
        """Начинает ветку после сообщения; после user сразу генерирует новый ответ."""
        try:
            conversation = accessible_conversation(conversation_id)
            task_control_enabled = task_control.is_enabled()
            if task_control_enabled and task_is_paused(conversation):
                return paused_task_response(conversation)
            checkpoint_id = json_body().get("checkpoint_id")
            source_message = next(
                (item for item in conversation.get("messages", []) if item.get("id") == checkpoint_id),
                None,
            )
            if source_message and source_message.get("technical", {}).get("local_command"):
                raise ValueError("Локальную команду состояния нельзя разветвить через модель.")
            if task_control_enabled and source_message:
                source_snapshot = source_message.get("technical", {}).get("task_state", {})
                current_state = ensure_task_state(conversation)
                stale_checkpoint = (
                    source_snapshot.get("stage_run_id") != current_state.get("stage_run_id")
                    if conversation.get("task_handoffs")
                    else bool(source_snapshot.get("stage") and source_snapshot.get("stage") != current_state.get("stage"))
                )
                if stale_checkpoint:
                    raise TaskTransitionError(
                        "Нельзя продолжить или перегенерировать сообщение из предыдущего этапа. "
                        "Используйте handoff текущего этапа."
                    )
            checkpoint = begin_branch(conversation, checkpoint_id)
            if checkpoint["role"] == "assistant":
                conversation = storage.save_conversation(conversation)
                return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201

            technical = checkpoint.get("technical", {})
            settings = AgentSettings.from_dict(technical.get("settings"))
            source = technical.get("configuration_source") or {
                "type": "custom", "preset_id": None, "preset_name": None,
            }
            update_summaries(conversation, chat_agent, stage_scoped=task_control_enabled)
            history, summary, facts, snapshot = request_history(conversation, stage_scoped=task_control_enabled)
            profile_id = checkpoint.get("author_profile_id") or conversation.get("owner_profile_id") or current_profile_id()
            memory_snapshot = memory_manager.context_snapshot(conversation, snapshot["mode"], profile_id)
            task_state_snapshot = deepcopy(ensure_task_state(conversation))
            handoff_snapshot = task_handoff_context(conversation)
            invariant_snapshot = invariant_context(conversation, profile_id)
            result = controlled_agent_reply(
                history=history,
                user_text=checkpoint["content"],
                settings=settings,
                summary=summary,
                facts=facts,
                memory_context=memory_snapshot,
                task_state=task_state_snapshot,
                handoff_context=handoff_snapshot,
                invariants=invariant_snapshot,
            )
            request_status = result.technical.get("request_status", "completed")
            retry_after_policy_block = technical.get("request_status") == "blocked"
            if retry_after_policy_block:
                usage = normalize_token_usage(result.technical.get("usage"))
                technical.update({
                    "request_status": request_status,
                    "token_usage": {
                        "context_tokens": usage["input_tokens"],
                        "cached_context_tokens": usage["cached_input_tokens"],
                        "uncached_context_tokens": usage["uncached_input_tokens"],
                    },
                    "context_management": snapshot,
                    "task_state": task_state_snapshot,
                    "task_handoff_context": handoff_snapshot,
                    "invariants": invariant_snapshot,
                })
            assistant = {
                "id": uuid.uuid4().hex,
                "exchange_id": checkpoint.get("exchange_id"),
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "created_at": now_iso(),
                "technical": {
                    **result.technical,
                    "request_status": request_status,
                    "configuration_source": source,
                    "settings": settings.to_dict(),
                    "context_management": snapshot,
                    "regenerated_from_checkpoint": checkpoint["id"],
                    "retry_after_policy_block": retry_after_policy_block,
                    "memory_context": memory_snapshot,
                    "profile_snapshot": deepcopy(memory_snapshot.get("profile_snapshot", {})),
                    "task_state": task_state_snapshot,
                    "task_handoff_context": handoff_snapshot,
                    "invariants": invariant_snapshot,
                    "task_control": {"enabled": task_control_enabled},
                },
            }
            append_tree_message(conversation, assistant, parent_id=checkpoint["id"])
            totals = dialogue_token_totals(conversation["messages"])
            conversation["token_totals"] = totals
            assistant["technical"]["dialogue_totals"] = totals
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except TaskTransitionError as error:
            return api_error(str(error), 409)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            logger.exception("Ошибка DeepSeek API при создании ветки | conversation_id=%s", conversation_id)
            return api_error(friendly_api_error(error), 502)

    @flask_app.patch("/api/conversations/<conversation_id>/branches/active")
    def change_active_branch(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            select_branch(conversation, data.get("checkpoint_id"), data.get("child_id"))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.post("/api/conversations/<conversation_id>/facts")
    def create_manual_fact(conversation_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            fact = add_fact(conversation, data.get("key"), data.get("value"))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "fact": fact, "conversation": with_token_totals(conversation)}), 201
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/conversations/<conversation_id>/facts/<fact_id>")
    def update_manual_fact(conversation_id: str, fact_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            data = json_body()
            fact = edit_fact(conversation, fact_id, data.get("key"), data.get("value"), data.get("locked", True))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "fact": fact, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except KeyError:
            return api_error("Факт не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/conversations/<conversation_id>/facts/<fact_id>")
    def remove_manual_fact(conversation_id: str, fact_id: str):
        try:
            conversation = accessible_conversation(conversation_id)
            delete_fact(conversation, fact_id)
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except (FileNotFoundError, PermissionError):
            return api_error("Диалог не найден.", 404)
        except KeyError:
            return api_error("Факт не найден.", 404)

    @flask_app.post("/api/import")
    def import_conversation():
        try:
            conversation = storage.import_bundle(json_body(), current_profile_id())
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (ValueError, OSError, json.JSONDecodeError) as error:
            return api_error(f"Не удалось импортировать файл: {error}", 400)

    @flask_app.post("/api/presets")
    def create_preset():
        try:
            return jsonify({"ok": True, "preset": preset_manager.create(json_body())}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/presets/<preset_id>")
    def update_preset(preset_id: str):
        try:
            return jsonify({"ok": True, "preset": preset_manager.update(preset_id, json_body())})
        except KeyError:
            return api_error("Пресет не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/presets/<preset_id>")
    def delete_preset(preset_id: str):
        try:
            preset_manager.delete(preset_id)
            return jsonify({"ok": True})
        except KeyError:
            return api_error("Пресет не найден.", 404)

    def run_scheduled_automation(task: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        if mcp_control.is_enabled() and "open-meteo" in task.get("mcp_servers", []):
            weather_mcp.start()
            weather_mcp.list_tools()
        profile_token = _profile_override.set(str(task["owner_profile_id"]))
        automation_token = _automation_run_override.set({
            "task_id": task["id"], "run_id": run["id"],
            "scheduled_for": run["scheduled_for"],
        })
        tools_token = _scheduled_tool_allowlist.set(set(task.get("allowed_tools", [])))
        try:
            client = flask_app.test_client()
            response = client.post(
                f"/api/conversations/{task['conversation_id']}/messages",
                json={"content": task["prompt"], "settings": task.get("settings") or {}},
            )
            payload = response.get_json(silent=True) or {}
            if response.status_code != 200 or not payload.get("ok"):
                raise RuntimeError(payload.get("error") or f"Внутренний запрос завершился с HTTP {response.status_code}.")
            messages = payload.get("conversation", {}).get("messages", [])
            assistant = next((item for item in reversed(messages) if item.get("role") == "assistant"), None)
            if not assistant:
                raise RuntimeError("Запланированный запуск не вернул ответ агента.")
            if assistant.get("technical", {}).get("request_status") != "completed":
                raise RuntimeError(assistant.get("content") or "Ответ автоматизации заблокирован.")
            return {"content": assistant.get("content", ""), "message_id": assistant.get("id")}
        finally:
            _scheduled_tool_allowlist.reset(tools_token)
            _automation_run_override.reset(automation_token)
            _profile_override.reset(profile_token)

    scheduler.set_runner(run_scheduled_automation)
    if scheduler_autostart:
        if mcp_control.is_enabled():
            scheduler_mcp.start()
            scheduler_mcp.list_tools()
        scheduler.start()
    if orchestration_autostart and mcp_control.is_enabled():
        for manager in (mediawiki_mcp, worldbank_mcp):
            manager.start()
            manager.list_tools()

    return flask_app


def json_body() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект.")
    return value


def require_message(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Введите сообщение.")
    content = value.strip()
    if len(content) > MAX_MESSAGE_LENGTH:
        raise ValueError("Сообщение длиннее 50 000 символов.")
    return content


def configuration_source(preset_id: Any, settings: AgentSettings, manager: PresetManager) -> dict[str, Any]:
    if not preset_id:
        return {"type": "custom", "preset_id": None, "preset_name": None}
    preset = manager.get(str(preset_id))
    source_type = "preset" if preset["settings"] == settings.to_dict() else "preset_modified"
    return {"type": source_type, "preset_id": preset["id"], "preset_name": preset["name"]}


def last_configuration(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("technical"), dict):
            return message["technical"]
    return None


def configuration_event_text(source: dict[str, Any]) -> str:
    if source["type"] == "preset":
        return f"Выбран пресет «{source['preset_name']}»."
    if source["type"] == "preset_modified":
        return f"Настройки пресета «{source['preset_name']}» изменены для следующего запроса."
    return "Для следующего запроса выбраны разовые настройки."


def api_error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def with_token_totals(conversation: dict[str, Any]) -> dict[str, Any]:
    """Добавляет вычисляемые представления, не изменяя сохранённый оригинал."""
    result = deepcopy(conversation)
    ensure_context_management(result)
    state = ensure_task_state(result)
    ensure_task_handoffs(result)
    stored_artifacts = ensure_artifacts(result)
    result["artifacts"] = [public_artifact(item) for item in stored_artifacts]
    result["active_artifacts"] = [public_artifact(item) for item in stored_artifacts if item.get("active") is True]
    result["token_totals"] = dialogue_token_totals(result.get("messages", []))
    visible = active_path_messages(result)
    result["visible_messages"] = visible
    result["active_path_token_totals"] = dialogue_token_totals(visible)
    result["branch_points"] = branch_points(result)
    result["summary_token_totals"] = summary_token_totals(result.get("summaries", []))
    result["facts_token_totals"] = facts_token_totals(result.get("fact_revisions", []))
    result["task_handoff_token_totals"] = handoff_token_totals(result.get("task_handoffs", []))
    result["active_task_handoffs"] = active_task_handoffs(result)
    result["active_stage_facts"] = deepcopy(current_stage_facts(result))
    result["current_stage_exchange_count"] = len(current_stage_exchanges(result))
    result["memory_token_totals"] = auxiliary_usage_totals(result.get("memory_extraction_revisions", []))
    result["task_state"]["allowed_events"] = allowed_task_events(result["task_state"])
    return result


def auxiliary_usage_totals(revisions: list[dict[str, Any]]) -> dict[str, int]:
    totals = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "uncached_input_tokens", "reasoning_tokens")}
    totals["request_count"] = 0
    for revision in revisions:
        technical = revision.get("technical", {}) if isinstance(revision, dict) else {}
        if revision.get("status") != "completed" or not isinstance(technical.get("usage"), dict):
            continue
        usage = normalize_token_usage(technical["usage"])
        totals["request_count"] += 1
        for key in usage:
            totals[key] += usage[key]
    return totals


def friendly_api_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    context_markers = (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "too many tokens",
        "token limit",
    )
    if any(marker in lowered for marker in context_markers):
        return (
            "Контекст диалога превысил лимит модели. "
            "Создайте новый диалог или сократите историю и повторите запрос."
        )
    if "api key" in lowered or "authentication" in lowered or "401" in lowered:
        return "DeepSeek отклонил API-ключ. Проверьте DEEPSEEK_API_KEY в файле .env."
    if "429" in lowered or "rate limit" in lowered:
        return "DeepSeek временно ограничил количество запросов. Попробуйте немного позже."
    if "timeout" in lowered or "timed out" in lowered:
        return "DeepSeek не успел ответить. Повторите запрос."
    if "connection" in lowered or "connect" in lowered or "network" in lowered:
        return "Не удалось соединиться с DeepSeek. Проверьте интернет, VPN или прокси."
    return f"Не удалось получить ответ DeepSeek: {text}"


_direct_log_paths = None
if __name__ == "__main__":
    from logging_setup import configure_logging

    _direct_log_paths = configure_logging(BASE_DIR / "logs")

app = create_app(voice_service=WhisperService(BASE_DIR, autostart=False), scheduler_autostart=False)


if __name__ == "__main__":
    app.extensions["whisper_service"].start()
    if app.extensions["mcp_control"].is_enabled():
        app.extensions["scheduler_mcp_manager"].start()
        app.extensions["scheduler_mcp_manager"].list_tools()
    app.extensions["scheduler_service"].start()
    if app.extensions["mcp_control"].is_enabled():
        for extension_name in ("mediawiki_mcp_manager", "worldbank_mcp_manager"):
            app.extensions[extension_name].start()
            app.extensions[extension_name].list_tools()
    logger.info("Приложение запущено напрямую через app.py | url=http://127.0.0.1:5000")
    print(f"Общий журнал: {_direct_log_paths['app']}")
    print(f"Журнал ошибок: {_direct_log_paths['errors']}")
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        app.extensions["scheduler_service"].stop()
        app.extensions["scheduler_mcp_manager"].stop()
        app.extensions["weather_mcp_manager"].stop()
        app.extensions["mediawiki_mcp_manager"].stop()
        app.extensions["worldbank_mcp_manager"].stop()
        app.extensions["mcp_manager"].stop()
        app.extensions["whisper_service"].stop()
        logger.info("Приложение остановлено")
