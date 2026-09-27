"""Долгоживущие stdio-соединения с локальными MCP-серверами."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import shutil
import sys
import threading
import os
from pathlib import Path
from typing import Any, Callable

from mcp import Client, StdioServerParameters


logger = logging.getLogger("deepseek_agent.mcp")
Command = tuple[str, dict[str, Any] | None, concurrent.futures.Future[Any]]


class StdioMCPManager:
    """Владеет одним MCP-процессом и одной сессией в отдельном async-потоке."""

    def __init__(self, *, command: str, args: list[str], process_name: str,
                 starting_message: str, installation_validator: Callable[[], None] | None = None,
                 workspace: Path | None = None, timeout_seconds: float = 20.0,
                 environment: dict[str, str] | None = None) -> None:
        self.command = command
        self.args = list(args)
        self.process_name = process_name
        self.starting_message = starting_message
        self.installation_validator = installation_validator
        self.workspace_dir = workspace.resolve() if workspace else None
        self.timeout_seconds = timeout_seconds
        self.environment = dict(environment or {})
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._commands: asyncio.Queue[Command] | None = None
        self._start_future: concurrent.futures.Future[dict[str, Any]] | None = None
        self._phase = "stopped"
        self._message = "MCP-сервер не запущен."
        self._error: str | None = None
        self._server_name: str | None = None
        self._server_version: str | None = None
        self._protocol_version: str | None = None
        self._tools: list[dict[str, Any]] = []
        atexit.register(self.stop)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._phase == "running":
                result = self._status_unlocked()
                result["already_running"] = True
                return result
            if self._phase == "stopping":
                raise RuntimeError("MCP-сервер сейчас останавливается. Повторите запуск через несколько секунд.")
            if self._phase == "starting" and self._start_future is not None:
                start_future = self._start_future
            else:
                if self.installation_validator:
                    self.installation_validator()
                self._phase = "starting"
                self._message = self.starting_message
                self._error = None
                self._tools = []
                start_future = concurrent.futures.Future()
                self._start_future = start_future
                self._thread = threading.Thread(target=self._thread_main, args=(start_future,), name=self.process_name, daemon=True)
                self._thread.start()
                logger.info("Запрошен запуск MCP | process=%s", self.process_name)
        try:
            return start_future.result(timeout=self.timeout_seconds + 5)
        except concurrent.futures.TimeoutError as error:
            raise RuntimeError("MCP-сервер не успел установить соединение.") from error

    def list_tools(self) -> dict[str, Any]:
        result = self._submit("list_tools")
        logger.info("MCP tools/list выполнен | server=%s | protocol=%s | tools=%s",
                    result.get("server", {}).get("name"), result.get("protocol_version"), result.get("tool_count"))
        return result

    def tools_for_model(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = list(self._tools)
        return cached or list(self.list_tools().get("tools") or [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            allowed = {tool["name"] for tool in self._tools}
        if name not in allowed:
            raise RuntimeError(f"MCP-инструмент {name!r} не зарегистрирован этим сервером.")
        return self._submit("call_tool", {"name": name, "arguments": arguments or {}})

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._phase == "stopped":
                return self._status_unlocked()
            if self._phase == "error" and not (self._thread and self._thread.is_alive()):
                self._phase = "stopped"
                self._message = "MCP-сервер остановлен."
                return self._status_unlocked()
            start_future = self._start_future if self._phase == "starting" else None
        if start_future is not None:
            try:
                start_future.result(timeout=self.timeout_seconds + 5)
            except Exception:
                with self._lock:
                    if not (self._thread and self._thread.is_alive()):
                        self._phase = "stopped"
                        self._message = "MCP-сервер остановлен."
                        return self._status_unlocked()
        with self._lock:
            if self._phase != "running":
                return self._status_unlocked()
            self._phase = "stopping"
            self._message = "Останавливаем MCP-сервер…"
        try:
            result = self._submit("stop", allow_stopping=True)
        except Exception as error:
            logger.warning("Ошибка остановки MCP: %s", error)
            with self._lock:
                result = self._status_unlocked()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
        return result

    def _submit(self, action: str, payload: dict[str, Any] | None = None, *, allow_stopping: bool = False) -> Any:
        with self._lock:
            allowed = self._phase == "running" or (allow_stopping and self._phase == "stopping")
            loop, commands = self._loop, self._commands
            if not allowed or loop is None or commands is None:
                raise RuntimeError("MCP-соединение не запущено. Сначала нажмите «Запустить MCP».")
            result_future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            loop.call_soon_threadsafe(commands.put_nowait, (action, payload, result_future))
        try:
            return result_future.result(timeout=self.timeout_seconds + 5)
        except concurrent.futures.TimeoutError as error:
            raise RuntimeError("MCP-сервер не ответил вовремя.") from error

    def _thread_main(self, start_future: concurrent.futures.Future[dict[str, Any]]) -> None:
        try:
            asyncio.run(self._serve(start_future))
        except BaseException as error:
            if not start_future.done():
                start_future.set_exception(self._public_error(error))
            self._record_failure(error)

    async def _serve(self, start_future: concurrent.futures.Future[dict[str, Any]]) -> None:
        stop_future: concurrent.futures.Future[Any] | None = None
        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env={**os.environ, **self.environment} if self.environment else None,
        )
        try:
            async with Client(params, read_timeout_seconds=self.timeout_seconds) as client:
                server_info = client.server_info
                with self._lock:
                    self._loop = asyncio.get_running_loop()
                    self._commands = asyncio.Queue()
                    self._server_name = getattr(server_info, "name", None)
                    self._server_version = getattr(server_info, "version", None)
                    self._protocol_version = client.protocol_version
                    self._phase = "running"
                    self._message = "MCP-соединение установлено."
                    self._error = None
                    connected = self._status_unlocked()
                start_future.set_result(connected)
                while True:
                    action, payload, result_future = await self._commands.get()
                    if action == "stop":
                        stop_future = result_future
                        break
                    try:
                        if action == "list_tools":
                            result_future.set_result(await self._list_all_tools(client))
                        elif action == "call_tool":
                            result_future.set_result(await self._call_tool(client, payload or {}))
                        else:
                            raise RuntimeError(f"Неизвестная команда MCP: {action}")
                    except BaseException as error:
                        result_future.set_exception(self._public_error(error))
        finally:
            with self._lock:
                self._loop = None
                self._commands = None
                self._start_future = None
                if self._phase in {"running", "stopping"}:
                    self._phase = "stopped"
                    self._message = "MCP-сервер остановлен."
                    self._error = None
                stopped = self._status_unlocked()
            if stop_future is not None and not stop_future.done():
                stop_future.set_result(stopped)
            logger.info("MCP-соединение закрыто | process=%s", self.process_name)

    async def _list_all_tools(self, client: Client) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor, cache_mode="reload")
            tools.extend({
                "name": tool.name, "title": getattr(tool, "title", None),
                "description": tool.description or "", "input_schema": tool.input_schema,
            } for tool in page.tools)
            cursor = page.next_cursor
            if cursor is None:
                break
        with self._lock:
            self._tools = tools
            self._message = f"Получено инструментов: {len(tools)}."
            status = self._status_unlocked()
        return {**status, "tool_count": len(tools), "tools": tools}

    async def _call_tool(self, client: Client, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "")
        arguments = payload.get("arguments") or {}
        result = await client.call_tool(name, arguments)
        structured = getattr(result, "structured_content", None)
        if structured is None:
            blocks = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                blocks.append(text if text is not None else item.model_dump(mode="json"))
            structured = blocks
        output = {"tool": name, "arguments": arguments, "is_error": bool(getattr(result, "is_error", False)), "result": structured}
        logger.info("MCP tools/call выполнен | server=%s | tool=%s | error=%s", self._server_name, name, output["is_error"])
        return output

    def _record_failure(self, error: BaseException) -> None:
        public_error = self._public_error(error)
        with self._lock:
            self._loop = None
            self._commands = None
            self._start_future = None
            self._phase = "error"
            self._message = str(public_error)
            self._error = str(public_error)
        logger.error("Ошибка MCP-соединения: %s", error, exc_info=True)

    @staticmethod
    def _public_error(error: BaseException) -> RuntimeError:
        text = str(error).strip() or error.__class__.__name__
        return RuntimeError(f"Ошибка MCP: {text}")

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "phase": self._phase, "connected": self._phase == "running", "message": self._message,
            "error": self._error, "server": {"name": self._server_name, "version": self._server_version},
            "protocol_version": self._protocol_version,
            "workspace": str(self.workspace_dir) if self.workspace_dir else None,
            "tool_count": len(self._tools),
        }


class MCPManager(StdioMCPManager):
    """Совместимый менеджер Filesystem MCP из Дня 16."""

    def __init__(self, project_dir: Path, timeout_seconds: float = 20.0) -> None:
        project_dir = Path(project_dir).resolve()
        workspace = (project_dir / "workspace").resolve()
        server_script = (project_dir / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js").resolve()

        def validate() -> None:
            if shutil.which("node") is None:
                raise RuntimeError("Node.js не найден. Установите Node.js 18+ и повторите запуск.")
            if not server_script.is_file():
                raise RuntimeError("Filesystem MCP Server не установлен. Выполните в папке проекта: npm install")
            if workspace.parent != project_dir:
                raise RuntimeError("Папка MCP workspace должна находиться внутри проекта.")
            workspace.mkdir(parents=True, exist_ok=True)

        super().__init__(command="node", args=[str(server_script), str(workspace)], process_name="filesystem-mcp",
                         starting_message="Запускаем Filesystem MCP Server…", installation_validator=validate,
                         workspace=workspace, timeout_seconds=timeout_seconds)


class OpenMeteoMCPManager(StdioMCPManager):
    """Менеджер собственного Open-Meteo MCP-сервера Дня 17."""

    def __init__(self, project_dir: Path, timeout_seconds: float = 30.0) -> None:
        project_dir = Path(project_dir).resolve()
        module_file = project_dir / "mcp_servers" / "open_meteo" / "server.py"

        def validate() -> None:
            if not module_file.is_file():
                raise RuntimeError("Не найден mcp_servers/open_meteo/server.py.")

        super().__init__(command=sys.executable, args=["-m", "mcp_servers.open_meteo.server"], process_name="open-meteo-mcp",
                         starting_message="Запускаем Open-Meteo MCP Server…", installation_validator=validate,
                         timeout_seconds=timeout_seconds)


class SchedulerMCPManager(StdioMCPManager):
    """Менеджер Scheduler MCP с общим для процесса Flask SQLite-файлом."""

    def __init__(self, project_dir: Path, database_path: Path, timeout_seconds: float = 20.0) -> None:
        project_dir = Path(project_dir).resolve()
        module_file = project_dir / "mcp_servers" / "scheduler" / "server.py"

        def validate() -> None:
            if not module_file.is_file():
                raise RuntimeError("Не найден mcp_servers/scheduler/server.py.")

        super().__init__(
            command=sys.executable,
            args=["-m", "mcp_servers.scheduler.server"],
            process_name="scheduler-mcp",
            starting_message="Запускаем Scheduler MCP Server…",
            installation_validator=validate,
            timeout_seconds=timeout_seconds,
            environment={"DEEPSEEK_SCHEDULER_DB": str(Path(database_path).resolve())},
        )


class NodePackageMCPManager(StdioMCPManager):
    """Запускает установленный в проекте Node.js MCP-сервер без npx и сети."""

    def __init__(self, project_dir: Path, *, package_path: str, process_name: str,
                 starting_message: str, environment: dict[str, str] | None = None,
                 timeout_seconds: float = 30.0) -> None:
        project_dir = Path(project_dir).resolve()
        server_script = (project_dir / "node_modules" / Path(package_path) / "dist" / "index.js").resolve()

        def validate() -> None:
            if shutil.which("node") is None:
                raise RuntimeError("Node.js не найден. Установите Node.js и повторите запуск.")
            if not server_script.is_file():
                raise RuntimeError(f"MCP-пакет {package_path} не установлен. Выполните в папке проекта: npm install")

        super().__init__(
            command="node",
            args=[str(server_script)],
            process_name=process_name,
            starting_message=starting_message,
            installation_validator=validate,
            timeout_seconds=timeout_seconds,
            environment=environment,
        )


class MediaWikiMCPManager(NodePackageMCPManager):
    """ProfessionalWiki MediaWiki MCP с анонимным read-only доступом к Wikipedia."""

    def __init__(self, project_dir: Path, timeout_seconds: float = 30.0) -> None:
        project_dir = Path(project_dir).resolve()
        config_file = (project_dir / "config" / "mediawiki_mcp.json").resolve()
        super().__init__(
            project_dir,
            package_path="@professional-wiki/mediawiki-mcp-server",
            process_name="mediawiki-mcp",
            starting_message="Запускаем ProfessionalWiki MediaWiki MCP Server…",
            environment={
                "CONFIG": str(config_file),
                "MCP_TRANSPORT": "stdio",
                "MCP_LOG_LEVEL": "error",
            },
            timeout_seconds=timeout_seconds,
        )


class WorldBankMCPManager(NodePackageMCPManager):
    """World Bank Open Data MCP без API-ключа."""

    def __init__(self, project_dir: Path, timeout_seconds: float = 40.0) -> None:
        super().__init__(
            project_dir,
            package_path="@cyanheads/worldbank-mcp-server",
            process_name="worldbank-mcp",
            starting_message="Запускаем World Bank MCP Server…",
            environment={"MCP_TRANSPORT_TYPE": "stdio", "MCP_LOG_LEVEL": "error"},
            timeout_seconds=timeout_seconds,
        )
