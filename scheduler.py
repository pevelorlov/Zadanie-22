"""SQLite-хранилище и фоновый исполнитель запланированных задач."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


TIMEZONE_NAME = "Asia/Novosibirsk"
TIMEZONE = timezone(timedelta(hours=7), name=TIMEZONE_NAME)
KNOWN_MCP_TOOLS = {
    "open-meteo": {
        "find_location",
        "get_current_weather",
        "get_hourly_forecast",
        "get_daily_forecast",
        "get_precipitation_window",
    },
}
logger = logging.getLogger("deepseek_agent.scheduler")


def scheduler_now() -> datetime:
    return datetime.now(TIMEZONE)


def to_iso(value: datetime) -> str:
    return value.astimezone(TIMEZONE).isoformat(timespec="seconds")


def parse_local_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Дата и время запуска не указаны.")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Используйте дату в формате YYYY-MM-DDTHH:MM:SS.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TIMEZONE)
    return parsed.astimezone(TIMEZONE)


def normalize_dependencies(mcp_servers: list[str] | None, allowed_tools: list[str] | None) -> tuple[list[str], list[str]]:
    servers = list(dict.fromkeys(str(item).strip().lower() for item in (mcp_servers or []) if str(item).strip()))
    tools = list(dict.fromkeys(str(item).strip() for item in (allowed_tools or []) if str(item).strip()))
    unknown_servers = [server for server in servers if server not in KNOWN_MCP_TOOLS]
    if unknown_servers:
        raise ValueError("Неизвестные MCP-серверы: " + ", ".join(unknown_servers) + ".")
    available = set().union(*(KNOWN_MCP_TOOLS[server] for server in servers)) if servers else set()
    if servers and not tools:
        raise ValueError("Для выбранного MCP-сервера укажите хотя бы один разрешённый инструмент.")
    unknown_tools = [tool for tool in tools if tool not in available]
    if unknown_tools:
        raise ValueError("Инструменты не принадлежат выбранным MCP-серверам: " + ", ".join(unknown_tools) + ".")
    return servers, tools


class SchedulerRepository:
    """Потокобезопасный CRUD расписания и истории запусков."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    schedule_type TEXT NOT NULL CHECK(schedule_type IN ('once', 'interval')),
                    run_at TEXT,
                    interval_seconds INTEGER,
                    next_run_at TEXT,
                    status TEXT NOT NULL CHECK(status IN ('enabled', 'paused', 'completed', 'missed')),
                    running INTEGER NOT NULL DEFAULT 0,
                    owner_profile_id TEXT,
                    conversation_id TEXT,
                    source_conversation_id TEXT,
                    project_id TEXT,
                    mcp_servers_json TEXT NOT NULL DEFAULT '[]',
                    allowed_tools_json TEXT NOT NULL DEFAULT '[]',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    last_run_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due
                    ON scheduled_tasks(status, running, next_run_at);
                CREATE TABLE IF NOT EXISTS scheduled_task_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
                    scheduled_for TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
                    result_text TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    conversation_message_id TEXT,
                    duration_ms INTEGER,
                    unread INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_runs_task
                    ON scheduled_task_runs(task_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_scheduled_runs_unread
                    ON scheduled_task_runs(unread, completed_at DESC);
                """
            )

    def create_task(
        self,
        *,
        title: str,
        prompt: str,
        schedule_type: str,
        run_at: str | None = None,
        delay_minutes: float | None = None,
        interval_minutes: float | None = None,
        mcp_servers: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        owner_profile_id: str | None = None,
        conversation_id: str | None = None,
        source_conversation_id: str | None = None,
        project_id: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        title = str(title or "").strip()
        prompt = str(prompt or "").strip()
        if not title or len(title) > 160:
            raise ValueError("Название задания должно содержать от 1 до 160 символов.")
        if not prompt or len(prompt) > 50_000:
            raise ValueError("Промпт задания должен содержать от 1 до 50000 символов.")
        schedule_type = str(schedule_type or "").strip().lower()
        if schedule_type not in {"once", "interval"}:
            raise ValueError("schedule_type должен быть once или interval.")
        now = scheduler_now()
        interval_seconds: int | None = None
        if schedule_type == "once":
            if delay_minutes is not None:
                delay = float(delay_minutes)
                if delay <= 0:
                    raise ValueError("delay_minutes должен быть больше нуля.")
                first_run = now + timedelta(minutes=delay)
            else:
                first_run = parse_local_datetime(str(run_at or ""))
                if first_run <= now:
                    raise ValueError("Разовый запуск должен находиться в будущем.")
            normalized_run_at = to_iso(first_run)
        else:
            interval = float(interval_minutes or 0)
            if interval < 5:
                raise ValueError("Минимальный период автоматического задания — 5 минут.")
            interval_seconds = max(300, int(round(interval * 60)))
            if run_at:
                first_run = parse_local_datetime(run_at)
                if first_run <= now:
                    first_run = self._next_interval(first_run, interval_seconds, now)
            elif delay_minutes is not None:
                delay = float(delay_minutes)
                if delay <= 0:
                    raise ValueError("delay_minutes должен быть больше нуля.")
                first_run = now + timedelta(minutes=delay)
            else:
                first_run = now + timedelta(seconds=interval_seconds)
            normalized_run_at = None
        servers, tools = normalize_dependencies(mcp_servers, allowed_tools)
        timestamp = to_iso(now)
        task_id = uuid.uuid4().hex
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO scheduled_tasks (
                    id, title, prompt, schedule_type, run_at, interval_seconds, next_run_at,
                    status, owner_profile_id, conversation_id, source_conversation_id,
                    project_id, mcp_servers_json, allowed_tools_json, settings_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'enabled', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id, title, prompt, schedule_type, normalized_run_at, interval_seconds,
                    to_iso(first_run), owner_profile_id, conversation_id, source_conversation_id,
                    project_id, json.dumps(servers, ensure_ascii=False),
                    json.dumps(tools, ensure_ascii=False), json.dumps(settings or {}, ensure_ascii=False),
                    timestamp, timestamp,
                ),
            )
        return self.get_task(task_id)

    def list_tasks(self, owner_profile_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM scheduled_tasks"
        params: tuple[Any, ...] = ()
        if owner_profile_id:
            query += " WHERE owner_profile_id = ?"
            params = (owner_profile_id,)
        query += " ORDER BY created_at DESC"
        with self._lock, self._connection() as connection:
            return [self._task(row) for row in connection.execute(query, params).fetchall()]

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("Запланированное задание не найдено.")
        return self._task(row)

    def attach_context(self, task_id: str, *, owner_profile_id: str, conversation_id: str,
                       source_conversation_id: str | None = None, project_id: str | None = None,
                       settings: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            changed = connection.execute(
                """UPDATE scheduled_tasks SET owner_profile_id = ?, conversation_id = ?,
                    source_conversation_id = ?, project_id = ?, settings_json = ?, updated_at = ?
                    WHERE id = ?""",
                (owner_profile_id, conversation_id, source_conversation_id, project_id,
                 json.dumps(settings or {}, ensure_ascii=False), to_iso(scheduler_now()), task_id),
            ).rowcount
        if not changed:
            raise FileNotFoundError("Запланированное задание не найдено.")
        return self.get_task(task_id)

    def update_task(self, task_id: str, **changes: Any) -> dict[str, Any]:
        current = self.get_task(task_id)
        title = str(changes.get("title", current["title"])).strip()
        prompt = str(changes.get("prompt", current["prompt"])).strip()
        if not title or len(title) > 160 or not prompt or len(prompt) > 50_000:
            raise ValueError("Проверьте название и промпт задания.")
        servers, tools = normalize_dependencies(
            changes.get("mcp_servers", current["mcp_servers"]),
            changes.get("allowed_tools", current["allowed_tools"]),
        )
        fields = {
            "title": title,
            "prompt": prompt,
            "mcp_servers_json": json.dumps(servers, ensure_ascii=False),
            "allowed_tools_json": json.dumps(tools, ensure_ascii=False),
            "updated_at": to_iso(scheduler_now()),
        }
        if "next_run_at" in changes and changes["next_run_at"]:
            value = parse_local_datetime(str(changes["next_run_at"]))
            if value <= scheduler_now():
                raise ValueError("Следующий запуск должен находиться в будущем.")
            fields["next_run_at"] = to_iso(value)
            if current["schedule_type"] == "once":
                fields["run_at"] = to_iso(value)
                fields["status"] = "enabled"
        if "interval_minutes" in changes and current["schedule_type"] == "interval":
            interval = float(changes["interval_minutes"])
            if interval < 5:
                raise ValueError("Минимальный период автоматического задания — 5 минут.")
            fields["interval_seconds"] = int(round(interval * 60))
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock, self._connection() as connection:
            connection.execute(
                f"UPDATE scheduled_tasks SET {assignments} WHERE id = ?",
                (*fields.values(), task_id),
            )
        return self.get_task(task_id)

    def set_status(self, task_id: str, status: str) -> dict[str, Any]:
        current = self.get_task(task_id)
        if current["running"]:
            raise ValueError("Нельзя менять состояние выполняющегося задания.")
        if status not in {"enabled", "paused"}:
            raise ValueError("Допустимы только enabled и paused.")
        if status == "enabled" and current["schedule_type"] == "once" and parse_local_datetime(current["next_run_at"]) <= scheduler_now():
            raise ValueError("Нельзя включить разовое задание с прошедшим временем.")
        next_run = current["next_run_at"]
        if status == "enabled" and current["schedule_type"] == "interval":
            next_run = to_iso(self._next_interval(
                parse_local_datetime(next_run), int(current["interval_seconds"]), scheduler_now()
            ))
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE scheduled_tasks SET status = ?, next_run_at = ?, running = 0, updated_at = ? WHERE id = ?",
                (status, next_run, to_iso(scheduler_now()), task_id),
            )
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> None:
        current = self.get_task(task_id)
        if current["running"]:
            raise ValueError("Нельзя удалить выполняющееся задание.")
        with self._lock, self._connection() as connection:
            changed = connection.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,)).rowcount
        if not changed:
            raise FileNotFoundError("Запланированное задание не найдено.")

    def request_run_now(self, task_id: str) -> dict[str, Any]:
        current = self.get_task(task_id)
        if current["running"]:
            raise ValueError("Задание уже выполняется.")
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE scheduled_tasks SET status = 'enabled', next_run_at = ?, updated_at = ? WHERE id = ?",
                (to_iso(scheduler_now()), to_iso(scheduler_now()), task_id),
            )
        return self.get_task(task_id)

    def skip_missed_after_restart(self) -> None:
        now = scheduler_now()
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM scheduled_tasks WHERE status = 'enabled' AND next_run_at <= ?",
                (to_iso(now),),
            ).fetchall()
            for row in rows:
                task = self._task(row)
                if task["schedule_type"] == "once":
                    connection.execute(
                        "UPDATE scheduled_tasks SET status = 'missed', running = 0, updated_at = ? WHERE id = ?",
                        (to_iso(now), task["id"]),
                    )
                else:
                    next_run = self._next_interval(
                        parse_local_datetime(task["next_run_at"]), int(task["interval_seconds"]), now
                    )
                    connection.execute(
                        "UPDATE scheduled_tasks SET next_run_at = ?, running = 0, updated_at = ? WHERE id = ?",
                        (to_iso(next_run), to_iso(now), task["id"]),
                    )

    def claim_due_task(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        now = scheduler_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM scheduled_tasks
                   WHERE status = 'enabled' AND running = 0 AND conversation_id IS NOT NULL
                     AND next_run_at <= ? ORDER BY next_run_at LIMIT 1""",
                (to_iso(now),),
            ).fetchone()
            if row is None:
                return None
            task = self._task(row)
            changed = connection.execute(
                "UPDATE scheduled_tasks SET running = 1, updated_at = ? WHERE id = ? AND running = 0",
                (to_iso(now), task["id"]),
            ).rowcount
            if not changed:
                return None
            run_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO scheduled_task_runs
                   (id, task_id, scheduled_for, started_at, status, unread)
                   VALUES (?, ?, ?, ?, 'running', 0)""",
                (run_id, task["id"], task["next_run_at"], to_iso(now)),
            )
        return task, self.get_run(run_id)

    def finish_run(self, run_id: str, *, result_text: str = "", error: str = "",
                   conversation_message_id: str | None = None) -> dict[str, Any]:
        run = self.get_run(run_id)
        task = self.get_task(run["task_id"])
        completed_at = scheduler_now()
        started_at = parse_local_datetime(run["started_at"])
        run_status = "failed" if error else "completed"
        if task["schedule_type"] == "once":
            task_status = "completed"
            next_run_at = task["next_run_at"]
        else:
            task_status = "enabled"
            next_run_at = to_iso(self._next_interval(
                parse_local_datetime(task["next_run_at"]), int(task["interval_seconds"]), completed_at
            ))
        with self._lock, self._connection() as connection:
            connection.execute(
                """UPDATE scheduled_task_runs SET completed_at = ?, status = ?, result_text = ?,
                    error = ?, conversation_message_id = ?, duration_ms = ?, unread = 1 WHERE id = ?""",
                (to_iso(completed_at), run_status, str(result_text)[:100_000], str(error)[:10_000],
                 conversation_message_id, max(0, int((completed_at - started_at).total_seconds() * 1000)), run_id),
            )
            connection.execute(
                """UPDATE scheduled_tasks SET status = ?, running = 0, next_run_at = ?,
                    last_run_at = ?, updated_at = ? WHERE id = ?""",
                (task_status, next_run_at, to_iso(completed_at), to_iso(completed_at), task["id"]),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM scheduled_task_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("Запуск не найден.")
        return dict(row)

    def list_runs(self, task_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM scheduled_task_runs"
        params: list[Any] = []
        if task_id:
            query += " WHERE task_id = ?"
            params.append(task_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connection() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def mark_notifications_read(self, owner_profile_id: str | None = None) -> int:
        with self._lock, self._connection() as connection:
            if owner_profile_id:
                result = connection.execute(
                    """UPDATE scheduled_task_runs SET unread = 0 WHERE unread = 1 AND task_id IN
                       (SELECT id FROM scheduled_tasks WHERE owner_profile_id = ?)""",
                    (owner_profile_id,),
                )
            else:
                result = connection.execute("UPDATE scheduled_task_runs SET unread = 0 WHERE unread = 1")
        return int(result.rowcount)

    def summary(self, owner_profile_id: str | None = None) -> dict[str, Any]:
        where = ""
        params: tuple[Any, ...] = ()
        if owner_profile_id:
            where = " WHERE owner_profile_id = ?"
            params = (owner_profile_id,)
        with self._lock, self._connection() as connection:
            tasks = connection.execute(
                "SELECT status, COUNT(*) count FROM scheduled_tasks" + where + " GROUP BY status", params
            ).fetchall()
            run_where = ""
            run_params: tuple[Any, ...] = ()
            if owner_profile_id:
                run_where = " WHERE task_id IN (SELECT id FROM scheduled_tasks WHERE owner_profile_id = ?)"
                run_params = (owner_profile_id,)
            runs = connection.execute(
                "SELECT status, COUNT(*) count FROM scheduled_task_runs" + run_where + " GROUP BY status", run_params
            ).fetchall()
            unread = connection.execute(
                "SELECT COUNT(*) FROM scheduled_task_runs" +
                (run_where + (" AND" if run_where else " WHERE") + " unread = 1"), run_params
            ).fetchone()[0]
            latest = connection.execute(
                "SELECT * FROM scheduled_task_runs" + run_where + " ORDER BY started_at DESC LIMIT 5", run_params
            ).fetchall()
        task_counts = {row["status"]: row["count"] for row in tasks}
        run_counts = {row["status"]: row["count"] for row in runs}
        return {
            "timezone": TIMEZONE_NAME,
            "tasks_total": sum(task_counts.values()),
            "tasks_by_status": task_counts,
            "runs_total": sum(run_counts.values()),
            "runs_by_status": run_counts,
            "unread_count": int(unread),
            "latest_runs": [dict(row) for row in latest],
        }

    @staticmethod
    def _next_interval(first: datetime, interval_seconds: int, after: datetime) -> datetime:
        candidate = first
        if candidate > after:
            return candidate
        elapsed = (after - candidate).total_seconds()
        steps = int(elapsed // interval_seconds) + 1
        return candidate + timedelta(seconds=steps * interval_seconds)

    @staticmethod
    def _task(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["running"] = bool(item.get("running"))
        item["mcp_servers"] = json.loads(item.pop("mcp_servers_json") or "[]")
        item["allowed_tools"] = json.loads(item.pop("allowed_tools_json") or "[]")
        item["settings"] = json.loads(item.pop("settings_json") or "{}")
        item["interval_minutes"] = (
            round(item["interval_seconds"] / 60, 4) if item.get("interval_seconds") else None
        )
        return item


class SchedulerService:
    """Проверяет наступившие задания, пока работает Flask-приложение."""

    def __init__(self, repository: SchedulerRepository,
                 runner: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
                 poll_seconds: float = 1.0) -> None:
        self.repository = repository
        self.runner = runner
        self.poll_seconds = max(0.1, float(poll_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def set_runner(self, runner: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]) -> None:
        self.runner = runner

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if self.runner is None:
                raise RuntimeError("Для Scheduler не настроен исполнитель заданий.")
            self.repository.skip_missed_after_restart()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="task-scheduler", daemon=True)
            self._thread.start()
            logger.info("Scheduler запущен | timezone=%s", TIMEZONE_NAME)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
        logger.info("Scheduler остановлен")

    def status(self) -> dict[str, Any]:
        thread = self._thread
        return {
            "running": bool(thread and thread.is_alive()),
            "timezone": TIMEZONE_NAME,
            **self.repository.summary(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            claimed = self.repository.claim_due_task()
            if claimed is None:
                self._stop.wait(self.poll_seconds)
                continue
            task, run = claimed
            try:
                result = self.runner(task, run) if self.runner else {}
                self.repository.finish_run(
                    run["id"], result_text=str(result.get("content") or ""),
                    conversation_message_id=result.get("message_id"),
                )
            except Exception as error:
                logger.error("Ошибка запланированного задания | task_id=%s", task["id"], exc_info=True)
                try:
                    self.repository.finish_run(run["id"], error=str(error))
                except Exception:
                    logger.error("Не удалось сохранить ошибку запуска | run_id=%s", run["id"], exc_info=True)
