"""Scheduler MCP: создание расписаний, управление ими и агрегированные результаты."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from scheduler import KNOWN_MCP_TOOLS, SchedulerRepository, TIMEZONE_NAME, scheduler_now, to_iso


DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / "data" / "scheduler.sqlite3"
DATABASE_PATH = Path(os.environ.get("DEEPSEEK_SCHEDULER_DB", str(DEFAULT_DATABASE))).resolve()
repository = SchedulerRepository(DATABASE_PATH)

server = MCPServer(
    name="scheduler-mcp",
    title="DeepSeek Scheduler",
    description=(
        "Локальные разовые и периодические задания DeepSeek. Данные сохраняются в SQLite, "
        f"часовой пояс — {TIMEZONE_NAME}."
    ),
    version="1.0.0",
)


@server.tool(structured_output=True)
def create_scheduled_task(
    title: str,
    prompt: str,
    schedule_type: str,
    run_at: str | None = None,
    delay_minutes: float | None = None,
    interval_minutes: float | None = None,
    mcp_servers: list[str] | None = None,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Создать задание. schedule_type: once или interval. Для once передай run_at в ISO-виде либо delay_minutes. Для interval передай interval_minutes не меньше 5 и при желании run_at/delay_minutes для первого запуска. Для погоды укажи mcp_servers=['open-meteo'] и нужные инструменты: find_location, get_current_weather, get_hourly_forecast, get_daily_forecast, get_precipitation_window. Сохрани в prompt самодостаточное будущее поручение, а не описание расписания."""
    return repository.create_task(
        title=title,
        prompt=prompt,
        schedule_type=schedule_type,
        run_at=run_at,
        delay_minutes=delay_minutes,
        interval_minutes=interval_minutes,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
    )


@server.tool(structured_output=True)
def list_scheduled_tasks() -> dict[str, Any]:
    """Получить все запланированные задания, их статусы, зависимости и ближайшие запуски."""
    tasks = repository.list_tasks()
    return {"count": len(tasks), "timezone": TIMEZONE_NAME, "tasks": tasks}


@server.tool(structured_output=True)
def get_scheduled_task(task_id: str) -> dict[str, Any]:
    """Получить одно запланированное задание и последние 20 запусков."""
    return {"task": repository.get_task(task_id), "runs": repository.list_runs(task_id, 20)}


@server.tool(structured_output=True)
def update_scheduled_task(
    task_id: str,
    title: str | None = None,
    prompt: str | None = None,
    next_run_at: str | None = None,
    interval_minutes: float | None = None,
    mcp_servers: list[str] | None = None,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Изменить текст, следующий запуск, период или набор разрешённых MCP-инструментов задания."""
    changes = {
        key: value for key, value in {
            "title": title, "prompt": prompt, "next_run_at": next_run_at,
            "interval_minutes": interval_minutes, "mcp_servers": mcp_servers,
            "allowed_tools": allowed_tools,
        }.items() if value is not None
    }
    return repository.update_task(task_id, **changes)


@server.tool(structured_output=True)
def pause_scheduled_task(task_id: str) -> dict[str, Any]:
    """Приостановить будущие запуски задания."""
    return repository.set_status(task_id, "paused")


@server.tool(structured_output=True)
def resume_scheduled_task(task_id: str) -> dict[str, Any]:
    """Возобновить задание. Пропущенные интервалы не выполняются задним числом."""
    return repository.set_status(task_id, "enabled")


@server.tool(structured_output=True)
def delete_scheduled_task(task_id: str) -> dict[str, Any]:
    """Удалить задание вместе с историей запусков. Диалог автоматизации остаётся в приложении."""
    repository.delete_task(task_id)
    return {"deleted": True, "task_id": task_id}


@server.tool(structured_output=True)
def run_scheduled_task_now(task_id: str) -> dict[str, Any]:
    """Поставить существующее задание на ближайший запуск, не дожидаясь расписания."""
    return repository.request_run_now(task_id)


@server.tool(structured_output=True)
def get_scheduled_task_runs(task_id: str, limit: int = 20) -> dict[str, Any]:
    """Получить историю запусков и сохранённые результаты одного задания."""
    repository.get_task(task_id)
    runs = repository.list_runs(task_id, limit)
    return {"task_id": task_id, "count": len(runs), "runs": runs}


@server.tool(structured_output=True)
def get_scheduler_summary() -> dict[str, Any]:
    """Вернуть агрегированную сводку: задания и запуски по статусам, непрочитанные и последние результаты."""
    return repository.summary()


@server.tool(structured_output=True)
def get_scheduler_capabilities() -> dict[str, Any]:
    """Показать текущее новосибирское время, виды расписания и доступные MCP-инструменты. Вызови перед созданием задания на календарную дату вроде «завтра в 8:00»."""
    return {
        "timezone": TIMEZONE_NAME,
        "current_time": to_iso(scheduler_now()),
        "schedule_types": ["once", "interval"],
        "minimum_interval_minutes": 5,
        "missed_run_policy": "skip",
        "available_mcp_servers": {name: sorted(tools) for name, tools in KNOWN_MCP_TOOLS.items()},
    }


if __name__ == "__main__":
    server.run("stdio")
