"""Проверки SQLite Scheduler, API и фонового выполнения без реального DeepSeek."""

from __future__ import annotations

import tempfile
import time
import unittest
import json
from datetime import timedelta
from pathlib import Path

from agent import Agent, AgentResult
from app import create_app
from scheduler import SchedulerRepository, SchedulerService, scheduler_now, to_iso
from test_app import FakeMCPManager, FakeProvider, FakeVoiceService


class SchedulerToolProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.tool_output = None

    def complete_with_tools(self, messages, settings, tools, tool_executor, max_rounds=6):
        self.asserted_tool_names = {tool["name"] for tool in tools}
        self.tool_output = tool_executor("create_scheduled_task", {
            "title": "Погода через час",
            "prompt": "Узнай погоду в Новосибирске и предложи одежду.",
            "schedule_type": "once",
            "delay_minutes": 60,
            "mcp_servers": ["open-meteo"],
            "allowed_tools": ["find_location", "get_current_weather"],
        })
        return AgentResult(
            content="Задание создано на час вперёд.", reasoning_content="",
            technical={"request_id": "scheduler-tool-test", "usage": {
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                "cached_input_tokens": 0, "uncached_input_tokens": 10, "reasoning_tokens": 0,
            }, "mcp_tool_calls": [{"name": "create_scheduled_task", "is_error": False}]},
        )


class BlockedSchedulerToolProvider(SchedulerToolProvider):
    def complete(self, messages, settings):
        if "строгий контроллер этапов" not in messages[0]["content"]:
            return super().complete(messages, settings)
        context = json.loads(messages[-1]["content"].rsplit("Политика:\n", 1)[1])
        return AgentResult(
            content=json.dumps({
                "allowed": False,
                "detected_action_type": "implementation",
                "checked_invariant_ids": [item["ref"] for item in context["invariants"]],
                "violated_invariant_ids": [],
                "stage_complete": False,
                "recommended_event": None,
                "required_artifacts": [],
                "explanation": "Ответ содержит отдельную реализацию.",
            }, ensure_ascii=False),
            reasoning_content="",
            technical={"usage": {}},
        )


class SchedulerRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = SchedulerRepository(Path(self.temp.name) / "scheduler.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_once_interval_history_and_aggregate_are_persisted(self):
        once = self.repository.create_task(
            title="Через час", prompt="Проверь погоду", schedule_type="once", delay_minutes=60,
            mcp_servers=["open-meteo"], allowed_tools=["get_current_weather"],
        )
        interval = self.repository.create_task(
            title="Каждый час", prompt="Сделай сводку", schedule_type="interval", interval_minutes=60,
        )
        self.assertEqual(once["mcp_servers"], ["open-meteo"])
        self.assertEqual(interval["interval_minutes"], 60)
        restored = SchedulerRepository(self.repository.database_path)
        self.assertEqual(len(restored.list_tasks()), 2)
        summary = restored.summary()
        self.assertEqual(summary["tasks_total"], 2)
        self.assertEqual(summary["tasks_by_status"]["enabled"], 2)

    def test_restart_skips_missed_runs(self):
        once = self.repository.create_task(
            title="Разово", prompt="Разовый запрос", schedule_type="once", delay_minutes=1,
        )
        interval = self.repository.create_task(
            title="Период", prompt="Периодический запрос", schedule_type="interval", interval_minutes=5,
        )
        past = to_iso(scheduler_now() - timedelta(minutes=20))
        with self.repository._connection() as connection:
            connection.execute("UPDATE scheduled_tasks SET next_run_at = ?", (past,))
        self.repository.skip_missed_after_restart()
        self.assertEqual(self.repository.get_task(once["id"])["status"], "missed")
        restored_interval = self.repository.get_task(interval["id"])
        self.assertEqual(restored_interval["status"], "enabled")
        self.assertGreater(restored_interval["next_run_at"], to_iso(scheduler_now()))
        self.assertEqual(self.repository.list_runs(), [])

    def test_dependencies_are_restricted_to_known_server_tools(self):
        with self.assertRaisesRegex(ValueError, "Неизвестные MCP-серверы"):
            self.repository.create_task(
                title="Опасное", prompt="Запрос", schedule_type="once", delay_minutes=5,
                mcp_servers=["unknown"], allowed_tools=[],
            )
        with self.assertRaisesRegex(ValueError, "не принадлежат"):
            self.repository.create_task(
                title="Чужой tool", prompt="Запрос", schedule_type="once", delay_minutes=5,
                mcp_servers=["open-meteo"], allowed_tools=["delete_everything"],
            )


class SchedulerApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repository = SchedulerRepository(self.root / "scheduler.sqlite3")
        self.service = SchedulerService(self.repository, poll_seconds=0.05)
        self.app = create_app(
            self.root, Agent(FakeProvider()), FakeVoiceService(), FakeMCPManager(),
            scheduler_service=self.service,
        )
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.service.stop()
        self.app.extensions["scheduler_mcp_manager"].stop()
        self.temp.cleanup()

    def test_crud_creates_dedicated_conversation_and_notifications(self):
        created = self.client.post("/api/scheduler/tasks", json={
            "title": "Проверка",
            "prompt": "Ответь кратко",
            "schedule_type": "once",
            "delay_minutes": 30,
            "settings": {},
        })
        self.assertEqual(created.status_code, 201)
        payload = created.get_json()
        task = payload["task"]
        self.assertTrue(task["conversation_id"])
        self.assertTrue(payload["conversation"]["title"].startswith("Автоматизация"))
        paused = self.client.post(f"/api/scheduler/tasks/{task['id']}/pause", json={})
        self.assertEqual(paused.get_json()["task"]["status"], "paused")
        resumed = self.client.post(f"/api/scheduler/tasks/{task['id']}/resume", json={})
        self.assertEqual(resumed.get_json()["task"]["status"], "enabled")
        state = self.client.get("/api/scheduler/state").get_json()
        self.assertEqual(len(state["tasks"]), 1)
        self.assertEqual(state["summary"]["unread_count"], 0)

    def test_due_task_runs_through_agent_and_is_saved_in_automation_dialog(self):
        self.service.start()
        created = self.client.post("/api/scheduler/tasks", json={
            "title": "Скорый запуск",
            "prompt": "Дай информационный ответ",
            "schedule_type": "once",
            "delay_minutes": 0.002,
            "settings": {},
        }).get_json()
        task = created["task"]
        deadline = time.monotonic() + 3
        runs = []
        while time.monotonic() < deadline:
            runs = self.repository.list_runs(task["id"])
            if runs and runs[0]["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertEqual(runs[0]["result_text"], "Тестовый ответ")
        self.assertEqual(self.repository.summary()["unread_count"], 1)
        conversation = self.app.extensions["json_storage"].get_conversation(task["conversation_id"])
        self.assertEqual(conversation["messages"][-1]["technical"]["scheduled_automation"]["task_id"], task["id"])
        read = self.client.post("/api/scheduler/notifications/read", json={}).get_json()
        self.assertEqual(read["summary"]["unread_count"], 0)

    def test_index_exposes_scheduler_workspace(self):
        text = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="scheduler-button"', text)
        self.assertIn('id="scheduler-dialog"', text)
        self.assertIn('id="scheduler-task-list"', text)
        self.assertIn("Пропущенные запуски не догоняются", text)

    def test_chat_agent_calls_scheduler_mcp_and_attaches_automation_dialog(self):
        provider = SchedulerToolProvider()
        self.app.extensions["chat_agent"].provider = provider
        manager = self.app.extensions["scheduler_mcp_manager"]
        manager.start()
        manager.list_tools()
        conversation = self.client.post("/api/conversations", json={"title": "Команды"}).get_json()["conversation"]
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Через час узнай погоду и посоветуй одежду.",
            "settings": {},
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("create_scheduled_task", provider.asserted_tool_names)
        self.assertFalse(provider.tool_output["is_error"])
        task = provider.tool_output["result"]
        self.assertTrue(task["conversation_id"])
        self.assertEqual(task["mcp_servers"], ["open-meteo"])
        self.assertEqual(task["allowed_tools"], ["find_location", "get_current_weather"])
        self.assertEqual(len(self.repository.list_tasks()), 1)

    def test_blocked_chat_answer_rolls_back_created_schedule_and_dialog(self):
        provider = BlockedSchedulerToolProvider()
        self.app.extensions["chat_agent"].provider = provider
        manager = self.app.extensions["scheduler_mcp_manager"]
        manager.start()
        manager.list_tools()
        conversation = self.client.post("/api/conversations", json={"title": "Команды"}).get_json()["conversation"]
        response = self.client.post(f"/api/conversations/{conversation['id']}/messages", json={
            "content": "Создай расписание и заодно реализуй программу.",
            "settings": {},
        })
        self.assertEqual(response.status_code, 200)
        assistant = response.get_json()["conversation"]["messages"][-1]
        self.assertEqual(assistant["technical"]["request_status"], "blocked")
        self.assertEqual(self.repository.list_tasks(), [])
        conversations = self.app.extensions["json_storage"].list_conversations()
        self.assertEqual([item["id"] for item in conversations], [conversation["id"]])


if __name__ == "__main__":
    unittest.main()
