"""Проверки собственных MCP-серверов без внешних HTTP-запросов."""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest import mock

from mcp.server.mcpserver.exceptions import ToolError

from mcp_manager import OpenMeteoMCPManager, SchedulerMCPManager, StdioMCPManager
from mcp_servers.open_meteo import server as meteo
from mcp_servers.openaq import server as openaq


class OpenMeteoToolTests(unittest.TestCase):
    def test_find_location_returns_compact_locations(self):
        with mock.patch.object(meteo, "get_json", return_value={"results": [{
            "id": 1, "name": "Новосибирск", "latitude": 55.03, "longitude": 82.92,
            "timezone": "Asia/Novosibirsk", "country": "Россия", "unused": "value",
        }]}):
            result = meteo.find_location("Новосибирск")
        self.assertEqual(result["locations"][0]["name"], "Новосибирск")
        self.assertNotIn("unused", result["locations"][0])

    def test_precipitation_window_aggregates_first_continuous_period(self):
        payload = {
            "hourly": {
                "time": ["10:00", "11:00", "12:00", "13:00"],
                "precipitation_probability": [10, 45, 80, 10],
                "precipitation": [0, 0.2, 1.1, 0],
            }
        }
        with mock.patch.object(meteo, "get_json", return_value=payload):
            result = meteo.get_precipitation_window(55, 83, probability_threshold=30)
        self.assertTrue(result["found"])
        self.assertEqual(result["window"]["start"], "11:00")
        self.assertEqual(result["window"]["end"], "12:00")
        self.assertEqual(result["window"]["peak_probability_percent"], 80)


class OpenAQToolTests(unittest.TestCase):
    def test_missing_key_has_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ToolError, "OPENAQ_API_KEY"):
                openaq.get_latest_measurements(123)


class MCPRegistrationTests(unittest.TestCase):
    def test_filesystem_write_file_creates_report_inside_allowed_workspace(self):
        project_dir = Path(__file__).resolve().parent
        server_script = project_dir / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js"
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            manager = StdioMCPManager(
                command="node", args=[str(server_script), str(workspace)],
                process_name="filesystem-write-test", starting_message="start", workspace=workspace,
            )
            try:
                manager.start()
                tools = manager.list_tools()["tools"]
                self.assertIn("write_file", {tool["name"] for tool in tools})
                report_path = workspace / "reports" / "pipeline-test.md"
                report_path.parent.mkdir(parents=True)
                output = manager.call_tool("write_file", {
                    "path": str(report_path), "content": "# Отчёт\n\nТемпература: 12 °C.\n",
                })
                self.assertFalse(output["is_error"])
                self.assertEqual(report_path.read_text(encoding="utf-8"), "# Отчёт\n\nТемпература: 12 °C.\n")
            finally:
                manager.stop()

    def test_each_server_registers_expected_tools(self):
        project_dir = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        definitions = {
            "open-meteo": (OpenMeteoMCPManager(project_dir), {
                "find_location", "get_current_weather", "get_hourly_forecast",
                "get_daily_forecast", "get_precipitation_window",
            }),
            "openalex": (StdioMCPManager(
                command=sys.executable, args=["-m", "mcp_servers.openalex.server"],
                process_name="openalex-test", starting_message="start",
            ), {"search_works", "get_work", "search_authors"}),
            "openaq": (StdioMCPManager(
                command=sys.executable, args=["-m", "mcp_servers.openaq.server"],
                process_name="openaq-test", starting_message="start",
            ), {"find_air_quality_locations", "get_location_details", "get_latest_measurements"}),
            "scheduler": (SchedulerMCPManager(project_dir, Path(temporary.name) / "scheduler.sqlite3"), {
                "create_scheduled_task", "list_scheduled_tasks", "get_scheduled_task",
                "update_scheduled_task", "pause_scheduled_task", "resume_scheduled_task",
                "delete_scheduled_task", "run_scheduled_task_now", "get_scheduled_task_runs",
                "get_scheduler_summary", "get_scheduler_capabilities",
            }),
        }
        for name, (manager, expected) in definitions.items():
            with self.subTest(server=name):
                try:
                    manager.start()
                    tools = manager.list_tools()["tools"]
                    self.assertEqual({tool["name"] for tool in tools}, expected)
                    self.assertTrue(all(tool["input_schema"]["type"] == "object" for tool in tools))
                finally:
                    manager.stop()


if __name__ == "__main__":
    unittest.main()
