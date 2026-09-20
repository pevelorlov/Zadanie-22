"""Постоянный локальный whisper.cpp-сервис для голосового ввода."""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


logger = logging.getLogger("deepseek_agent.voice")

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".mp4"}


class WhisperService:
    """Запускает whisper-server один раз и держит модель загруженной до выхода."""

    def __init__(self, project_dir: Path, autostart: bool = True) -> None:
        self.project_dir = project_dir
        self.server_path = Path(os.getenv(
            "WHISPER_SERVER_PATH",
            project_dir / "tools" / "whisper" / "bin" / "whisper-server.exe",
        ))
        self.model_path = Path(os.getenv(
            "WHISPER_MODEL_PATH",
            project_dir / "models" / "ggml-large-v3-turbo.bin",
        ))
        self.ffmpeg_dir = Path(os.getenv(
            "WHISPER_FFMPEG_DIR",
            project_dir / "tools" / "ffmpeg" / "bin",
        ))
        self.host = "127.0.0.1"
        self.port = int(os.getenv("WHISPER_PORT", "8091"))
        self.language = os.getenv("WHISPER_LANGUAGE", "ru")
        self.runtime_path = project_dir / "logs" / "whisper-runtime.json"
        self.lock_path = project_dir / "logs" / "whisper-runtime.lock"
        self._process: subprocess.Popen[str] | None = None
        self._server_pid: int | None = None
        self._reused = False
        self._start_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._phase = "stopped"
        self._message = "Whisper ещё не запущен."
        self._last_log = ""
        atexit.register(self.stop)
        if autostart and os.getenv("WHISPER_AUTOSTART", "true").lower() not in {"0", "false", "no"}:
            self.start()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        with self._lock:
            if self._server_pid and self._pid_running(self._server_pid):
                return
            if self._start_thread and self._start_thread.is_alive():
                return
            missing = []
            if not self.server_path.is_file():
                missing.append(f"не найден {self.server_path.name}")
            if not self.model_path.is_file():
                missing.append(f"не найдена модель {self.model_path.name}")
            if missing:
                self._phase = "not_installed"
                self._message = "Whisper не установлен: " + ", ".join(missing) + "."
                logger.warning("Whisper не запущен | reason=%s", ", ".join(missing))
                return
            self._phase = "starting"
            self._message = "Загружаем large-v3-turbo в память RX 6600…"
            self._start_thread = threading.Thread(target=self._start_worker, daemon=True)
            self._start_thread.start()
            logger.info("Запрошен запуск Whisper | port=%s | language=%s", self.port, self.language)

    def _start_worker(self) -> None:
        try:
            with self._runtime_lock():
                existing_pid = self._find_reusable_server()
                if existing_pid:
                    self._attach(existing_pid, reused=True)
                    self._wait_until_ready(existing_pid)
                    return
                listener_pid = self._listener_pid(self.port)
                if listener_pid:
                    raise RuntimeError(
                        f"Порт {self.port} занят посторонним процессом PID {listener_pid}; "
                        "второй Whisper не запущен."
                    )
                self._launch_server()
        except Exception as error:
            self.stop()
            with self._lock:
                self._phase = "error"
                self._message = f"Не удалось запустить Whisper: {error}"
            logger.exception("Ошибка запуска Whisper | port=%s", self.port)

    def _launch_server(self) -> None:
        environment = os.environ.copy()
        if self.ffmpeg_dir.is_dir():
            environment["PATH"] = str(self.ffmpeg_dir) + os.pathsep + environment.get("PATH", "")
        # whisper.cpp on Windows still passes the model name through a narrow
        # native string. A relative ASCII-only path keeps projects in folders
        # such as "Задание 9" working correctly.
        model_argument = os.path.relpath(self.model_path, self.server_path.parent)
        command = [
            str(self.server_path),
            "--host", self.host,
            "--port", str(self.port),
            "--model", model_argument,
            "--language", self.language,
            "--convert",
        ]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=self.server_path.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )
        with self._lock:
            self._process = process
        self._attach(process.pid, reused=False)
        logger.info("Процесс Whisper запущен | pid=%s | port=%s", process.pid, self.port)
        threading.Thread(target=self._capture_output, args=(process,), daemon=True).start()
        self._wait_until_ready(process.pid)

    def _attach(self, server_pid: int, *, reused: bool) -> None:
        runtime = self._load_runtime()
        clients = self._live_client_pids(runtime)
        if os.getpid() not in clients:
            clients.append(os.getpid())
        self._server_pid = server_pid
        self._reused = reused
        self._write_runtime(server_pid, clients)

    def _wait_until_ready(self, server_pid: int) -> None:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not self._pid_running(server_pid):
                code = self._process.returncode if self._process else "неизвестен"
                raise RuntimeError(self._last_log or f"whisper-server завершился с кодом {code}")
            if self._server_is_ready():
                with self._lock:
                    self._phase = "ready"
                    self._message = self._ready_message()
                action = "переиспользован" if self._reused else "готов"
                logger.info("Whisper %s | pid=%s | port=%s", action, server_pid, self.port)
                return
            time.sleep(0.5)
        raise RuntimeError("модель не загрузилась за 180 секунд")

    def _capture_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                with self._lock:
                    self._last_log = stripped[-500:]
        if self._phase == "ready" and process.poll() is not None:
            with self._lock:
                self._phase = "error"
                self._message = f"Whisper неожиданно завершился с кодом {process.returncode}."
            logger.error("Whisper неожиданно завершился | code=%s", process.returncode)

    def status(self) -> dict[str, Any]:
        with self._lock:
            process_running = bool(self._server_pid and self._pid_running(self._server_pid))
            if self._phase == "ready" and not process_running:
                self._phase = "error"
                self._message = "Whisper неожиданно завершился. Нажмите запуск повторно."
            return {
                "phase": self._phase,
                "ready": self._phase == "ready" and process_running,
                "message": self._message,
                "model": "large-v3-turbo",
                "backend": "Vulkan",
                "language": self.language,
                "pid": self._server_pid,
                "port": self.port,
                "reused": self._reused,
                "server_path": str(self.server_path),
                "model_path": str(self.model_path),
                "last_log": self._last_log if self._phase == "error" else "",
            }

    def transcribe(self, audio: bytes, filename: str, content_type: str) -> dict[str, Any]:
        started = time.perf_counter()
        if not audio:
            raise ValueError("Получена пустая аудиозапись.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise ValueError("Аудиозапись больше 25 МБ. Запишите более короткое сообщение.")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError("Неподдерживаемый формат аудио.")
        if not self.status()["ready"]:
            raise RuntimeError(self.status()["message"])
        logger.info(
            "Распознавание начато | bytes=%s | extension=%s | content_type=%s",
            len(audio), suffix, content_type,
        )

        boundary = uuid.uuid4().hex
        safe_name = Path(filename).name.replace('"', "")
        fields = {
            "response_format": "json",
            "language": self.language,
            "temperature": "0.0",
            "temperature_inc": "0.2",
        }
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            f'Content-Type: {content_type or "application/octet-stream"}\r\n\r\n'.encode()
        )
        parts.extend([audio, b"\r\n", f"--{boundary}--\r\n".encode()])
        request_body = b"".join(parts)
        inference_request = urllib.request.Request(
            self.base_url + "/inference",
            data=request_body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(inference_request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Whisper не смог распознать запись: {error}") from error

        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("В записи не удалось распознать речь.")
        logger.info(
            "Распознавание завершено | chars=%s | elapsed=%.3f",
            len(text), time.perf_counter() - started,
        )
        return {"text": text, "language": self.language}

    def stop(self) -> None:
        process = None
        server_pid = None
        should_terminate = False
        with self._lock:
            process = self._process
            server_pid = self._server_pid
            self._process = None
            self._server_pid = None
            self._phase = "stopped"
            self._message = "Whisper остановлен, видеопамять освобождена."
        if not server_pid:
            return
        with self._runtime_lock():
            runtime = self._load_runtime()
            if int(runtime.get("server_pid") or 0) == server_pid:
                clients = [pid for pid in self._live_client_pids(runtime) if pid != os.getpid()]
                if clients:
                    self._write_runtime(server_pid, clients)
                    logger.info("Whisper остаётся для других приложений | pid=%s | clients=%s", server_pid, clients)
                else:
                    should_terminate = True
                    self.runtime_path.unlink(missing_ok=True)
            elif process and process.pid == server_pid:
                # Реестр мог быть повреждён или удалён, но Popen однозначно
                # подтверждает, что этот дочерний процесс запущен нами.
                should_terminate = True
        if not should_terminate:
            return
        logger.info("Остановка Whisper | pid=%s", server_pid)
        if process and process.pid == server_pid and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Whisper не остановился за 5 секунд; принудительное завершение | pid=%s", server_pid)
                process.kill()
        else:
            self._terminate_server_pid(server_pid)
        logger.info("Whisper остановлен | pid=%s", server_pid)

    def _ready_message(self) -> str:
        mode = "переиспользован" if self._reused else "запущен"
        return f"Whisper готов · Vulkan · RX 6600 · large-v3-turbo · PID {self._server_pid} · порт {self.port} · {mode}"

    def _find_reusable_server(self) -> int | None:
        runtime = self._load_runtime()
        runtime_pid = int(runtime.get("server_pid") or 0)
        runtime_port = int(runtime.get("port") or 0)
        if (
            runtime_pid and runtime_port == self.port
            and runtime.get("server_path") == str(self.server_path.resolve())
            and runtime.get("model_path") == str(self.model_path.resolve())
            and self._is_project_server(runtime_pid)
        ):
            return runtime_pid
        listener_pid = self._listener_pid(self.port)
        if listener_pid and self._is_project_server(listener_pid) and self._server_is_ready():
            return listener_pid
        return None

    def _is_project_server(self, process_id: int) -> bool:
        actual = self._process_path(process_id)
        return bool(actual and Path(actual).resolve() == self.server_path.resolve() and self._pid_running(process_id))

    def _server_is_ready(self) -> bool:
        try:
            request = urllib.request.Request(self.base_url + "/", method="GET")
            with urllib.request.urlopen(request, timeout=1.5) as response:
                return "whisper.cpp" in str(response.headers.get("Server", "")).lower()
        except (OSError, urllib.error.URLError):
            return False

    def _load_runtime(self) -> dict[str, Any]:
        try:
            value = json.loads(self.runtime_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_runtime(self, server_pid: int, client_pids: list[int]) -> None:
        self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": 1,
            "server_pid": server_pid,
            "port": self.port,
            "server_path": str(self.server_path.resolve()),
            "model_path": str(self.model_path.resolve()),
            "client_pids": sorted(set(client_pids)),
            "updated_at": time.time(),
        }
        temporary = self.runtime_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.runtime_path)

    def _live_client_pids(self, runtime: dict[str, Any]) -> list[int]:
        values = runtime.get("client_pids", [])
        return [int(pid) for pid in values if str(pid).isdigit() and self._pid_running(int(pid))]

    @contextlib.contextmanager
    def _runtime_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0)
                if lock_file.tell() == 0 and self.lock_path.stat().st_size == 0:
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            lock_file.close()

    @staticmethod
    def _listener_pid(port: int) -> int | None:
        if os.name != "nt":
            return None
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        pattern = re.compile(rf"^\s*TCP\s+127\.0\.0\.1:{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.MULTILINE)
        match = pattern.search(result.stdout)
        return int(match.group(1)) if match else None

    @staticmethod
    def _pid_running(process_id: int) -> bool:
        if process_id <= 0:
            return False
        if os.name != "nt":
            try:
                os.kill(process_id, 0)
                return True
            except OSError:
                return False
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    @staticmethod
    def _process_path(process_id: int) -> str:
        if os.name != "nt":
            return ""
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return ""
        try:
            size = ctypes.c_ulong(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return buffer.value
            return ""
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def _terminate_server_pid(self, process_id: int) -> None:
        if not self._is_project_server(process_id):
            logger.warning("Whisper PID не завершён: путь процесса не совпал | pid=%s", process_id)
            return
        if os.name == "nt":
            handle = ctypes.windll.kernel32.OpenProcess(0x0001 | 0x00100000, False, process_id)
            if not handle:
                return
            try:
                ctypes.windll.kernel32.TerminateProcess(handle, 0)
                ctypes.windll.kernel32.WaitForSingleObject(handle, 5000)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        else:
            os.kill(process_id, signal.SIGTERM)
