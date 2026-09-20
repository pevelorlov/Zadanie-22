"""Запуск DeepSeek Agent с автоматическим открытием браузера и логированием."""

import logging
import os
import platform
import threading
import webbrowser
import socket
from pathlib import Path

from logging_setup import configure_logging


BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("deepseek_agent.startup")


def find_available_port(start: int = 5000, stop: int = 5100) -> int:
    """Возвращает первый свободный локальный порт в указанном диапазоне."""
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Не найден свободный порт в диапазоне {start}–{stop}.")


def open_browser(url: str) -> None:
    webbrowser.open(url)


def main() -> None:
    log_paths = configure_logging(BASE_DIR / "logs")
    from app import create_app

    app = create_app()

    port = find_available_port()
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.0, open_browser, args=(url,)).start()
    if port != 5000:
        print(f"Порт 5000 занят другой программой. Используется свободный порт {port}.")
        logger.warning("Порт 5000 занят; выбран порт %s", port)
    print(f"DeepSeek Agent запущен: {url}")
    print(f"Общий журнал: {log_paths['app']}")
    print(f"Журнал ошибок: {log_paths['errors']}")
    print("Чтобы остановить сервер, нажмите Ctrl+C.\n")
    logger.info(
        "Приложение запускается | url=%s | pid=%s | python=%s | os=%s",
        url, os.getpid(), platform.python_version(), platform.platform(),
    )
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Получена команда остановки приложения")
    finally:
        app.extensions["whisper_service"].stop()
        logger.info("Приложение остановлено")


if __name__ == "__main__":
    main()
