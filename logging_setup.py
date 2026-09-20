"""Безопасное файловое логирование с ротацией и маскированием секретов."""

from __future__ import annotations

import logging
import os
import re
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | request=%(request_id)s | %(threadName)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5
_HANDLER_MARKER = "_deepseek_agent_handler"
_request_id: ContextVar[str] = ContextVar("deepseek_agent_request_id", default="-")
_REDACTIONS = (
    (re.compile(r"(?i)(DEEPSEEK_API_KEY\s*[=:]\s*)\S+"), r"\1<скрыто>"),
    (re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)\S+"), r"\1<скрыто>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "<скрытый-api-ключ>"),
)


def redact_secrets(value: Any) -> str:
    """Маскирует распространённые формы API-ключей в диагностическом тексте."""
    text = str(value)
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Форматирует запись, не позволяя ключам попасть в сообщение или traceback."""

    def format(self, record: logging.LogRecord) -> str:
        original_message, original_args = record.msg, record.args
        try:
            record.msg = redact_secrets(record.getMessage())
            record.args = ()
            return super().format(record)
        finally:
            record.msg, record.args = original_message, original_args

    def formatException(self, exc_info: Any) -> str:  # noqa: N802 - API logging.Formatter
        return redact_secrets(super().formatException(exc_info))


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def set_request_id(value: str) -> Token[str]:
    return _request_id.set(value)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


def configure_logging(log_dir: Path) -> dict[str, Path]:
    """Подключает консоль, основной журнал и отдельный журнал ошибок."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    app_log = log_dir / "app.log"
    error_log = log_dir / "errors.log"
    level_name = os.getenv("DEEPSEEK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()

    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter = RedactingFormatter(LOG_FORMAT, DATE_FORMAT)
    request_filter = RequestContextFilter()
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            app_log, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT,
            encoding="utf-8", delay=False,
        ),
        RotatingFileHandler(
            error_log, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT,
            encoding="utf-8", delay=False,
        ),
        logging.StreamHandler(),
    ]
    handlers[0].setLevel(level)
    handlers[1].setLevel(logging.ERROR)
    handlers[2].setLevel(level)
    for handler in handlers:
        setattr(handler, _HANDLER_MARKER, True)
        handler.addFilter(request_filter)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(min(level, logging.ERROR))
    logging.captureWarnings(True)
    logging.getLogger("deepseek_agent.logging").info(
        "Логирование настроено | level=%s | app_log=%s | error_log=%s",
        logging.getLevelName(level), app_log, error_log,
    )
    return {"app": app_log, "errors": error_log}
