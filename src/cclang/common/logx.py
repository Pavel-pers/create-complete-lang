from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterable, Optional

# --- Вспомогательное ---------------------------------------------------------

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# Поля LogRecord по умолчанию — чтобы корректно выделить extra
_LOG_RECORD_DEFAULT_ATTRS = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process",
}

# Глобальный контекст для логов (run_id, pipeline и др.)
_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("_LOG_CONTEXT", default={})

def _now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

# --- Форматтеры --------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Простой JSON‑форматтер без внешних зависимостей."""
    def __init__(self, *, indent: Optional[int] = None):
        super().__init__()
        self._indent = indent

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }
        # Добавим контекст + extra
        for k, v in record.__dict__.items():
            if k not in _LOG_RECORD_DEFAULT_ATTRS and not k.startswith("_"):
                data[k] = v
        return json.dumps(data, ensure_ascii=False, indent=self._indent)


class ConsoleFormatter(logging.Formatter):
    """Компактный человекочитаемый форматтер с подсветкой уровней (ANSI)."""
    COLORS = {
        "DEBUG": "\033[37m",     # серый
        "INFO": "\033[36m",      # бирюзовый
        "WARNING": "\033[33m",   # жёлтый
        "ERROR": "\033[31m",     # красный
        "CRITICAL": "\033[41m",  # красный фон
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        color = self.COLORS.get(level, "")
        reset = self.RESET if color else ""
        # Короткая строка контекста из extra (run_id и pipeline показываем первыми)
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in _LOG_RECORD_DEFAULT_ATTRS and not k.startswith("_")}
        # Сортировка ключей: сначала run_id, pipeline, service, потом остальные по алфавиту
        ordered_keys = [k for k in ["run_id", "pipeline", "service"] if k in extra] + sorted([k for k in extra if k not in {"run_id","pipeline","service"}])
        ctx = " ".join(f"{k}={extra[k]}" for k in ordered_keys)
        msg = record.getMessage()
        return f"{ts} {color}{level:<8}{reset} {record.name} {ctx} - {msg}"

# --- Контекст и адаптеры -----------------------------------------------------

@dataclass(frozen=True)
class _ContextToken:
    previous: Dict[str, Any]

@contextmanager
def use_context(**pairs: Any):
    """Контекст-менеджер: добавляет поля (например, run_id, pipeline) на время блока."""
    current = dict(_LOG_CONTEXT.get())
    previous = dict(current)
    current.update(pairs)
    token = _LOG_CONTEXT.set(current)
    try:
        yield _ContextToken(previous=previous)
    finally:
        # Восстанавливаем прежний контекст
        _LOG_CONTEXT.reset(token)

class ContextFilter(logging.Filter):
    """Добавляет контекстные поля из ContextVar в каждый LogRecord."""
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get()
        for k, v in ctx.items():
            setattr(record, k, v)
        return True

class BoundLogger(logging.LoggerAdapter):
    """LoggerAdapter с удобным .bind(**extra) и слиянием extra-данных."""
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})

    def bind(self, **extra: Any) -> "BoundLogger":
        merged = dict(self.extra)
        merged.update(extra)
        return BoundLogger(self.logger, merged)

    def process(self, msg, kwargs):
        event_extra = kwargs.pop("extra", {}) or {}
        merged = dict(self.extra)
        merged.update(event_extra)
        if merged:
            kwargs["extra"] = merged
        return msg, kwargs

# --- Фильтры -----------------------------------------------------------------
class MaxLevelFilter(logging.Filter):
    """Пропускает только записи <= max_level (включительно)."""
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level

class MinLevelFilter(logging.Filter):
    """Пропускает только записи >= min_level."""
    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level

# --- Инициализация -----------------------------------------------------------

def setup_logging(
    *,
    service: Optional[str] = None,
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    file: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    json_indent: Optional[int] = None,
    propagate_library_logs: bool = True,
) -> logging.Logger:
    """Глобальная настройка логов. Вызывать один раз при старте приложения."""
    # Читаем окружение
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    level_no = _LEVELS.get(level_name, logging.INFO)

    fmt = (fmt or os.getenv("LOG_FORMAT") or "console").lower()
    log_file = file or os.getenv("LOG_FILE")
    max_bytes = int(max_bytes or os.getenv("LOG_MAX_BYTES") or 10_485_760)  # 10 MiB
    backup_count = int(backup_count or os.getenv("LOG_BACKUP_COUNT") or 5)

    root = logging.getLogger()
    root.setLevel(level_no)
    # Удаляем старые хэндлеры, чтобы не дублировать вывод при повторной инициализации
    for h in list(root.handlers):
        root.removeHandler(h)

    # Общий фильтр контекста
    ctx_filter = ContextFilter()
    formatter = JsonFormatter(indent=json_indent) if fmt == "json" else ConsoleFormatter()

    # === stdout: всё до WARNING включительно (INFO/WARNING) ===
    sh_out = logging.StreamHandler(stream=sys.stdout)
    sh_out.setLevel(logging.DEBUG)                 # пусть проходит всё
    sh_out.addFilter(ctx_filter)
    sh_out.addFilter(MaxLevelFilter(logging.WARNING))
    sh_out.setFormatter(formatter)
    root.addHandler(sh_out)

    # === stderr: только ERROR и выше ===
    sh_err = logging.StreamHandler(stream=sys.stderr)
    sh_err.setLevel(logging.ERROR)                 # отсекаем ниже ERROR
    sh_err.addFilter(ctx_filter)
    sh_err.addFilter(MinLevelFilter(logging.ERROR))
    sh_err.setFormatter(formatter)
    root.addHandler(sh_err)

    # Файл (если указан)
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(level_no)
        fh.addFilter(ctx_filter)
        # Файл удобнее писать в JSON
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)

    # Перехват warnings и библиотек
    logging.captureWarnings(True)
    if propagate_library_logs:
        for lib in ("urllib3", "requests", "aiohttp", "asyncio"):
            logging.getLogger(lib).setLevel(max(level_no, logging.WARNING))

    # Установим базовый контекст (service)
    if service:
        current = dict(_LOG_CONTEXT.get())
        current["service"] = service
        _LOG_CONTEXT.set(current)

    return root

def get_logger(name: Optional[str] = None) -> BoundLogger:
    """Получить связанный логгер-адаптер с поддержкой .bind(**extra)."""
    logger = logging.getLogger(name if name else "")
    return BoundLogger(logger)

__all__ = [
    "setup_logging",
    "get_logger",
    "use_context",
    "BoundLogger",
]