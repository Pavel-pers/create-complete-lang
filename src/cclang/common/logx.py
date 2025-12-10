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
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# --- Helpers ----------------------------------------------------------------

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# Default LogRecord fields so we can separate extra
_LOG_RECORD_DEFAULT_ATTRS = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process",
}

# Global log context (run_id, pipeline, etc.)
_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("_LOG_CONTEXT", default={})

def _now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

# --- Formatters --------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter without external deps."""
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
        # Add context + extra
        for k, v in record.__dict__.items():
            if k not in _LOG_RECORD_DEFAULT_ATTRS and not k.startswith("_"):
                data[k] = v
        return json.dumps(data, ensure_ascii=False, indent=self._indent, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable formatter with ANSI level coloring."""
    COLORS = {
        "DEBUG": "\033[37m",     # gray
        "INFO": "\033[36m",      # cyan
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        color = self.COLORS.get(level, "")
        reset = self.RESET if color else ""
        # Compact context string from extra (run_id / pipeline first)
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in _LOG_RECORD_DEFAULT_ATTRS and not k.startswith("_")}
        # Sort keys: run_id, pipeline, service first, then alphabetically
        ordered_keys = [k for k in ["run_id", "pipeline", "service"] if k in extra] + sorted([k for k in extra if k not in {"run_id","pipeline","service"}])
        ctx = " ".join(f"{k}={extra[k]}" for k in ordered_keys)
        msg = record.getMessage()
        return f"{ts} {color}{level:<8}{reset} {record.name} {ctx} - {msg}"

# --- Context and adapters ----------------------------------------------------

@dataclass(frozen=True)
class _ContextToken:
    previous: Dict[str, Any]

@contextmanager
def use_context(**pairs: Any):
    """Context manager: temporarily add fields (run_id, pipeline, etc.) to all log records."""
    current = dict(_LOG_CONTEXT.get())
    previous = dict(current)
    current.update(pairs)
    token = _LOG_CONTEXT.set(current)
    try:
        yield _ContextToken(previous=previous)
    finally:
        # Restore previous context
        _LOG_CONTEXT.reset(token)

class ContextFilter(logging.Filter):
    """Inject ContextVar fields into each LogRecord."""
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CONTEXT.get()
        for k, v in ctx.items():
            setattr(record, k, v)
        return True

class BoundLogger(logging.LoggerAdapter):
    """LoggerAdapter with convenient .bind(**extra) and merged extra payloads."""
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

    def exception(self, msg, *args, exc: BaseException | None = None, **kwargs):
        """
        Log an ERROR with traceback and, when debug is enabled, attach structured error fields.

        Extra fields:
        - error_type: exception class name
        - error_message: str(exc)
        """
        exc_obj = exc or sys.exc_info()[1]
        extra = kwargs.pop("extra", {}) or {}
        if self.logger.isEnabledFor(logging.DEBUG) and exc_obj is not None:
            extra.setdefault("error_type", type(exc_obj).__name__)
            extra.setdefault("error_message", str(exc_obj))
        kwargs["exc_info"] = exc_obj or True
        return super().log(logging.ERROR, msg, *args, extra=extra, **kwargs)

# --- Filters -----------------------------------------------------------------
class MaxLevelFilter(logging.Filter):
    """Pass only records with level <= max_level."""
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level

class MinLevelFilter(logging.Filter):
    """Pass only records with level >= min_level."""
    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level

# --- Initialization ----------------------------------------------------------

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
    """Configure logging for CLI/apps; should be called once."""
    # Read environment
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    level_no = _LEVELS.get(level_name, logging.INFO)

    fmt = (fmt or os.getenv("LOG_FORMAT") or "console").lower()
    log_file = file or os.getenv("LOG_FILE")
    max_bytes = int(max_bytes or os.getenv("LOG_MAX_BYTES") or 10_485_760)  # 10 MiB
    backup_count = int(backup_count or os.getenv("LOG_BACKUP_COUNT") or 5)

    root = logging.getLogger()
    root.setLevel(level_no)
    # Remove existing handlers to avoid duplication on re-init
    for h in list(root.handlers):
        root.removeHandler(h)

    # Shared context filter
    ctx_filter = ContextFilter()
    formatter = JsonFormatter(indent=json_indent) if fmt == "json" else ConsoleFormatter()

    # === stdout: everything up to WARNING (INFO/WARNING) ===
    sh_out = logging.StreamHandler(stream=sys.stdout)
    sh_out.setLevel(logging.DEBUG)                 # allow everything in, filter later
    sh_out.addFilter(ctx_filter)
    sh_out.addFilter(MaxLevelFilter(logging.WARNING))
    sh_out.setFormatter(formatter)
    root.addHandler(sh_out)

    # === stderr: only ERROR and above ===
    sh_err = logging.StreamHandler(stream=sys.stderr)
    sh_err.setLevel(logging.ERROR)                 # drop below ERROR
    sh_err.addFilter(ctx_filter)
    sh_err.addFilter(MinLevelFilter(logging.ERROR))
    sh_err.setFormatter(formatter)
    root.addHandler(sh_err)

    # File (if provided)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(level_no)
        fh.addFilter(ctx_filter)
        # JSON is easier to parse in files
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)

    # Capture warnings and quiet noisy libraries
    logging.captureWarnings(True)
    if propagate_library_logs:
        for lib in ("urllib3", "requests", "aiohttp", "asyncio"):
            logging.getLogger(lib).setLevel(max(level_no, logging.WARNING))

    # Set baseline context (service)
    if service:
        current = dict(_LOG_CONTEXT.get())
        current["service"] = service
        _LOG_CONTEXT.set(current)

    return root

def get_logger(name: Optional[str] = None) -> BoundLogger:
    """Return BoundLogger with .bind(**extra) support."""
    logger = logging.getLogger(name if name else "")
    return BoundLogger(logger)

__all__ = [
    "setup_logging",
    "get_logger",
    "use_context",
    "BoundLogger",
]
