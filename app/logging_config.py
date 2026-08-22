"""
Centralized logging configuration.

Call `configure_logging()` once at process startup (done automatically by
`app.api.main` and `app.cli`). Every module then just does:

    import logging
    logger = logging.getLogger(__name__)

and log records inherit the handlers/formatters configured here, instead of
scattering `print()` calls through the codebase.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.config import settings

_CONFIGURED = False


class _RequestContextFilter(logging.Filter):
    """Injects a `session_id` field (default '-') so the formatter can
    always reference it, even for log records emitted outside a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    if settings.log_json:
        try:
            from pythonjsonlogger import jsonlogger

            fmt = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(session_id)s %(message)s"
            )
        except ImportError:  # pragma: no cover - optional dependency
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | session=%(session_id)s | %(message)s"
            )
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | session=%(session_id)s | %(message)s"
        )

    ctx_filter = _RequestContextFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(ctx_filter)
    root.addHandler(console)

    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.addFilter(ctx_filter)
        root.addHandler(file_handler)

    # Quiet down noisy third-party loggers unless we're in DEBUG.
    if level > logging.DEBUG:
        for noisy in ("httpx", "urllib3", "sentence_transformers", "faiss.loader"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured (level=%s)", settings.log_level)
