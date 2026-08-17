from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from datetime import UTC, datetime

_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(authorization|password|token|api[_-]?key)\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*")


class SafeJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "event": _redact(record.getMessage()),
        }
        for name in (
            "request_id",
            "method",
            "route",
            "status",
            "duration_ms",
            "actor_id",
            "error_code",
            "reason",
        ):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = _redact(str(value)) if isinstance(value, str) else value
        if record.exc_info and record.exc_info[2] is not None:
            payload["stack"] = [
                {
                    "file": frame.filename,
                    "line": frame.lineno,
                    "function": frame.name,
                }
                for frame in traceback.extract_tb(record.exc_info[2])
            ]
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("runfold_server")
    if any(getattr(handler, "_runfold_handler", False) for handler in logger.handlers):
        logger.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SafeJsonFormatter())
    handler._runfold_handler = True  # type: ignore[attr-defined]
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(level)


def _redact(value: str) -> str:
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _SENSITIVE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
