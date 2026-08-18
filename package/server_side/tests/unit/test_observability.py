from __future__ import annotations

import json
import logging
import sys

from runfold_server.observability import SafeJsonFormatter


def test_formatter_redacts_secrets_and_omits_exception_message() -> None:
    try:
        raise ValueError("password=plain-secret Bearer opaque-token")
    except ValueError:
        exception = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="authorization:Bearer visible-token api_key=visible-key",
        args=(),
        exc_info=exception,
    )

    formatted = SafeJsonFormatter().format(record)
    payload = json.loads(formatted)

    assert "visible-token" not in formatted
    assert "visible-key" not in formatted
    assert "plain-secret" not in formatted
    assert payload["exception_type"] == "ValueError"
    assert payload["stack"]

