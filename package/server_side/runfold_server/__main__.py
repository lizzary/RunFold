from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping, Sequence

import uvicorn

from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings
from runfold_server.errors import StartupError
from runfold_server.observability import configure_logging

_LOGGER = logging.getLogger("runfold_server")


def main(argv: Sequence[str] | None = None, environment: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runfold_server")
    parser.add_argument("--workers", type=int, default=1, help="must remain 1")
    arguments = parser.parse_args(argv)
    env = os.environ if environment is None else environment
    configure_logging()
    try:
        _validate_single_worker(arguments.workers, env)
        settings = load_settings(env)
        application = bootstrap(settings)
    except StartupError as error:
        _LOGGER.error("startup_rejected", extra={"error_code": error.code})
        return 2
    except Exception:
        _LOGGER.exception("startup_failed")
        return 1

    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        workers=1,
        access_log=False,
    )
    return 0


def _validate_single_worker(workers: int, environment: Mapping[str, str]) -> None:
    configured = [str(workers)]
    for name in ("UVICORN_WORKERS", "WEB_CONCURRENCY"):
        if name in environment:
            configured.append(environment[name])
    try:
        counts = [int(value) for value in configured]
    except ValueError as error:
        raise StartupError("invalid_worker_count", "Worker count must be exactly 1") from error
    if any(count != 1 for count in counts):
        raise StartupError("single_worker_required", "RunFold requires exactly one worker")


if __name__ == "__main__":
    raise SystemExit(main())
