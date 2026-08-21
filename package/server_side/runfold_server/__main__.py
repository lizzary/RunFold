from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from runfold_server.bootstrap import bootstrap, compact_index, rebuild_index
from runfold_server.config import load_settings
from runfold_server.errors import StartupError
from runfold_server.observability import configure_logging

_LOGGER = logging.getLogger("runfold_server")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runfold_server")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("serve", "rebuild-index", "compact-index"),
        default="serve",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML configuration file (default: ./config.yaml)",
    )
    parser.add_argument(
        "--actor",
        help="active direct system_admin username required by rebuild-index",
    )
    parser.add_argument("--workers", type=int, default=1, help="must remain 1")
    arguments = parser.parse_args(argv)
    configure_logging()
    try:
        _validate_single_worker(arguments.workers)
        settings = load_settings(arguments.config)
        if arguments.command == "rebuild-index":
            if arguments.actor is None:
                raise StartupError(
                    "maintenance_actor_required",
                    "rebuild-index requires --actor",
                )
            asyncio.run(rebuild_index(settings, arguments.actor))
            return 0
        if arguments.actor is not None:
            raise StartupError(
                "unexpected_maintenance_actor",
                "--actor is only valid with rebuild-index",
            )
        if arguments.command == "compact-index":
            compact_index(settings)
            return 0
        application = bootstrap(settings)
    except StartupError as error:
        _LOGGER.error(
            "startup_rejected",
            extra={"error_code": error.code, "reason": error.safe_message},
        )
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


def _validate_single_worker(workers: int) -> None:
    if workers != 1:
        raise StartupError("single_worker_required", "RunFold requires exactly one worker")


if __name__ == "__main__":
    raise SystemExit(main())
