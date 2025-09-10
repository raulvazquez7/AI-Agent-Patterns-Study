# ruff: noqa: E402
import os

if not (
    (disable_truststore := os.getenv("DISABLE_TRUSTSTORE"))
    and disable_truststore.lower() == "true"
):
    import truststore  # noqa: F401

    truststore.inject_into_ssl()  # noqa: F401

import asyncio
import contextlib
import json
import logging.config
import pathlib
import signal
from contextlib import asynccontextmanager
from typing import cast

import structlog

from langgraph_runtime.database import pool_stats
from langgraph_runtime.lifespan import lifespan
from langgraph_runtime.metrics import get_metrics

logger = structlog.stdlib.get_logger(__name__)


async def health_and_metrics_server():
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    port = int(os.getenv("PORT", "8080"))

    async def health_endpoint(request):
        return JSONResponse({"status": "ok"})

    async def metrics_endpoint(request):
        metrics = get_metrics()
        worker_metrics = cast(dict[str, int], metrics["workers"])
        workers_max = worker_metrics["max"]
        workers_active = worker_metrics["active"]
        workers_available = worker_metrics["available"]

        project_id = os.getenv("LANGSMITH_HOST_PROJECT_ID")
        revision_id = os.getenv("LANGSMITH_HOST_REVISION_ID")

        metrics_lines = [
            "# HELP lg_api_workers_max The maximum number of workers available.",
            "# TYPE lg_api_workers_max gauge",
            f'lg_api_workers_max{{project_id="{project_id}", revision_id="{revision_id}"}} {workers_max}',
            "# HELP lg_api_workers_active The number of currently active workers.",
            "# TYPE lg_api_workers_active gauge",
            f'lg_api_workers_active{{project_id="{project_id}", revision_id="{revision_id}"}} {workers_active}',
            "# HELP lg_api_workers_available The number of available (idle) workers.",
            "# TYPE lg_api_workers_available gauge",
            f'lg_api_workers_available{{project_id="{project_id}", revision_id="{revision_id}"}} {workers_available}',
        ]

        metrics_lines.extend(
            pool_stats(
                project_id=project_id,
                revision_id=revision_id,
            )
        )

        return PlainTextResponse(
            "\n".join(metrics_lines),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    app = Starlette(
        routes=[
            Route("/ok", health_endpoint),
            Route("/metrics", metrics_endpoint),
        ]
    )

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="error",
        access_log=False,
    )
    # Server will run indefinitely until the process is terminated
    server = uvicorn.Server(config)

    logger.info(f"Health and metrics server started at http://0.0.0.0:{port}")
    await server.serve()


async def entrypoint(
    grpc_port: int | None = None,
    entrypoint_name: str = "python-queue",
    cancel_event: asyncio.Event | None = None,
):
    from langgraph_api import logging as lg_logging
    from langgraph_api.api import user_router

    lg_logging.set_logging_context({"entrypoint": entrypoint_name})
    tasks: set[asyncio.Task] = set()

    original_lifespan = user_router.router.lifespan_context if user_router else None

    @asynccontextmanager
    async def combined_lifespan(
        app, with_cron_scheduler=False, grpc_port=None, taskset=None
    ):
        async with lifespan(
            app,
            with_cron_scheduler=with_cron_scheduler,
            grpc_port=grpc_port,
            taskset=taskset,
            cancel_event=cancel_event,
        ):
            if original_lifespan:
                async with original_lifespan(app):
                    yield
            else:
                yield

    async with combined_lifespan(
        None, with_cron_scheduler=False, grpc_port=grpc_port, taskset=tasks
    ):
        tasks.add(asyncio.create_task(health_and_metrics_server()))
        await asyncio.gather(*tasks)


async def main(grpc_port: int | None = None, entrypoint_name: str = "python-queue"):
    """Run the queue entrypoint and shut down gracefully on SIGTERM/SIGINT."""
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.warning("Received termination signal, initiating graceful shutdown")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    except (NotImplementedError, RuntimeError):
        signal.signal(signal.SIGTERM, lambda *_: _handle_signal())

    entry_task = asyncio.create_task(
        entrypoint(
            grpc_port=grpc_port,
            entrypoint_name=entrypoint_name,
            cancel_event=stop_event,
        )
    )
    # Handle the case where the entrypoint errors out
    entry_task.add_done_callback(lambda _: stop_event.set())
    await stop_event.wait()

    logger.warning("Cancelling queue entrypoint task")
    entry_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await entry_task


if __name__ == "__main__":
    from langgraph_api import config

    config.IS_QUEUE_ENTRYPOINT = True
    with open(pathlib.Path(__file__).parent.parent / "logging.json") as file:
        loaded_config = json.load(file)
        logging.config.dictConfig(loaded_config)
    try:
        import uvloop  # type: ignore[unresolved-import]

        uvloop.install()
    except ImportError:
        pass
    # run the entrypoint
    asyncio.run(main())
