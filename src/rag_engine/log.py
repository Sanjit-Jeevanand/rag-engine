import contextvars
import logging
import sys
import uuid

import structlog

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def bind_request_id(request_id: str | None = None) -> str:
    rid = request_id or str(uuid.uuid4())
    _request_id.set(rid)
    return rid


def _add_request_id(
    logger: structlog.types.WrappedLogger,
    method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    rid = _request_id.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    processors: list[structlog.types.Processor] = [
        _add_request_id,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
