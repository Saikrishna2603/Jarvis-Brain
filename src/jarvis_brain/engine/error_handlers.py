from fastapi import Request
from fastapi.responses import JSONResponse

from jarvis_platform.observability.event_logger import observability_event_logger


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a consistent safe response for unexpected errors."""
    observability_event_logger.log_error(
        "unhandled_exception",
        str(exc),
        metadata={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "type": "internal_server_error",
                "message": "An internal error occurred.",
            },
        },
    )
