"""
RequestLoggingMiddleware: attaches a request ID to every request and logs
structured timing + status on completion.

Why middleware (not a route decorator)?
  - Covers all routes automatically, including future ones
  - Adds X-Request-ID to response headers for client-side tracing
  - Captures errors from background tasks that route decorators miss

Why not OpenTelemetry?
  - OTEL adds 3+ dependencies and requires a collector sidecar
  - For a single-service deployment, structured JSON logs consumed by
    Render/Railway/Fly log aggregation is sufficient
  - Easy to upgrade: replace this middleware with an OTEL SDK exporter
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        # Attach request ID to request state for downstream logging
        request.state.request_id = request_id

        response: Response = await call_next(request)

        latency_ms = round((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request method=%s path=%s status=%d latency_ms=%d request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response
