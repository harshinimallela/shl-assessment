"""
SHL Assessment Recommender API
================================
Multi-agent conversational system for SHL Individual Test Solution selection.
Powered by Groq (free tier) — llama-3.3-70b-versatile.

Architecture: Supervisor → [Guardrail → Intent → Route → Agent → Validate]
API Contract: GET /health, POST /chat  (schema non-negotiable per SHL spec)
"""
from __future__ import annotations

import logging

from groq import APIError as GroqAPIError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.dependencies import supervisor_dep
from app.middleware.logging import RequestLoggingMiddleware
from app.models.api import ChatRequest, ChatResponse
from app.supervisor import Supervisor

# ── Logging setup ─────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational AI for recommending SHL Individual Test Solutions. "
        "Powered by Groq (free tier) — Llama 3.3 70B. "
        "Multi-agent pipeline: Guardrail → Intent → Retrieval → Recommend/Compare/Refine."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Readiness probe. Returns HTTP 200 when the service is ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    req: ChatRequest,
    supervisor: Supervisor = Depends(supervisor_dep),
) -> ChatResponse:
    """
    Stateless conversational endpoint.

    Send full conversation history on every request.
    Service stores no per-conversation state.
    """
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        response, trace = supervisor.handle(messages)
        logger.debug("trace=%s", trace)
        return response
    except GroqAPIError as e:
        logger.error("Groq API error: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
    except Exception as e:
        logger.error("Unexpected error in /chat: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
