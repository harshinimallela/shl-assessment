# ── Stage 1: Builder ───────────────────────────────────────────────────────────
# Install dependencies in an isolated layer.
# Separating builder from runtime halves the final image size (~900MB → ~500MB)
# because build tools (gcc, pip cache) are discarded.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer if unchanged
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user: defence in depth against container escape
RUN useradd --no-create-home --shell /bin/false appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY catalog.json .

# Groq is an API — no model download needed.
# Pre-download sentence-transformers models for FAISS retrieval:
# Sentence-transformers caches models to ~/.cache/huggingface
# Pre-download at build time so container starts without internet access
# (remove this RUN line if you prefer runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

USER appuser

EXPOSE 8000

# Health check — Render/Railway/Fly use this to determine readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
