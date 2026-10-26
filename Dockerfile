# syntax=docker/dockerfile:1.7
#
# Two stages (as in the AI20K D12 deployment lab): the builder has uv, the runtime
# image only gets the virtualenv and the code. Dev tools are not
# installed. Optional features are picked at build time:
#
#   docker build --build-arg EXTRAS="rerank a2a deep" -t ai-kit .

# ---------------------------------------------------------------- builder
FROM python:3.13.2-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# No compilers: every locked dependency ships a wheel, and --no-build below makes that a
# build error instead of a surprise. Add build-essential here if a dependency ever needs it.
WORKDIR /app
ARG EXTRAS=""

# Locked dependencies first: this layer is reused until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-build --no-install-project $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)

# ---------------------------------------------------------------- runtime
FROM python:3.13.2-slim AS runtime

ARG APP_ENV=production
ENV APP_ENV=${APP_ENV} \
    PORT=8000 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

# A fixed, unprivileged uid: escaping the app must not mean root on the host.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
RUN chmod +x /app/scripts/docker-entrypoint.sh && mkdir -p /app/logs && chown appuser:appuser /app/logs
USER appuser

EXPOSE 8000

# python:slim has no curl. /health answers 503 when the database is down, which fails the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
# PORT comes from the platform (Cloud Run, Render, Railway set it). On SIGTERM, uvicorn stops
# accepting connections and gives in-flight requests 30 s to finish before exiting.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --timeout-graceful-shutdown 30"]
