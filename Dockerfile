# RecoverAI: dashboard + FastAPI + Celery worker/beat.
# Start with PROCESS=api|worker|beat (see scripts/start.sh).

FROM node:22-bookworm-slim AS dashboard
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
# Empty API URL = same origin as the FastAPI host that serves dist/.
ENV VITE_API_URL=""
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/backend:/app \
    PORT=8010 \
    PROCESS=api \
    EXECUTION_MODE=shadow \
    CELERY_TASK_ALWAYS_EAGER=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 recoverai

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY ml /app/ml
COPY scripts /app/scripts
COPY deploy /app/deploy
COPY --from=dashboard /dashboard/dist /app/dashboard/dist

RUN sed -i 's/\r$//' /app/scripts/start.sh \
    && chmod +x /app/scripts/start.sh \
    && python /app/deploy/check_runtime_files.py \
    && chown -R recoverai:recoverai /app

USER recoverai
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8010'))"

CMD ["sh", "/app/scripts/start.sh", "api"]
