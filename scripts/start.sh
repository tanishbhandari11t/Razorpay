#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/backend:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}/backend"

PROCESS="${1:-${PROCESS:-api}}"
PORT="${PORT:-8010}"

if [ ! -f "${ROOT}/ml/artifacts/recovery_model_v1.json" ]; then
  echo "RecoverAI: missing ml/artifacts/recovery_model_v1.json." >&2
  echo "The API will not boot until the frozen model files are present." >&2
fi

case "${PROCESS}" in
  api|web)
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker)
    exec celery -A app.workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
    ;;
  beat)
    exec celery -A app.workers.celery_app:celery_app beat --loglevel=INFO --schedule /tmp/celerybeat-schedule
    ;;
  *)
    echo "Unknown process: ${PROCESS} (expected api, worker, or beat)" >&2
    exit 1
    ;;
esac
