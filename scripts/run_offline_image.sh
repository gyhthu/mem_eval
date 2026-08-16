#!/usr/bin/env bash
set -euo pipefail

IMAGE_ARCHIVE="${1:-}"
IMAGE_TAG="${MEM_EVAL_IMAGE:-mem-eval:latest}"
HOST_PORT="${PLATFORM_PORT:-8501}"
CONTAINER_NAME="${MEM_EVAL_CONTAINER:-mem-eval}"

if [[ -z "${IMAGE_ARCHIVE}" || ! -f "${IMAGE_ARCHIVE}" ]]; then
  echo "Usage: bash scripts/run_offline_image.sh <mem-eval-image.tar>" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run the offline image." >&2
  exit 1
fi
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container ${CONTAINER_NAME} already exists. Start or remove it explicitly first." >&2
  exit 1
fi

docker load -i "${IMAGE_ARCHIVE}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  -p "${HOST_PORT}:8501" \
  "${IMAGE_TAG}"

echo "Mem Eval is starting at http://127.0.0.1:${HOST_PORT}"
echo "Health check: curl http://127.0.0.1:${HOST_PORT}/_stcore/health"
