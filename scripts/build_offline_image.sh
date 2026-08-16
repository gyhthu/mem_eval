#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_PLATFORM="${1:-linux/amd64}"
OUTPUT_FILE="${2:-mem-eval-${TARGET_PLATFORM#linux/}.tar}"
IMAGE_TAG="${MEM_EVAL_IMAGE:-mem-eval:latest}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the offline image." >&2
  exit 1
fi

cd "${REPO_ROOT}"
docker buildx build \
  --platform "${TARGET_PLATFORM}" \
  --load \
  -f Dockerfile \
  -t "${IMAGE_TAG}" \
  .
docker save "${IMAGE_TAG}" -o "${OUTPUT_FILE}"

echo "Offline image written to: ${REPO_ROOT}/${OUTPUT_FILE}"
