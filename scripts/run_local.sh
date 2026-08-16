#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PLATFORM_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m streamlit run src/eval_platform/app.py \
  --server.address "${PLATFORM_HOST:-0.0.0.0}" \
  --server.port "${PLATFORM_PORT:-8501}" \
  --server.headless true
