#!/usr/bin/env bash
set -euo pipefail

bash scripts/fetch_model.sh

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
