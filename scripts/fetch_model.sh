#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-model/resnet50_mura_finetuned.keras}"

if [ -z "${MODEL_URL:-}" ]; then
  echo "MODEL_URL is not set."
  echo "Set MODEL_URL to the GitHub Release asset URL for the .keras file."
  exit 1
fi

mkdir -p "$(dirname "$MODEL_PATH")"

if [ -f "$MODEL_PATH" ]; then
  echo "Model already exists at $MODEL_PATH"
  exit 0
fi

echo "Downloading model to $MODEL_PATH"
curl -L --fail --retry 3 --retry-delay 5 "$MODEL_URL" -o "$MODEL_PATH"

echo "Model downloaded:"
ls -lh "$MODEL_PATH"
