#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_URL="${ONDAMM_OLLAMA_URL:-http://127.0.0.1:11434}"
CHAT_MODEL="${ONDAMM_OLLAMA_MODEL:-qwen3.8:27b-mlx}"
EMBED_MODEL="${ONDAMM_OLLAMA_EMBED_MODEL:-embeddinggemma}"
NUM_CTX="${ONDAMM_OLLAMA_NUM_CTX:-16384}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama CLI가 없습니다. https://ollama.com/download/mac 에서 공식 앱을 설치하세요." >&2
  exit 2
fi

if ! curl --fail --silent --max-time 2 "${OLLAMA_URL}/api/version" >/dev/null; then
  echo "${OLLAMA_URL}의 Ollama daemon에 연결할 수 없습니다." >&2
  echo "macOS에서는 Ollama.app을 실행하세요: open -a Ollama" >&2
  exit 3
fi

CHAT_DIGEST="$(ollama list | awk -v model="${CHAT_MODEL}" 'NR > 1 && $1 == model {print $2; exit}')"
if [[ -z "${CHAT_DIGEST}" ]]; then
  echo "로컬 chat/vision model이 없습니다: ${CHAT_MODEL}" >&2
  echo "ollama pull ${CHAT_MODEL}" >&2
  exit 4
fi

if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "${EMBED_MODEL}:latest" \
  && ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "${EMBED_MODEL}"; then
  echo "로컬 embedding model이 없습니다: ${EMBED_MODEL}" >&2
  echo "ollama pull ${EMBED_MODEL}" >&2
  exit 5
fi

export ONDAMM_LLM_PROVIDER=ollama
export ONDAMM_OLLAMA_URL="${OLLAMA_URL}"
export ONDAMM_OLLAMA_MODEL="${CHAT_MODEL}"
export ONDAMM_OLLAMA_MODEL_DIGEST="${ONDAMM_OLLAMA_MODEL_DIGEST:-${CHAT_DIGEST}}"
export ONDAMM_OLLAMA_EMBED_MODEL="${EMBED_MODEL}"
export ONDAMM_OLLAMA_NUM_CTX="${NUM_CTX}"

exec bash "${ROOT_DIR}/scripts/ondamm_web.sh" "$@"
