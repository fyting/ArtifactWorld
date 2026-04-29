#!/usr/bin/env bash
# Stage-1: artifact heatmap / noisemap prediction (modified LTX-Video-Trainer).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

STAGE1_ROOT_DEFAULT="${ARTIFACTWORLD_ROOT}/stages/stage1"
STAGE1_ROOT="${LTX_STAGE1_ROOT:-${STAGE1_ROOT_DEFAULT}}"

if [[ ! -f "${STAGE1_ROOT}/scripts/validate.py" ]]; then
    echo "Stage-1 code not found at: ${STAGE1_ROOT}" >&2
    echo "Please ensure ${ARTIFACTWORLD_ROOT}/stages/stage1 exists or set LTX_STAGE1_ROOT." >&2
    exit 1
fi

CFG="$(_expand_config stage1_infer.yaml)"
cleanup() { rm -f "${CFG}"; }
trap cleanup EXIT

export PYTHONPATH="${STAGE1_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${STAGE1_ROOT}"
exec python scripts/validate.py "${CFG}" "$@"
