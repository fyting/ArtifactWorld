#!/usr/bin/env bash
# Stage-2: noisemap-guided restoration with auxiliary latent fusion (modified LTX-Video-Trainer).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

STAGE2_ROOT_DEFAULT="${ARTIFACTWORLD_ROOT}/stages/stage2"
STAGE2_ROOT="${LTX_STAGE2_ROOT:-${STAGE2_ROOT_DEFAULT}}"

if [[ ! -f "${STAGE2_ROOT}/scripts/validate.py" ]]; then
    echo "Stage-2 code not found at: ${STAGE2_ROOT}" >&2
    echo "Please ensure ${ARTIFACTWORLD_ROOT}/stages/stage2 exists or set LTX_STAGE2_ROOT." >&2
    exit 1
fi

CFG="$(_expand_config stage2_infer.yaml)"
cleanup() { rm -f "${CFG}"; }
trap cleanup EXIT

export PYTHONPATH="${STAGE2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${STAGE2_ROOT}"
exec python scripts/validate.py "${CFG}" "$@"
