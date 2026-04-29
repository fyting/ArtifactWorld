#!/usr/bin/env bash
# End-to-end example: preprocess → Stage-1 → Stage-2 (edit paths / GPU as needed).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
PROC_OUT="${ARTIFACTWORLD_ROOT}/workspace/processed"
STAGE1_OUT="${ARTIFACTWORLD_ROOT}/outputs/stage1"
STAGE2_OUT="${ARTIFACTWORLD_ROOT}/outputs/stage2"

mkdir -p "${ARTIFACTWORLD_ROOT}/workspace"

echo "[1/3] process_videos.py -> ${PROC_OUT}"
python "${ARTIFACTWORLD_ROOT}/tools/process_videos.py" \
    --gt-dir "${GT_DIR:?set GT_DIR}" \
    --artifact-dir "${ARTIFACT_DIR:?set ARTIFACT_DIR}" \
    --output-dir "${PROC_OUT}"

echo "[2/3] stage1 validate (GPU ${GPU})"
CUDA_VISIBLE_DEVICES="${GPU}" "${SCRIPT_DIR}/run_stage1_infer.sh" \
    --input-folder "${PROC_OUT}" \
    --output-folder "${STAGE1_OUT}"

echo "[3/3] stage2 validate (GPU ${GPU})"
CUDA_VISIBLE_DEVICES="${GPU}" "${SCRIPT_DIR}/run_stage2_infer.sh" \
    --input-folder "${PROC_OUT}" \
    --noisemap-videos-folder "${STAGE1_OUT}/samples" \
    --output-folder "${STAGE2_OUT}"

echo "Done. Restored videos: ${STAGE2_OUT}/samples"
