# shellcheck shell=bash
# Resolve ArtifactWorld repo root (parent of scripts/)
export ARTIFACTWORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_expand_config() {
    local name="$1"
    local src="${ARTIFACTWORLD_ROOT}/configs/${name}"
    local dst
    dst="$(mktemp)"
    local weights_root="${ARTIFACTWORLD_WEIGHTS_ROOT:-${ARTIFACTWORLD_ROOT}/weights}"
    local ltx_main_root="${ARTIFACTWORLD_LTX_MAIN_ROOT:-${ARTIFACTWORLD_ROOT}/weights/LTX-Video}"
    local ltx_097_dev_root="${ARTIFACTWORLD_LTX_097_DEV_ROOT:-${weights_root}/LTX-Video-0.9.7-dev}"
    # 13B VAE. NOTE: Lightricks/LTX-Video's own vae/ is the old 2B VAE and does NOT
    # work with the 13B 0.9.8-distilled transformer; the 13B VAE (identical file in
    # LTX-Video-0.9.7-dev / -0.9.7-distilled / -0.9.8-13B-distilled) must be used.
    local ltx_vae_root="${ARTIFACTWORLD_LTX_VAE_ROOT:-${weights_root}/LTX-Video-13B-vae}"
    sed -e "s|__AW_ROOT__|${ARTIFACTWORLD_ROOT}|g" \
        -e "s|__WEIGHTS_ROOT__|${weights_root}|g" \
        -e "s|__LTX_MAIN_ROOT__|${ltx_main_root}|g" \
        -e "s|__LTX_097_DEV_ROOT__|${ltx_097_dev_root}|g" \
        -e "s|__LTX_VAE_ROOT__|${ltx_vae_root}|g" \
        "${src}" >"${dst}"
    echo "${dst}"
}
