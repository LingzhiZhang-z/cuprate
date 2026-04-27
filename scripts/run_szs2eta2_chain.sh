#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROOT="${ROOT:-results_szs2eta2}"
NP="${NP:-4}"
NS="${NS:-2 3 4 5 6}"
U="${U:-1.0}"
MODE="SzS2eta2"
SCOPE="nonnegative"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

U_TOKEN="$(printf "%.4f" "${U}")"

format_t() {
    local cents="$1"
    if (( cents < 100 )); then
        printf "0.%02d00" "${cents}"
    else
        printf "%d.%04d" "$(( cents / 100 ))" "$(( (cents % 100) * 100 ))"
    fi
}

result_path() {
    local n="$1"
    local t="$2"
    local workflow="$3"
    printf "%s/block_main/N_%d_nelec_%d_U_%s_t_%s/mode_twoSz_twoS_eta_0/workflow_%s/results.json" \
        "${ROOT}" "${n}" "${n}" "${U_TOKEN}" "${t}" "${workflow}"
}

run_main() {
    local n="$1"
    local t="$2"
    local workflow="$3"
    local cache_mode="$4"
    local seed_results="${5:-}"
    local output
    output="$(result_path "${n}" "${t}" "${workflow}")"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output}" ]]; then
        echo "[skip] N=${n} T=${t} workflow=${workflow} output=${output}"
        return
    fi

    local cmd=(
        "${MPIRUN_BIN}" "-np" "${NP}"
        "${PYTHON_BIN}" "-m" "cuprate.main"
        "ROOT=${ROOT}"
        "N=${n}"
        "U=${U}"
        "T=${t}"
        "MODE=${MODE}"
        "SCOPE=${SCOPE}"
        "workflow=${workflow}"
        "CACHE_MODE=${cache_mode}"
    )
    if [[ -n "${seed_results}" ]]; then
        cmd+=("SEED_RESULTS=${seed_results}")
    fi

    echo "[run] N=${n} T=${t} workflow=${workflow} CACHE_MODE=${cache_mode}"
    if [[ -n "${seed_results}" ]]; then
        echo "[seed] ${seed_results}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf "  %q" "${cmd[@]}"
        printf "\n"
    else
        "${cmd[@]}"
    fi
}

echo "[config] ROOT=${ROOT} NP=${NP} NS=${NS} U=${U} T_steps=${START_STEP}..${END_STEP}"

for n in ${NS}; do
    previous_adiabatic=""
    for (( step = START_STEP; step <= END_STEP; step++ )); do
        cents=$(( step * 2 ))
        t="$(format_t "${cents}")"

        run_main "${n}" "${t}" "occ" "save"
        run_main "${n}" "${t}" "greedy_multi" "load"

        if (( step == START_STEP )); then
            seed_results="$(result_path "${n}" "${t}" "occ")"
        else
            seed_results="${previous_adiabatic}"
        fi

        if [[ "${DRY_RUN}" != "1" && ! -f "${seed_results}" ]]; then
            echo "error: missing adiabatic seed: ${seed_results}" >&2
            exit 1
        fi

        run_main "${n}" "${t}" "adiabatic" "load" "${seed_results}"
        previous_adiabatic="$(result_path "${n}" "${t}" "adiabatic")"
    done
done
