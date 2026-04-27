#!/usr/bin/env bash
set -euo pipefail
#
# Drive lce + embed across (Nmax=2..6) x (workflow) x (t) for the
# embed-vs-perturbation convergence plot.
#
# Per (Nmax, workflow, t):
#   - Use truncated seed file from scripts/gen_nmax_seeds.py
#     (Nmax=6 reuses the existing untagged seed)
#   - Run python -m cuprate.lce  (writes block_lce/N_<Nmax>_.../seed_<token>/)
#   - Run python -m cuprate.embed (writes block_embed/...)
#
# SKIP_EXISTING=1 to skip combinations whose embed two_site.txt already
# exists. DRY_RUN=1 to print commands without executing.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROOT="${ROOT:-results_szs2eta2}"
NMAXES="${NMAXES:-2 3 4 5 6}"
WORKFLOWS="${WORKFLOWS:-occ greedy_multi adiabatic}"
U="${U:-1.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
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

seed_path() {
    local workflow="$1"
    local nmax="$2"
    local t="$3"
    if (( nmax == 6 )); then
        printf "seed_sets/szs2eta2_%s_T%s.txt" "${workflow}" "${t}"
    else
        printf "seed_sets/szs2eta2_%s_Nmax%d_T%s.txt" "${workflow}" "${nmax}" "${t}"
    fi
}

embed_two_site_path() {
    local workflow="$1"
    local nmax="$2"
    local t="$3"
    local seed_token
    if (( nmax == 6 )); then
        seed_token="seed_szs2eta2_${workflow}_T${t}"
    else
        seed_token="seed_szs2eta2_${workflow}_Nmax${nmax}_T${t}"
    fi
    printf "%s/block_embed/N_%d_nelec_%d_U_%s_t_%s/%s/two_site.txt" \
        "${ROOT}" "${nmax}" "${nmax}" "${U_TOKEN}" "${t}" "${seed_token}"
}

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

# Step 1: generate truncated seeds (idempotent)
echo "[step] generate truncated seeds for Nmax=2..5"
run "${PYTHON_BIN}" scripts/gen_nmax_seeds.py --root "${ROOT}"

# Step 2: lce + embed for each (Nmax, workflow, t)
total=0
done_count=0
skip_count=0

for nmax in ${NMAXES}; do
    for workflow in ${WORKFLOWS}; do
        for step in $(seq "${START_STEP}" "${END_STEP}"); do
            cents=$(( step * 2 ))
            t="$(format_t "${cents}")"
            total=$(( total + 1 ))

            seed="$(seed_path "${workflow}" "${nmax}" "${t}")"
            embed_out="$(embed_two_site_path "${workflow}" "${nmax}" "${t}")"

            if [[ "${SKIP_EXISTING}" == "1" && -f "${embed_out}" ]]; then
                skip_count=$(( skip_count + 1 ))
                continue
            fi

            if [[ ! -f "${ROOT}/${seed}" ]]; then
                echo "[skip] missing seed ${seed}"
                skip_count=$(( skip_count + 1 ))
                continue
            fi

            echo "[run] Nmax=${nmax} workflow=${workflow} t=${t}"
            run "${PYTHON_BIN}" -m cuprate.lce \
                "ROOT=${ROOT}" "N=${nmax}" "U=${U}" "T=${t}" "SEED_SET=${seed}"
            run "${PYTHON_BIN}" -m cuprate.embed \
                "ROOT=${ROOT}" "N=${nmax}" "U=${U}" "T=${t}" "SEED_SET=${seed}"
            done_count=$(( done_count + 1 ))
        done
    done
done

echo "[done] total=${total} ran=${done_count} skipped=${skip_count}"
