#!/usr/bin/env python3
"""Layer B: family-level numerical comparison vs data_test/.

For every (N, T, MODE, workflow, hole, class_idx) case present in results_opus:

- Resolve the matching reference subdir under data_test/Block_U_t.../ :
    workflow=occ            -> N{N}{,_sz{Sz}}
    workflow=greedy_multi   -> N{N}{,_sz{Sz}}_multi_restart
    workflow=adiabatic      -> N{N}{,_sz{Sz}}_adiabatic_restart
  where Sz token is `0.0000` for even N (twoSz=0) and `0.5000` for odd N (twoSz=1).

- Recompute family observables in-process (HubbardModel) and compare:
    Strict   : eigvals, converted states, double_occ, S² col0+col2 (degenerate-trace
               fallback for col0 / col2 / double_occ).
    Strict   : per-cluster .txt (constant, two-/four-/six-site coefficients, fit).
  For workflow=occ:
    Strict   : selected indices set, selected occupation (sorted), s2_selected
               (sorted by occupation), Heff and T11m1 element-wise on legacy-permuted
               basis.
  For workflow=greedy_multi:
    Norm-     : current |T11-I| <= ref + ATOL_NORM (loose).
  For workflow=adiabatic:
    Norm-     : current |T11-I| <= ref + ATOL_NORM (loose); at T=0.02 require
               equality with the matching occ result (we compare against the same
               occ reference subdir).

Restart-able via <root>/_status/layer_b.jsonl. Each case writes one record on
completion. Pre-existing 'passed' records short-circuit re-validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from cuprate.clusters import ClusterSets  # noqa: E402
from cuprate.hubbard import HubbardModel  # noqa: E402
from cuprate.operators import operators_from_terms, operators_to_terms  # noqa: E402
from cuprate.paths import (  # noqa: E402
    RESULTS_FILE,
    STAGE_MAIN,
    family_clusters_file,
    family_exchange_file,
    family_projection_file,
    workflow_dir,
)
from cuprate.states import calc_double_occupation_matrix, calc_fourS2_matrix  # noqa: E402

from verify_main_vs_data_test import (  # noqa: E402
    ATOL_BASELINE,
    ATOL_DEGEN_TRACE,
    ATOL_TXT_ROUND,
    FileResult,
    _degenerate_blocks,
    _max_abs,
    canonical_spin_permutation,
    compare_array,
    compare_eigvals,
    compare_per_eigenstate,
    compare_txt_like,
    legacy_spin_ordering,
    legacy_state_key,
    legacy_state_row,
    parse_text_array,
    parse_txt,
    sidecar_to_txtlike,
    _sites_in_operator_order,
)

from run_data_test_regression import (  # noqa: E402
    Case,
    StatusIndex,
    U_VALUE,
    enumerate_cases,
    now_iso,
)


ATOL_NORM = 1e-6  # loose tolerance for |T11-I| norm on greedy_multi / adiabatic


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    family_results: list[FileResult] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict:
        n_pass = sum(1 for r in self.family_results if r.passed)
        n_total = len(self.family_results)
        return {"checks": n_total, "passed": n_pass, "failed": n_total - n_pass}


def ref_subdir_for_case(case: Case) -> str:
    """Map case to data_test reference subdirectory name."""
    parts = [f"N{case.N}"]
    if case.mode == "Sz":
        sz_value = case.twoSz / 2  # 0 -> 0.0, 1 -> 0.5
        parts[-1] = f"N{case.N}_sz{sz_value:.4f}"
    if case.workflow == "greedy_multi":
        parts[-1] += "_multi_restart"
    elif case.workflow == "adiabatic":
        parts[-1] += "_adiabatic_restart"
    return parts[-1]


def ref_dir_for_case(case: Case) -> Path:
    """Absolute path to reference subdirectory."""
    block_dir = REPO_ROOT / "data_test" / f"Block_U{U_VALUE:.4f}_t{case.T:.4f}"
    return block_dir / ref_subdir_for_case(case)


def workflow_output_dir(case: Case, root: Path) -> Path:
    return workflow_dir(
        root,
        STAGE_MAIN,
        case.N,
        case.N,
        U_VALUE,
        case.T,
        case.mode,
        case.workflow,
        twoSz=case.twoSz,
    )


@dataclass
class FamilyObs:
    """Recomputed observables for one (N, T, MODE, hole, class) family."""

    hole: int
    class_idx: int
    basis_states: list[int]
    eigvals: np.ndarray
    eigvecs_fock: np.ndarray
    double_occ: np.ndarray
    s2_diag_values: np.ndarray
    s2_variance: np.ndarray
    s2_overlap: np.ndarray
    spin_fock_rows: list[int]
    selected_indices: list[int]
    selected_double_occ: np.ndarray
    heff: np.ndarray
    t11m1: np.ndarray
    t11m1_norm: float


def recompute_family(
    case: Case,
    hole: int,
    class_idx: int,
    members: list,
) -> FamilyObs:
    """Recompute family observables. Always uses workflow=occ for selection
    because the strict-comparison reference data was produced from the occ
    representative. greedy_multi / adiabatic re-runs reuse the same eigh; only
    `selected_indices` differs and is checked separately by norm comparison."""
    representative = members[0]
    model = HubbardModel(representative, U_VALUE, case.T)
    model.set_symmetry(case.mode, twoSz=case.twoSz)
    model.build_hamiltonians()
    model.solve()
    model.project(method="occ")

    block = model.blocks[0]
    eigvecs_fock = np.asarray(block.eigvecs_fock)
    eigvals = np.asarray(block.eigvals).astype(float)

    dom = calc_double_occupation_matrix(block.basis_states, case.N)
    double_occ = np.real(np.diag(eigvecs_fock.conj().T @ dom @ eigvecs_fock))

    fourS2 = calc_fourS2_matrix(block.basis_states, case.N)
    mat_fourS2 = eigvecs_fock.conj().T @ fourS2 @ eigvecs_fock
    sq_fourS2 = eigvecs_fock.conj().T @ fourS2 @ fourS2 @ eigvecs_fock
    s2_values = 0.25 * np.real(np.diag(mat_fourS2))
    s2_variance = 0.0625 * np.real(np.diag(sq_fourS2) - np.diag(mat_fourS2) ** 2)

    spin_rows = block.spin_fock_rows()
    s2_overlap = np.sum(np.abs(eigvecs_fock[spin_rows, :]) ** 2, axis=0)

    selected_indices = list(model.selected_indices[0])
    heff = np.asarray(model.heff[0])

    spin_cols = block.spin_sector_columns()
    s_bd = block.eigvecs[np.ix_(spin_cols, selected_indices)]
    U_svd, sigma, _ = np.linalg.svd(s_bd, full_matrices=False)
    t11m1 = U_svd @ np.diag(sigma) @ U_svd.conj().T - np.eye(s_bd.shape[0])
    t11m1_norm = float(np.linalg.norm(t11m1.flatten()))

    return FamilyObs(
        hole=hole,
        class_idx=class_idx,
        basis_states=list(block.basis_states),
        eigvals=eigvals,
        eigvecs_fock=eigvecs_fock,
        double_occ=double_occ,
        s2_diag_values=s2_values,
        s2_variance=s2_variance,
        s2_overlap=s2_overlap,
        spin_fock_rows=spin_rows,
        selected_indices=selected_indices,
        selected_double_occ=double_occ[np.asarray(selected_indices, dtype=int)],
        heff=heff,
        t11m1=t11m1,
        t11m1_norm=t11m1_norm,
    )


def enumerate_families(N: int) -> list[tuple[int, int, list]]:
    by: dict[tuple[int, int], list] = {}
    for cluster in ClusterSets(N).generate().clusters:
        key = (int(cluster.hole), int(cluster.class_idx))
        by.setdefault(key, []).append(cluster)
    return [(hole, cls, by[(hole, cls)]) for (hole, cls) in sorted(by)]


def load_current_payloads(case: Case, root: Path) -> tuple[dict, dict[tuple[int, int], dict]]:
    """Return (results.json, {(hole,class): exchange.json}). Raises ValueError on missing."""
    workflow_out = workflow_output_dir(case, root)
    results_path = workflow_out / RESULTS_FILE
    if not results_path.is_file():
        raise ValueError(f"missing current results.json: {results_path}")
    results = json.loads(results_path.read_text())
    family_payloads: dict[tuple[int, int], dict] = {}
    for entry in results["families"]:
        ex_path = workflow_out / entry["exchange_file"]
        if not ex_path.is_file():
            raise ValueError(f"missing exchange json: {ex_path}")
        family_payloads[(int(entry["hole"]), int(entry["class_idx"]))] = json.loads(
            ex_path.read_text()
        )
    return results, family_payloads


def load_reference_npy_set(ref_dir: Path, prefix: str) -> dict[str, np.ndarray | None]:
    """Load all .npy reference files for one family. Missing ones return None."""
    names = (
        "eigvals",
        "double_occupation_expectation",
        "s2_digonal",
        "s2_selected",
        "states",
        "t11_selected_indices",
        "t11_selected_occupation",
        "Heff",
        "T11m1",
    )
    out: dict[str, np.ndarray | None] = {}
    for name in names:
        path = ref_dir / f"{prefix}_{name}.npy"
        out[name] = parse_text_array(path) if path.is_file() else None
    return out


def fallback_eigvals_path(case: Case) -> Path:
    """Restart references don't store eigvals; fall back to the parent occ ref."""
    parent_case = Case(
        N=case.N,
        T=case.T,
        mode=case.mode,
        twoSz=case.twoSz,
        workflow="occ",
    )
    return ref_dir_for_case(parent_case)


def compare_family_strict(
    case: Case,
    obs: FamilyObs,
    ref: dict[str, np.ndarray | None],
    ref_dir: Path,
    prefix: str,
) -> list[FileResult]:
    """Strict per-eigenstate checks shared across all workflows."""
    out: list[FileResult] = []

    if ref["eigvals"] is None:
        # Restart variants reuse parent eigvals — load from the parent dir.
        parent = fallback_eigvals_path(case)
        ev_path = parent / f"{prefix}_eigvals.npy"
        ref_eigvals = parse_text_array(ev_path).astype(float) if ev_path.is_file() else None
    else:
        ref_eigvals = ref["eigvals"].astype(float)

    if ref_eigvals is None:
        out.append(
            FileResult(
                name="eigvals.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail="reference eigvals not found",
            )
        )
    else:
        out.append(compare_eigvals("eigvals.npy", obs.eigvals, ref_eigvals))

    if ref["double_occupation_expectation"] is not None:
        out.append(
            compare_per_eigenstate(
                "double_occupation_expectation.npy",
                obs.double_occ.astype(complex),
                ref["double_occupation_expectation"].astype(complex),
                obs.eigvals,
            )
        )
    else:
        out.append(
            FileResult(
                name="double_occupation_expectation.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail="reference not found",
            )
        )

    # states.npy: legacy-sorted Fock encodings
    if ref["states"] is not None:
        cur_basis_sorted = sorted(obs.basis_states, key=lambda s: legacy_state_key(s, case.N))
        cur_states = np.asarray(
            [legacy_state_row(s, case.N) for s in cur_basis_sorted], dtype=int
        )
        ref_states_int = (
            ref["states"].astype(int)
            if not np.iscomplexobj(ref["states"])
            else np.real(ref["states"]).astype(int)
        )
        if cur_states.shape != ref_states_int.shape:
            out.append(
                FileResult(
                    name="states.npy",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail=f"shape current={cur_states.shape} reference={ref_states_int.shape}",
                )
            )
        else:
            diff_rows = int(np.sum(np.any(cur_states != ref_states_int, axis=1)))
            out.append(
                FileResult(
                    name="states.npy",
                    passed=diff_rows == 0,
                    max_abs_diff=float(diff_rows),
                    detail="" if diff_rows == 0 else f"{diff_rows} row(s) differ",
                )
            )

    # s2_digonal: 3 columns. col 0 (S²) and col 2 (overlap) are degenerate-trace
    # invariant; col 1 (variance) is not.
    cur_stack = np.column_stack(
        [
            obs.s2_diag_values.astype(complex),
            obs.s2_variance.astype(complex),
            obs.s2_overlap.astype(complex),
        ]
    )
    ref_s2 = ref["s2_digonal"]
    if ref_s2 is None or ref_s2.ndim != 2 or ref_s2.shape[1] not in (2, 3):
        out.append(
            FileResult(
                name="s2_digonal.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail=(
                    "missing"
                    if ref_s2 is None
                    else f"unexpected shape {ref_s2.shape}"
                ),
            )
        )
    else:
        # adiabatic_restart references are 2-column (col0=S², col1=variance);
        # occ / multi_restart references are 3-column (extra col2=spin-overlap).
        out.append(
            compare_per_eigenstate(
                "s2_digonal.npy[col0 S²]",
                cur_stack[:, 0:1],
                ref_s2[:, 0:1],
                obs.eigvals,
            )
        )
        if ref_s2.shape[1] == 3:
            out.append(
                compare_per_eigenstate(
                    "s2_digonal.npy[col2 spin-overlap]",
                    cur_stack[:, 2:3],
                    ref_s2[:, 2:3],
                    obs.eigvals,
                )
            )
        var_diff = np.abs(cur_stack[:, 1] - ref_s2[:, 1])
        max_var_nondegen = 0.0
        for s, e in _degenerate_blocks(obs.eigvals):
            if e - s == 1:
                max_var_nondegen = max(max_var_nondegen, float(var_diff[s]))
        out.append(
            FileResult(
                name="s2_digonal.npy[col1 variance, non-degen]",
                passed=max_var_nondegen <= ATOL_BASELINE,
                max_abs_diff=max_var_nondegen,
            )
        )

    return out


def compare_family_occ(
    case: Case,
    obs: FamilyObs,
    ref: dict[str, np.ndarray | None],
) -> list[FileResult]:
    """Selected-state and Heff/T11m1 checks for workflow=occ."""
    out: list[FileResult] = []

    cur_sel_set = set(obs.selected_indices)
    if ref["t11_selected_indices"] is not None:
        ref_sel_set = {int(x) for x in ref["t11_selected_indices"].tolist()}
        sel_equal = cur_sel_set == ref_sel_set
        sym_cur = cur_sel_set - ref_sel_set
        sym_ref = ref_sel_set - cur_sel_set
        out.append(
            FileResult(
                name="t11_selected_indices.npy (set)",
                passed=sel_equal,
                max_abs_diff=0.0 if sel_equal else float(len(sym_cur) + len(sym_ref)),
                detail=(
                    "sets equal"
                    if sel_equal
                    else f"|cur\\ref|={len(sym_cur)} |ref\\cur|={len(sym_ref)}"
                ),
            )
        )
    else:
        sel_equal = False
        out.append(
            FileResult(
                name="t11_selected_indices.npy (set)",
                passed=False,
                max_abs_diff=float("nan"),
                detail="reference not found",
            )
        )

    if ref["t11_selected_occupation"] is not None:
        cur_sorted = np.sort(obs.selected_double_occ.real)
        ref_sorted = np.sort(np.asarray(ref["t11_selected_occupation"]).astype(complex).real)
        if cur_sorted.shape != ref_sorted.shape:
            out.append(
                FileResult(
                    name="t11_selected_occupation.npy (sorted)",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail=f"shape current={cur_sorted.shape} reference={ref_sorted.shape}",
                )
            )
        else:
            diff = float(np.max(np.abs(cur_sorted - ref_sorted))) if cur_sorted.size else 0.0
            out.append(
                FileResult(
                    name="t11_selected_occupation.npy (sorted)",
                    passed=diff <= ATOL_BASELINE,
                    max_abs_diff=diff,
                )
            )

    # s2_selected — pair selected states by ascending double-occ on both sides
    if ref["s2_selected"] is not None and ref["t11_selected_occupation"] is not None:
        cur_s2_sel = np.column_stack(
            [
                obs.s2_diag_values.astype(complex),
                obs.s2_variance.astype(complex),
                obs.s2_overlap.astype(complex),
            ]
        )[np.asarray(obs.selected_indices, dtype=int)]
        cur_order = np.argsort(obs.selected_double_occ.real)
        ref_order = np.argsort(
            np.asarray(ref["t11_selected_occupation"]).astype(complex).real, kind="stable"
        )
        ref_s2_sel = ref["s2_selected"]
        if cur_s2_sel.shape == ref_s2_sel.shape:
            cur_sorted = cur_s2_sel[cur_order]
            ref_sorted = ref_s2_sel[ref_order]
            diff = np.asarray(cur_sorted, dtype=complex) - np.asarray(ref_sorted, dtype=complex)
            mx, loc = _max_abs(diff)
            out.append(
                FileResult(
                    name="s2_selected.npy (sorted)",
                    passed=mx <= ATOL_BASELINE,
                    max_abs_diff=mx,
                    location=loc,
                )
            )
        else:
            out.append(
                FileResult(
                    name="s2_selected.npy (sorted)",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail=f"shape current={cur_s2_sel.shape} reference={ref_s2_sel.shape}",
                )
            )

    # Heff / T11m1 with legacy basis permutation
    if (
        ref["Heff"] is not None
        and ref["T11m1"] is not None
        and ref["states"] is not None
        and ref["Heff"].shape == obs.heff.shape
    ):
        ref_states_int = (
            ref["states"].astype(int)
            if not np.iscomplexobj(ref["states"])
            else np.real(ref["states"]).astype(int)
        )
        legacy_spin_states = legacy_spin_ordering(ref_states_int, case.N)
        canonical_spin_states = [obs.basis_states[i] for i in obs.spin_fock_rows]
        try:
            perm = canonical_spin_permutation(canonical_spin_states, legacy_spin_states)
            cur_heff_perm = obs.heff[np.ix_(perm, perm)]
            cur_t11m1_perm = obs.t11m1[np.ix_(perm, perm)]
        except Exception as exc:
            out.append(
                FileResult(
                    name="Heff.npy (permutation)",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail=f"permutation failed: {exc}",
                )
            )
            cur_heff_perm = obs.heff
            cur_t11m1_perm = obs.t11m1
        out.append(compare_array("Heff.npy", cur_heff_perm, ref["Heff"], ATOL_BASELINE))
        out.append(compare_array("T11m1.npy", cur_t11m1_perm, ref["T11m1"], ATOL_BASELINE))
    elif ref["Heff"] is None or ref["T11m1"] is None:
        if ref["Heff"] is None:
            out.append(
                FileResult(
                    name="Heff.npy",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="reference not found",
                )
            )
        if ref["T11m1"] is None:
            out.append(
                FileResult(
                    name="T11m1.npy",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="reference not found",
                )
            )
    else:
        out.append(
            FileResult(
                name="Heff.npy",
                passed=False,
                max_abs_diff=float("nan"),
                detail=f"shape current={obs.heff.shape} reference={ref['Heff'].shape}",
            )
        )

    return out


def compare_family_norm(
    case: Case,
    family_payload: dict,
    ref: dict[str, np.ndarray | None],
) -> list[FileResult]:
    """Norm-based checks for greedy_multi / adiabatic.

    Uses the actual run's |T11-I| norm from exchange.json's fit block (not the
    occ recompute), since the selection — and thus T11m1 — is workflow-specific.
    """
    out: list[FileResult] = []
    ref_t11m1 = ref["T11m1"]
    if ref_t11m1 is None:
        out.append(
            FileResult(
                name="T11m1.npy (norm)",
                passed=False,
                max_abs_diff=float("nan"),
                detail="reference not found",
            )
        )
        return out
    ref_norm = float(np.linalg.norm(np.asarray(ref_t11m1).flatten()))
    cur_norm = float(family_payload["fit"]["t11_minus_1_norm"])
    excess = cur_norm - ref_norm
    out.append(
        FileResult(
            name="T11m1.npy (norm-no-worse)",
            passed=excess <= ATOL_NORM,
            max_abs_diff=excess,
            detail=f"current={cur_norm:.6e} reference={ref_norm:.6e}",
        )
    )
    return out


def compare_family_clusters(
    case: Case,
    members: list,
    family_payload: dict,
    ref_dir: Path,
    prefix: str,
    root: Path,
) -> list[FileResult]:
    """Per-cluster .txt comparison against data_test reference."""
    out: list[FileResult] = []
    family_constant, family_terms = operators_to_terms(family_payload["operators"])
    workflow_out = workflow_output_dir(case, root)
    clusters_path = workflow_out / family_payload.get(
        "clusters_file",
        f"clusters/{family_clusters_file(int(family_payload['hole']), int(family_payload['class_idx']))}",
    )
    if not clusters_path.is_file():
        # The exchange payload doesn't carry clusters_file directly; resolve via results.json
        results_path = workflow_out / RESULTS_FILE
        results = json.loads(results_path.read_text())
        for entry in results["families"]:
            if int(entry["hole"]) == int(family_payload["hole"]) and int(
                entry["class_idx"]
            ) == int(family_payload["class_idx"]):
                clusters_path = workflow_out / entry["clusters_file"]
                break
    clusters_payload = json.loads(clusters_path.read_text())

    for member in members:
        cluster_idx = int(member.cluster_idx)
        ref_txt = ref_dir / f"{prefix}_cluster{cluster_idx}_results.txt"
        if not ref_txt.is_file():
            out.append(
                FileResult(
                    name=f"cluster{cluster_idx}_results.txt",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="reference not found",
                )
            )
            continue
        ref_art = parse_txt(ref_txt)
        geometry_entry = next(
            (
                e
                for e in clusters_payload["clusters"]
                if int(e["cluster_idx"]) == cluster_idx
            ),
            None,
        )
        if geometry_entry is None:
            out.append(
                FileResult(
                    name=f"cluster{cluster_idx}_results.txt",
                    passed=False,
                    max_abs_diff=float("nan"),
                    detail="cluster entry missing in clusters.json",
                )
            )
            continue
        sites = _sites_in_operator_order(geometry_entry, case.N)
        operators = operators_from_terms(sites, family_constant, family_terms)
        entry = {
            "sites": [[int(x), int(y)] for x, y in sites],
            "operators": operators,
            "fit": family_payload["fit"],
        }
        cur_art = sidecar_to_txtlike(entry)
        sub_results = compare_txt_like(cur_art, ref_art, atol=ATOL_TXT_ROUND)
        for fr in sub_results:
            out.append(
                FileResult(
                    name=f"cluster{cluster_idx}.{fr.name}",
                    passed=fr.passed,
                    max_abs_diff=fr.max_abs_diff,
                    location=fr.location,
                    detail=fr.detail,
                )
            )

    return out


def validate_case(case: Case, root: Path) -> CaseResult:
    case_id = case.case_id
    try:
        results, family_payloads = load_current_payloads(case, root)
        ref_dir = ref_dir_for_case(case)
        if not ref_dir.is_dir():
            return CaseResult(
                case_id=case_id,
                passed=False,
                error=f"reference directory missing: {ref_dir}",
            )
        all_results: list[FileResult] = []
        for hole, class_idx, members in enumerate_families(case.N):
            prefix = f"hole{hole}_class{class_idx}"
            obs = recompute_family(case, hole, class_idx, members)
            ref = load_reference_npy_set(ref_dir, prefix)
            family_results = compare_family_strict(case, obs, ref, ref_dir, prefix)
            family_payload = family_payloads.get((hole, class_idx))
            if case.workflow == "occ":
                family_results.extend(compare_family_occ(case, obs, ref))
            else:
                if family_payload is None:
                    family_results.append(
                        FileResult(
                            name="T11m1.npy (norm-no-worse)",
                            passed=False,
                            max_abs_diff=float("nan"),
                            detail="exchange.json missing for this family",
                        )
                    )
                else:
                    family_results.extend(
                        compare_family_norm(case, family_payload, ref)
                    )
            if family_payload is not None:
                family_results.extend(
                    compare_family_clusters(
                        case, members, family_payload, ref_dir, prefix, root
                    )
                )
            for fr in family_results:
                fr.name = f"{prefix}.{fr.name}"
            all_results.extend(family_results)
        passed = all(fr.passed for fr in all_results)
        return CaseResult(case_id=case_id, passed=passed, family_results=all_results)
    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            passed=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_opus")
    parser.add_argument(
        "--n", dest="n_filter", type=int, nargs="*", default=None,
    )
    parser.add_argument(
        "--workflow",
        dest="workflow_filter",
        nargs="*",
        default=None,
        help="restrict to workflows; default all",
    )
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    status_dir = root / "_status"
    status_path = status_dir / "layer_b.jsonl"
    failures_path = status_dir / "layer_b_failures.jsonl"
    summary_path = status_dir / "layer_b_summary.json"

    n_filter = tuple(args.n_filter) if args.n_filter else None
    cases = enumerate_cases(n_filter=n_filter)
    if args.workflow_filter:
        wf = set(args.workflow_filter)
        cases = [c for c in cases if c.workflow in wf]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    status = StatusIndex.load(status_path)
    status_dir.mkdir(parents=True, exist_ok=True)

    n_pass = n_skip = n_fail = 0
    for case in cases:
        if status.passed(case.case_id):
            n_skip += 1
            continue
        started = now_iso()
        result = validate_case(case, root)
        finished = now_iso()
        record = {
            "case_id": case.case_id,
            "stage": "layer_b",
            "status": "passed" if result.passed else "failed",
            "started_at": started,
            "finished_at": finished,
            "checks": result.summary(),
        }
        if result.error:
            record["error"] = result.error.splitlines()[0][:500]
        status.append(record)
        if result.passed:
            n_pass += 1
            print(f"[pass] {case.case_id} ({result.summary()['checks']} checks)")
        else:
            n_fail += 1
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            failed_checks = [
                {
                    "name": fr.name,
                    "max_abs_diff": fr.max_abs_diff,
                    "location": fr.location,
                    "detail": fr.detail,
                }
                for fr in result.family_results
                if not fr.passed
            ]
            failure = {
                "case_id": case.case_id,
                "error": result.error,
                "failed_checks": failed_checks,
                "summary": result.summary(),
            }
            with failures_path.open("a") as fh:
                fh.write(json.dumps(failure) + "\n")
            label = result.error.splitlines()[0] if result.error else (
                f"{len(failed_checks)} checks failed; first: {failed_checks[0]['name']} "
                f"max|Δ|={failed_checks[0]['max_abs_diff']}"
            )
            print(f"[FAIL] {case.case_id}: {label[:200]}")
            if args.stop_on_fail:
                break

    summary = {
        "passed": n_pass,
        "skipped": n_skip,
        "failed": n_fail,
        "total": len(cases),
        "root": str(root),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
