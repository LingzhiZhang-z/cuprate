#!/usr/bin/env python3
"""Regression diagnostic plotter (gnuplot batch).

Plot families:
  block : per (block, family), 1x4 multiplot [eigvals, <D>, ||T11-I||, relerr]
          1x5 if workflow is adiabatic (adds adjacent-T overlap)
  total : per combo, 1x2 multiplot [relerr per family, T11 total per family]
          1x3 if workflow is adiabatic (adds overlap total per family)

Aggregations used in `total`:
  - T11 total = sqrt(sum_b ||T11_b - I||^2)  (Frobenius decomposes on block-diag)
  - Overlap total = sum_b sum|M_b|^2 / sum_b dim_b  (weighted average by spin_dim)

Display convention (each axis sorted by its own value at each T):
  - All eigvals       : sorted ascending by eigenvalue (eigh's natural order)
  - All <D>           : sorted ascending by <D>      (column i = i-th lowest <D>)
  - Selected eigvals  : sorted ascending by selected eigenvalue
  - Selected <D>      : sorted ascending by selected <D>
  Cross-panel red-line identity is NOT preserved between eigvals_sel and dc_sel.

Y-range conventions:
  - relerr, overlap : fixed [0, 1] (linear, not log)
  - eigvals, <D>, T11 : autoscale (must include all bands)

Outputs to <root>/_plots/. Pipeline:
  1. Python walks the regression matrix and emits .dat files under _plots/data/
  2. Python writes a single all_plots.gp template referencing those .dat files
  3. gnuplot is invoked once to render all PNGs

Usage:
  python scripts/plot_regression.py --root results_opus
  python scripts/plot_regression.py --plot block
  python scripts/plot_regression.py --n 4 5 --workflow occ --plot total
  python scripts/plot_regression.py --no-render        # data + .gp only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (SRC_DIR, SCRIPTS_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from cuprate.clusters import ClusterSets  # noqa: E402
from cuprate.paths import (  # noqa: E402
    STAGE_MAIN,
    family_exchange_file,
    family_projection_file,
    main_data_dir,
    mode_token,
    workflow_dir,
)
from cuprate.states import count_double_occ  # noqa: E402


U_VALUE = 1.0
T_VALUES: list[float] = [round(0.02 * i, 2) for i in range(1, 31)]


# ---------------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Combo:
    N: int
    mode: str
    twoSz: int | None
    twoS: int | None
    scope: str
    workflow: str

    @property
    def mode_tok(self) -> str:
        return mode_token(self.mode, twoSz=self.twoSz, twoS=self.twoS, scope=self.scope)

    @property
    def label(self) -> str:
        return f"N{self.N}_{self.mode_tok}_{self.workflow}"


def enumerate_combos(
    n_filter: tuple[int, ...] | None,
    mode_filter: tuple[str, ...] | None,
    workflow_filter: tuple[str, ...] | None,
) -> Iterable[Combo]:
    n_default = (2, 3, 4, 5, 6)
    n_values = n_filter or n_default
    modes = mode_filter or ("full", "Sz", "SzS2", "SzS2eta2")
    wfs = workflow_filter or ("occ", "greedy_multi", "adiabatic")
    for N in n_values:
        for mode in modes:
            if mode == "full":
                for wf in wfs:
                    yield Combo(N, "full", None, None, "nonnegative", wf)
            elif mode == "Sz":
                if N <= 5:
                    twoSz = 0 if N % 2 == 0 else 1
                    for wf in wfs:
                        yield Combo(N, "Sz", twoSz, None, "nonnegative", wf)
            elif mode == "SzS2":
                if N >= 2 and N <= 6:
                    for wf in wfs:
                        yield Combo(N, "SzS2", None, None, "nonnegative", wf)
            elif mode == "SzS2eta2":
                if N >= 2 and N <= 6:
                    for wf in wfs:
                        yield Combo(N, "SzS2eta2", None, None, "nonnegative", wf)


def list_families(N: int) -> list[tuple[int, int, list]]:
    by: dict[tuple[int, int], list] = {}
    for cluster in ClusterSets(N).generate().clusters:
        by.setdefault((int(cluster.hole), int(cluster.class_idx)), []).append(cluster)
    return [(h, c, by[(h, c)]) for (h, c) in sorted(by)]


def case_workflow_dir(combo: Combo, T: float, root: Path) -> Path:
    return workflow_dir(
        root, STAGE_MAIN, combo.N, combo.N, U_VALUE, T,
        combo.mode, combo.workflow,
        twoSz=combo.twoSz, twoS=combo.twoS, scope=combo.scope,
    )


def discover_blocks(combo: Combo, root: Path) -> list[str]:
    """Block labels present in this combo's projection.npz files."""
    families = list_families(combo.N)
    if not families:
        return []
    for T in T_VALUES:
        wd = case_workflow_dir(combo, T, root)
        for hole, class_idx, _ in families:
            npz_path = wd / "artifacts" / family_projection_file(hole, class_idx)
            if npz_path.is_file():
                with np.load(npz_path) as data:
                    return [str(lbl) for lbl in data["block_labels"].tolist()]
    return []


def parse_label_txt(path: Path) -> tuple[list[int], np.ndarray]:
    lines = path.read_text().splitlines()
    header_tokens = lines[0].split()
    # Old header: N nelec twoSz twoS n_states n_eigvals          (6 tokens)
    # New header: N nelec twoSz twoS eta n_states n_eigvals      (7 tokens)
    if len(header_tokens) == 6:
        n_states = int(header_tokens[4])
        n_eigvals = int(header_tokens[5])
    elif len(header_tokens) == 7:
        n_states = int(header_tokens[5])
        n_eigvals = int(header_tokens[6])
    else:
        raise ValueError(
            f"label.txt has {len(header_tokens)} header tokens; expected 6 or 7"
        )
    body = " ".join(lines[1:]).split()
    basis_states = [int(x) for x in body[:n_states]]
    eigvals = np.array([float(x) for x in body[n_states:n_states + n_eigvals]])
    return basis_states, eigvals


def _fmt(v: float) -> str:
    if v != v or np.isinf(v):
        return "NaN"
    return f"{v:.10e}"


def _data_dir_name(mode: str) -> str:
    if mode == "full":
        return "DATA"
    if mode == "Sz":
        return "DATA_twoSz"
    if mode == "SzS2eta2":
        return "DATA_twoSz_twoS_eta_0"
    return "DATA_twoSz_twoS"


def _cluster_cache_dir(combo: Combo, T: float, root: Path, hole: int, class_idx: int, cluster_idx: int) -> Path:
    cache_root = main_data_dir(root, combo.N, combo.N, U_VALUE, T, combo.mode)
    return cache_root / f"hole{hole}_class{class_idx}_idx{cluster_idx}"


def _parse_block_label(label: str) -> tuple[int | None, int | None]:
    if label == "full":
        return (None, None)
    parts = label.split("_")
    twoSz = twoS = None
    i = 0
    while i < len(parts):
        if parts[i] == "twoSz" and i + 1 < len(parts):
            tok = parts[i + 1]
            twoSz = -int(tok[1:]) if tok.startswith("n") else int(tok)
            i += 2
        elif parts[i] == "twoS" and i + 1 < len(parts):
            tok = parts[i + 1]
            twoS = -int(tok[1:]) if tok.startswith("n") else int(tok)
            i += 2
        else:
            i += 1
    return (twoSz, twoS)


# ---------------------------------------------------------------------------
# Data emit: T11
# ---------------------------------------------------------------------------

def emit_t11_data(combo: Combo, root: Path, data_dir: Path) -> list[tuple[str, int, int]]:
    """Per (block, family): write `t11_<combo>_h<h>c<c>_<block>.dat` with [t/U, t11_norm].

    Returns [(block_label, hole, class_idx)] for the gnuplot generator.
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return []
    out: list[tuple[str, int, int]] = []
    for block_label in blocks:
        for hole, class_idx, _ in families:
            rows: list[tuple[float, float]] = []
            for T in T_VALUES:
                wd = case_workflow_dir(combo, T, root)
                ex_path = wd / "exchanges" / family_exchange_file(hole, class_idx)
                if not ex_path.is_file():
                    rows.append((T / U_VALUE, float("nan")))
                    continue
                ex = json.loads(ex_path.read_text())
                norm = float("nan")
                for blk in ex.get("projection", {}).get("blocks", []):
                    if str(blk.get("block")) == block_label:
                        norm = float(blk["t11_minus_1_norm"])
                        break
                rows.append((T / U_VALUE, norm))
            out_path = data_dir / f"t11_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with out_path.open("w") as fh:
                fh.write("# t/U  t11_minus_1_norm\n")
                for t_val, norm in rows:
                    fh.write(f"{_fmt(t_val)} {_fmt(norm)}\n")
            out.append((block_label, hole, class_idx))
    return out


# ---------------------------------------------------------------------------
# Data emit: relative_error (block-independent)
# ---------------------------------------------------------------------------

def emit_relerr_data(combo: Combo, root: Path, data_dir: Path) -> list[tuple[int, int]]:
    """Per combo: write `relerr_<combo>.dat` with [t/U, err per family].

    relative_error is a workflow-level fit metric (per family, NOT per block).
    Returns the family list [(hole, class_idx)] used as columns.
    """
    families = list_families(combo.N)
    if not families:
        return []
    fam_pairs = [(h, c) for h, c, _ in families]
    rows: list[list[float]] = []
    have_any = False
    for T in T_VALUES:
        row = [T / U_VALUE]
        wd = case_workflow_dir(combo, T, root)
        for hole, class_idx in fam_pairs:
            ex_path = wd / "exchanges" / family_exchange_file(hole, class_idx)
            if ex_path.is_file():
                ex = json.loads(ex_path.read_text())
                err = float(ex["fit"]["relative_error"])
                have_any = True
            else:
                err = float("nan")
            row.append(err)
        rows.append(row)
    if not have_any:
        return []
    out_path = data_dir / f"relerr_{combo.label}.dat"
    with out_path.open("w") as fh:
        fh.write("# t/U")
        for hole, class_idx in fam_pairs:
            fh.write(f"  err_h{hole}c{class_idx}")
        fh.write("\n")
        for row in rows:
            fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return fam_pairs


# ---------------------------------------------------------------------------
# Data emit: T11 aggregated across blocks (combo-level)
# ---------------------------------------------------------------------------

def emit_t11_total_data(combo: Combo, root: Path, data_dir: Path) -> list[tuple[int, int]]:
    """Per combo: write `t11_total_<combo>.dat` with sqrt(sum of squares) per family.

    For each (T, family), aggregates per-block t11_minus_1_norm:
        t11_total[T, f] = sqrt(sum_b ||T11_b - I||^2)
    Frobenius norm of (T11 - I) on the block-diagonal full matrix decomposes this way.
    Returns the family list as columns.
    """
    families = list_families(combo.N)
    if not families:
        return []
    fam_pairs = [(h, c) for h, c, _ in families]
    rows: list[list[float]] = []
    have_any = False
    for T in T_VALUES:
        row = [T / U_VALUE]
        wd = case_workflow_dir(combo, T, root)
        for hole, class_idx in fam_pairs:
            ex_path = wd / "exchanges" / family_exchange_file(hole, class_idx)
            if not ex_path.is_file():
                row.append(float("nan"))
                continue
            ex = json.loads(ex_path.read_text())
            sum_sq = 0.0
            count = 0
            for blk in ex.get("projection", {}).get("blocks", []):
                norm = blk.get("t11_minus_1_norm")
                if norm is None:
                    continue
                v = float(norm)
                sum_sq += v * v
                count += 1
            if count > 0:
                row.append(float(np.sqrt(sum_sq)))
                have_any = True
            else:
                row.append(float("nan"))
        rows.append(row)
    if not have_any:
        return []
    out_path = data_dir / f"t11_total_{combo.label}.dat"
    with out_path.open("w") as fh:
        fh.write("# t/U")
        for h, c in fam_pairs:
            fh.write(f"  t11_total_h{h}c{c}")
        fh.write("\n")
        for row in rows:
            fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return fam_pairs


# ---------------------------------------------------------------------------
# Data emit: per-block triple [eigvals, <D>] (T11 reuses emit_t11_data output)
# ---------------------------------------------------------------------------

def emit_block_triple_data(
    combo: Combo, root: Path, data_dir: Path
) -> list[tuple[str, int, int, int, int]]:
    """Per (block, family): write 4 dat files for the eigvals/<D> subplots.

    Each axis sorts ascending by its own value at each T (size order):
      eigvals_all_<...>.dat : t/U + ev_0 ... ev_{n-1}     (sorted by eigenvalue at each T)
      eigvals_sel_<...>.dat : t/U + sev_0 ... sev_{k-1}   (sorted by selected eigenvalue)
      dc_all_<...>.dat      : t/U + dc_0 ... dc_{n-1}     (sorted by <D> at each T)
      dc_sel_<...>.dat      : t/U + sdc_0 ... sdc_{k-1}   (sorted by selected <D>)

    Note: red line i in eigvals_sel and dc_sel may correspond to different
    eigenstates because the two panels use different sort keys.

    Returns [(block_label, hole, class_idx, n_dim, k_sel)].
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return []

    out: list[tuple[str, int, int, int, int]] = []
    for block_idx, block_label in enumerate(blocks):
        for hole, class_idx, members in families:
            cluster_idx = int(members[0].cluster_idx)

            ev_all_rows: list[list[float]] = []
            ev_sel_rows: list[list[float]] = []
            dc_all_rows: list[list[float]] = []
            dc_sel_rows: list[list[float]] = []
            n_dim: int | None = None
            k_sel: int | None = None
            D_diag: np.ndarray | None = None

            for T in T_VALUES:
                cache = _cluster_cache_dir(combo, T, root, hole, class_idx, cluster_idx)
                label_path = cache / f"{block_label}_label.txt"
                data_path = cache / f"{block_label}_data.npz"
                if not label_path.is_file() or not data_path.is_file():
                    continue
                try:
                    basis_states, eigvals = parse_label_txt(label_path)
                except Exception:
                    continue
                with np.load(data_path) as cached:
                    if "eigvecs" not in cached.files:
                        continue
                    eigvecs = np.asarray(cached["eigvecs"])
                    if "basis_transform" in cached.files:
                        bt = np.asarray(cached["basis_transform"])
                        eigvecs_fock = bt @ eigvecs
                    else:
                        eigvecs_fock = eigvecs

                if D_diag is None or len(D_diag) != len(basis_states):
                    D_diag = np.array(
                        [count_double_occ(s, combo.N) for s in basis_states], dtype=float
                    )
                if n_dim is None:
                    n_dim = len(eigvals)

                # <psi_i|D|psi_i> = sum_k |psi_i(k)|^2 * D(k,k)
                dc = np.sum(np.abs(eigvecs_fock) ** 2 * D_diag[:, None], axis=0).real

                wd = case_workflow_dir(combo, T, root)
                npz_path = wd / "artifacts" / family_projection_file(hole, class_idx)
                selected = np.array([], dtype=int)
                if npz_path.is_file():
                    with np.load(npz_path) as proj:
                        sel_key = f"block_{block_idx}_selected_indices"
                        if sel_key in proj.files:
                            selected = np.asarray(proj[sel_key], dtype=int)
                if k_sel is None:
                    k_sel = int(selected.size)

                # All eigvals, energy order, padded to n_dim
                ev_all_pad = list(eigvals) + [float("nan")] * max(0, n_dim - len(eigvals))
                ev_all_rows.append([T / U_VALUE] + ev_all_pad[:n_dim])

                # All <D>, sorted by <D>, padded
                dc_sorted = np.sort(dc)
                dc_all_pad = list(dc_sorted) + [float("nan")] * max(0, n_dim - len(dc_sorted))
                dc_all_rows.append([T / U_VALUE] + dc_all_pad[:n_dim])

                # Selected: each axis sorts ascending by its own value (size order).
                # NOTE: this means red line i in eigvals_sel and dc_sel may NOT
                # refer to the same selected eigenstate (they use different sort keys).
                if selected.size:
                    sev = sorted(float(v) for v in eigvals[selected])
                    sdc = sorted(float(v) for v in dc[selected])
                else:
                    sev = []
                    sdc = []
                # Pad to k_sel
                sev_pad = sev + [float("nan")] * max(0, k_sel - len(sev))
                sdc_pad = sdc + [float("nan")] * max(0, k_sel - len(sdc))
                ev_sel_rows.append([T / U_VALUE] + sev_pad[:k_sel])
                dc_sel_rows.append([T / U_VALUE] + sdc_pad[:k_sel])

            if not ev_all_rows or n_dim is None:
                continue
            if k_sel is None:
                k_sel = 0

            ev_all_path = data_dir / f"eigvals_all_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with ev_all_path.open("w") as fh:
                fh.write("# t/U" + "".join(f"  ev_{i}" for i in range(n_dim)) + "\n")
                for row in ev_all_rows:
                    fh.write(" ".join(_fmt(v) for v in row) + "\n")

            dc_all_path = data_dir / f"dc_all_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with dc_all_path.open("w") as fh:
                fh.write("# t/U" + "".join(f"  dc_{i}" for i in range(n_dim)) + "\n")
                for row in dc_all_rows:
                    fh.write(" ".join(_fmt(v) for v in row) + "\n")

            if k_sel > 0:
                ev_sel_path = data_dir / f"eigvals_sel_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                with ev_sel_path.open("w") as fh:
                    fh.write("# t/U" + "".join(f"  sev_{i}" for i in range(k_sel)) + "\n")
                    for row in ev_sel_rows:
                        fh.write(" ".join(_fmt(v) for v in row) + "\n")

                dc_sel_path = data_dir / f"dc_sel_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                with dc_sel_path.open("w") as fh:
                    fh.write("# t/U" + "".join(f"  sdc_{i}" for i in range(k_sel)) + "\n")
                    for row in dc_sel_rows:
                        fh.write(" ".join(_fmt(v) for v in row) + "\n")

            out.append((block_label, hole, class_idx, n_dim, k_sel))
    return out


# ---------------------------------------------------------------------------
# Data emit: coupling (kept from previous version)
# ---------------------------------------------------------------------------

def emit_coupling_data(
    combo: Combo, root: Path, data_dir: Path
) -> tuple[list[tuple[str, list[tuple[int, int]]]], list[tuple[int, int]]]:
    """Adjacent-T overlap of selected eigvecs.

      coupling_<combo>_<block>.dat       : per (block) all families as columns
      coupling_total_<combo>.dat         : per combo, per family aggregated:
          ovl_total[T, f] = sum_b sum|M_b|^2 / sum_b dim_b
        i.e. weighted average over blocks by their selected dimension.

    Returns (per_block_list, fam_pairs_for_total).
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return [], []
    fam_pairs = [(h, c) for h, c, _ in families]
    n_T = len(T_VALUES)
    total_acc: list[list[tuple[float, int]]] = [
        [(0.0, 0) for _ in fam_pairs] for _ in range(n_T)
    ]

    out: list[tuple[str, list[tuple[int, int]]]] = []
    for block_idx, block_label in enumerate(blocks):
        rows = []
        prev = {fam: None for fam in fam_pairs}
        for T_idx, T in enumerate(T_VALUES):
            wd = case_workflow_dir(combo, T, root)
            row_overlaps = {fam: float("nan") for fam in fam_pairs}
            for fam_idx, (hole, class_idx) in enumerate(fam_pairs):
                npz_path = wd / "artifacts" / family_projection_file(hole, class_idx)
                if not npz_path.is_file():
                    prev[(hole, class_idx)] = None
                    continue
                with np.load(npz_path) as data:
                    if f"block_{block_idx}_eigvecs_fock" not in data.files:
                        prev[(hole, class_idx)] = None
                        continue
                    eigvecs = np.asarray(data[f"block_{block_idx}_eigvecs_fock"])
                    selected = np.asarray(data[f"block_{block_idx}_selected_indices"], dtype=int)
                cur = eigvecs[:, selected] if selected.size else eigvecs[:, :0]
                p = prev[(hole, class_idx)]
                if p is not None and p.shape == cur.shape and cur.size > 0:
                    M = p.conj().T @ cur
                    sum_M2 = float(np.sum(np.abs(M) ** 2))
                    dim_b = cur.shape[1]
                    row_overlaps[(hole, class_idx)] = sum_M2 / dim_b
                    prev_M2, prev_dim = total_acc[T_idx][fam_idx]
                    total_acc[T_idx][fam_idx] = (prev_M2 + sum_M2, prev_dim + dim_b)
                prev[(hole, class_idx)] = cur
            if T > T_VALUES[0] + 1e-9:
                row = [T / U_VALUE]
                for hole, class_idx in fam_pairs:
                    row.append(row_overlaps[(hole, class_idx)])
                rows.append(row)
        out_path = data_dir / f"coupling_{combo.label}_{block_label}.dat"
        with out_path.open("w") as fh:
            fh.write("# t/U")
            for h, c in fam_pairs:
                fh.write(f"  overlap_h{h}c{c}")
            fh.write("\n")
            for row in rows:
                fh.write(" ".join(_fmt(v) for v in row) + "\n")
        out.append((block_label, fam_pairs))

    total_path = data_dir / f"coupling_total_{combo.label}.dat"
    with total_path.open("w") as fh:
        fh.write("# t/U")
        for h, c in fam_pairs:
            fh.write(f"  overlap_total_h{h}c{c}")
        fh.write("\n")
        for T_idx, T in enumerate(T_VALUES):
            if T <= T_VALUES[0] + 1e-9:
                continue
            row = [T / U_VALUE]
            any_data = False
            for fam_idx in range(len(fam_pairs)):
                sum_M2, total_dim = total_acc[T_idx][fam_idx]
                if total_dim > 0:
                    row.append(sum_M2 / total_dim)
                    any_data = True
                else:
                    row.append(float("nan"))
            if any_data:
                fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return out, fam_pairs


# ---------------------------------------------------------------------------
# Data emit: per-block relative_error (from JSON or recomputed locally)
# ---------------------------------------------------------------------------

def _recompute_block_relerr(
    combo: Combo,
    T: float,
    root: Path,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    exchange_json: dict,
    block_label: str,
) -> float:
    """Recompute per-block relative error for legacy JSONs missing the field.

    Reuses Block.downfold and Block._spin_operators on cached _data.npz + _label.txt.
    Bond list and fit coefficient vector are reconstructed from JSON.operators
    (so this does not depend on cluster.generate_bonds matching at runtime).
    """
    from cuprate.manifold import Block  # imported lazily; only legacy fallback uses it

    operators = exchange_json.get("operators")
    if not operators:
        return float("nan")

    bonds: list[tuple[int, ...]] = []
    const = operators.get("constant_term") or {}
    x_list: list[complex] = [
        complex(float(const.get("real", 0.0)), float(const.get("imag", 0.0)))
    ]
    for group in operators.get("groups", []):
        for term in group.get("terms", []):
            sites = term.get("sites")
            if sites is None:
                continue
            bonds.append(tuple(int(s) for s in sites))
            coef = term.get("coefficient") or {}
            x_list.append(
                complex(float(coef.get("real", 0.0)), float(coef.get("imag", 0.0)))
            )
    x = np.array(x_list, dtype=complex)

    selected: list[int] | None = None
    twoSz = twoS = eta = None
    for blk in exchange_json.get("projection", {}).get("blocks", []):
        if str(blk.get("block")) == block_label:
            selected = [int(idx) for idx in blk.get("selected_indices", [])]
            twoSz = blk.get("twoSz")
            twoS = blk.get("twoS")
            eta = blk.get("eta")
            break
    if selected is None:
        return float("nan")

    cache = _cluster_cache_dir(combo, T, root, hole, class_idx, cluster_idx)
    if not (cache / f"{block_label}_data.npz").exists():
        return float("nan")
    try:
        block = Block.load(cache, twoSz=twoSz, twoS=twoS, eta=eta)
    except Exception:
        return float("nan")
    if block.eigvecs is None or block.eigvals is None:
        return float("nan")

    try:
        heff, _ = block.downfold(selected)
        A_i = block._spin_operators(bonds)
    except Exception:
        return float("nan")

    b_i = heff.flatten()
    if A_i.shape[1] != x.size or A_i.shape[0] != b_i.size:
        return float("nan")
    residual_i = A_i @ x - b_i
    b_i_norm = float(np.linalg.norm(b_i))
    if b_i_norm <= 0:
        return 0.0
    return float(np.linalg.norm(residual_i)) / b_i_norm


def emit_per_block_relerr_data(
    combo: Combo, root: Path, data_dir: Path
) -> list[tuple[str, int, int]]:
    """Per (block, family): write `relerr_block_<combo>_h<h>c<c>_<block>.dat`.

    Reads `projection.blocks[i].relative_error` from JSON when present.
    Otherwise falls back to local recomputation from cached Block + JSON operators.
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return []
    out: list[tuple[str, int, int]] = []
    for block_label in blocks:
        for hole, class_idx, members in families:
            cluster_idx = int(members[0].cluster_idx)
            rows: list[tuple[float, float]] = []
            for T in T_VALUES:
                wd = case_workflow_dir(combo, T, root)
                ex_path = wd / "exchanges" / family_exchange_file(hole, class_idx)
                if not ex_path.is_file():
                    rows.append((T / U_VALUE, float("nan")))
                    continue
                ex = json.loads(ex_path.read_text())
                relerr = float("nan")
                for blk in ex.get("projection", {}).get("blocks", []):
                    if str(blk.get("block")) == block_label:
                        if "relative_error" in blk:
                            relerr = float(blk["relative_error"])
                        else:
                            relerr = _recompute_block_relerr(
                                combo, T, root, hole, class_idx, cluster_idx, ex, block_label
                            )
                        break
                rows.append((T / U_VALUE, relerr))
            out_path = data_dir / f"relerr_block_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with out_path.open("w") as fh:
                fh.write("# t/U  relerr_block\n")
                for t_val, relerr in rows:
                    fh.write(f"{_fmt(t_val)} {_fmt(relerr)}\n")
            out.append((block_label, hole, class_idx))
    return out


# ---------------------------------------------------------------------------
# Gnuplot templates
# ---------------------------------------------------------------------------

GNUPLOT_PREAMBLE = r"""set datafile missing 'NaN'
set grid back lc rgb '#cccccc'
"""


def render_gp_block_panels(
    combo: Combo,
    block_label: str,
    hole: int,
    class_idx: int,
    family_idx: int,
    n_dim: int,
    k_sel: int,
    ev_all: Path,
    dc_all: Path,
    t11: Path,
    relerr_block_dat: Path,
    coupling_dat: Path | None,
    ev_sel: Path | None,
    dc_sel: Path | None,
    png: Path,
    is_adiabatic: bool,
) -> str:
    """1x4 (or 1x5 for adiabatic) multiplot per (block, family).

    Panels: eigvals | <D> | ||T11 - I|| | per-block relerr | (overlap if adiabatic).
    `relerr_block_dat` is a single-column dat for THIS (block, family).
    `coupling_dat` (adiabatic only) has per-family columns; `family_idx` selects.
    """
    title = (
        f"N={combo.N}  {combo.mode_tok}  {combo.workflow}  "
        f"h{hole}c{class_idx}  block={block_label}"
    )
    family_col = 2 + family_idx  # used by coupling_dat lookup only

    plot1_all = (
        f"for [i=2:{n_dim + 1}] '{ev_all}' using 1:i with linespoints "
        "lw 0.4 ps 0.3 lc rgb '#888888' notitle"
    )
    plot1_sel = (
        f", for [i=2:{k_sel + 1}] '{ev_sel}' using 1:i with linespoints "
        "lw 1.4 pt 7 ps 0.6 lc rgb '#d62728' notitle"
    ) if (ev_sel is not None and k_sel > 0) else ""
    plot1 = f"plot {plot1_all}{plot1_sel}"

    plot2_all = (
        f"for [i=2:{n_dim + 1}] '{dc_all}' using 1:i with linespoints "
        "lw 0.4 ps 0.3 lc rgb '#888888' notitle"
    )
    plot2_sel = (
        f", for [i=2:{k_sel + 1}] '{dc_sel}' using 1:i with linespoints "
        "lw 1.4 pt 7 ps 0.6 lc rgb '#d62728' notitle"
    ) if (dc_sel is not None and k_sel > 0) else ""
    plot2 = f"plot {plot2_all}{plot2_sel}"

    plot3 = (
        f"plot '{t11}' using 1:2 with linespoints lw 1.4 pt 7 ps 0.6 "
        "lc rgb '#1f77b4' notitle"
    )
    plot4 = (
        f"plot '{relerr_block_dat}' using 1:2 with linespoints "
        "lw 1.4 pt 7 ps 0.6 lc rgb '#2ca02c' notitle"
    )

    if is_adiabatic and coupling_dat is not None:
        n_panels = 5
        plot5 = (
            f"plot '{coupling_dat}' using 1:{family_col} with linespoints "
            "lw 1.4 pt 7 ps 0.6 lc rgb '#9467bd' notitle"
        )
        panel5 = f"""
set ylabel 'overlap'
set yrange [0:1]
{plot5}
"""
    else:
        n_panels = 4
        panel5 = ""

    width = 500 * n_panels

    return f"""
set terminal pngcairo size {width},500 enhanced font ',11'
set output '{png}'
set multiplot layout 1,{n_panels} title "{title}" font ',12' noenhanced
set xlabel 't/U'
set xrange [0:0.62]
unset logscale y
set format y '%g'

set ylabel 'eigenvalue'
set autoscale y
{plot1}

set ylabel '<D>'
set autoscale y
{plot2}

set ylabel '||T11 - I||'
set autoscale y
{plot3}

set ylabel 'relative error' noenhanced
set yrange [0:1]
{plot4}
{panel5}
unset multiplot
unset output
"""


def render_gp_combo_total(
    combo: Combo,
    families: list[tuple[int, int]],
    relerr_dat: Path,
    t11_total_dat: Path,
    coupling_total_dat: Path | None,
    png: Path,
    is_adiabatic: bool,
) -> str:
    """1x2 (or 1x3 for adiabatic) multiplot per combo.

    Panels: relerr per family | T11_total per family | (overlap_total per family if adiabatic).
    """
    title = f"N={combo.N}  {combo.mode_tok}  {combo.workflow}  TOTAL"

    relerr_specs = ", ".join(
        f"'{relerr_dat}' using 1:{2 + i} with linespoints lw 1.4 pt 7 ps 0.6 title 'h{h}c{c}'"
        for i, (h, c) in enumerate(families)
    )
    t11_specs = ", ".join(
        f"'{t11_total_dat}' using 1:{2 + i} with linespoints lw 1.4 pt 7 ps 0.6 title 'h{h}c{c}'"
        for i, (h, c) in enumerate(families)
    )

    if is_adiabatic and coupling_total_dat is not None:
        n_panels = 3
        ovl_specs = ", ".join(
            f"'{coupling_total_dat}' using 1:{2 + i} with linespoints lw 1.4 pt 7 ps 0.6 title 'h{h}c{c}'"
            for i, (h, c) in enumerate(families)
        )
        panel3 = f"""
set ylabel 'overlap (sum|M|^2 / sum dim)' noenhanced
set yrange [0:1]
plot {ovl_specs}
"""
    else:
        n_panels = 2
        panel3 = ""

    width = 600 * n_panels

    return f"""
set terminal pngcairo size {width},500 enhanced font ',11'
set output '{png}'
set multiplot layout 1,{n_panels} title "{title}" font ',12' noenhanced
set xlabel 't/U'
set xrange [0:0.62]
unset logscale y
set format y '%g'
set key outside right top vertical Left reverse samplen 2 spacing 1.0

set ylabel 'relative error' noenhanced
set yrange [0:1]
plot {relerr_specs}

set ylabel '||T11 - I||_F (sum-of-sq)' noenhanced
set autoscale y
plot {t11_specs}
{panel3}
unset multiplot
unset output
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_opus")
    parser.add_argument(
        "--plot", nargs="*",
        default=["block", "total"],
        choices=["block", "total"],
    )
    parser.add_argument("--n", dest="n_filter", type=int, nargs="*", default=None)
    parser.add_argument("--mode", dest="mode_filter", nargs="*", default=None,
                        choices=["full", "Sz", "SzS2", "SzS2eta2"])
    parser.add_argument("--workflow", dest="workflow_filter", nargs="*", default=None,
                        choices=["occ", "greedy_multi", "adiabatic"])
    parser.add_argument("--no-render", action="store_true",
                        help="emit data + .gp but don't invoke gnuplot")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    plots_root = root / "_plots"
    data_dir = plots_root / "data"
    block_dir = plots_root / "block"
    total_dir = plots_root / "total"
    for d in (data_dir, block_dir, total_dir):
        d.mkdir(parents=True, exist_ok=True)

    combos = list(enumerate_combos(
        tuple(args.n_filter) if args.n_filter else None,
        tuple(args.mode_filter) if args.mode_filter else None,
        tuple(args.workflow_filter) if args.workflow_filter else None,
    ))

    gp_blocks: list[str] = []
    t_extract_start = time.perf_counter()
    n_block = n_total = 0

    want_block = "block" in args.plot
    want_total = "total" in args.plot

    for combo in combos:
        is_adiabatic = combo.workflow == "adiabatic"
        families = list_families(combo.N)
        fam_pairs = [(h, c) for h, c, _ in families]
        fam_idx_map = {(h, c): i for i, (h, c) in enumerate(fam_pairs)}

        # Common data sources for both block and total plots.
        relerr_fam_pairs = emit_relerr_data(combo, root, data_dir)
        relerr_dat = data_dir / f"relerr_{combo.label}.dat"
        # Coupling: only needed if any plot uses overlap (adiabatic only).
        coupling_block_list: list[tuple[str, list[tuple[int, int]]]] = []
        coupling_total_fam_pairs: list[tuple[int, int]] = []
        if is_adiabatic and (want_block or want_total):
            coupling_block_list, coupling_total_fam_pairs = emit_coupling_data(
                combo, root, data_dir
            )

        if want_block:
            t11_index = {
                (b, h, c): data_dir / f"t11_{combo.label}_h{h}c{c}_{b}.dat"
                for (b, h, c) in emit_t11_data(combo, root, data_dir)
            }
            relerr_block_index = {
                (b, h, c): data_dir / f"relerr_block_{combo.label}_h{h}c{c}_{b}.dat"
                for (b, h, c) in emit_per_block_relerr_data(combo, root, data_dir)
            }
            for (block_label, hole, class_idx, n_dim, k_sel) in emit_block_triple_data(
                combo, root, data_dir
            ):
                ev_all = data_dir / f"eigvals_all_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                dc_all = data_dir / f"dc_all_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                ev_sel = data_dir / f"eigvals_sel_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                dc_sel = data_dir / f"dc_sel_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                t11 = t11_index.get((block_label, hole, class_idx))
                relerr_block = relerr_block_index.get((block_label, hole, class_idx))
                if t11 is None or not t11.exists():
                    continue
                if relerr_block is None or not relerr_block.exists():
                    continue
                if (hole, class_idx) not in fam_idx_map:
                    continue
                family_idx = fam_idx_map[(hole, class_idx)]
                coupling_dat = (
                    data_dir / f"coupling_{combo.label}_{block_label}.dat"
                    if is_adiabatic else None
                )
                if coupling_dat is not None and not coupling_dat.exists():
                    coupling_dat = None
                png = block_dir / f"{combo.label}_h{hole}c{class_idx}_{block_label}.png"
                gp_blocks.append("\n# === block panels ===\n")
                gp_blocks.append(GNUPLOT_PREAMBLE)
                gp_blocks.append(render_gp_block_panels(
                    combo, block_label, hole, class_idx, family_idx,
                    n_dim, k_sel,
                    ev_all, dc_all, t11, relerr_block, coupling_dat,
                    ev_sel if k_sel > 0 else None,
                    dc_sel if k_sel > 0 else None,
                    png, is_adiabatic,
                ))
                n_block += 1

        if want_total:
            t11_total_fam_pairs = emit_t11_total_data(combo, root, data_dir)
            t11_total_dat = data_dir / f"t11_total_{combo.label}.dat"
            coupling_total_dat = (
                data_dir / f"coupling_total_{combo.label}.dat"
                if is_adiabatic else None
            )
            if not relerr_dat.exists() or not t11_total_dat.exists() or not fam_pairs:
                continue
            png = total_dir / f"{combo.label}.png"
            gp_blocks.append("\n# === combo total ===\n")
            gp_blocks.append(GNUPLOT_PREAMBLE)
            gp_blocks.append(render_gp_combo_total(
                combo, fam_pairs, relerr_dat, t11_total_dat, coupling_total_dat,
                png, is_adiabatic,
            ))
            n_total += 1

    t_extract = time.perf_counter() - t_extract_start

    gp_path = plots_root / "all_plots.gp"
    gp_path.write_text("".join(gp_blocks))

    print(f"Data extracted in {t_extract:.1f}s")
    print(f"Plots queued: block={n_block}, total={n_total}")
    print(f"Gnuplot script: {gp_path}")

    if args.no_render:
        return 0

    t_render_start = time.perf_counter()
    log_path = plots_root / "render.log"
    with log_path.open("w") as log:
        proc = subprocess.run(
            ["gnuplot", str(gp_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=plots_root.parent,
        )
    t_render = time.perf_counter() - t_render_start
    print(f"Gnuplot rendered in {t_render:.1f}s -> log: {log_path}")
    if proc.returncode != 0:
        print(f"gnuplot returned {proc.returncode}; check {log_path}")
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
