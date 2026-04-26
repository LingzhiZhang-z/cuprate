#!/usr/bin/env python3
"""Regression diagnostic plotter (gnuplot batch).

Generates three plot families from regression outputs:
  norm     : ||T11-I|| and relative_error vs t/U per (N, mode, workflow, block)
  coupling : adjacent-T selected-subspace overlap vs t/U per (N, mode, workflow, block)
  occ      : eigenvalue spectrum + double-occupation per (N, mode, workflow, hole, class, block)

Outputs to <root>/_plots/. Pipeline:
  1. Python walks the regression matrix and emits .dat files under _plots/data/
  2. Python writes a single all_plots.gp template referencing those .dat files
  3. gnuplot is invoked once to render all PNGs

Usage:
  python scripts/plot_regression.py --root results_opus
  python scripts/plot_regression.py --plot norm coupling
  python scripts/plot_regression.py --n 4 5 --workflow occ --plot norm
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
    RESULTS_FILE,
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
                # Half-filling: even N -> twoSz=0, odd N -> twoSz=1
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
    tokens = path.read_text().split()
    n_states = int(tokens[5])
    n_eigvals = int(tokens[6])
    basis_states = [int(x) for x in tokens[7:7 + n_states]]
    idx = 7 + n_states
    eigvals = np.array([float(x) for x in tokens[idx:idx + n_eigvals]])
    return basis_states, eigvals


def _fmt(v: float) -> str:
    if v != v or np.isinf(v):
        return "NaN"
    return f"{v:.10e}"


# ---------------------------------------------------------------------------
# Plot A: ||T11-I|| and relative_error
# ---------------------------------------------------------------------------

def emit_norm_data(combo: Combo, root: Path, data_dir: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    """For each block, write a .dat with one row per T and 2 columns per family.
    Returns [(block_label, families_list)] for the gnuplot generator.
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return []
    out: list[tuple[str, list[tuple[int, int]]]] = []
    for block_label in blocks:
        rows = []
        for T in T_VALUES:
            row = [T / U_VALUE]
            wd = case_workflow_dir(combo, T, root)
            for hole, class_idx, _ in families:
                norm = err = float("nan")
                ex_path = wd / "exchanges" / family_exchange_file(hole, class_idx)
                if ex_path.is_file():
                    ex = json.loads(ex_path.read_text())
                    err = float(ex["fit"]["relative_error"])
                    for blk in ex.get("projection", {}).get("blocks", []):
                        if str(blk.get("block")) == block_label:
                            norm = float(blk["t11_minus_1_norm"])
                            break
                row.extend([norm, err])
            rows.append(row)
        out_path = data_dir / f"norm_{combo.label}_{block_label}.dat"
        with out_path.open("w") as fh:
            fh.write("# t/U")
            for hole, class_idx, _ in families:
                fh.write(f"  norm_h{hole}c{class_idx}  err_h{hole}c{class_idx}")
            fh.write("\n")
            for row in rows:
                fh.write(" ".join(_fmt(v) for v in row) + "\n")
        fam_pairs = [(h, c) for h, c, _ in families]
        out.append((block_label, fam_pairs))
    return out


def emit_coupling_data(combo: Combo, root: Path, data_dir: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    """For each block, compute adjacent-T overlap of selected eigenvecs per family.
    Output: row per T pair (midpoint t/U, overlap per family).
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return []
    out = []
    for block_idx, block_label in enumerate(blocks):
        rows = []
        prev = {fam: None for fam in [(h, c) for h, c, _ in families]}
        for T in T_VALUES:
            wd = case_workflow_dir(combo, T, root)
            row_overlaps = {fam: float("nan") for fam in prev}
            for hole, class_idx, _ in families:
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
                    row_overlaps[(hole, class_idx)] = float(np.sum(np.abs(M) ** 2)) / cur.shape[1]
                prev[(hole, class_idx)] = cur
            # Skip the first T (no previous to compare)
            if T > T_VALUES[0] + 1e-9:
                row = [T / U_VALUE]
                for hole, class_idx, _ in families:
                    row.append(row_overlaps[(hole, class_idx)])
                rows.append(row)
        out_path = data_dir / f"coupling_{combo.label}_{block_label}.dat"
        with out_path.open("w") as fh:
            fh.write("# t/U")
            for hole, class_idx, _ in families:
                fh.write(f"  overlap_h{hole}c{class_idx}")
            fh.write("\n")
            for row in rows:
                fh.write(" ".join(_fmt(v) for v in row) + "\n")
        fam_pairs = [(h, c) for h, c, _ in families]
        out.append((block_label, fam_pairs))
    return out


# ---------------------------------------------------------------------------
# Plot C: eigvals + double-occ
# ---------------------------------------------------------------------------

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


def emit_occ_data(combo: Combo, root: Path, data_dir: Path) -> list[tuple[str, int, int]]:
    """For each (block, hole, class), write eigvals.dat and dc.dat.

    eigvals.dat: row T  ev_0  ev_1  ...  ev_{n-1}   (n = block dimension)
    dc.dat:       row T  dc_0  dc_1  ...  flag_0  flag_1  ...

    Returns [(block_label, hole, class_idx)] tuples for gnuplot generator.
    """
    families = list_families(combo.N)
    blocks = discover_blocks(combo, root)
    if not blocks or not families:
        return []
    out: list[tuple[str, int, int]] = []
    for block_idx, block_label in enumerate(blocks):
        for hole, class_idx, members in families:
            cluster_idx = int(members[0].cluster_idx)
            ev_rows = []
            dc_rows = []
            n_dim = None
            D_diag = None  # cache: depends only on basis_states, fixed across T
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
                if D_diag is None:
                    D_diag = np.array(
                        [count_double_occ(s, combo.N) for s in basis_states], dtype=float
                    )
                    n_dim = len(eigvals)
                # ⟨ψ_i|D|ψ_i⟩ = Σ_k |ψ_i(k)|² · D(k,k); D is diagonal so O(N²) instead of O(N³)
                dc = np.sum(np.abs(eigvecs_fock) ** 2 * D_diag[:, None], axis=0).real
                # Locate selected_indices for this block from projection.npz
                wd = case_workflow_dir(combo, T, root)
                npz_path = wd / "artifacts" / family_projection_file(hole, class_idx)
                selected = np.array([], dtype=int)
                if npz_path.is_file():
                    with np.load(npz_path) as proj:
                        sel_key = f"block_{block_idx}_selected_indices"
                        if sel_key in proj.files:
                            selected = np.asarray(proj[sel_key], dtype=int)
                ev_padded = list(eigvals) + [float("nan")] * max(0, n_dim - len(eigvals))
                dc_padded = list(dc) + [float("nan")] * max(0, n_dim - len(dc))
                flags = [0.0] * len(ev_padded)
                for s in selected:
                    if 0 <= int(s) < len(flags):
                        flags[int(s)] = 1.0
                ev_rows.append([T / U_VALUE] + ev_padded[:n_dim])
                dc_rows.append([T / U_VALUE] + dc_padded[:n_dim] + flags[:n_dim])
            if not ev_rows or n_dim is None:
                continue
            ev_path = data_dir / f"eigvals_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with ev_path.open("w") as fh:
                fh.write("# t/U" + "".join(f"  ev_{i}" for i in range(n_dim)) + "\n")
                for row in ev_rows:
                    fh.write(" ".join(_fmt(v) for v in row) + "\n")
            dc_path = data_dir / f"dc_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with dc_path.open("w") as fh:
                fh.write(
                    "# t/U"
                    + "".join(f"  dc_{i}" for i in range(n_dim))
                    + "".join(f"  flag_{i}" for i in range(n_dim))
                    + "\n"
                )
                for row in dc_rows:
                    fh.write(" ".join(_fmt(v) for v in row) + "\n")
            dc_selected_path = data_dir / f"dc_selected_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
            with dc_selected_path.open("w") as fh:
                fh.write("# t/U  selected_dc\n")
                for row in dc_rows:
                    t_value = row[0]
                    dc_values = row[1:1 + n_dim]
                    flags = row[1 + n_dim:1 + 2 * n_dim]
                    for dc_value, flag in zip(dc_values, flags):
                        if flag == 1.0:
                            fh.write(f"{_fmt(t_value)} {_fmt(dc_value)}\n")
            out.append((block_label, hole, class_idx))
    return out


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
# Gnuplot template
# ---------------------------------------------------------------------------

GNUPLOT_HEADER = r"""set datafile missing 'NaN'
set terminal pngcairo size 1200,500 enhanced font ',11'
set key outside right top vertical Left reverse samplen 2 spacing 1.0
set grid back lc rgb '#cccccc'
"""

GNUPLOT_HEADER_SQUARE = r"""set datafile missing 'NaN'
set terminal pngcairo size 800,600 enhanced font ',11'
set key outside right top vertical Left reverse samplen 2 spacing 1.0
set grid back lc rgb '#cccccc'
"""


def render_gp_norm(combo: Combo, block_label: str, families: list[tuple[int, int]],
                   data_path: Path, out_path: Path) -> str:
    """Two-panel: left ||T11-I|| linear, right relative_error log."""
    fam_count = len(families)
    plot_norm_specs = []
    plot_err_specs = []
    for idx, (h, c) in enumerate(families):
        col_norm = 2 + 2 * idx
        col_err = 3 + 2 * idx
        title = f"h{h}c{c}"
        plot_norm_specs.append(f"'{data_path}' using 1:{col_norm} with linespoints lw 1.4 pt 7 ps 0.6 title '{title}'")
        plot_err_specs.append(f"'{data_path}' using 1:{col_err} with linespoints lw 1.4 pt 7 ps 0.6 title '{title}'")
    title = f"||T11-I|| and rel.err  N={combo.N} {combo.mode_tok} {combo.workflow} block={block_label}"
    return f"""
set output '{out_path}'
set multiplot layout 1,2 title "{title}" font ',12'
set xlabel 't/U'
set xrange [0:0.62]
unset logscale y
set ylabel '||T11-I||'
plot {", ".join(plot_norm_specs)}
set ylabel 'relative_error'
set logscale y
set format y '%.0e'
plot {", ".join(plot_err_specs)}
unset logscale y
set format y '%g'
unset multiplot
unset output
"""


def render_gp_coupling(combo: Combo, block_label: str, families: list[tuple[int, int]],
                       data_path: Path, out_path: Path) -> str:
    plot_specs = []
    for idx, (h, c) in enumerate(families):
        col = 2 + idx
        plot_specs.append(f"'{data_path}' using 1:{col} with linespoints lw 1.4 pt 7 ps 0.6 title 'h{h}c{c}'")
    title = f"Adjacent-T subspace overlap  N={combo.N} {combo.mode_tok} {combo.workflow} block={block_label}"
    return f"""
set output '{out_path}'
set title "{title}" font ',12'
set xlabel 't/U'
set ylabel '|<sel(T-{0xCE94:c}) | sel(T)>|² / spin_dim'
set yrange [0:1.1]
set xrange [0:0.62]
unset logscale y
set arrow 1 from graph 0,first 1 to graph 1,first 1 nohead lc rgb '#aaaaaa' dt 2
plot {", ".join(plot_specs)}
unset arrow 1
unset title
unset output
"""


def render_gp_eigvals(combo: Combo, block_label: str, hole: int, class_idx: int, n_dim: int,
                      data_path: Path, out_path: Path) -> str:
    title = f"Eigenvalue spectrum  N={combo.N} {combo.mode_tok} {combo.workflow} h{hole}c{class_idx} block={block_label}"
    if n_dim == 1:
        plot_line = f"plot '{data_path}' using 1:2 with lines lw 1.4 lc rgb '#1f77b4' notitle"
    else:
        plot_line = (
            f"plot for [i=2:{n_dim + 1}] '{data_path}' using 1:i:(real(i-2)/{n_dim - 1}) "
            "with lines lw 0.8 lc palette notitle"
        )
    return f"""
set output '{out_path}'
set title "{title}" font ',12'
set xlabel 't/U'
set ylabel 'eigenvalue'
set xrange [0:0.62]
unset logscale y
set palette defined (0 '#1f77b4', 0.5 '#2ca02c', 1 '#d62728')
unset colorbox
unset key
{plot_line}
set key outside right top vertical Left reverse samplen 2 spacing 1.0
unset title
unset output
"""


def render_gp_dc(
    combo: Combo,
    block_label: str,
    hole: int,
    class_idx: int,
    n_dim: int,
    data_path: Path,
    selected_path: Path | None,
    out_path: Path,
) -> str:
    title = f"Double-occ ⟨D⟩  N={combo.N} {combo.mode_tok} {combo.workflow} h{hole}c{class_idx} block={block_label}\\n(red=selected, gray=all)"
    plot_specs = [
        (
            f"for [i=2:{n_dim + 1}] '{data_path}' using 1:i "
            "with points pt 1 ps 0.5 lc rgb '#cccccc' notitle"
        )
    ]
    if selected_path is not None:
        plot_specs.append(
            f"'{selected_path}' using 1:2 with points pt 7 ps 0.7 lc rgb '#d62728' notitle"
        )
    return f"""
set output '{out_path}'
set title "{title}" font ',12'
set xlabel 't/U'
set ylabel '<D>'
set xrange [0:0.62]
unset logscale y
unset key
plot {", ".join(plot_specs)}
set key outside right top vertical Left reverse samplen 2 spacing 1.0
unset title
unset output
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_opus")
    parser.add_argument("--plot", nargs="*", default=["norm", "coupling", "occ"],
                        choices=["norm", "coupling", "occ"])
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
    norm_dir = plots_root / "norm"
    coupling_dir = plots_root / "coupling"
    occ_dir = plots_root / "occ"
    for d in (data_dir, norm_dir, coupling_dir, occ_dir):
        d.mkdir(parents=True, exist_ok=True)

    combos = list(enumerate_combos(
        tuple(args.n_filter) if args.n_filter else None,
        tuple(args.mode_filter) if args.mode_filter else None,
        tuple(args.workflow_filter) if args.workflow_filter else None,
    ))

    gp_blocks: list[str] = [GNUPLOT_HEADER]

    t_extract_start = time.perf_counter()
    n_norm = n_coupling = n_eigvals = n_dc = 0

    for combo in combos:
        if "norm" in args.plot:
            for block_label, fams in emit_norm_data(combo, root, data_dir):
                dat = data_dir / f"norm_{combo.label}_{block_label}.dat"
                png = norm_dir / f"{combo.label}_{block_label}.png"
                gp_blocks.append(render_gp_norm(combo, block_label, fams, dat, png))
                n_norm += 1
        if "coupling" in args.plot:
            for block_label, fams in emit_coupling_data(combo, root, data_dir):
                dat = data_dir / f"coupling_{combo.label}_{block_label}.dat"
                png = coupling_dir / f"{combo.label}_{block_label}.png"
                gp_blocks.append("\n# === single-panel ===\n")
                gp_blocks.append(GNUPLOT_HEADER_SQUARE)
                gp_blocks.append(render_gp_coupling(combo, block_label, fams, dat, png))
                gp_blocks.append("\n# === resume two-panel default ===\n")
                gp_blocks.append(GNUPLOT_HEADER)
                n_coupling += 1
        if "occ" in args.plot:
            for block_label, hole, class_idx in emit_occ_data(combo, root, data_dir):
                ev_dat = data_dir / f"eigvals_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                dc_dat = data_dir / f"dc_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                dc_selected_dat = data_dir / f"dc_selected_{combo.label}_h{hole}c{class_idx}_{block_label}.dat"
                ev_png = occ_dir / f"eigvals_{combo.label}_h{hole}c{class_idx}_{block_label}.png"
                dc_png = occ_dir / f"dc_{combo.label}_h{hole}c{class_idx}_{block_label}.png"
                # Read header to learn n_dim
                with ev_dat.open() as fh:
                    header = fh.readline()
                n_dim = header.count("ev_")
                selected_path = (
                    dc_selected_dat
                    if dc_selected_dat.exists() and len(dc_selected_dat.read_text().splitlines()) > 1
                    else None
                )
                gp_blocks.append("\n# === single-panel ===\n")
                gp_blocks.append(GNUPLOT_HEADER_SQUARE)
                gp_blocks.append(render_gp_eigvals(combo, block_label, hole, class_idx, n_dim, ev_dat, ev_png))
                gp_blocks.append(render_gp_dc(
                    combo,
                    block_label,
                    hole,
                    class_idx,
                    n_dim,
                    dc_dat,
                    selected_path,
                    dc_png,
                ))
                gp_blocks.append("\n# === resume two-panel default ===\n")
                gp_blocks.append(GNUPLOT_HEADER)
                n_eigvals += 1
                n_dc += 1

    t_extract = time.perf_counter() - t_extract_start

    gp_path = plots_root / "all_plots.gp"
    gp_path.write_text("".join(gp_blocks))

    print(f"Data extracted in {t_extract:.1f}s")
    print(f"Plots queued: norm={n_norm}, coupling={n_coupling}, eigvals={n_eigvals}, dc={n_dc}")
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
