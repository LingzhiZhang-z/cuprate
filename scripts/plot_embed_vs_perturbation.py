#!/usr/bin/env python3
"""Plot cuprate embed couplings vs analytical perturbation theory.

Output families (assuming U=1 eV so that J in code units = J/eV; values are
multiplied by 1000 for meV to match the JPS-style figures):

  absolute_meV/   J1, Jc   — y in meV, x=t/U, multi-N cuprate + PT reference.
                            Jc panel shows edge (solid) + diag (dashed) + PT.
  ratio_to_J1/    J2, J3   — y = J/J1 (cuprate), x=t/U, multi-N + PT ratio.
  comparison/     R panel  — y = J_cuprate / J_pert at fixed Nmax,
                            5 lines (J1, J2, J3, Jc edge, Jc diag).
  absolute/       remaining vectors and clusters that have no PT formula
                            (pure cuprate, multi-N, log y if needed).

Reads embed outputs from
  <root>/block_embed/N_<Nmax>_…/seed_szs2eta2_<workflow>_Nmax<n>_T<t>/
and Nmax=6's untagged
  <root>/block_embed/N_6_…/seed_szs2eta2_<workflow>_T<t>/

Outputs PNGs to <root>/_plots/embed_pert/{absolute_meV,ratio_to_J1,comparison,absolute}/.

Usage:
  python scripts/plot_embed_vs_perturbation.py --root results_szs2eta2
  python scripts/plot_embed_vs_perturbation.py --workflow occ --nmax 4 6 --no-render
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _pert_formulas as pert  # noqa: E402


U_VALUE = 1.0
T_VALUES: list[float] = [round(0.02 * i, 2) for i in range(1, 31)]
WORKFLOWS = ("occ", "greedy_multi", "adiabatic")
NMAX_VALUES = (2, 3, 4, 5, 6)
MEV_PER_U = 1000.0  # J in code energy units → meV when U=1 eV.

# Two-site vectors with named perturbation references.
J_LABELS: dict[tuple[int, int], tuple[str, callable]] = {
    (1, 0): ("J1", pert.J_NN),
    (1, 1): ("J2", pert.J_NNN),
    (2, 0): ("J3", pert.J_3rd),
}

# JPS-style palette: matches the colors in user's slide deck.
NMAX_COLORS = {
    2: "#8b4513",   # brown
    3: "#9467bd",   # purple
    4: "#d62728",   # red
    5: "#2ca02c",   # green
    6: "#1f77b4",   # blue
    7: "#17becf",   # cyan
}
PT_COLOR = "#ff7f0e"  # thick orange — the perturbation reference

# Cuprate t/U region of interest (La2CuO4 ≈ 0.10, SrCuO2 ≈ 0.155).
CUPRATES_T_LOW = 0.10
CUPRATES_T_HIGH = 0.155


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t_token(t: float) -> str:
    return f"{t:.4f}"


def _fmt(v: float) -> str:
    if v != v or math.isinf(v):
        return "NaN"
    return f"{v:.10e}"


def embed_dir(root: Path, workflow: str, nmax: int, t: float) -> Path:
    """Path to the embed seed-stage directory for (workflow, Nmax, t)."""
    if nmax == 6:
        seed_token = f"seed_szs2eta2_{workflow}_T{t_token(t)}"
    else:
        seed_token = f"seed_szs2eta2_{workflow}_Nmax{nmax}_T{t_token(t)}"
    return (
        root
        / "block_embed"
        / f"N_{nmax}_nelec_{nmax}_U_{U_VALUE:.4f}_t_{t_token(t)}"
        / seed_token
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_TWO_SITE_RE = re.compile(
    r"^\s*\d+\s+\((-?\d+),(-?\d+)\)\s+(\S+)\s+(\S+)\s*$"
)


def parse_two_site(path: Path) -> dict[tuple[int, int], float]:
    """Parse two_site.txt -> {(dx, dy): real_value}."""
    out: dict[tuple[int, int], float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        m = _TWO_SITE_RE.match(line)
        if not m:
            continue
        dx, dy = int(m.group(1)), int(m.group(2))
        out[(dx, dy)] = float(m.group(3))
    return out


_SITES_RE = re.compile(r"\((-?\d+),(-?\d+)\)")


def parse_cluster_file(path: Path) -> tuple[list[tuple[int, int]], list[tuple[str, float]]]:
    """Parse one cluster file -> (sites_list, [(pairing_label, real_value)])."""
    sites: list[tuple[int, int]] = []
    rows: list[tuple[str, float]] = []
    if not path.is_file():
        return sites, rows
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("sites:"):
            sites = [(int(a), int(b)) for a, b in _SITES_RE.findall(s)]
        elif s.startswith("(S"):
            close = s.rfind(")")
            label = s[: close + 1]
            tail = s[close + 1 :].split()
            if len(tail) >= 2:
                rows.append((label, float(tail[0])))
    return sites, rows


def is_plaquette(sites: list[tuple[int, int]]) -> bool:
    """True iff the 4 sites form a 1x1 square (2x2 plaquette)."""
    if len(sites) != 4:
        return False
    xs = [x for x, _ in sites]
    ys = [y for _, y in sites]
    nx = [x - min(xs) for x in xs]
    ny = [y - min(ys) for y in ys]
    canonical = sorted((nx[i], ny[i]) for i in range(4))
    return canonical == [(0, 0), (0, 1), (1, 0), (1, 1)]


def find_plaquette_file(cluster_dir: Path) -> Path | None:
    if not cluster_dir.is_dir():
        return None
    for path in sorted(cluster_dir.glob("N4_*.txt")):
        sites, _ = parse_cluster_file(path)
        if is_plaquette(sites):
            return path
    return None


def extract_jc_values(plaquette_file: Path) -> tuple[float, float] | None:
    """Return (jc_edge_avg, jc_diagonal) from a plaquette N4 cluster file.

    Sites must be canonical (0,0),(0,1),(1,0),(1,1) -> S0,S1,S2,S3.
    Pairings:
        (S0 S1)(S2 S3) — vertical edges
        (S0 S2)(S1 S3) — horizontal edges
        (S0 S3)(S1 S2) — diagonal cross
    """
    sites, rows = parse_cluster_file(plaquette_file)
    if not is_plaquette(sites) or len(rows) != 3:
        return None
    by_label = {label: value for label, value in rows}
    edge1 = by_label.get("(S0 S1)(S2 S3)")
    edge2 = by_label.get("(S0 S2)(S1 S3)")
    diag = by_label.get("(S0 S3)(S1 S2)")
    if edge1 is None or edge2 is None or diag is None:
        return None
    return 0.5 * (edge1 + edge2), diag


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_combo(root: Path, workflow: str, nmax: int, t: float) -> dict | None:
    """Load all coupling values for one (workflow, nmax, t). None if missing."""
    edir = embed_dir(root, workflow, nmax, t)
    two_site = parse_two_site(edir / "two_site.txt")
    if not two_site:
        return None
    plaquette = find_plaquette_file(edir / "clusters")
    jc = extract_jc_values(plaquette) if plaquette is not None else None
    return {"two_site": two_site, "jc": jc}


def build_cache(
    root: Path, workflow: str, nmaxes: tuple[int, ...]
) -> dict[tuple[int, float], dict]:
    """Pre-load all (nmax, t) combos for a workflow."""
    cache: dict[tuple[int, float], dict] = {}
    for nmax in nmaxes:
        for t in T_VALUES:
            combo = load_combo(root, workflow, nmax, t)
            if combo is not None:
                cache[(nmax, t)] = combo
    return cache


# ---------------------------------------------------------------------------
# Emit dat files
# ---------------------------------------------------------------------------

def emit_J1_absolute_meV(
    cache: dict, workflow: str, nmaxes: tuple[int, ...], data_dir: Path
) -> Path:
    dat_path = data_dir / f"abs_meV_J1_{workflow}.dat"
    with dat_path.open("w") as fh:
        fh.write("# t/U")
        for nmax in nmaxes:
            fh.write(f"  N{nmax}/meV")
        fh.write("  PT/meV\n")
        for t in T_VALUES:
            row = [t / U_VALUE]
            for nmax in nmaxes:
                two_site = cache.get((nmax, t), {}).get("two_site", {})
                val = two_site.get((1, 0))
                row.append(val * MEV_PER_U if val is not None else float("nan"))
            row.append(pert.J_NN(t, U_VALUE) * MEV_PER_U)
            fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return dat_path


def emit_Jc_absolute_meV(
    cache: dict, workflow: str, nmaxes: tuple[int, ...], data_dir: Path
) -> Path | None:
    has_any = any(cache.get((nmax, t), {}).get("jc") is not None
                  for nmax in nmaxes for t in T_VALUES)
    if not has_any:
        return None
    dat_path = data_dir / f"abs_meV_Jc_{workflow}.dat"
    with dat_path.open("w") as fh:
        fh.write("# t/U")
        for nmax in nmaxes:
            fh.write(f"  edge_N{nmax}/meV  diag_N{nmax}/meV")
        fh.write("  edge_PT/meV  diag_PT/meV\n")
        for t in T_VALUES:
            row = [t / U_VALUE]
            for nmax in nmaxes:
                jc = cache.get((nmax, t), {}).get("jc")
                if jc is None:
                    row.extend([float("nan"), float("nan")])
                else:
                    edge_val, diag_val = jc
                    row.append(edge_val * MEV_PER_U)
                    row.append(diag_val * MEV_PER_U)
            row.append(pert.Jc_edge(t, U_VALUE) * MEV_PER_U)
            row.append(pert.Jc_diagonal(t, U_VALUE) * MEV_PER_U)
            fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return dat_path


def emit_ratio_to_J1(
    cache: dict, workflow: str, label: str, vector: tuple[int, int],
    pert_fn: callable, nmaxes: tuple[int, ...], data_dir: Path,
) -> Path:
    dat_path = data_dir / f"ratio_J1_{label}_{workflow}.dat"
    with dat_path.open("w") as fh:
        fh.write("# t/U")
        for nmax in nmaxes:
            fh.write(f"  {label}/J1_N{nmax}")
        fh.write(f"  {label}/J1_PT\n")
        for t in T_VALUES:
            row = [t / U_VALUE]
            for nmax in nmaxes:
                two_site = cache.get((nmax, t), {}).get("two_site", {})
                j1 = two_site.get((1, 0))
                jx = two_site.get(vector)
                if j1 is None or jx is None or j1 == 0:
                    row.append(float("nan"))
                else:
                    row.append(jx / j1)
            j1_pt = pert.J_NN(t, U_VALUE)
            jx_pt = pert_fn(t, U_VALUE)
            row.append(jx_pt / j1_pt if j1_pt != 0 else float("nan"))
            fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return dat_path


def emit_comparison_R(
    cache: dict, workflow: str, nmax: int, data_dir: Path
) -> Path:
    """One panel comparing R = J_cuprate / J_4thPT for J1, J2, J3, Jc∥, Jc× at fixed N."""
    dat_path = data_dir / f"comparison_R_N{nmax}_{workflow}.dat"
    with dat_path.open("w") as fh:
        fh.write("# t/U  J1_R  J2_R  J3_R  Jc_edge_R  Jc_diag_R\n")
        for t in T_VALUES:
            row = [t / U_VALUE]
            two_site = cache.get((nmax, t), {}).get("two_site", {})
            jc = cache.get((nmax, t), {}).get("jc")
            for label, vector, pert_fn in (
                ("J1", (1, 0), pert.J_NN),
                ("J2", (1, 1), pert.J_NNN),
                ("J3", (2, 0), pert.J_3rd),
            ):
                val = two_site.get(vector)
                pv = pert_fn(t, U_VALUE)
                row.append(val / pv if (val is not None and pv != 0) else float("nan"))
            if jc is None:
                row.extend([float("nan"), float("nan")])
            else:
                edge_val, diag_val = jc
                edge_pt = pert.Jc_edge(t, U_VALUE)
                diag_pt = pert.Jc_diagonal(t, U_VALUE)
                row.append(edge_val / edge_pt if edge_pt != 0 else float("nan"))
                row.append(diag_val / diag_pt if diag_pt != 0 else float("nan"))
            fh.write(" ".join(_fmt(v) for v in row) + "\n")
    return dat_path


def emit_other_two_site_absolute(
    cache: dict, workflow: str, nmaxes: tuple[int, ...], data_dir: Path
) -> dict[str, Path]:
    """Absolute meV plots for two-site vectors not in J_LABELS."""
    extra_vectors: set[tuple[int, int]] = set()
    for entry in cache.values():
        for vector in entry.get("two_site", {}):
            if vector not in J_LABELS:
                extra_vectors.add(vector)
    out: dict[str, Path] = {}
    for vector in sorted(extra_vectors,
                         key=lambda v: (v[0]**2 + v[1]**2, v[0], v[1])):
        label = f"J_{vector[0]}_{vector[1]}"
        dat_path = data_dir / f"abs_meV_{label}_{workflow}.dat"
        with dat_path.open("w") as fh:
            fh.write("# t/U")
            for nmax in nmaxes:
                fh.write(f"  N{nmax}/meV")
            fh.write("\n")
            for t in T_VALUES:
                row = [t / U_VALUE]
                for nmax in nmaxes:
                    two_site = cache.get((nmax, t), {}).get("two_site", {})
                    val = two_site.get(vector)
                    row.append(val * MEV_PER_U if val is not None else float("nan"))
                fh.write(" ".join(_fmt(v) for v in row) + "\n")
        out[label] = dat_path
    return out


# ---------------------------------------------------------------------------
# Gnuplot templates
# ---------------------------------------------------------------------------

GNUPLOT_PREAMBLE = r"""set datafile missing 'NaN'
set terminal pngcairo size 1000,750 enhanced font ',14'
set border lw 1.5
set grid back lc rgb '#dddddd'
set tics scale 1.0 nomirror
set key inside left top vertical Left reverse samplen 1.5 spacing 1.1 font ',12'
"""


def cuprates_marker_gp(y_low_frac: float = 0.0) -> str:
    """Gnuplot snippet that draws a 'Cuprates' bar between the cuprate t/U range.

    y_low_frac: graph-relative position of the bar (0 = bottom, 0.05 = just above x axis).
    """
    return rf"""
set arrow 99 from {CUPRATES_T_LOW},graph({y_low_frac:.3f}) to {CUPRATES_T_HIGH},graph({y_low_frac:.3f}) nohead lw 4 lc rgb 'black' front
set label 99 "Cuprates" at {0.5*(CUPRATES_T_LOW+CUPRATES_T_HIGH)},graph({y_low_frac+0.04:.3f}) center font ',12' tc rgb 'black' front
"""


def clear_cuprates_marker_gp() -> str:
    return "unset arrow 99\nunset label 99\n"


def render_absolute_J1(
    workflow: str, nmaxes: tuple[int, ...], dat: Path, png: Path
) -> str:
    """J1 absolute panel: multi-N + thick orange PT.  Style mirrors page 9 of the slides."""
    plot_specs = []
    for idx, nmax in enumerate(nmaxes, start=2):
        color = NMAX_COLORS.get(nmax, "#000000")
        plot_specs.append(
            f"'{dat}' using 1:{idx} with linespoints "
            f"lw 1.6 pt 7 ps 0.7 lc rgb '{color}' title 'N={nmax}'"
        )
    pt_col = 2 + len(nmaxes)
    plot_specs.append(
        f"'{dat}' using 1:{pt_col} with lines "
        f"lw 3.5 lc rgb '{PT_COLOR}' title 'PT'"
    )
    title = "{/:Bold J_1}  ({/:Italic U} = 1 eV)"
    return f"""
set output '{png}'
set title "{title}" font ',16' enhanced
set label 2 "workflow: {workflow}" at graph 0.99,1.04 right font ',11' tc rgb '#555555' noenhanced front
set xlabel '{{/:Italic t/U}}' font ',15' enhanced
set ylabel '{{/:Italic J_1}}/meV' font ',15' enhanced
set xrange [0:0.22]
set autoscale y
unset logscale y
set label 1 "{{/:Italic J_1}} = 4{{/:Italic t}}^2/{{/:Italic U}} (1 - 6{{/:Italic t}}^2/{{/:Italic U}}^2)" \
    at graph 0.55,0.92 center font ',13' tc rgb '{PT_COLOR}' front
{cuprates_marker_gp(y_low_frac=0.02)}
plot {", ".join(plot_specs)}
{clear_cuprates_marker_gp()}
unset label 1
unset label 2
unset title
unset output
"""


def render_absolute_Jc(
    workflow: str, nmaxes: tuple[int, ...], dat: Path, png: Path
) -> str:
    """Jc absolute panel: edges (solid) + diag (dashed) per N + 2 PT lines (orange)."""
    plot_specs = []
    for idx, nmax in enumerate(nmaxes):
        col_edge = 2 + 2 * idx
        col_diag = 3 + 2 * idx
        color = NMAX_COLORS.get(nmax, "#000000")
        plot_specs.append(
            f"'{dat}' using 1:{col_edge} with linespoints "
            f"lw 1.6 pt 7 ps 0.7 lc rgb '{color}' title 'N={nmax}'"
        )
        plot_specs.append(
            f"'{dat}' using 1:{col_diag} with linespoints "
            f"lw 1.6 pt 5 ps 0.55 lc rgb '{color}' dt 2 notitle"
        )
    pt_edge_col = 2 + 2 * len(nmaxes)
    pt_diag_col = 3 + 2 * len(nmaxes)
    plot_specs.append(
        f"'{dat}' using 1:{pt_edge_col} with lines "
        f"lw 3.5 lc rgb '{PT_COLOR}' title 'PT'"
    )
    plot_specs.append(
        f"'{dat}' using 1:{pt_diag_col} with lines "
        f"lw 3.5 lc rgb '{PT_COLOR}' dt 2 notitle"
    )
    title = "{/:Bold J_c}  ({/:Italic U} = 1 eV) — solid: edge ‖, dashed: diagonal ×"
    return f"""
set output '{png}'
set title "{title}" font ',16' enhanced
set label 2 "workflow: {workflow}" at graph 0.99,1.04 right font ',11' tc rgb '#555555' noenhanced front
set xlabel '{{/:Italic t/U}}' font ',15' enhanced
set ylabel '{{/:Italic J_c}}/meV' font ',15' enhanced
set xrange [0:0.22]
set autoscale y
unset logscale y
set label 1 "{{/:Italic J_c}} = ±80{{/:Italic t}}^4/{{/:Italic U}}^3" \
    at graph 0.6,0.94 center font ',13' tc rgb '{PT_COLOR}' front
set arrow 50 from graph 0,first 0 to graph 1,first 0 nohead lc rgb '#888888' dt 2 lw 1
{cuprates_marker_gp(y_low_frac=0.02)}
plot {", ".join(plot_specs)}
{clear_cuprates_marker_gp()}
unset arrow 50
unset label 1
unset label 2
unset title
unset output
"""


def render_ratio_to_J1(
    workflow: str, label: str, nmaxes: tuple[int, ...], dat: Path, png: Path
) -> str:
    """J2/J1 or J3/J1 ratio panel: multi-N + PT thick orange.  Mirrors page 10."""
    plot_specs = []
    for idx, nmax in enumerate(nmaxes, start=2):
        color = NMAX_COLORS.get(nmax, "#000000")
        plot_specs.append(
            f"'{dat}' using 1:{idx} with linespoints "
            f"lw 1.6 pt 7 ps 0.7 lc rgb '{color}' title 'N={nmax}'"
        )
    pt_col = 2 + len(nmaxes)
    plot_specs.append(
        f"'{dat}' using 1:{pt_col} with lines "
        f"lw 3.5 lc rgb '{PT_COLOR}' title 'PT'"
    )
    formula_label = "{/:Italic J_2/J_1}" if label == "J2" else "{/:Italic J_3/J_1}"
    title = f"{{/:Bold {label}/J_1}}  ({{/:Italic U}} = 1 eV)"
    return f"""
set output '{png}'
set title "{title}" font ',16' enhanced
set label 2 "workflow: {workflow}" at graph 0.99,1.04 right font ',11' tc rgb '#555555' noenhanced front
set xlabel '{{/:Italic t/U}}' font ',15' enhanced
set ylabel '{formula_label}' font ',15' enhanced
set xrange [0:0.22]
set autoscale y
unset logscale y
set label 1 "{{/:Italic J_2 = J_3}} = 4{{/:Italic t}}^4/{{/:Italic U}}^3" \
    at graph 0.5,0.92 center font ',13' tc rgb '{PT_COLOR}' front
{cuprates_marker_gp(y_low_frac=0.02)}
plot {", ".join(plot_specs)}
{clear_cuprates_marker_gp()}
unset label 1
unset label 2
unset title
unset output
"""


def render_comparison_R(
    workflow: str, nmax: int, dat: Path, png: Path
) -> str:
    """Single panel: R = J_cuprate / J_4thPT for J1, J2, J3, Jc∥, Jc× at fixed Nmax."""
    series = [
        (2, "J_1", "#d62728", "lines linewidth 2 pt 7 ps 0.6", "solid"),
        (3, "J_2", "#2ca02c", "lines linewidth 2 pt 7 ps 0.6", "solid"),
        (4, "J_3", "#1f77b4", "lines linewidth 2 pt 7 ps 0.6", "solid"),
        (5, "J_{c‖}", "#9467bd", "lines linewidth 2 pt 7 ps 0.6", "solid"),
        (6, "J_{c×}", "#9467bd", "lines linewidth 2 pt 5 ps 0.55", "dashed"),
    ]
    plot_specs = []
    for col, label, color, _style, dash in series:
        dt_clause = "" if dash == "solid" else " dt 2"
        plot_specs.append(
            f"'{dat}' using 1:{col} with linespoints "
            f"lw 2.0 pt 7 ps 0.7 lc rgb '{color}'{dt_clause} title '{label}'"
        )
    title = f"{{/:Bold Comparison: cuprate / 4th-PT}}  N={nmax}  ({{/:Italic U}} = 1 eV)"
    return f"""
set output '{png}'
set title "{title}" font ',16' enhanced
set label 2 "workflow: {workflow}" at graph 0.99,1.04 right font ',11' tc rgb '#555555' noenhanced front
set xlabel '{{/:Italic t/U}}' font ',15' enhanced
set ylabel 'R = J_{{cuprate}} / J_{{4th-PT}}' font ',15' enhanced
set xrange [0:0.22]
set yrange [0:2.6]
unset logscale y
set arrow 50 from graph 0,first 1 to graph 1,first 1 nohead lc rgb '#888888' dt 2 lw 1
{cuprates_marker_gp(y_low_frac=0.02)}
plot {", ".join(plot_specs)}
{clear_cuprates_marker_gp()}
unset arrow 50
unset label 2
unset title
unset output
"""


def render_other_absolute(
    workflow: str, label: str, nmaxes: tuple[int, ...], dat: Path, png: Path
) -> str:
    """Plain absolute meV plot for vectors with no PT formula."""
    plot_specs = []
    for idx, nmax in enumerate(nmaxes, start=2):
        color = NMAX_COLORS.get(nmax, "#000000")
        plot_specs.append(
            f"'{dat}' using 1:{idx} with linespoints "
            f"lw 1.6 pt 7 ps 0.7 lc rgb '{color}' title 'N={nmax}'"
        )
    title = f"{label} (no PT)  (U = 1 eV)"
    return f"""
set output '{png}'
set title "{title}" font ',16' noenhanced
set label 2 "workflow: {workflow}" at graph 0.99,1.04 right font ',11' tc rgb '#555555' noenhanced front
set xlabel 't/U' font ',15'
set ylabel '{label} / meV' font ',15' noenhanced
set xrange [0:0.22]
set autoscale y
unset logscale y
plot {", ".join(plot_specs)}
unset label 2
unset title
unset output
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "results_szs2eta2")
    parser.add_argument("--workflow", nargs="*", default=None, choices=WORKFLOWS)
    parser.add_argument("--nmax", type=int, nargs="*", default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    workflows = tuple(args.workflow) if args.workflow else WORKFLOWS
    nmaxes = tuple(sorted(args.nmax)) if args.nmax else NMAX_VALUES
    nmax_for_comparison = max(nmaxes)

    plots_root = root / "_plots" / "embed_pert"
    data_dir = plots_root / "data"
    abs_meV_dir = plots_root / "absolute_meV"
    ratio_dir = plots_root / "ratio_to_J1"
    cmp_dir = plots_root / "comparison"
    other_dir = plots_root / "absolute"
    for d in (data_dir, abs_meV_dir, ratio_dir, cmp_dir, other_dir):
        d.mkdir(parents=True, exist_ok=True)

    gp_blocks: list[str] = [GNUPLOT_PREAMBLE]
    t_extract = time.perf_counter()
    n_meV = n_ratio = n_cmp = n_other = 0

    for workflow in workflows:
        cache = build_cache(root, workflow, nmaxes)
        if not cache:
            print(f"[skip] no cached data for workflow={workflow}")
            continue

        # Absolute meV: J1 (always) + Jc (if plaquette data exists).
        j1_dat = emit_J1_absolute_meV(cache, workflow, nmaxes, data_dir)
        png = abs_meV_dir / f"J1_{workflow}.png"
        gp_blocks.append("\n# === absolute meV J1 ===\n")
        gp_blocks.append(render_absolute_J1(workflow, nmaxes, j1_dat, png))
        n_meV += 1

        jc_dat = emit_Jc_absolute_meV(cache, workflow, nmaxes, data_dir)
        if jc_dat is not None:
            png = abs_meV_dir / f"Jc_{workflow}.png"
            gp_blocks.append("\n# === absolute meV Jc ===\n")
            gp_blocks.append(render_absolute_Jc(workflow, nmaxes, jc_dat, png))
            n_meV += 1

        # Ratio to J1: J2/J1, J3/J1.
        for label, vector, pert_fn in (
            ("J2", (1, 1), pert.J_NNN),
            ("J3", (2, 0), pert.J_3rd),
        ):
            dat = emit_ratio_to_J1(cache, workflow, label, vector, pert_fn,
                                    nmaxes, data_dir)
            png = ratio_dir / f"{label}_over_J1_{workflow}.png"
            gp_blocks.append(f"\n# === ratio {label}/J1 ===\n")
            gp_blocks.append(render_ratio_to_J1(workflow, label, nmaxes, dat, png))
            n_ratio += 1

        # Comparison panel R = J_cuprate / J_PT at fixed Nmax (largest available).
        cmp_dat = emit_comparison_R(cache, workflow, nmax_for_comparison, data_dir)
        png = cmp_dir / f"R_N{nmax_for_comparison}_{workflow}.png"
        gp_blocks.append("\n# === comparison R ===\n")
        gp_blocks.append(render_comparison_R(workflow, nmax_for_comparison, cmp_dat, png))
        n_cmp += 1

        # Other two-site vectors (long-range) — pure absolute, no PT.
        other_outputs = emit_other_two_site_absolute(cache, workflow, nmaxes, data_dir)
        for label, dat in other_outputs.items():
            if dat.stat().st_size <= 100:
                continue
            png = other_dir / f"{label}_{workflow}.png"
            gp_blocks.append(f"\n# === other absolute {label} ===\n")
            gp_blocks.append(render_other_absolute(workflow, label, nmaxes, dat, png))
            n_other += 1

    gp_path = plots_root / "all_plots.gp"
    gp_path.write_text("".join(gp_blocks))
    extract_dt = time.perf_counter() - t_extract
    print(f"Data extracted in {extract_dt:.1f}s")
    print(f"Plots queued: meV={n_meV}, ratio_to_J1={n_ratio}, comparison={n_cmp}, other={n_other}")
    print(f"Gnuplot script: {gp_path}")

    if args.no_render:
        return 0

    log_path = plots_root / "render.log"
    t_render = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(
            ["gnuplot", str(gp_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=plots_root.parent,
        )
    render_dt = time.perf_counter() - t_render
    print(f"Gnuplot rendered in {render_dt:.1f}s -> {log_path}")
    if proc.returncode != 0:
        print(f"gnuplot exit {proc.returncode}; check {log_path}")
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
