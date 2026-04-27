#!/usr/bin/env python3
"""Simple JPS-style plots from embedded SzS2eta2 couplings.

This script reads production embed outputs under:

  <root>/block_embed/N_<Nmax>_nelec_<Nmax>_U_1.0000_t_<T>/seed_szs2eta2_<workflow>_.../

and writes four PNGs similar to the result plots in the JPS slides:

  fig9_dominant.png       Jc and J1
  fig10_subdominant.png   J2/J1 and J3/J1
  fig11_pt_ratio.png      selected couplings divided by 4th-order PT
  fig12_n4_terms.png      two largest non-plaquette N4 terms divided by J2
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


U_VALUE = 1.0
WORKFLOWS = ("occ", "greedy_multi", "adiabatic")
DEFAULT_NMAXES = (2, 3, 4, 5, 6)

COLORS = {
    2: "#b85c00",
    3: "#8a2be2",
    4: "#ff0000",
    5: "#00cc22",
    6: "#0048ff",
}

TWO_SITE_RE = re.compile(r"^\s*\d+\s+\((-?\d+),(-?\d+)\)\s+(\S+)\s+(\S+)\s*$")
SITES_RE = re.compile(r"\((-?\d+),(-?\d+)\)")
PAIR_RE = re.compile(r"\(S(\d+)\s+S(\d+)\)")


def j1_pt(t: float, u: float = U_VALUE) -> float:
    return 4.0 * t**2 / u - 24.0 * t**4 / u**3


def j2_pt(t: float, u: float = U_VALUE) -> float:
    return 4.0 * t**4 / u**3


def j3_pt(t: float, u: float = U_VALUE) -> float:
    return 4.0 * t**4 / u**3


def jc_parallel_pt(t: float, u: float = U_VALUE) -> float:
    return 80.0 * t**4 / u**3


def jc_cross_pt(t: float, u: float = U_VALUE) -> float:
    return -80.0 * t**4 / u**3


def t_token(t: float) -> str:
    return f"{t:.4f}"


def t_values(tmax: float) -> list[float]:
    n_steps = int(round(tmax / 0.02))
    return [round(0.02 * i, 2) for i in range(1, n_steps + 1)]


def embed_dir(root: Path, workflow: str, nmax: int, t: float) -> Path | None:
    base = (
        root
        / "block_embed"
        / f"N_{nmax}_nelec_{nmax}_U_{U_VALUE:.4f}_t_{t_token(t)}"
    )
    candidates = [
        base / f"seed_szs2eta2_{workflow}_Nmax{nmax}_T{t_token(t)}",
        base / f"seed_szs2eta2_{workflow}_T{t_token(t)}",
    ]
    for path in candidates:
        if (path / "embed_results.json").is_file():
            return path
    return None


def parse_two_site(path: Path) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = TWO_SITE_RE.match(line)
        if match is None:
            continue
        dx, dy = int(match.group(1)), int(match.group(2))
        out[(dx, dy)] = float(match.group(3))
    return out


def parse_cluster_file(path: Path) -> tuple[list[tuple[int, int]], list[tuple[str, float]]]:
    sites: list[tuple[int, int]] = []
    terms: list[tuple[str, float]] = []
    if not path.is_file():
        return sites, terms
    for line in path.read_text().splitlines():
        text = line.strip()
        if text.startswith("sites:"):
            sites = [(int(x), int(y)) for x, y in SITES_RE.findall(text)]
        elif text.startswith("(S"):
            close = text.rfind(")")
            label = text[: close + 1]
            fields = text[close + 1 :].split()
            if fields:
                terms.append((label, float(fields[0])))
    return sites, terms


def normalized_sites(sites: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in sites)
    min_y = min(y for _, y in sites)
    return tuple(sorted((x - min_x, y - min_y) for x, y in sites))


def is_plaquette(sites: list[tuple[int, int]]) -> bool:
    return normalized_sites(sites) == ((0, 0), (0, 1), (1, 0), (1, 1))


def pair_coords(sites: list[tuple[int, int]], label: str) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    pairs = []
    for a, b in PAIR_RE.findall(label):
        p = sites[int(a)]
        q = sites[int(b)]
        pairs.append(tuple(sorted((p, q))))
    return tuple(sorted(pairs))


def n4_key(sites: list[tuple[int, int]], label: str) -> tuple:
    min_x = min(x for x, _ in sites)
    min_y = min(y for _, y in sites)

    def norm(p: tuple[int, int]) -> tuple[int, int]:
        return (p[0] - min_x, p[1] - min_y)

    norm_pairs = []
    for pair in pair_coords(sites, label):
        norm_pairs.append(tuple(sorted((norm(pair[0]), norm(pair[1])))))
    return (normalized_sites(sites), tuple(sorted(norm_pairs)))


def key_label(key: tuple) -> str:
    shape, pairs = key
    pair_text = " ".join(f"{a}-{b}" for a, b in pairs)
    return f"shape={shape}, pairs={pair_text}"


def extract_jc(cluster_dir: Path) -> tuple[float, float] | None:
    if not cluster_dir.is_dir():
        return None
    for path in sorted(cluster_dir.glob("N4_*.txt")):
        sites, rows = parse_cluster_file(path)
        if not sites or not is_plaquette(sites):
            continue
        edge_terms = []
        diag_terms = []
        for label, value in rows:
            distances = [
                abs(a[0] - b[0]) + abs(a[1] - b[1])
                for a, b in pair_coords(sites, label)
            ]
            if distances == [1, 1]:
                edge_terms.append(value)
            else:
                diag_terms.append(value)
        if edge_terms and diag_terms:
            return float(np.mean(edge_terms)), float(np.mean(diag_terms))
    return None


def extract_n4_terms(cluster_dir: Path, *, include_plaquette: bool = False) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    if not cluster_dir.is_dir():
        return out
    for path in sorted(cluster_dir.glob("N4_*.txt")):
        sites, rows = parse_cluster_file(path)
        if not sites:
            continue
        if is_plaquette(sites) and not include_plaquette:
            continue
        for label, value in rows:
            out[n4_key(sites, label)] = value
    return out


def load_combo(root: Path, workflow: str, nmax: int, t: float) -> dict | None:
    directory = embed_dir(root, workflow, nmax, t)
    if directory is None:
        return None
    two_site = parse_two_site(directory / "two_site.txt")
    if not two_site:
        return None
    return {
        "two_site": two_site,
        "jc": extract_jc(directory / "clusters"),
        "n4": extract_n4_terms(directory / "clusters"),
    }


def series(
    data: dict[tuple[int, float], dict],
    nmax: int,
    ts: list[float],
    getter,
) -> list[float]:
    out = []
    for t in ts:
        combo = data.get((nmax, t))
        out.append(float("nan") if combo is None else getter(combo, t))
    return out


def ratio(value: float | None, denom: float) -> float:
    if value is None or denom == 0.0:
        return float("nan")
    return value / denom


def setup_axis(ax, *, xlabel: str = "t/U", ylabel: str = "", xlim: tuple[float, float]):
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def plot_dominant(data, outdir: Path, workflow: str, nmaxes: tuple[int, ...], ts: list[float], tmax: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    t_dense = np.linspace(0.0, tmax, 300)

    ax = axes[0]
    for nmax in nmaxes:
        color = COLORS.get(nmax)
        jcp = series(data, nmax, ts, lambda combo, _: combo["jc"][0] * 1000.0 if combo["jc"] else float("nan"))
        jcx = series(data, nmax, ts, lambda combo, _: combo["jc"][1] * 1000.0 if combo["jc"] else float("nan"))
        if not np.all(np.isnan(jcp)):
            ax.plot(ts, jcp, "o-", color=color, label=f"N={nmax} Jc parallel")
        if not np.all(np.isnan(jcx)):
            ax.plot(ts, jcx, "o--", color=color, label=f"N={nmax} Jc cross")
    ax.plot(t_dense, [jc_parallel_pt(t) * 1000.0 for t in t_dense], color="#f5a000", linewidth=2.3, label="PT parallel")
    ax.plot(t_dense, [jc_cross_pt(t) * 1000.0 for t in t_dense], color="#f5a000", linewidth=2.3, linestyle="--", label="PT cross")
    setup_axis(ax, ylabel="Jc / meV", xlim=(0.0, tmax))
    ax.set_title("Four-site plaquette exchange")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for nmax in nmaxes:
        vals = series(data, nmax, ts, lambda combo, _: combo["two_site"].get((1, 0), float("nan")) * 1000.0)
        ax.plot(ts, vals, "o-", color=COLORS.get(nmax), label=f"N={nmax}")
    ax.plot(t_dense, [j1_pt(t) * 1000.0 for t in t_dense], color="#f5a000", linewidth=2.5, label="PT")
    setup_axis(ax, ylabel="J1 / meV", xlim=(0.0, tmax))
    ax.set_title("Nearest-neighbor exchange")
    ax.legend(fontsize=8)

    fig.suptitle(f"Dominant exchanges, workflow={workflow}")
    fig.savefig(outdir / "fig9_dominant.png", dpi=220)
    plt.close(fig)


def plot_subdominant(data, outdir: Path, workflow: str, nmaxes: tuple[int, ...], ts: list[float], tmax: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    t_dense = np.linspace(0.0, tmax, 300)
    pt_ratio = [ratio(j2_pt(t), j1_pt(t)) for t in t_dense]

    for ax, vector, title, ylabel in (
        (axes[0], (1, 1), "Second-neighbor exchange", "J2 / J1"),
        (axes[1], (2, 0), "Third-neighbor exchange", "J3 / J1"),
    ):
        for nmax in nmaxes:
            vals = series(
                data,
                nmax,
                ts,
                lambda combo, _: ratio(combo["two_site"].get(vector), combo["two_site"].get((1, 0), 0.0)),
            )
            ax.plot(ts, vals, "o-", color=COLORS.get(nmax), label=f"N={nmax}")
        ax.plot(t_dense, pt_ratio, color="#f5a000", linewidth=2.5, label="PT")
        setup_axis(ax, ylabel=ylabel, xlim=(0.0, tmax))
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(f"Subdominant two-site exchanges, workflow={workflow}")
    fig.savefig(outdir / "fig10_subdominant.png", dpi=220)
    plt.close(fig)


def plot_pt_ratio(data, outdir: Path, workflow: str, nmax: int, ts: list[float], tmax: float):
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    curves = [
        ("J1", "#ff0000", "-", lambda combo, t: ratio(combo["two_site"].get((1, 0)), j1_pt(t))),
        ("J2", "#00cc22", "-", lambda combo, t: ratio(combo["two_site"].get((1, 1)), j2_pt(t))),
        ("J3", "#0048ff", "-", lambda combo, t: ratio(combo["two_site"].get((2, 0)), j3_pt(t))),
        ("Jc cross", "#8a00cc", "--", lambda combo, t: ratio(combo["jc"][1] if combo["jc"] else None, jc_cross_pt(t))),
        ("Jc parallel", "#8a00cc", "-", lambda combo, t: ratio(combo["jc"][0] if combo["jc"] else None, jc_parallel_pt(t))),
    ]
    for label, color, linestyle, getter in curves:
        vals = series(data, nmax, ts, getter)
        ax.plot(ts, vals, marker="o", linestyle=linestyle, color=color, label=label)
    ax.axhline(1.0, color="#777777", linestyle=":", linewidth=1.2)
    setup_axis(ax, ylabel="non-perturbative / PT", xlim=(0.0, tmax))
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Comparison with 4th-order PT, workflow={workflow}, N={nmax}")
    ax.legend(fontsize=8)
    fig.savefig(outdir / "fig11_pt_ratio.png", dpi=220)
    plt.close(fig)


def select_n4_terms(data, nmax: int, t_ref: float) -> list[tuple]:
    combo = data.get((nmax, t_ref))
    if combo is None:
        return []
    j2 = combo["two_site"].get((1, 1))
    if j2 is None or j2 == 0.0:
        return []
    ranked = sorted(
        combo["n4"],
        key=lambda key: abs(combo["n4"][key] / j2),
        reverse=True,
    )
    return ranked[:2]


def plot_n4_terms(data, outdir: Path, workflow: str, nmaxes: tuple[int, ...], ts: list[float], tmax: float):
    ref_nmax = max(nmaxes)
    keys = select_n4_terms(data, ref_nmax, ts[-1])
    if not keys:
        print("warning: no non-plaquette N4 terms found; skipping fig12")
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.8), constrained_layout=True)
    if len(keys) == 1:
        axes = [axes]

    details = []
    for idx, (ax, key) in enumerate(zip(axes, keys), start=1):
        details.append({"panel": idx, "key": key_label(key)})
        for nmax in nmaxes:
            vals = series(
                data,
                nmax,
                ts,
                lambda combo, _: ratio(combo["n4"].get(key), combo["two_site"].get((1, 1), 0.0)),
            )
            ax.plot(ts, vals, "o-", color=COLORS.get(nmax), label=f"N={nmax}")
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        setup_axis(ax, ylabel="J_N4 / J2", xlim=(0.0, tmax))
        ax.set_title(f"N4 term {idx}")
        ax.legend(fontsize=8)

    fig.suptitle(f"Leading non-plaquette four-site terms, workflow={workflow}")
    fig.savefig(outdir / "fig12_n4_terms.png", dpi=220)
    plt.close(fig)

    details_path = outdir / "fig12_n4_terms_selected.txt"
    lines = ["# selected non-plaquette N4 terms", f"# workflow={workflow} ref_N={ref_nmax} ref_t={ts[-1]:.4f}"]
    for item in details:
        lines.append(f"panel {item['panel']}: {item['key']}")
    details_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results_szs2eta2"))
    parser.add_argument("--workflow", default="adiabatic", choices=WORKFLOWS)
    parser.add_argument("--nmax", type=int, nargs="*", default=list(DEFAULT_NMAXES))
    parser.add_argument("--tmax", type=float, default=0.20)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    nmaxes = tuple(sorted(args.nmax))
    ts = t_values(args.tmax)
    outdir = root / "_plots" / "jps_simple" / args.workflow
    outdir.mkdir(parents=True, exist_ok=True)

    data: dict[tuple[int, float], dict] = {}
    missing = []
    for nmax in nmaxes:
        for t in ts:
            combo = load_combo(root, args.workflow, nmax, t)
            if combo is None:
                missing.append((nmax, t))
            else:
                data[(nmax, t)] = combo

    if not data:
        raise SystemExit(f"no embed data found under {root}")

    plot_dominant(data, outdir, args.workflow, nmaxes, ts, args.tmax)
    plot_subdominant(data, outdir, args.workflow, nmaxes, ts, args.tmax)
    plot_pt_ratio(data, outdir, args.workflow, max(nmaxes), ts, args.tmax)
    plot_n4_terms(data, outdir, args.workflow, nmaxes, ts, args.tmax)

    print(f"wrote plots to {outdir}")
    if missing:
        print(f"missing combos: {len(missing)}")
        for nmax, t in missing[:12]:
            print(f"  N={nmax} T={t:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
