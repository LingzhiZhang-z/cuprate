"""Result formatting and artifact writing for the main Hubbard workflow."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from cuprate.io import (
    build_projection_analysis_artifact,
    build_spin_coupling_artifact_from_coeffs,
    build_spin_coupling_artifact_from_terms,
    build_spin_coupling_terms_from_catalog,
    deserialize_complex,
    write_cluster_points,
)


def _fmt_real(value) -> str:
    """Format a scalar coefficient as a real number; scientific for small values."""
    x = float(np.real(value))
    if x == 0.0 or abs(x) >= 1e-4:
        return f"{x:.10f}"
    return f"{x:.6e}"


def _pair_operator_str(pair_indices) -> str:
    parts = []
    for k in range(0, len(pair_indices), 2):
        i, j = pair_indices[k], pair_indices[k + 1]
        parts.append(f"(S{i}·S{j})")
    return "".join(parts)


def _site_list_str(sites) -> str:
    return "{" + ", ".join(str(s) for s in sites) + "}"


def _write_spin_coupling_text(
    f,
    artifact: dict,
    *,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    cluster,
) -> None:
    fit = artifact["fit"]
    groups = artifact["operators"]["groups"]
    constant_term = deserialize_complex(artifact["operators"]["constant_term"])

    f.write("=== Summary ===\n")
    f.write(f"Hole: {hole}  Class: {class_idx}  Cluster: {cluster_idx}\n")
    f.write(f"N={len(cluster)}  Sites: {' '.join(f'({x},{y})' for x, y in cluster)}\n")
    f.write(
        f"R\u00b2: {_fmt_real(fit['r_squared'])}  "
        f"|T11-I|: {_fmt_real(fit['t11_minus_1_norm'])}  "
        f"Overlap: {'n/a' if fit['overlap'] is None else _fmt_real(fit['overlap'])}\n"
    )

    f.write("\n=== Two-site couplings ===\n")
    for group in groups:
        if group["arity"] != 2 or not group["terms"]:
            continue
        dx, dy = map(int, group["vector"])
        f.write(f"{group['label']}  vector ({dx},{dy}):\n")
        for term in group["terms"]:
            site1, site2 = map(int, term["sites"])
            coefficient = deserialize_complex(term["coefficient"])
            f.write(f"    Sites {site1}-{site2}:  {_fmt_real(coefficient)}\n")

    for arity, section_name in ((4, "Four-site"), (6, "Six-site"), (8, "Eight-site")):
        multi_groups = [group for group in groups if group["arity"] == arity]
        if not multi_groups:
            continue
        f.write(f"\n=== {section_name} couplings ===\n")
        for group in multi_groups:
            support_sites = sorted({int(site) for term in group["terms"] for site in term["sites"]})
            f.write(f"Group {group['label']}: sites {_site_list_str(support_sites)}\n")
            for term in group["terms"]:
                coefficient = deserialize_complex(term["coefficient"])
                f.write(f"    {_pair_operator_str(term['sites'])}:  {_fmt_real(coefficient)}\n")

    f.write("\n=== Fit quality ===\n")
    f.write(f"Constant term:   {_fmt_real(constant_term)}\n")
    f.write(f"Relative error:  {_fmt_real(fit['relative_error'])}\n")
    f.write(f"Residual:        {_fmt_real(fit['residual'])}\n")
    f.write(f"R\u00b2:              {_fmt_real(fit['r_squared'])}\n")
    f.write(f"|T11-I|:         {_fmt_real(fit['t11_minus_1_norm'])}\n")
    f.write(f"Overlap:         {'n/a' if fit['overlap'] is None else _fmt_real(fit['overlap'])}\n")
    f.write(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def save_spin_coupling_result(
    result_dir: str,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    rank: int,
    cluster_time: float,
    cluster,
    spin_operator_catalog,
    coeffs,
    error: tuple[float, float, float],
    t11m1_norm: float,
    overlap: float | None,
) -> dict:
    base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
    artifact = build_spin_coupling_artifact_from_coeffs(
        cluster,
        spin_operator_catalog,
        coeffs,
        error,
        t11m1_norm,
        overlap,
        hole=hole,
        class_idx=class_idx,
        cluster_idx=cluster_idx,
        metadata={
            "rank": rank,
            "computation_time_s": float(cluster_time),
        },
    )
    with open(f"{base_filename}_results.txt", "w") as f:
        _write_spin_coupling_text(
            f,
            artifact,
            hole=hole,
            class_idx=class_idx,
            cluster_idx=cluster_idx,
            cluster=cluster,
        )
    with open(f"{base_filename}_results.json", "w") as f:
        json.dump(artifact, f, indent=2)
    return artifact


def save_projection_result(
    result_dir: str,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    rank: int,
    cluster_time: float,
    cluster,
    spin_operator_catalog,
    model,
) -> dict:
    base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
    operator_artifact = build_spin_coupling_artifact_from_terms(
        cluster,
        build_spin_coupling_terms_from_catalog(spin_operator_catalog),
        hole=hole,
        class_idx=class_idx,
        cluster_idx=cluster_idx,
    )
    with open(f"{base_filename}_results.txt", "w") as f:
        f.write(f"Hole: {hole}\n")
        f.write(f"Class: {class_idx}\n")
        f.write(f"Cluster: {cluster_idx}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"Computation time: {cluster_time:.6f} seconds\n")
        write_cluster_points(f, cluster)

        f.write("\n=== Bond Structure ===")
        for group in operator_artifact["operators"]["groups"]:
            if group["arity"] != 2:
                continue
            dx, dy = map(int, group["vector"])
            f.write(f"\nBond vector ({dx}, {dy}), {group['label']}:")
            if not group["terms"]:
                f.write(" (not present in cluster)\n")
                continue
            f.write("\n")
            for idx, term in enumerate(group["terms"]):
                site1, site2 = map(int, term["sites"])
                x1, y1 = cluster[site1]
                x2, y2 = cluster[site2]
                f.write(f"    {idx}: Sites {site1}-{site2} ({x1},{y1})-({x2},{y2})\n")

        four_site_groups = [group for group in operator_artifact["operators"]["groups"] if group["arity"] == 4]
        if four_site_groups:
            f.write("\nFour-site bonds:\n")
            for group in four_site_groups:
                f.write(f"    Group {group['label']}:\n")
                for term_idx, term in enumerate(group["terms"], start=1):
                    f.write(f"      type {term_idx}: Four-site bond: {term['sites']}\n")
                f.write("\n")

        six_site_groups = [group for group in operator_artifact["operators"]["groups"] if group["arity"] == 6]
        if six_site_groups:
            f.write("\nSix-site bonds:\n")
            for group in six_site_groups:
                f.write(f"    Group {group['label']}:\n")
                for term_idx, term in enumerate(group["terms"], start=1):
                    f.write(f"      type {term_idx}: Six-site bond: {term['sites']}\n")
                f.write("\n")

        df = model.downfold
        f.write("\n=== Projection Diagnostics ===\n")
        f.write("Spin couplings are not reported for MODE=fixed_sz_s2.\n")
        f.write("This mode fixes a single total-spin sector and does not determine unique SU(2)-invariant couplings.\n")
        f.write(f"T11-1 norm: {df.t11m1_norm:.10f}\n")
        if df.overlap is not None:
            f.write(f"Overlap: {df.overlap:.10f}\n")
        f.write(f"Selected state count: {len(df.selected_indices)}\n")
        f.write(f"Heff dimension: {df.heff.shape[0]} x {df.heff.shape[1]}\n")
        f.write("\nSelected eigenstate indices:\n")
        f.write(" ".join(str(int(idx)) for idx in np.atleast_1d(df.selected_indices)) + "\n")
        f.write("\nSelected double occupation expectation:\n")
        f.write(" ".join(f"{value.real:.10f}" for value in np.atleast_1d(df.selected_occupation)) + "\n")

        if model.S2 is not None and model.S2.diag is not None:
            f.write("\nSelected S^2 diagnostics:\n")
            for idx in np.atleast_1d(df.selected_indices):
                f.write(
                    f"  idx {int(idx)}: "
                    f"S2={model.S2.diag[int(idx)]:.10f}, "
                    f"error={model.S2.error[int(idx)].real:.10e}\n"
                )

        f.write(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    artifact = build_projection_analysis_artifact(
        cluster,
        model,
        hole=hole,
        class_idx=class_idx,
        cluster_idx=cluster_idx,
        metadata={
            "rank": rank,
            "computation_time_s": float(cluster_time),
        },
    )
    with open(f"{base_filename}_results.json", "w") as f:
        json.dump(artifact, f, indent=2)
    return artifact
