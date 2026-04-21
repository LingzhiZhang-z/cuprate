"""Artifact writing for the current stage-1 runtime."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from cuprate.io import (
    SpinCouplingTerms,
    build_spin_coupling_artifact_from_terms,
    serialize_complex,
    write_cluster_points,
)


def _fmt_real(value) -> str:
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


def _bond_vector(cluster, bond) -> tuple[int, int]:
    site1, site2 = map(int, bond[:2])
    dx = abs(cluster[site2][0] - cluster[site1][0])
    dy = abs(cluster[site2][1] - cluster[site1][1])
    return tuple(sorted((dx, dy), reverse=True))


def _spin_coupling_terms_from_project(cluster, bond_groups, coeffs) -> SpinCouplingTerms:
    two_site: dict[tuple[int, int], list[list]] = {}
    four_site: list[list] = []
    six_site: list[list] = []
    eight_site: list[list] = []

    for group, group_coeffs in zip(bond_groups, coeffs[1:]):
        if not group:
            continue
        arity = len(group[0])
        entries = [[*map(int, bond), coeff] for bond, coeff in zip(group, group_coeffs)]
        if arity == 2:
            vector = _bond_vector(cluster, group[0])
            two_site.setdefault(vector, []).extend(entries)
        elif arity == 4:
            four_site.extend(entries)
        elif arity == 6:
            six_site.extend(entries)
        elif arity == 8:
            eight_site.extend(entries)
        else:
            raise ValueError(f"Unsupported bond arity for reporting: {arity}")

    return SpinCouplingTerms(
        constant=coeffs[0],
        two_site=two_site,
        four_site=four_site,
        six_site=six_site,
        eight_site=eight_site,
    )


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
    constant_term = complex(
        artifact["operators"]["constant_term"]["real"],
        artifact["operators"]["constant_term"]["imag"],
    )

    f.write(f"Hole: {hole}\n")
    f.write(f"Class: {class_idx}\n")
    f.write(f"Cluster: {cluster_idx}\n")
    f.write(f"Rank: {artifact['metadata']['rank']}\n")
    f.write(f"Computation time: {artifact['metadata']['computation_time_s']:.6f} seconds\n")
    write_cluster_points(f, cluster)

    f.write("\n=== Bond Structure ===\n")
    for group in groups:
        if group["arity"] == 2:
            dx, dy = map(int, group["vector"])
            if group["terms"]:
                f.write(f"Bond vector ({dx}, {dy}), {group['label']}:\n")
                for idx, term in enumerate(group["terms"]):
                    site1, site2 = map(int, term["sites"])
                    x1, y1 = cluster[site1]
                    x2, y2 = cluster[site2]
                    f.write(f"    {idx}: Sites {site1}-{site2} ({x1},{y1})-({x2},{y2})\n")
            else:
                f.write(f"Bond vector ({dx}, {dy}), {group['label']}: (not present in cluster)\n")

    multi_site_labels = {4: "Four-site bond", 6: "Six-site bond", 8: "Eight-site bond"}
    for arity, header in multi_site_labels.items():
        multi_groups = [group for group in groups if group["arity"] == arity]
        if not multi_groups:
            continue
        f.write(f"\n{header}\n")
        for group in multi_groups:
            for term_idx, term in enumerate(group["terms"], start=1):
                f.write(
                    f"    class 0 type {term_idx} : {_pair_operator_str(term['sites'])}\n"
                )

    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {_fmt_real(constant_term)}\n")
    for group in groups:
        if group["arity"] == 2:
            dx, dy = map(int, group["vector"])
            if group["terms"]:
                f.write(f"\nBond vector ({dx}, {dy}), {group['label']}:\n")
                for idx, term in enumerate(group["terms"]):
                    coeff = complex(term["coefficient"]["real"], term["coefficient"]["imag"])
                    site1, site2 = map(int, term["sites"])
                    f.write(f"    {idx}: Sites {site1}-{site2}: {coeff.real:.10f}  +  {coeff.imag:.10f}i\n")
            else:
                f.write(f"\nBond vector ({dx}, {dy}), {group['label']}: (not present in cluster)\n")
        elif group["terms"]:
            f.write(f"\n{group['label']}\n")
            for term_idx, term in enumerate(group["terms"], start=1):
                coeff = complex(term["coefficient"]["real"], term["coefficient"]["imag"])
                f.write(
                    f"    class 0 type {term_idx} : {_pair_operator_str(term['sites'])}: "
                    f"{coeff.real:.10f}  +  {coeff.imag:.10f}i\n"
                )

    f.write("\n=== Individual Fit Error ===\n")
    f.write(f"Relative Error: {fit['relative_error']:.10f}\n")
    f.write(f"Residual: {fit['residual']:.10f}\n")
    f.write(f"R^2: {fit['r_squared']:.10f}\n")
    f.write(f"T11-1 norm: {fit['t11_minus_1_norm']:.10f}\n")
    f.write(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def save_spin_coupling_result(
    result_dir: str,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    rank: int,
    cluster_time: float,
    cluster,
    bond_groups,
    coeffs,
    error: tuple[float, float, float],
    t11m1_norm: float,
    overlap: float | None,
) -> dict:
    base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
    artifact = build_spin_coupling_artifact_from_terms(
        cluster,
        _spin_coupling_terms_from_project(cluster, bond_groups, coeffs),
        hole=hole,
        class_idx=class_idx,
        cluster_idx=cluster_idx,
        fit={
            "relative_error": float(error[0]),
            "residual": float(error[1]),
            "r_squared": float(error[2]),
            "t11_minus_1_norm": float(t11m1_norm),
            "overlap": None if overlap is None else float(overlap),
        },
        metadata={
            "rank": int(rank),
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
    projection,
) -> dict:
    base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
    artifact = {
        "hole": int(hole),
        "class_idx": int(class_idx),
        "cluster_idx": int(cluster_idx),
        "sites": [[int(x), int(y)] for x, y in cluster],
        "projection": {
            "t11_minus_1_norm": float(projection.t11m1_norm),
            "overlap": None if projection.overlap is None else float(projection.overlap),
            "selected_state_count": int(len(projection.selected_indices)),
            "heff_dimension": [int(projection.heff.shape[0]), int(projection.heff.shape[1])],
            "selected_indices": [int(idx) for idx in projection.selected_indices],
            "double_occupation_expectation": [
                float(value.real) for value in projection.selected_occupation
            ],
        },
        "metadata": {
            "rank": int(rank),
            "computation_time_s": float(cluster_time),
        },
    }

    with open(f"{base_filename}_results.txt", "w") as f:
        f.write(f"Hole: {hole}\n")
        f.write(f"Class: {class_idx}\n")
        f.write(f"Cluster: {cluster_idx}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"Computation time: {cluster_time:.6f} seconds\n")
        write_cluster_points(f, cluster)
        f.write("\n=== Projection Diagnostics ===\n")
        f.write("Spin couplings are not reported for MODE=fixed_sz_s2.\n")
        f.write(f"T11-1 norm: {projection.t11m1_norm:.10f}\n")
        f.write(f"Selected state count: {len(projection.selected_indices)}\n")
        f.write(f"Heff dimension: {projection.heff.shape[0]} x {projection.heff.shape[1]}\n")
        f.write("\nSelected eigenstate indices:\n")
        f.write(" ".join(str(int(idx)) for idx in projection.selected_indices) + "\n")
        f.write("\nSelected double occupation expectation:\n")
        f.write(" ".join(f"{value.real:.10f}" for value in projection.selected_occupation) + "\n")
        f.write("\nSelected S^2 diagnostics:\n")
        for idx in projection.selected_indices:
            s2_value = projection.s2_diag[int(idx), 0]
            s2_error = projection.s2_diag[int(idx), 1]
            f.write(
                f"  idx {int(idx)}: S2={s2_value.real:.10f}, error={s2_error.real:.10e}\n"
            )
        f.write(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    with open(f"{base_filename}_results.json", "w") as f:
        json.dump(artifact, f, indent=2)
    return artifact
