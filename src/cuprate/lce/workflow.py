"""Linked-cluster expansion workflow implementation."""

import json
import os
import sys
import time
from datetime import datetime

from cuprate.io import (
    LCE_CLI_SPEC,
    block_root_dir,
    build_run_suffixes,
    build_consolidated_results,
    build_spin_coupling_artifact_from_terms,
    consolidated_results_path,
    deserialize_complex,
    lce_root_dir,
    load_consolidated_results,
    load_spin_coupling_catalog,
    spin_coupling_terms_from_artifact,
    write_cluster_points,
    RESULT_KIND_SPIN_COUPLINGS,
    parse_lce_cli_args,
    print_cli_help,
)

from .core import collect_subgraph_data, subtract_subgraph_contributions


def _fmt_real(value) -> str:
    x = float(value.real)
    if x == 0.0 or abs(x) >= 1e-4:
        return f"{x:.10f}"
    return f"{x:.6e}"


def _pair_operator_str(pair_indices) -> str:
    parts = []
    for idx in range(0, len(pair_indices), 2):
        site1, site2 = pair_indices[idx], pair_indices[idx + 1]
        parts.append(f"(S{site1}·S{site2})")
    return "".join(parts)


def _write_lce_couplings(f, artifact):
    f.write("\n=== Individual Bond Coefficients ===\n")
    constant_term = deserialize_complex(artifact["operators"]["constant_term"])
    f.write(f"Constant term: {_fmt_real(constant_term)}\n")

    groups = artifact["operators"]["groups"]
    for group in groups:
        if group["arity"] != 2:
            continue
        dx, dy = map(int, group["vector"])
        if not group["terms"]:
            f.write(f"\nBond vector ({dx}, {dy}), {group['label']}: (not present in cluster)\n")
            continue
        f.write(f"\nBond vector ({dx}, {dy}), {group['label']}:\n")
        for idx, term in enumerate(group["terms"]):
            site1, site2 = map(int, term["sites"])
            coefficient = deserialize_complex(term["coefficient"])
            f.write(f"    {idx}: Sites {site1}-{site2}: {_fmt_real(coefficient)}\n")

    for arity, section_name in ((4, "Four-site bond"), (6, "Six-site bond"), (8, "Eight-site bond")):
        multi_groups = [group for group in groups if group["arity"] == arity]
        if not multi_groups:
            continue
        f.write(f"\n{section_name}\n")
        for group in multi_groups:
            f.write(f"    Group {group['label']}:\n")
            for term_idx, term in enumerate(group["terms"], start=1):
                coefficient = deserialize_complex(term["coefficient"])
                f.write(
                    f"      type {term_idx}: {_pair_operator_str(term['sites'])}: {_fmt_real(coefficient)}\n"
                )
            f.write("\n")


def _validate_complete_block_catalog(clusters, block_root: str, run_dir: str, n_max: int) -> None:
    missing = [n for n in range(2, n_max + 1) if n not in clusters]
    if not missing:
        return

    missing_dirs = "\n".join(
        f"  {block_root}/N{n}{run_dir}"
        for n in missing
    )
    raise FileNotFoundError(
        "LCE requires spin-coupling results for every cluster size from 2 up to "
        f"N={n_max}. Missing upstream Block runs:\n{missing_dirs}"
    )


def _process_site_count(
    clusters,
    data,
    block_root: str,
    lce_root: str,
    run_dir: str,
    nsites: int,
) -> str:
    block_run_dir = f"{block_root}/N{nsites}{run_dir}"
    if not os.path.exists(block_run_dir):
        raise FileNotFoundError(f"Missing upstream Block run directory: {block_run_dir}")
    block_payload = load_consolidated_results(block_run_dir)
    output_dir = f"{lce_root}/N{nsites}{run_dir}"
    os.makedirs(output_dir, exist_ok=True)
    cluster_entries = []
    entry_by_id = {
        (entry["hole"], entry["class_idx"], entry["cluster_idx"]): entry
        for entry in block_payload["clusters"]
    }

    for hole in clusters[nsites]:
        for class_idx in clusters[nsites][hole]:
            for rank_idx, cluster in clusters[nsites][hole][class_idx].items():
                entry = entry_by_id[(hole, class_idx, rank_idx)]
                error = entry["fit"]["relative_error"]
                t11 = entry["fit"]["t11_minus_1_norm"]

                cluster_data = data[nsites][hole][class_idx][rank_idx]
                cluster_data["terms"] = spin_coupling_terms_from_artifact(entry)
                subtract_subgraph_contributions(cluster_data, data)

                base_filename = f"{output_dir}/hole{hole}_class{class_idx}_cluster{rank_idx}"
                artifact = build_spin_coupling_artifact_from_terms(
                    cluster,
                    cluster_data["terms"],
                    hole=hole,
                    class_idx=class_idx,
                    cluster_idx=rank_idx,
                    metadata={"rank": 0, "computation_time_s": 0.0},
                )
                cluster_entries.append(artifact)
                with open(f"{base_filename}_results.txt", "w") as f:
                    write_cluster_points(f, cluster)
                    _write_lce_couplings(f, artifact)
                    f.write("\n")
                    f.write(f"This file is read from {consolidated_results_path(block_run_dir)}.\n")
                    f.write("Please check the following values, we did not check the fitting accuracy!\n")
                    f.write(f"Relative Error: {error}\n")
                    f.write(f"T11-1 Norm: {t11}\n")
                with open(f"{base_filename}_results.json", "w") as f:
                    json.dump(artifact, f, indent=2)

    with open(consolidated_results_path(output_dir), "w") as f:
        json.dump(
            build_consolidated_results(
                RESULT_KIND_SPIN_COUPLINGS,
                block_payload["run_params"],
                cluster_entries,
            ),
            f,
            indent=2,
        )
    print(f"Now is saving the results in {output_dir}")
    return output_dir


def main(params):
    base_dir, run_dir, _ = build_run_suffixes(params)
    block_root = block_root_dir(base_dir)
    lce_root = lce_root_dir(base_dir)
    print(f"Working directory: {lce_root}/N{params.N}{run_dir}")
    if not os.path.exists(block_root):
        raise FileNotFoundError(f"Missing upstream Block root directory: {block_root}")

    clusters = load_spin_coupling_catalog(block_root, run_dir, params.N)
    _validate_complete_block_catalog(clusters, block_root, run_dir, params.N)
    data = collect_subgraph_data(clusters)
    for nsites in clusters:
        if nsites >= 2:
            _process_site_count(
                clusters,
                data,
                block_root,
                lce_root,
                run_dir,
                nsites,
            )


def run_cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_cli_help(LCE_CLI_SPEC)
        sys.exit(0)
    params = parse_lce_cli_args()
    t0 = time.time()
    print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main(params)
    t1 = time.time()
    print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task is finished in {t1 - t0} seconds")


if __name__ == "__main__":
    run_cli()
