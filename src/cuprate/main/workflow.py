"""Main Hubbard workflow implementation on top of the current ED core."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from math import comb

from cuprate.clusters import ClusterSets
from cuprate.io import (
    MAIN_CLI_SPEC,
    build_consolidated_results,
    build_path_spec,
    consolidated_results_path,
    parse_main_cli_args,
    print_cli_help,
    resolve_mode_spec,
    RESULT_KIND_PROJECTION_ANALYSIS,
)
from cuprate.mpi import comm, is_root, rank, size

from .solver import cluster_process_work_item


def analyze_space(N: int) -> None:
    dim = comb(2 * N, N)
    dim_eff = 2 ** N
    print(f"Total dimension: {dim}")
    print(f"Total effective dimension: {dim_eff}")

    twoSz_values = range(0 if N % 2 == 0 else 1, N + 1, 2)
    print("Sz sectors (dim, dim_eff):")
    for twoSz in twoSz_values:
        nup = (N + twoSz) // 2
        ndown = N - nup
        dim_sz = comb(N, nup) * comb(N, ndown)
        dim_eff_sz = comb(N, nup)
        print(f"  twoSz={twoSz}: dim={dim_sz}, dim_eff={dim_eff_sz}")


def _cluster_families(cluster_sets: ClusterSets) -> dict[tuple[int, int], list]:
    families = defaultdict(list)
    for cluster in cluster_sets.clusters:
        families[(int(cluster.hole), int(cluster.class_idx))].append(cluster)
    return {
        key: sorted(clusters, key=lambda cluster: int(cluster.cluster_idx))
        for key, clusters in sorted(families.items())
    }


def distribute_work(work_items: list, rank: int, size: int) -> list:
    total_items = len(work_items)
    items_per_rank = total_items // size
    remainder = total_items % size

    work_items_for_rank = []
    base_items = items_per_rank * size
    for idx in range(rank, base_items, size):
        work_items_for_rank.append((idx, work_items[idx]))
    if remainder > 0 and rank >= size - remainder:
        extra_item_idx = base_items + (rank - (size - remainder))
        work_items_for_rank.append((extra_item_idx, work_items[extra_item_idx]))
    return work_items_for_rank


def main(params, mode_spec):
    spec = build_path_spec(params)
    if is_root():
        os.makedirs(spec.data_dir, exist_ok=True)
        os.makedirs(spec.output_dir, exist_ok=True)
        os.makedirs(spec.tmp_dir, exist_ok=True)
        print(f"Working directory: {spec.output_dir}")

    t0_total = time.time()
    cluster_families = None
    work_items = None

    if is_root():
        analyze_space(params.N)
        print("Computing clusters...")
        cluster_sets = ClusterSets(params.N).generate()
        cluster_sets.print_info()
        cluster_families = _cluster_families(cluster_sets)
        work_items = list(cluster_families)
        print(f"\nPrepared {len(work_items)} work items for distribution")
        print("Broadcasting cluster families...")

    comm.Barrier()
    cluster_families = comm.bcast(cluster_families, root=0)
    work_items = comm.bcast(work_items, root=0)

    rank_work_items = distribute_work(work_items, rank, size)
    if is_root():
        print("Starting computation..." if not params.restart else "Restart requested...")
        print(f"Selection workflow: {params.workflow or 'occ'}")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Local work items on rank 0: {len(rank_work_items)}")
        print(f"Up to now, time cost: {time.time() - t0_total:.1f}s\n")

    rank_cluster_entries = []
    for work_idx, (hole, class_idx) in rank_work_items:
        t0 = time.time()
        rank_cluster_entries.extend(
            cluster_process_work_item(
                hole,
                class_idx,
                cluster_families,
                params,
                spec,
                mode_spec,
                rank,
            )
        )
        print(
            f"Rank {rank} completed task {work_idx + 1}/{len(work_items)} in {time.time() - t0:.1f}s",
            flush=True,
        )

    comm.Barrier()
    gathered_cluster_entries = comm.gather(rank_cluster_entries, root=0)

    if is_root():
        payload = build_consolidated_results(
            mode_spec.result_kind,
            {
                "N": int(params.N),
                "U": float(params.U),
                "t": float(params.t),
                "mode": str(params.mode),
                **({"twoSz": int(params.twoSz)} if params.twoSz is not None else {}),
                **({"twoS": int(params.twoS)} if params.twoS is not None else {}),
                **({"workflow": str(params.workflow)} if params.workflow is not None else {}),
            },
            [entry for rank_entries in gathered_cluster_entries for entry in rank_entries],
        )
        with open(consolidated_results_path(spec.output_dir), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nTotal execution time: {time.time() - t0_total:.6f} seconds")

    comm.Barrier()


def run_cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_cli_help(MAIN_CLI_SPEC)
        sys.exit(0)

    params = parse_main_cli_args()
    mode_spec = resolve_mode_spec(params)

    if is_root():
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"Using parameters: N={params.N}, U={params.U}, t={params.t}, "
            f"workflow={params.workflow or 'occ'}, twoSz={params.twoSz}, "
            f"twoS={params.twoS}, restart={params.restart}"
        )
        if mode_spec.result_kind == RESULT_KIND_PROJECTION_ANALYSIS:
            print("MODE=fixed_sz_s2 runs projection analysis only; spin couplings will not be reported.")

    main(params, mode_spec)

    if is_root():
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_cli()
