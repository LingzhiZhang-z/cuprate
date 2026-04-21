"""Main Hubbard workflow implementation."""

import os
import sys
import time
import json
from datetime import datetime
from math import comb

from cuprate.mpi import comm, rank, size, is_root
from cuprate.io import (
    MAIN_CLI_SPEC,
    Params,
    build_path_spec,
    build_consolidated_results,
    consolidated_results_path,
    parse_main_cli_args,
    print_cli_help,
    resolve_mode_spec,
    RESULT_KIND_PROJECTION_ANALYSIS,
)
from cuprate.clusters import Clusters_Square
from .solver import cluster_process_work_item


# ============================================================
# Space Analysis
# ============================================================

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

    print("Sz, S^2 sectors (dim, dim_eff):")
    for twoSz in twoSz_values:
        for twoS in range(twoSz, N + 1, 2):
            k = (N - twoS) // 2
            dim_sz_s = comb(N, k) - (comb(N, k - 1) if k > 0 else 0)
            S2_val = twoS * (twoS + 2) / 4
            dim_eff_sz_s = comb(N, k) - (comb(N, k - 1) if k > 0 else 0)
            print(
                f"  twoSz={twoSz}, twoS={twoS}, S2={S2_val:.2f}: "
                f"dim={dim_sz_s}, dim_eff={dim_eff_sz_s}"
            )

# ============================================================
# Workflow
# ============================================================


def distribute_work(work_items: list, rank: int, size: int) -> list:
    """Distribute work items among MPI ranks using round-robin with extra tasks to later ranks."""
    total_items = len(work_items)
    items_per_rank = total_items // size
    remainder = total_items % size

    work_items_for_rank = []
    base_items = items_per_rank * size

    for i in range(rank, base_items, size):
        work_items_for_rank.append((i, work_items[i]))

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
    clusters = None
    work_items = None

    if is_root():
        analyze_space(params.N)
        print("Computing clusters...")
        clusters = Clusters_Square(params.N)
        clusters.compute_clusters(if_print_time=True)
        clusters.classify_clusters(if_print_time=True)
        clusters.print_info()
        work_items = [
            (hole, class_idx)
            for hole in range(len(clusters.clusters_classified))
            for class_idx in range(len(clusters.clusters_classified[hole]))
        ]
        print(f"\nPrepared {len(work_items)} work items for distribution")
        print("Broadcasting clusters...")

    comm.Barrier()

    clusters = comm.bcast(clusters, root=0)
    work_items = comm.bcast(work_items, root=0)

    rank_work_items = distribute_work(work_items, rank, size)
    if is_root():
        print(f"Restart from previous computation: {params.restart}") if params.restart else print("Starting computation...")
        print(f"We adopt the \"{params.workflow}\" workflow to calculate T11")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}")
        print(f"Up to now, time cost: {time.time() - t0_total:.1f}s\n")
        if params.workflow == "adiabatic":
            if params.delta is None:
                raise ValueError("delta is not set")
            params_previous = Params(
                N=params.N,
                U=params.U,
                t=params.t - params.delta,
                mode=params.mode,
                twoSz=params.twoSz,
                twoS=params.twoS,
                match_spin_sectors=params.match_spin_sectors,
                workflow=params.workflow,
                restart=params.restart,
            )
            spec_previous = build_path_spec(params_previous)
            print(
                f"In the adiabatic process, read data from {spec_previous.data_dir}",
                flush=True,
            )

    rank_cluster_entries = []
    for work_idx, (hole, class_idx) in rank_work_items:
        t0 = time.time()
        rank_cluster_entries.extend(
            cluster_process_work_item(hole, class_idx, clusters, params, spec, mode_spec, rank)
        )

        if is_root():
            print(
                f"Rank 0 completed task {work_idx + 1} / {len(rank_work_items)}, "
                f"time: {time.time() - t0:.1f}s",
                flush=True,
            )

    comm.Barrier()

    gathered_cluster_entries = comm.gather(rank_cluster_entries, root=0)

    if is_root():
        consolidated_payload = build_consolidated_results(
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
            json.dump(consolidated_payload, f, indent=2)
        print(f"\nTotal execution time: {time.time() - t0_total:.6f} seconds")

    comm.Barrier()


def run_cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print_cli_help(MAIN_CLI_SPEC)
        sys.exit(0)

    params = parse_main_cli_args()
    mode_spec = resolve_mode_spec(params)

    if is_root():
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"Using parameters: N={params.N}, U={params.U}, t={params.t}, "
            f"workflow={params.workflow}, twoSz={params.twoSz}, "
            f"twoS={params.twoS}, restart={params.restart}"
        )
        if mode_spec.result_kind == RESULT_KIND_PROJECTION_ANALYSIS:
            print("MODE=fixed_sz_s2 runs projection analysis only; spin couplings will not be reported.")
        if params.match_spin_sectors:
            print("MATCH_SPIN_SECTORS=true: low-energy states will be selected with spin-sector multiplicity matching.")

    main(params, mode_spec)

    if is_root():
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_cli()
