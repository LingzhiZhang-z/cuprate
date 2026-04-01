import os
import sys
import time
import numpy as np
from datetime import datetime
from math import comb
from typing import List, Tuple

from cuprate.mpi import comm, rank, size, is_root, barrier, broadcast
from cuprate.io import (
    Params, PathSpec, build_path_spec, resolve_mode_spec,
    setup_work_environment, filter_work_items, check_and_print_adiabatic_info,
    write_file, write_result_artifact, load_array_compat,
    build_spin_coupling_artifact_from_coeffs, build_projection_analysis_artifact,
    find_bond_vector, write_cluster_points, get_all_possible_vectors,
    iter_cli_assignments, parse_bool_arg, tune_sz,
    MODE_FULL, MODE_BLOCK_SZ_FULL, MODE_BLOCK_SZS2_FULL,
    RESULT_KIND_SPIN_COUPLINGS, RESULT_KIND_PROJECTION_ANALYSIS,
)
from cuprate.hubbard import Hubbard_SingleBand
from cuprate.clusters import Clusters_Square, bond_analysis_spin, canonical_bond_type


# ============================================================
# Space Analysis
# ============================================================

def comb_safe(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def analyze_space(N: int) -> None:
    Smax = N * 0.5
    dim = comb(2 * N, N)
    dim_eff = 2 ** N
    n_half = N * 0.5

    Nups = [N - i for i in range(N + 1)]
    Ndos = [i for i in range(N + 1)]
    sz_states = []
    sz_s2_states = []
    for idx_sz in range(N + 1):
        sz = (Nups[idx_sz] - Ndos[idx_sz]) * 0.5
        dim_sz = comb_safe(N, Nups[idx_sz]) * comb_safe(N, Ndos[idx_sz])
        dim_eff_sz = comb_safe(N, Nups[idx_sz])

        sum_dim_sz_s = 0
        sum_dim_eff_sz_s = 0
        for idx_s in range(int(Smax - abs(sz)) + 1):
            s = abs(sz) + idx_s
            s2 = s * (s + 1)
            dim_sz_s = comb_safe(N, int(n_half + s)) * comb_safe(N, int(n_half - s)) - comb_safe(N, int(n_half + s + 1)) * comb_safe(N, int(n_half - s - 1))
            dim_eff_sz_s = comb_safe(N, int(n_half - s)) - comb_safe(N, int(n_half - s - 1))
            sz_s2_states.append((sz, s, s2, dim_sz_s, dim_eff_sz_s))
            sum_dim_sz_s += dim_sz_s
            sum_dim_eff_sz_s += dim_eff_sz_s
        sz_states.append((sz, dim_sz, dim_eff_sz, sum_dim_sz_s, sum_dim_eff_sz_s))

    print(f"Total dimension: {dim}")
    print(f"Total effective dimension: {dim_eff}")
    print("Sz sectors (dim, dim_eff):")
    for sz, dim_sz, dim_eff_sz, sum_dim_sz_s, sum_dim_eff_sz_s in sz_states:
        print(f"  sz={sz:.1f}: dim={dim_sz}, dim_eff={dim_eff_sz}")
    print("Sz, S^2 sectors (dim, dim_eff):")
    for sz, s, s2, dim_sz_s, dim_eff_sz_s in sz_s2_states:
        print(f"  sz={sz:.1f}, s={s:.1f}, s2={s2:.2f}: dim={dim_sz_s}, dim_eff={dim_eff_sz_s}")


# ============================================================
# CLI Parsing
# ============================================================

DEFAULT_N = 3
DEFAULT_U = 1.0
DEFAULT_T = 0.1


def print_help() -> None:
    print("Usage: python -m cuprate.main [KEY=VALUE]...")
    print("  N: number of sites (default: 3)")
    print("  U: Hubbard U parameter (default: 1.0)")
    print("  T: hopping parameter (default: 0.1)")
    print("  MODE: one of full, fixed_sz, block_sz_full, fixed_sz_s2, block_szs2_full")
    print("  SZ_IDX/S_IDX: sector indices for fixed_sz and fixed_sz_s2 modes")
    print("  MATCH_SPIN_SECTORS: true/false")


def parse_arguments(rank: int = 0) -> Params:
    """Parse command-line arguments and return a Params dataclass."""
    params = Params(N=DEFAULT_N, U=DEFAULT_U, t=DEFAULT_T)

    for key, value in iter_cli_assignments(sys.argv):
        if key == 'N':
            params.N = int(value)
        elif key == 'U':
            params.U = float(value)
        elif key == 'T':
            params.t = float(value)
        elif key == 'MODE':
            params.mode = value.lower()
        elif key == 'SZ_IDX':
            params.sz_idx = int(value)
        elif key == 'S_IDX':
            params.s_idx = int(value)
        elif key == 'T2':
            params.t2 = float(value)
        elif key == 'T3':
            params.t3 = float(value)
        elif key == 'SZ':
            params.sz = float(value)
        elif key == 'S2':
            params.s2 = float(value)
        elif key == 'S2_FIX':
            params.s2_fix = parse_bool_arg(value)
        elif key == 'TYPE':
            params.type = value
        elif key == 'BLOCK':
            params.block = value
        elif key == 'SELECT':
            params.select = value
        elif key == 'DELTA':
            params.delta = float(value)
        elif key == 'DELTA2':
            params.delta2 = float(value)
        elif key == 'RESTART':
            params.restart = parse_bool_arg(value)
        elif key == 'TYPE_DELTA':
            params.type_delta = value
        elif key == 'NCELL':
            params.Ncell = int(value)
        elif key == 'NCUT':
            params.Ncut = int(value)
        elif key == 'RATIO':
            params.ratio = float(value)
        elif key == 'MATCH_SPIN_SECTORS':
            params.match_spin_sectors = parse_bool_arg(value)

    params.sz = tune_sz(params.sz, params.N, rank=rank)
    spec = resolve_mode_spec(params)
    params.mode = spec.mode
    params.sz = spec.sz
    params.sz_idx = spec.sz_idx
    params.s = spec.s
    params.s_idx = spec.s_idx
    params.result_kind = spec.result_kind
    params.supports_spin_couplings = spec.supports_spin_couplings
    if spec.mode == "fixed_sz_s2":
        params.s2_fix = True
    return params


# ============================================================
# Workflow
# ============================================================

def prepare_clusters_and_tasks(params: Params) -> Tuple[Clusters_Square, List[Tuple[int, int]]]:
    print("Computing clusters...")
    clusters = Clusters_Square(params['N'])
    clusters.compute_clsuters(t2=params['t2'], if_print_time=True)
    clusters.classify_clusters(t2=params['t2'], if_print_time=True)
    clusters.print_info()

    work_items_all = [
        (hole, class_idx)
        for hole in range(len(clusters.clusters_classified))
        for class_idx in range(len(clusters.clusters_classified[hole]))
    ]
    print(f"\nPrepared {len(work_items_all)} work items for distribution")

    return clusters, work_items_all


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


# ============================================================
# Reporter
# ============================================================

def write_basic_info(f, hole: int, class_idx: int, cluster_idx: int, rank: int, cluster_time: float) -> None:
    """Write basic information about the calculation."""
    f.write(f"Hole: {hole}\n")
    f.write(f"Class: {class_idx}\n")
    f.write(f"Cluster: {cluster_idx}\n")
    f.write(f"Rank: {rank}\n")
    f.write(f"Computation time: {cluster_time:.6f} seconds\n")


def write_coefficients(f, cluster, bonds, coeffs_list, error, T11m1_norm: float, overlap) -> None:
    """Write individual bond coefficients."""
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {coeffs_list[0].real:.10f}\n")

    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in coeffs_by_vector:
                coeffs_by_vector[(dx, dy)] = []
            coeffs_by_vector[(dx, dy)].append((class_idx, bond_group, coeffs_list[class_idx + 1]))

    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if (dx, dy) in coeffs_by_vector:
            for class_idx, bond_group, coeffs in coeffs_by_vector[(dx, dy)]:
                f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}:\n")
                for idx, (bond, coef) in enumerate(zip(bond_group, coeffs)):
                    site1, site2 = bond
                    f.write(f"    {idx}: Sites {site1}-{site2}: {coef.real:.10f}  +  {coef.imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}: (not present in cluster)\n")

    f.write("\nFour-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 4:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx // 3 + 1} type {idx % 3 + 1} : Four-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}): {coeffs_list[class_idx + 1][idx].real:.10f}  +  {coeffs_list[class_idx + 1][idx].imag:.10f}i\n")
                if (idx + 1) % 3 == 0:
                    f.write("\n")

    f.write("\nSix-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 6:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx // 15 + 1} type {idx % 15 + 1} : Six-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}) * ({bond[4]}-{bond[5]}): {coeffs_list[class_idx + 1][idx].real:.10f}  +  {coeffs_list[class_idx + 1][idx].imag:.10f}i\n")
                if (idx + 1) % 15 == 0:
                    f.write("\n")

    f.write("\n=== Individual Fit Error ===\n")
    f.write(f"Relative Error: {error[0]:.10f}\n")
    f.write(f"Residual: {error[1]:.10f}\n")
    f.write(f"R^2: {error[2]:.10f}\n")
    f.write(f"T11-1 norm: {T11m1_norm:.10f}\n")
    if overlap is not None:
        f.write(f"Overlap: {overlap:.10f}\n")


def write_bond_structure(f, cluster, bonds) -> None:
    """Write bond structure information."""
    f.write("\n=== Bond Structure ===")

    bonds_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in bonds_by_vector:
                bonds_by_vector[(dx, dy)] = []
            bonds_by_vector[(dx, dy)].append((class_idx, bond_group))

    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}:")
        if (dx, dy) in bonds_by_vector:
            f.write("\n")
            for class_idx, bond_group in bonds_by_vector[(dx, dy)]:
                for idx, bond in enumerate(bond_group):
                    site1, site2 = bond
                    x1, y1 = cluster[site1]
                    x2, y2 = cluster[site2]
                    f.write(f"    {idx}: Sites {site1}-{site2} ({x1},{y1})-({x2},{y2})\n")
        else:
            f.write(" (not present in cluster)\n")

    square_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 4]
    if square_bonds:
        f.write("\nFour-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            square_bonds_in_class = [bond for bond in bond_group if len(bond) == 4]
            if square_bonds_in_class:
                for idx, bond in enumerate(square_bonds_in_class):
                    f.write(f"    {idx // 3 + 1} type {idx % 3 + 1}: Four-site bond: {bond}\n")
                    if (idx + 1) % 3 == 0:
                        f.write("\n")

    six_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 6]
    if six_bonds:
        f.write("\nSix-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            six_bonds_in_class = [bond for bond in bond_group if len(bond) == 6]
            if six_bonds_in_class:
                for idx, bond in enumerate(six_bonds_in_class):
                    f.write(f"    {idx // 15 + 1} type {idx % 15 + 1}: Six-site bond: {bond}\n")
                    if (idx + 1) % 15 == 0:
                        f.write("\n")


def write_results_to_file(
    f,
    hole,
    class_idx,
    cluster_idx,
    rank,
    cluster_time,
    cluster,
    bonds,
    coeffs,
    error,
    T11m1_norm,
    overlap,
) -> None:
    """Write all results to a file in a structured format."""
    write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time)
    write_cluster_points(f, cluster)
    write_bond_structure(f, cluster, bonds)
    write_coefficients(f, cluster, bonds, coeffs, error, T11m1_norm, overlap)


def write_projection_diagnostics(f, model) -> None:
    f.write("\n=== Projection Diagnostics ===\n")
    f.write("Spin couplings are not reported for MODE=fixed_sz_s2.\n")
    f.write("This mode fixes a single total-spin sector and does not determine unique SU(2)-invariant couplings.\n")
    f.write(f"T11-1 norm: {model.T11m1_norm:.10f}\n")
    if model.overlap is not None:
        f.write(f"Overlap: {model.overlap:.10f}\n")
    f.write(f"Selected state count: {len(model.t11_selected_indices)}\n")
    f.write(f"Heff dimension: {model.Heff.shape[0]} x {model.Heff.shape[1]}\n")

    f.write("\nSelected eigenstate indices:\n")
    indices = " ".join(str(int(idx)) for idx in np.atleast_1d(model.t11_selected_indices))
    f.write(f"{indices}\n")

    f.write("\nSelected double occupation expectation:\n")
    occupations = " ".join(f"{value.real:.10f}" for value in np.atleast_1d(model.t11_selected_occupation))
    f.write(f"{occupations}\n")

    if hasattr(model, "S2_diag") and model.S2_diag is not None:
        f.write("\nSelected S^2 diagnostics:\n")
        for idx in np.atleast_1d(model.t11_selected_indices):
            f.write(
                f"  idx {int(idx)}: "
                f"S2={model.S2_diag[int(idx)]:.10f}, "
                f"error={model.S2_error[int(idx)].real:.10e}\n"
            )


def write_projection_results_to_file(
    f,
    hole,
    class_idx,
    cluster_idx,
    rank,
    cluster_time,
    cluster,
    bonds,
    model,
) -> None:
    write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time)
    write_cluster_points(f, cluster)
    write_bond_structure(f, cluster, bonds)
    write_projection_diagnostics(f, model)


def cluster_save_results(
    result_dir: str,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    rank: int,
    cluster_time: float,
    cluster,
    bonds,
    coeffs,
    error,
    T11m1_norm: float,
    overlap,
) -> None:
    base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
    with open(f"{base_filename}_results.txt", 'w') as f:
        write_results_to_file(
            f,
            hole,
            class_idx,
            cluster_idx,
            rank,
            cluster_time,
            cluster,
            bonds,
            coeffs,
            error,
            T11m1_norm,
            overlap,
        )
    write_result_artifact(
        f"{base_filename}_results.txt",
        build_spin_coupling_artifact_from_coeffs(
            cluster,
            bonds,
            coeffs,
            error,
            T11m1_norm,
            overlap,
            metadata={
                "hole": hole,
                "class_idx": class_idx,
                "cluster_idx": cluster_idx,
                "rank": rank,
                "cluster_time_seconds": float(cluster_time),
            },
        ),
    )
    write_file(f"{base_filename}_results.txt", f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def cluster_save_projection_results(
    result_dir: str,
    hole: int,
    class_idx: int,
    cluster_idx: int,
    rank: int,
    cluster_time: float,
    cluster,
    bonds,
    model,
) -> None:
    base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
    with open(f"{base_filename}_results.txt", 'w') as f:
        write_projection_results_to_file(
            f,
            hole,
            class_idx,
            cluster_idx,
            rank,
            cluster_time,
            cluster,
            bonds,
            model,
        )
    write_result_artifact(
        f"{base_filename}_results.txt",
        build_projection_analysis_artifact(
            cluster,
            model,
            metadata={
                "hole": hole,
                "class_idx": class_idx,
                "cluster_idx": cluster_idx,
                "rank": rank,
                "cluster_time_seconds": float(cluster_time),
            },
        ),
    )
    write_file(f"{base_filename}_results.txt", f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# Solver
# ============================================================

def find_restart_path(spec: PathSpec, hole: int, class_idx: int) -> str:
    """查找 restart 文件路径"""
    path = f"{spec.data_dir}/hole{hole}_class{class_idx}"
    if not os.path.exists(f"{path}_eigvals.npy"):
        raise FileNotFoundError(f"The directory does not exist: {path}!\nPlease run it in advance...")
    return path


def cluster_process_work_item(hole: int, class_idx: int, clusters, params: Params, rank: int) -> None:
    params_cluster = {}
    params_cluster['hole'] = hole
    params_cluster['class_idx'] = class_idx
    params_cluster['rank'] = rank

    spec = params.get('path_spec')
    if spec is None:
        spec = build_path_spec(params)

    base_filename = f"{spec.output_dir}/hole{hole}_class{class_idx}"
    if not params['restart']:
        model = cluster_process(clusters.clusters_classified[hole][class_idx][0], params, params_cluster)
        np.save(f"{base_filename}_eigvals.npy", model.eigvals)
        np.save(f"{base_filename}_eigvecs.npy", model.eigvecs)
    else:
        restart_path = find_restart_path(spec, hole, class_idx)
        model = cluster_process(
            clusters.clusters_classified[hole][class_idx][0],
            params,
            params_cluster,
            restart_path,
        )

    model.save_data(base_filename)
    _save_results_for_model(model, clusters, params, spec.output_dir, hole, class_idx, rank)
    model.clear()
    del model


def _save_spin_coupling_results(
    model,
    clusters,
    params: Params,
    output_dir: str,
    hole: int,
    class_idx: int,
    rank: int,
) -> None:
    for cluster_idx in range(len(clusters.clusters_classified[hole][class_idx])):
        t0_cluster = time.time()

        cluster = clusters.clusters_classified[hole][class_idx][cluster_idx]
        bonds = bond_analysis_spin(cluster)
        coeffs, error = model.calc_spin_coeff(bonds, params['s2'])
        cluster_save_results(
            output_dir,
            hole,
            class_idx,
            cluster_idx,
            rank,
            time.time() - t0_cluster,
            cluster,
            bonds,
            coeffs,
            error,
            model.T11m1_norm,
            model.overlap,
        )


def _save_projection_only_results(model, clusters, output_dir: str, hole: int, class_idx: int, rank: int) -> None:
    for cluster_idx in range(len(clusters.clusters_classified[hole][class_idx])):
        t0_cluster = time.time()

        cluster = clusters.clusters_classified[hole][class_idx][cluster_idx]
        bonds = bond_analysis_spin(cluster)
        cluster_save_projection_results(
            output_dir,
            hole,
            class_idx,
            cluster_idx,
            rank,
            time.time() - t0_cluster,
            cluster,
            bonds,
            model,
        )


def _save_results_for_model(
    model,
    clusters,
    params: Params,
    output_dir: str,
    hole: int,
    class_idx: int,
    rank: int,
) -> None:
    if model.result_kind == RESULT_KIND_SPIN_COUPLINGS:
        _save_spin_coupling_results(model, clusters, params, output_dir, hole, class_idx, rank)
        return
    if model.result_kind == RESULT_KIND_PROJECTION_ANALYSIS:
        _save_projection_only_results(model, clusters, output_dir, hole, class_idx, rank)
        return
    raise ValueError(f"Unsupported result kind: {model.result_kind}")


def cluster_process(cluster, params: Params, params_cluster: dict, restart_path: str = None):
    """构建 Hubbard 模型，对角化，提取有效哈密顿量。"""
    mode_spec = resolve_mode_spec(params)
    model = Hubbard_SingleBand(params['N'], params['U'], params['t'])
    model.set_mode_spec(mode_spec)

    if params.get('match_spin_sectors'):
        if mode_spec.mode in (MODE_FULL, MODE_BLOCK_SZ_FULL):
            model.enable_match_spin_sectors()

    if params.get('path_spec') is not None:
        model.tmp_dir = params['path_spec'].tmp_dir

    bonds = bond_analysis_spin(cluster)
    if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 1:
        model.set_bonds_by_class(bonds[0], params['t'])
    if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 2:
        model.set_bonds_by_class(bonds[0], params['t2'])
    elif len(bonds) >= 2 and canonical_bond_type(cluster, bonds[1][0]) == 2:
        model.set_bonds_by_class(bonds[1], params['t2'])

    model.set_states(nsites=params['N'], nelec=params['N'], sz_set=mode_spec.sz)
    model.set_select_mode(params['select'])

    if model.if_block_szs2:
        model.save_blocks(params['N'])
        model.load_blocks(params['N'])
        model.construct_transform_matrix(params['N'])

    if params['restart']:
        model.restart(restart_path)
    else:
        model.calc_hamiltonian()
        model.solve()

    model.calc_s2()
    model.calc_heff_halffilled(params, params_cluster)
    return model


# ============================================================
# Main Function And Entry Point
# ============================================================

FORCE_DISTRIBUTE = True


def main(params):
    spec = setup_work_environment(params, rank=rank)
    params['path_spec'] = spec
    params['result_dir'] = spec.output_dir

    t0_total = time.time()
    clusters = None
    work_items = None

    if is_root():
        analyze_space(params['N'])
        clusters, work_items_all = prepare_clusters_and_tasks(params)
        work_items = filter_work_items(work_items_all, params, FORCE_DISTRIBUTE)
        print("Broadcasting clusters...")

    barrier()

    clusters = broadcast(clusters, root=0)
    work_items = broadcast(work_items, root=0)

    rank_work_items = distribute_work(work_items, rank, size)
    if is_root(size - 1):
        print(f"Restart from previous computation: {params['restart']}") if params['restart'] else print("Starting computation...")
        print(f"We adopt the \"{params['type']}\" scheme to calculate T11")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}")
        print(f"Up to now, time cost: {time.time() - t0_total:.1f}s\n")
        check_and_print_adiabatic_info(params)

    for work_idx, (hole, class_idx) in rank_work_items:
        t0 = time.time()
        cluster_process_work_item(hole, class_idx, clusters, params, rank)

        if is_root(size - 1):
            print(
                f"Rank {size - 1} completed task {work_idx + 1} / {len(rank_work_items)}, "
                f"time: {time.time() - t0:.1f}s",
                flush=True,
            )

    barrier()

    if is_root():
        print(f"\nTotal execution time: {time.time() - t0_total:.6f} seconds")

    barrier()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)

    params = parse_arguments(rank=rank)

    if is_root():
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using parameters: N={params['N']}, U={params['U']}, t={params['t']}, t2={params['t2']}, type={params['type']}, sz={params['sz']}, s={params['s']}, restart={params['restart']}")
        if not params["supports_spin_couplings"]:
            print("MODE=fixed_sz_s2 runs projection analysis only; spin couplings will not be reported.")
        if params.get("match_spin_sectors"):
            print("MATCH_SPIN_SECTORS=true: low-energy states will be selected with spin-sector multiplicity matching.")

    main(params)

    if is_root():
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
