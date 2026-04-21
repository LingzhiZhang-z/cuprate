"""Solver-side helpers for the main Hubbard workflow."""

from __future__ import annotations

import os
import time

import numpy as np

from cuprate import downfolding, sectors
from cuprate.clusters import build_spin_operator_catalog, generate_bonds
from cuprate.hubbard import HubbardModel
from cuprate.io import (
    MODE_BLOCK_SZ_FULL,
    MODE_FULL,
    PathSpec,
    RESULT_KIND_PROJECTION_ANALYSIS,
    RESULT_KIND_SPIN_COUPLINGS,
    Params,
)

from .reporting import (
    save_projection_result,
    save_spin_coupling_result,
)


def cluster_process_work_item(
    hole: int,
    class_idx: int,
    clusters,
    params: Params,
    spec: PathSpec,
    mode_spec,
    rank: int,
) -> list[dict]:
    params_cluster = {"hole": hole, "class_idx": class_idx, "rank": rank}
    cluster_family = clusters.clusters_classified[hole][class_idx]

    base_filename = f"{spec.output_dir}/hole{hole}_class{class_idx}"
    data_filename = f"{spec.data_dir}/hole{hole}_class{class_idx}"
    if not params.restart:
        model = cluster_process(cluster_family[0], params, params_cluster, spec, mode_spec)
        np.save(f"{data_filename}_eigvals.npy", model.eigvals)
        np.save(f"{data_filename}_eigvecs.npy", model.eigvecs)
    else:
        restart_path = data_filename
        if not os.path.exists(f"{restart_path}_eigvals.npy"):
            raise FileNotFoundError(
                f"The directory does not exist: {restart_path}!\nPlease run it in advance..."
            )
        model = cluster_process(
            cluster_family[0],
            params,
            params_cluster,
            spec,
            mode_spec,
            restart_path,
        )

    model.save_data(base_filename)
    cluster_entries = []
    for cluster_idx, cluster in enumerate(cluster_family):
        t0_cluster = time.time()
        spin_operator_catalog = build_spin_operator_catalog(cluster)
        if model.result_kind == RESULT_KIND_SPIN_COUPLINGS:
            coeffs, error = downfolding.calc_spin_coeff(
                model.downfold.heff,
                model.states[:model.dimspin],
                model.dimspin,
                spin_operator_catalog,
            )
            cluster_entries.append(
                save_spin_coupling_result(
                    spec.output_dir,
                    hole,
                    class_idx,
                    cluster_idx,
                    rank,
                    time.time() - t0_cluster,
                    cluster,
                    spin_operator_catalog,
                    coeffs,
                    error,
                    model.downfold.t11m1_norm,
                    model.downfold.overlap,
                )
            )
        elif model.result_kind == RESULT_KIND_PROJECTION_ANALYSIS:
            cluster_entries.append(
                save_projection_result(
                    spec.output_dir,
                    hole,
                    class_idx,
                    cluster_idx,
                    rank,
                    time.time() - t0_cluster,
                    cluster,
                    spin_operator_catalog,
                    model,
                )
            )
        else:
            raise ValueError(f"Unsupported result kind: {model.result_kind}")

    model.clear()
    return cluster_entries


def cluster_process(
    cluster,
    params: Params,
    params_cluster: dict,
    spec: PathSpec,
    mode_spec,
    restart_path: str = None,
):
    """Build the Hubbard model, diagonalise it, and extract the effective Hamiltonian."""
    model = HubbardModel(params.N, params.U, params.t)
    model.set_mode_spec(mode_spec)

    if params.match_spin_sectors:
        if mode_spec.mode in (MODE_FULL, MODE_BLOCK_SZ_FULL):
            model.enable_match_spin_sectors()

    model.tmp_dir = spec.tmp_dir

    nn_bonds = generate_bonds(cluster)[0]
    model.add_hopping_bonds(nn_bonds, params.t)

    model.set_states(nsites=params.N, nelec=params.N, twoSz_set=mode_spec.twoSz)
    model.set_select_mode(params.select)

    if model.use_S2_blocks:
        model.load_blocks()
        sectors.construct_transform_matrix(params.N, model.sz_sectors, model.S2_sectors)

    if params.restart:
        model.eigvals = np.load(f"{restart_path}_eigvals.npy", allow_pickle=False)
        model.eigvecs = np.load(f"{restart_path}_eigvecs.npy", allow_pickle=False)
    else:
        model.calc_hamiltonian()
        model.solve()

    model.calc_S2()
    model.calc_heff_halffilled(params, params_cluster)
    return model
