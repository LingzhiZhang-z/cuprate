import os
import time
import numpy as np

from cuprate.models.hubbard import Hubbard_SingleBand
from cuprate.clusters.square import bond_analysis_spin, canonical_bond_type
from cuprate.shared.io import PathSpec, build_path_spec
from cuprate.main.params import Params
from cuprate.main.reporter import cluster_save_results


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

    # 获取 PathSpec
    spec = params.get('path_spec')
    if spec is None:
        spec = build_path_spec(params)

    try:
        # Process first cluster to get model
        base_filename = f"{spec.output_dir}/hole{hole}_class{class_idx}"
        if not params['restart']:
            model = cluster_process(clusters.clusters_classified[hole][class_idx][0], params, params_cluster)

            np.save(f"{base_filename}_eigvals.npy", model.eigvals)
            np.save(f"{base_filename}_eigvecs.npy", model.eigvecs)
        else:
            restart_path = find_restart_path(spec, hole, class_idx)
            model = cluster_process(clusters.clusters_classified[hole][class_idx][0], params, params_cluster, restart_path)

        model.save_data(base_filename)

        # Process all clusters in this class
        for cluster_idx in range(len(clusters.clusters_classified[hole][class_idx])):
            t0_cluster = time.time()

            bonds = bond_analysis_spin(clusters.clusters_classified[hole][class_idx][cluster_idx])
            coeffs, error = model.calc_spin_coeff(bonds, params['s2'])
            # Save individual cluster results
            cluster_save_results(spec.output_dir, hole, class_idx, cluster_idx, rank,
                               time.time() - t0_cluster,
                               clusters.clusters_classified[hole][class_idx][cluster_idx],
                               bonds,
                               coeffs, error, model.T11m1_norm, model.overlap)
        # Clean up
        model.clear()
        del model
    except Exception as e:
        raise RuntimeError(f"Failed to process work item: {str(e)}")


def cluster_process(cluster, params: Params, params_cluster: dict, restart_path: str = None):
    try:
        model = Hubbard_SingleBand(params['N'], params['U'], params['t'])

        # 注入 PathSpec 中的 tmp_dir 以确保路径一致
        if params.get('path_spec') is not None:
            model.tmp_dir = params['path_spec'].tmp_dir

        bonds = bond_analysis_spin(cluster)
        # Setting the type of bonds, if None do nothing
        if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 1:
            model.set_bonds_by_class(bonds[0], params['t'])

        if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 2:
            model.set_bonds_by_class(bonds[0], params['t2'])
        elif len(bonds) >= 2 and canonical_bond_type(cluster, bonds[1][0]) == 2:
            model.set_bonds_by_class(bonds[1], params['t2'])

        model.set_block(params['block'])
        model.set_states(nsites=params['N'], nelec=params['N'], sz_set=params['sz'])
        model.load_blocks(params['N'])
        model.construct_transform_matrix(params['N'])

        if params['restart']:
            model.restart(restart_path)
        else:
            model.calc_hamiltonian()
            model.solve(model.s2_Us)

        model.calc_s2()
        model.calc_heff_halffilled(params, params_cluster)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to process cluster: {str(e)}")
