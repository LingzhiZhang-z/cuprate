import numpy as np
import os
import glob
import shutil
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class PathSpec:
    """集中管理文件路径的配置结构"""
    work_dir: str        # e.g. Block
    base_dir: str        # e.g. U1.0000_t0.1000_tp0.0500
    data_dir: str        # e.g. N3_sz0.0_s0.0
    run_dir: str         # e.g. N3_sz0.0_s0.0_adiabatic_restart
    output_dir: str      # e.g. work_dir/base_dir/run_dir
    tmp_dir: str         # e.g. work_dir/base_dir/run_dir/tmp

def build_path_spec(params) -> PathSpec:
    """根据参数构建标准路径规范"""
    # 0. 构建 work_dir (工作目录)
    work_dir = "./Block"

    # 1. 构建 base_dir (物理参数层)
    # 例如: U1.0000_t0.1000
    # 如果 t2 存在，则添加 tp 后缀
    base_dir = f"U{params['U']:.4f}_t{params['t']:.4f}"
    if params['t2'] is not None:
        base_dir = f"{base_dir}_tp{params['t2']:.4f}"

    # 2. 构建 run_dir (运行参数层)
    # 例如: N3_sz0.0
    # 如果 sz 存在，则添加 sz 后缀
    # 如果 s 存在，则添加 s 后缀
    # 如果 type 存在，则添加 type 后缀, 同时 data_dir 不添加 type 后缀
    suffix_spin = ""
    if params['sz'] is not None:
        suffix_spin = f"_sz{params['sz']:.1f}"
    if params['s'] is not None:
        suffix_spin = f"_s{params['s']:.1f}"

    suffix_type = ""
    if params['type'] is not None:
        suffix_type = f"_{params['type']}"
    if params['restart']:
        suffix_type = f"{suffix_type}_restart"
    run_dir = f"N{params['N']}{suffix_spin}${suffix_type}"
    
    # 3. 组合完整路径
    output_dir = f"{work_dir}/{base_dir}/{run_dir}"
    
    # 4. 构建 tmp 前缀
    tmp_dir = f"{work_dir}/{base_dir}/{run_dir}/tmp"

    # 5. 构建 data_dir
    data_dir = f"{work_dir}/{base_dir}/N{params['N']}{suffix_spin}"
    
    return PathSpec(
        work_dir=work_dir,
        base_dir=base_dir,
        run_dir=run_dir,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        data_dir=data_dir
    )

def setup_params(U=1.0, t=0.1, t2=None, sz=None, s=None, type=None, restart=False):
    params = {}
    params['U'] = U
    params['t'] = t
    params['t2'] = t2
    params['sz'] = sz
    params['s'] = s
    params['type'] = type
    params['restart'] = restart
    return params


def setup_work_environment(params, rank=0) -> PathSpec:
    """Setup the working environment and parameters using PathSpec"""
    spec = build_path_spec(params)
    
    if rank == 0:
        # 1. 创建 tmp 目录结构
        # spec.tmp_dir 格式为 work_dir/base_dir/run_dir/tmp
        # 我们需要确保其所在的目录存在
        tmp_parent = os.path.dirname(spec.tmp_dir)
        os.makedirs(tmp_parent, exist_ok=True)
        
        # 2. 创建输出目录
        if not params['restart']:
            os.makedirs(spec.output_dir, exist_ok=True)
            print(f"Working directory: {spec.output_dir}")
        else:
            os.makedirs(spec.restart_dir, exist_ok=True)
            print(f"Working directory (restart): {spec.restart_dir}")

    return spec

    return None

def filter_work_items(work_items_all, params, force_distribute=True):
    """
    Filter work items based on restart conditions.
    If restart is enabled and not force_distribute, it checks previous results.
    Good results are copied to the restart directory, and only unfinished/bad items are returned.
    """
    if not params['restart'] or force_distribute:
        return work_items_all

    work_items = []
    spec = params.get('path_spec')
    if spec:
        restart_dir = spec.restart_dir
    else:
        # Fallback
        result_dir = params.get('result_dir', '')
        restart_dir = f"{result_dir}_restart"

    if spec:
        base_dir = f"{spec.work_dir}/{spec.base_dir}"
    else:
        base_dir = f"Block_U{params['U']:.4f}_t{params['t']:.4f}"
        if params.get('t2') is not None:
            base_dir = f"{base_dir}_tp{params['t2']:.4f}"

    restart_dir_abs = os.path.abspath(restart_dir)
    for idx, (hole, class_idx) in enumerate(work_items_all):
        # Construct search pattern for previous results
        # Using standard structure Block_U.../N...*/...
        pattern = f"{base_dir}/N{params['N']}*/hole{hole}_class{class_idx}_cluster0_results.txt"
        files = glob.glob(pattern)
        
        errors = []
        t11s = []
        for file in files:
            errors.append(read_key(file, "Relative") or 100000)
            t11s.append(read_key(file, "T11") or 100000)
        
        if not t11s:
             work_items.append(((hole, class_idx)))
             continue

        idx_selected = t11s.index(min(t11s))
        if not errors or errors[idx_selected] > 0.05:
            work_items.append(((hole, class_idx)))
        else:
            base_path = files[idx_selected].replace("_cluster0_results.txt", "")
            
            # Check self-copying loop
            base_path_abs = os.path.abspath(base_path)
            if base_path_abs.startswith(restart_dir_abs + os.sep):
                continue
            
            # Copy previous good results
            for src in glob.glob(f"{base_path}*"):
                shutil.copy2(src, restart_dir)
            # Original code deleted them
            for suffix in ("_eigvals.npy", "_eigvecs.npy"):
                for path in glob.glob(f"{restart_dir}/hole{hole}_class{class_idx}*{suffix}"):
                    os.remove(path)
            write_file(f"{restart_dir}/restart_flag.txt", f"cp {base_path}* ")
            
    return work_items


def check_and_print_adiabatic_info(params):
    if params['type'] != 'adiabatic':
        return
    if params['delta'] is None:
        raise ValueError(f"delta is not set")
    if params['delta2'] is None:
        params['delta2'] = 0.0

    t1_previous = params['t'] - params['delta']
    t2_previous = params['t2'] - params['delta2'] if params['t2'] is not None else None
    
    params_previous = setup_params(U=params['U'], t=t1_previous, t2=t2_previous, sz=params['sz'], s=params['s'], type=params['type'], restart=params['restart'])
    spec_previous = build_path_spec(params_previous)
    print(f"In the adiabatic process, read data from {spec_previous.data_dir}", flush=True)

def read_key(filename, key):
    """Read a specific key from a file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(key):
                    tmp = line.split()
                    return float(tmp[-1])
    except FileNotFoundError:
        return None
    return None

def write_file(filename, strings):
    with open(filename, "a") as f:
        f.write(f"{strings}\n")

def find_bond_vector(bond, cluster):
    """Find the bond vector of a bond"""
    site1, site2 = bond
    x1, y1 = cluster[site1]
    x2, y2 = cluster[site2]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    return sorted([dx, dy], reverse=True)

def write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time):
    """Write basic information about the calculation."""
    f.write(f"Hole: {hole}\n")
    f.write(f"Class: {class_idx}\n")
    f.write(f"Cluster: {cluster_idx}\n")
    f.write(f"Rank: {rank}\n")
    f.write(f"Computation time: {cluster_time:.6f} seconds\n")

def write_cluster_points(f, cluster):
    """Write cluster points information."""
    f.write("\n=== Cluster Points ===\n")
    for i, point in enumerate(cluster):
        f.write(f"Point {i}: {point}\n")

def get_all_possible_vectors(cluster, max_distance=None):
    """Get all possible bond vectors in the first quadrant below y=x."""
    if max_distance is None:
        max_distance = len(cluster)
    vectors = []
    for dx in range(max_distance):
        for dy in range(dx + 1):
            if dx == 0 and dy == 0:
                continue
            vectors.append((dx, dy))
    return sorted(vectors, key=lambda v: v[0]**2 + v[1]**2)


def write_coefficients(f, cluster, bonds, coeffs_list, error, T11m1_norm, overlap):
    """Write individual bond coefficients."""
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {coeffs_list[0].real:.10f}\n")
    
    # Create a dictionary to store coefficients by vector
    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in coeffs_by_vector:
                coeffs_by_vector[(dx, dy)] = []
            coeffs_by_vector[(dx, dy)].append((class_idx, bond_group, coeffs_list[class_idx+1]))
    
    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if (dx, dy) in coeffs_by_vector:
            for class_idx, bond_group, coeffs in coeffs_by_vector[(dx, dy)]:
                f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:\n")
                for idx, (bond, coef) in enumerate(zip(bond_group, coeffs)):
                    site1, site2 = bond
                    f.write(f"    {idx}: Sites {site1}-{site2}: {coef.real:.10f}  +  {coef.imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}: (not present in cluster)\n")
    
    f.write(f"\nFour-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 4:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//3+1} type {idx%3+1} : Four-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}): {coeffs_list[class_idx+1][idx].real:.10f}  +  {coeffs_list[class_idx+1][idx].imag:.10f}i\n")
                if (idx+1)%3==0:
                    f.write("\n")

    f.write(f"\nSix-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 6:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//15+1} type {idx%15+1} : Six-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}) * ({bond[4]}-{bond[5]}): {coeffs_list[class_idx+1][idx].real:.10f}  +  {coeffs_list[class_idx+1][idx].imag:.10f}i\n")
                if (idx+1)%15==0:
                    f.write("\n")
    
    """Write error information."""
    f.write("\n=== Individual Fit Error ===\n")
    f.write(f"Relative Error: {error[0]:.10f}\n")
    f.write(f"Residual: {error[1]:.10f}\n")
    f.write(f"R^2: {error[2]:.10f}\n")
    f.write(f"T11-1 norm: {T11m1_norm:.10f}\n")
    if overlap is not None:
        f.write(f"Overlap: {overlap:.10f}\n")
    


def write_errors(f, individual_error, class_error):
    """Write error information."""
    f.write("\n=== Individual Fit Error ===\n")
    np.savetxt(f, [individual_error])
    
    f.write("\n=== Class-Averaged Fit Error ===\n")
    np.savetxt(f, [class_error])

def write_bond_structure(f, cluster, bonds):
    """Write bond structure information."""
    f.write("\n=== Bond Structure ===")
    
    # Create a dictionary to store bonds by their vector
    bonds_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in bonds_by_vector:
                bonds_by_vector[(dx, dy)] = []
            bonds_by_vector[(dx, dy)].append((class_idx, bond_group))
    
    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:")
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
    
    # Output square bonds if any
    square_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 4]
    if square_bonds:
        f.write("\nFour-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            square_bonds_in_class = [bond for bond in bond_group if len(bond) == 4]
            if square_bonds_in_class:
                for idx, bond in enumerate(square_bonds_in_class):
                    f.write(f"    {idx//3+1} type {idx%3+1}: Four-site bond: {bond}\n")
                    if (idx+1)%3==0:
                        f.write("\n")

    # Output square bonds if any
    six_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 6]
    if six_bonds:
        f.write("\nSix-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            six_bonds_in_class = [bond for bond in bond_group if len(bond) == 6]
            if six_bonds_in_class:
                for idx, bond in enumerate(square_bonds_in_class):
                    f.write(f"    {idx//15+1} type {idx%15+1}: Six-site bond: {bond}\n")
                    if (idx+1)%15==0:
                        f.write("\n")

def write_results_to_file(f, hole, class_idx, cluster_idx, rank, cluster_time, 
                         cluster, bonds, coeffs, error, T11m1_norm, overlap):
    """Write all results to a file in a structured format."""
    write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time)
    write_cluster_points(f, cluster)
    write_bond_structure(f, cluster, bonds)
    write_coefficients(f, cluster, bonds, coeffs, error, T11m1_norm, overlap)


def read_previous(filename):
    eigvals = np.load(f"{filename}_eigvals.npy", allow_pickle=False)
    eigvecs = np.load(f"{filename}_eigvecs.npy", allow_pickle=False)
    selected_indices = np.load(f"{filename}_selected_indices.npy", dtype=int)
    return eigvals, eigvecs, selected_indices

def setup_work_environment_previous(params):
    if params['delta'] is None:
        raise ValueError(f"delta is not set")
    if params['delta2'] is None:
        params['delta2'] = 0.0
    t1_previous = params['t'] - params['delta']
    t2_previous = params['t2'] - params['delta2'] if params['t2'] is not None else None
            
    result_dir1 = f"Block_U{params['U']:.4f}_t{t1_previous:.4f}"
    if t2_previous is not None:
        result_dir1 = f"{result_dir1}_tp{t2_previous:.4f}"

    result_dir2 = f"N{params['N']}"
    result_dir3 = f"N{params['N']}"
    if params['sz'] is not None:
        result_dir2 = f"{result_dir2}_sz{params['sz']:.4f}"
        result_dir3 = f"{result_dir3}_sz{params['sz']:.4f}"
    if params['s2'] is not None:
        result_dir2 = f"{result_dir2}_s{params['s2']:.0f}"
        result_dir3 = f"{result_dir3}_s{params['s2']:.0f}"
    if params['type'] is not None:
        result_dir2 = f"{result_dir2}_{params['type']}"

    return result_dir1, result_dir2, result_dir3
