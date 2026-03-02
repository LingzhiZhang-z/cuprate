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
    base_dir = f"U{params['U']:.4f}_t{params['t']:.4f}"
    if params['t2'] is not None:
        base_dir = f"{base_dir}_tp{params['t2']:.4f}"

    # 2. 构建 run_dir (运行参数层)
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
        tmp_parent = os.path.dirname(spec.tmp_dir)
        os.makedirs(tmp_parent, exist_ok=True)

        if not params['restart']:
            os.makedirs(spec.output_dir, exist_ok=True)
            print(f"Working directory: {spec.output_dir}")
        else:
            os.makedirs(spec.restart_dir, exist_ok=True)
            print(f"Working directory (restart): {spec.restart_dir}")

    return spec

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

            base_path_abs = os.path.abspath(base_path)
            if base_path_abs.startswith(restart_dir_abs + os.sep):
                continue

            for src in glob.glob(f"{base_path}*"):
                shutil.copy2(src, restart_dir)
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
