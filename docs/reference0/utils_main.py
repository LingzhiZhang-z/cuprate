import os
import sys
import time
import glob
import psutil
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from datetime import datetime
from Hubbard_SingleBand import *
from Clusters_Square import *
from spin_operators import *
from utils_state import *
from utils_math import *
from utils_file import *

#
# 参数解析（科研脚本常用）：保留在 utils_main，方便其它脚本直接复用
#
DEFAULT_N = 3
DEFAULT_U = 1.0
DEFAULT_T = 0.1

def parse_arguments(rank=0):
    params = {}
    params['N'] = DEFAULT_N
    params['U'] = DEFAULT_U
    params['t'] = DEFAULT_T
    params['t2'] = None
    params['t3'] = None
    params['sz'] = None
    params['s2'] = None
    params['s2_fix'] = False
    params['type'] = None
    params['delta'] = None
    params['delta2'] = None
    params['restart'] = False
    params['type_delta'] = None

    params['Ncell'] = None
    params['Ncut'] = None
    params['ratio'] = None

    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, value = arg.split('=', 1)
        key = key.upper()
        if key == 'N'.upper():
            params['N'] = int(value)
        elif key == 'U'.upper():
            params['U'] = float(value)
        elif key == 'T'.upper():
            params['t'] = float(value)
        elif key == 'T2'.upper():
            params['t2'] = float(value)
        elif key == 'T3'.upper():
            params['t3'] = float(value)
        elif key == 'SZ'.upper():
            params['sz'] = float(value)
        elif key == 'S2'.upper():
            params['s2'] = float(value)
        elif key == 'S2_FIX'.upper():
            params['s2_fix'] = bool(value)
        elif key == 'TYPE'.upper():
            params['type'] = value
        elif key == 'DELTA'.upper():
            params['delta'] = float(value)
        elif key == 'DELTA2'.upper():
            params['delta2'] = float(value)
        elif key == 'RESTART'.upper():
            params['restart'] = bool(value)
        elif key == 'TYPE_DELTA'.upper():
            params['type_delta'] = value
        elif key == 'NCELL'.upper():
            params['Ncell'] = int(value)
        elif key == 'NCUT'.upper():
            params['Ncut'] = int(value)
        elif key == 'RATIO'.upper():
            params['ratio'] = float(value)

    # 统一 sz（若未指定则自动调节/选择）
    params['sz'] = tune_sz(params['sz'], params['N'], rank=rank)
    return params

class Monitor:
    def __init__(self, rank):
        self.rank = rank
        self.memory_records = [] # 存 (stage_name, memory_gb)
        self.time_records = []   # 存 (stage_name, duration)
        
        # 初始记录
        self.record_memory("initial")

    def record_memory(self, stage_name):
        """记录当前内存使用"""
        usage = get_memory_usage() # 复用现有的函数
        self.memory_records.append((stage_name, usage))

    def record_time(self, stage_name, duration):
        """记录计算耗时"""
        self.time_records.append((stage_name, duration))

    def save_with_total_time(self, path, total_time, comm, root=0):
         """
         收集所有 Rank 的数据并在 Root 节点保存文件
         """
         # Gather data
         all_memory = comm.gather(self.memory_records, root=root)
         all_times = comm.gather(self.time_records, root=root)
         
         if self.rank == root:
             write_timing_status(path, all_times, total_time)
             write_memory_status(path, all_memory)
             print("Timing and Memory statistics written.")

def analyze_space(N):
    Smax=N*0.5
    dim = comb(2*N, N)
    dim_eff = 2**N
    n_half = N * 0.5


    Nups=[ N-i for i in range(N+1)]
    Ndos=[ i   for i in range(N+1)]
    sz_states = []
    sz_s2_states = []
    for idx_sz in range(N+1):
        sz = (Nups[idx_sz]-Ndos[idx_sz])*0.5
        dim_sz = comb_safe(N, Nups[idx_sz]) * comb_safe(N, Ndos[idx_sz])
        dim_eff_sz = comb_safe(N, Nups[idx_sz])

        sum_dim_sz_s = 0
        sum_dim_eff_sz_s = 0
        for idx_s in range(int(Smax-abs(sz))+1):
            s=abs(sz)+idx_s
            s2 = s * (s + 1)
            dim_sz_s = comb_safe(N, int(n_half+s)) * comb_safe(N, int(n_half-s)) - comb_safe(N, int(n_half+s+1)) * comb_safe(N, int(n_half-s-1))
            dim_eff_sz_s = comb_safe(N, int(n_half-s)) - comb_safe(N, int(n_half-s-1))
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

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def write_memory_status(path, all_memory):
    """Write memory usage statistics to a file"""
    with open(os.path.join(path, "memory_status.txt"), "w") as f:
        f.write("Memory Usage Statistics (MB)\n")
        f.write("=" * 50 + "\n\n")
        
        for rank_idx, rank_memory in enumerate(all_memory):
            f.write(f"Rank {rank_idx} Memory Usage:\n")
            f.write("-" * 30 + "\n")
            for stage, memory in rank_memory:
                f.write(f"{stage}: {memory:.2f} MB\n")
            f.write("\n")
                
            # Calculate memory differences
            if len(rank_memory) > 1:
                f.write("Memory Changes:\n")
                for i in range(1, len(rank_memory)):
                    prev_stage, prev_memory = rank_memory[i-1]
                    curr_stage, curr_memory = rank_memory[i]
                    diff = curr_memory - prev_memory
                    f.write(f"{prev_stage} -> {curr_stage}: {diff:+.2f} MB\n")
            f.write("\n")

def write_timing_status(path, all_times, total_time):
    """Write timing statistics to a file"""
    with open(os.path.join(path, "timing_status.txt"), "w") as f:
        f.write("Timing Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Write total execution time
        f.write(f"Total execution time: {total_time:.6f} seconds\n\n")
        
        # Write per-rank timing information
        for rank_idx, rank_times in enumerate(all_times):
            f.write(f"Rank {rank_idx} Timing:\n")
            f.write("-" * 30 + "\n")
            for stage, time in rank_times:
                f.write(f"{stage}: {time:.6f} seconds\n")
            f.write("\n")

def prepare_clusters_and_tasks(params):
    print("Computing clusters...")
    clusters = Clusters_Square(params['N'])
    clusters.compute_clsuters(t2=params['t2'], if_print_time=True)
    clusters.classify_clusters(t2=params['t2'], if_print_time=True)
    clusters.print_info()
    
    # Prepare work items
    work_items_all = [(hole, class_idx) 
                 for hole in range(len(clusters.clusters_classified))
                 for class_idx in range(len(clusters.clusters_classified[hole]))]
    print(f"\nPrepared {len(work_items_all)} work items for distribution")
    
    return clusters, work_items_all

def distribute_work(work_items, rank, size):
    """Distribute work items among MPI ranks using round-robin with extra tasks to later ranks"""
    total_items = len(work_items)
    items_per_rank = total_items // size
    remainder = total_items % size
    
    # Use round-robin distribution for the base items
    work_items_for_rank = []
    base_items = items_per_rank * size  # Number of items distributed in round-robin
    
    # Add base round-robin items
    for i in range(rank, base_items, size):
        work_items_for_rank.append((i, work_items[i]))
    
    # Add extra items to later ranks if there are remainder items
    if remainder > 0 and rank >= size - remainder:
        extra_item_idx = base_items + (rank - (size - remainder))
        work_items_for_rank.append((extra_item_idx, work_items[extra_item_idx]))
    
    return work_items_for_rank


def write_data(x_list, data, filename):
    nx=len(x_list)
    ny=len(data)
    with open(filename, "w") as f:
        for i in range(nx):
            f.write(f"{x_list[i]:.6f}")
            for j in range(ny):
                f.write(f" {data[j][i]:.12f}")
            f.write("\n")

def plot_data(x_list, data, filename, labels=None, ylims=None, xlims=None):
    if labels is None:
        labels = [f"hole{idx}" for idx in range(len(data))]
    if xlims is None:
        xlims = [0.0000, 1.000]
    if ylims is None:
        ylims = [0.0, 1.0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for idx in range(len(data)):
        ax.plot(x_list, data[idx], label=labels[idx], linewidth=1.0)
        ax.scatter(x_list, data[idx], linewidth=1.0)
    ax.set_title("Overlap of all clusters")
    ax.set_xlabel("t/U")
    ax.set_ylabel("Overlap")
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def find_restart_path(spec: PathSpec, hole, class_idx) -> str:
    """查找 restart 文件路径"""
    path = f"{spec.data_dir}/hole{hole}_class{class_idx}"
    if not os.path.exists(f"{path}_eigvals.npy"):
        raise FileNotFoundError(f"The directory does not exist: {path}!\nPlease run it in advance...")
    return path

def cluster_process_work_item(hole, class_idx, clusters, params, rank):
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

def cluster_process(cluster, params, params_cluster, restart_path=None):
    try:
        model = Hubbard_SingleBand(params['N'], params['U'], params['t'])
        
        # 注入 PathSpec 中的 tmp_dir 以确保路径一致
        if 'path_spec' in params:
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

def cluster_save_results(result_dir, hole, class_idx, cluster_idx, rank, cluster_time, 
                        cluster, bonds, 
                        coeffs, error, T11m1_norm, overlap):
    try:
        base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
        with open(f"{base_filename}_results.txt", 'w') as f:
            write_results_to_file(f, hole, class_idx, cluster_idx, rank, cluster_time, 
                                cluster, bonds, coeffs, error, T11m1_norm, overlap)
        write_file(f"{base_filename}_results.txt", f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        raise IOError(f"Failed to save cluster results: {str(e)}")
