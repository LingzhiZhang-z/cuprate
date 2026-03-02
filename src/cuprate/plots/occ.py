'''
This program is used to calculate the figure of double occupation
'''

import os
import sys
import time
import numpy as np
from math import comb
from datetime import datetime
import matplotlib.pyplot as plt
from cuprate.mpi import comm, rank, size, is_root, barrier, broadcast
from cuprate.clusters.square import Clusters_Square
from cuprate.main.cli import parse_arguments
from cuprate.main.workflow import distribute_work
from cuprate.shared.io import setup_work_environment


# Monitoring configuration
MONITOR_MEMORY = True  # Set to False to disable detailed memory monitoring
MONITOR_TIME = True    # Set to False to disable detailed timing monitoring

def setup_work_environment_occ(params, rank=0):
    """Setup the working environment and parameters"""
    result_dir = f"Block_Plot/N{params['N']}"
    
    if params['sz'] is not None:
        result_dir = f"{result_dir}_sz{params['sz']:.4f}"
    if params['s2'] is not None:
        result_dir = f"{result_dir}_s{params['s2']:.0f}"
    if params['ratio'] is not None:
        result_dir = f"{result_dir}_ratio{params['ratio']:.4f}"

    suffix=""
    if params['type'] is not None:
        suffix = f"{suffix}_{params['type']}"
    if params['restart']:
        suffix = f"{suffix}_restart"

    if rank == 0:
        os.makedirs(f"{result_dir}", exist_ok=True)
        os.makedirs(f"{result_dir}/data", exist_ok=True)
        print(f"Working directory: {result_dir}")
    
    return result_dir, suffix


def write_data(x_list, data, filename):
    with open(f"{filename}", "w") as f:
        for idx in range(min(len(x_list), len(data))):
            f.write(f" {x_list[idx]:.6f}\t")
            for idx_level in range(len(data[idx])):
                f.write(f" {data[idx][idx_level].real:.6f}\t")
            f.write("\n")


def plot_same(x_list, data1, data2, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, filename="fig.png"):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        if xlim is None:
            xlim=(0.2000, 0.6000)
        if ylim is None:
            data_real = np.array(data1).real
            min_y, max_y = np.min(data_real), np.max(data_real)
            delta = max_y-min_y
            ylim=(min_y-0.1*delta, max_y+0.1*delta)

        if xlabel is None:
            xlabel="t/U"
        if ylabel is None:
            xlabel="y"
        if title is None:
            title="y of all states"

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        for idx_level in range(len(data1[0])):
            y_list = []
            for idx in range(len(x_list)):
                y_list.append(data1[idx][idx_level].real)
            ax.plot(x_list, y_list, label=f"level{idx_level}", linewidth=0.2, alpha=0.5)
            ax.scatter(x_list, y_list, s=1, marker="x", alpha=0.5)

        for idx_level in range(len(data2[0])):
            y_list = []
            for idx in range(len(x_list)):
                y_list.append(data2[idx][idx_level].real)
            ax.plot(x_list, y_list, label=f"level{idx_level}", linewidth=1.4)
            ax.scatter(x_list, y_list, s=16)
        fig.tight_layout()
        fig.savefig(f"{filename}", dpi=300)
        plt.close(fig)


def plot_diff(x_list, data1, data2, title1=None, title2=None, xlabel=None, ylabel=None, xlim=None, ylim=None, filename="fig.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 创建一行两列的子图
        
        if xlim is None:
            xlim=(0.2000, 0.6000)
        if ylim is None:
            data_real = np.array(data1).real
            min_y, max_y = np.min(data_real), np.max(data_real)
            delta = max_y-min_y
            ylim=(min_y-0.1*delta, max_y+0.1*delta)

        if xlabel is None:
            xlabel="t/U"
        if ylabel is None:
            xlabel="y"
        if title1 is None:
            title1="y of all states"
        if title2 is None:
            title2="y of selected states"

        ax1.set_title(title1)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])

        ax2.set_title(title2)
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylim(ylim[0], ylim[1])
        
        for idx_level in range(len(data1[0])):
            y_list = []
            for idx in range(len(x_list)):
                y_list.append(data1[idx][idx_level].real)
            ax1.plot(x_list, y_list, label=f"level{idx_level}", linewidth=0.8)
        #ax1.legend()

        for idx_level in range(len(data2[0])):
            y_list = []
            for idx in range(len(x_list)):
                y_list.append(data2[idx][idx_level].real)
            ax2.plot(x_list, y_list, label=f"level{idx_level}", linewidth=0.8)
        #ax2.legend()
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

def main(params):
    # Setup environment
    result_dir, suffix = setup_work_environment_occ(params, rank)
    if params['sz'] is None:
        dimspin=2**params['N']
    else:
        dimspin=comb(params['N'], int(params['N']/2))
    
    # Initialize variables for all ranks
    clusters = None
    work_items = None
    U=1.0
    t_list = list(np.linspace(0.2050, 0.6000, 80))
    
    # Compute clusters on rank 0
    if is_root():
        print("Computing clusters...")
        clusters = Clusters_Square(params['N'])
        clusters.compute_clsuters(t2=params['ratio'], if_print_time=True)
        clusters.classify_clusters(t2=params['ratio'], if_print_time=True)
        clusters.print_info()
        
        # Prepare work items
        work_items = [(hole, class_idx) 
                     for hole in range(len(clusters.clusters_classified))
                     for class_idx in range(len(clusters.clusters_classified[hole]))]
        print(f"\nPrepared {len(work_items)} work items for distribution")
        print("Broadcasting clusters...")
    
    # Ensure all ranks are ready to receive data
    barrier()
    
    # Broadcast data to all ranks
    # Rank 0 sends the data, other ranks receive it
    clusters = broadcast(clusters, root=0)
    work_items = broadcast(work_items, root=0)
    
    # Process work items
    rank_work_items = distribute_work(work_items, rank, size)
    
    if is_root(size-1):
        print("Starting computation...")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}\n")
    
    
    for work_idx, (hole, class_idx) in rank_work_items:
        # Process work item
        t0 = time.time()

        suffix_fig=""
        if params['sz'] is not None:
            suffix_fig=f"{suffix_fig}_sz{params['sz']:.4f}"
        if params['s2'] is not None:
            suffix_fig=f"{suffix_fig}_s{params['s2']:.0f}"
        if params['type'] is not None:
            suffix_fig=f"{suffix_fig}_{params['type']}"
        if params['restart']:
            suffix_fig=f"{suffix_fig}_restart"
        suffix_fig=f"{suffix_fig}_hole{hole}_class{class_idx}"


        eigvals_list=[]
        double_occ_list=[]
        eigvals_selected_list=[]
        double_occ_selected_list=[]
        x_list=[]
        for idx in range(len(t_list)):
            params['t'] = float(t_list[idx])
            if params['ratio'] is not None:
                params['t2'] = params['t'] * params['ratio']
            else:
                params['t2'] = None
            result_dir1, result_dir2, result_dir3 = setup_work_environment(params, rank=1)

            dir_occ=f"Block_dense/{result_dir1}/{result_dir2}"
            if params['restart']:
                dir_occ=f"{dir_occ}_restart"
            dir_eig=f"Block_dense/{result_dir1}/{result_dir3}"

            if not os.path.exists(f"{dir_occ}/timing_status.txt"):
                continue
            if not os.path.exists(f"{dir_eig}/timing_status.txt"):
                continue
            occ_filename = f"{dir_occ}/hole{hole}_class{class_idx}"
            eig_filename = f"{dir_eig}/hole{hole}_class{class_idx}"
            try:
                try:
                    eigvals=np.load(f"{eig_filename}_eigvals.npy", allow_pickle=False)/U
                except:
                    eigvals=np.loadtxt(f"{eig_filename}_eigvals.npy", dtype=float)/U
                double_occ=np.loadtxt(f"{occ_filename}_double_occupation_expectation.npy", dtype=complex)
                sorted_indices_occ=np.argsort(double_occ)

                selected_indices=np.loadtxt(f"{occ_filename}_t11_selected_indices.npy", dtype=float).astype(int)
                selected_indices_sorted=np.argsort(double_occ[selected_indices])
                selected_indices_sorted=selected_indices[selected_indices_sorted]

                eigvals_list.append((eigvals).copy())
                double_occ_list.append((double_occ[sorted_indices_occ]).copy())
                eigvals_selected_list.append((eigvals[selected_indices_sorted]).copy())
                double_occ_selected_list.append((double_occ[selected_indices_sorted]).copy())
                x_list.append(params['t']/U)
            except:
                continue

        #plot_diff(x_list, eigvals_list, eigvals_sorted_list, filename=f"{result_dir}/eigvals_diff_hole{hole}_class{class_idx}.png", ylabel="energy", title1="Energy of all states", title2="Energy of selected states")
        #plot_diff(x_list, double_occ_list, double_occ_sorted_list, filename=f"{result_dir}/double_occ_diff_hole{hole}_class{class_idx}.png", ylabel="sites with double_occ", title1="Sites with double_occ of all states", title2="Sites with double_occ of selected states")
        plot_same(x_list, eigvals_list, eigvals_selected_list, filename=f"{result_dir}/eigvals{suffix_fig}.png", ylabel="energy", title="Energy of all states")
        plot_same(x_list, double_occ_list, double_occ_selected_list, filename=f"{result_dir}/dc{suffix_fig}.png", ylabel="sites with double_occ", title="Sites with double_occ of all states")
        
        write_data(x_list, double_occ_list       , filename=f"{result_dir}/data/data_dc{suffix_fig}.txt")
        write_data(x_list, eigvals_list          , filename=f"{result_dir}/data/data_eigvals{suffix_fig}.txt")
        write_data(x_list, double_occ_selected_list, filename=f"{result_dir}/data/data_dc_selected{suffix_fig}.txt")
        write_data(x_list, eigvals_selected_list   , filename=f"{result_dir}/data/data_eigvals_selected{suffix_fig}.txt")
        

        # Print progress for rank 0
        if is_root(size-1):
            print(f"Rank {size-1} completed task {work_idx+1} / {len(rank_work_items)}, time: {time.time()-t0:.1f}s")

    # Ensure all ranks have finished their work before gathering results
    barrier()
                    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        if is_root():
            print("Usage: python main_occ.py [N=n] [type=type] [restart=restart]")
        sys.exit(0)
    
    params = parse_arguments()
    if is_root():
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using parameters: N={params['N']}")
    main(params)
    if is_root():
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


