'''
This program is used to calculate the figure of double occupation
'''

import os
import sys
import time
import numpy as np
from math import comb
from mpi4py import MPI
from datetime import datetime
import matplotlib.pyplot as plt
from Clusters_Square import *
from utils_main import parse_arguments, distribute_work, setup_work_environment, read_key


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Monitoring configuration
MONITOR_MEMORY = True  # Set to False to disable detailed memory monitoring
MONITOR_TIME = True    # Set to False to disable detailed timing monitoring

def setup_work_environment_norm(params, rank=0):
    """Setup the working environment and parameters"""
    result_dir = f"Block_Plot/N{params['N']}"
    if params['ratio'] is not None:
        result_dir = f"{result_dir}_ratio{params['ratio']:.4f}"
    
    if params['sz'] is not None:
        result_dir = f"{result_dir}_sz{params['sz']:.4f}"
    if params['s2'] is not None:
        result_dir = f"{result_dir}_s{params['s2']:.0f}"

    suffix=""
    if params['type'] is not None:
        suffix = f"{suffix}_{params['type']}"
    if params['restart']:
        suffix = f"{suffix}_restart"

    if rank == 0:
        os.makedirs(f"{result_dir}", exist_ok=True)
        print(f"Working directory: {result_dir}")
    
    
    return result_dir, suffix


def main(params):
    # Setup environment
    result_dir, suffix = setup_work_environment_norm(params)
    if params['sz'] is None:
        nstate=2**params['N']
    else:
        nstate=comb(params['N'], int(params['N']/2))

    # Initialize variables for all ranks
    clusters = None
    work_items = None
    # Compute clusters on rank 0
    if rank == 0:
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
    comm.Barrier()
    
    # Broadcast data to all ranks
    # Rank 0 sends the data, other ranks receive it
    clusters = comm.bcast(clusters, root=0)
    work_items = comm.bcast(work_items, root=0)
    
    # Process work items
    rank_work_items = distribute_work(work_items, rank, size)
    
    if rank == 0:
        print("Starting computation...")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}\n")
    
    U=1.0
    t_list = list(np.linspace(0.2050, 0.6000, 80))
    errors_list=[]
    t11s_list=[]
    overlaps_list=[]
    for work_idx, (hole, class_idx) in rank_work_items:
        # Process work item
        t0 = time.time()
        x_list=[]
        errors=[]
        t11s=[]
        overlaps=[]
        for idx in range(len(t_list)):
            params['t'] = float(t_list[idx])
            if params['ratio'] is not None:
                params['t2'] = params['t'] * params['ratio']
            data_dir1, data_dir2, data_dir3 = setup_work_environment(params, 1)
            if params['restart']:
                data_dir2 = f"{data_dir2}_restart"
            base_filename = f"Block_dense/{data_dir1}/{data_dir2}/hole{hole}_class{class_idx}_cluster0_results.txt"
            if os.path.exists(base_filename):
                try:
                    errors.append(read_key(base_filename, "Relative"))
                    t11s.append(read_key(base_filename, "T11")/nstate)
                    if params['type'] == "adiabatic":
                        overlaps.append(read_key(base_filename, "Overlap"))
                    x_list.append(params['t']/U)
                except:
                    print(f"Fail to read results from {base_filename}.")
                    continue
        errors_list.append(errors)
        t11s_list.append(t11s)
        overlaps_list.append(overlaps)

        # Print progress for rank 0
        if rank == 0:
            print(f"Rank 0 completed task {work_idx+1} / {len(rank_work_items)}, time: {time.time()-t0:.1f}s")

    # Ensure all ranks have finished their work before gathering results
    comm.Barrier()

    # Save data to files
    if rank == 0:
        # Save t11s data with x in first column
        with open(f"{result_dir}/t11s_data{suffix}.txt", "w") as f:
            f.write("# T11-1 values\n")
            for idx in range(len(t11s_list)):
                work_idx, (hole, class_idx) = rank_work_items[idx]
                f.write(f"# hole{hole}_class{class_idx}")
            f.write("\n")
            
            for i, x in enumerate(x_list):
                f.write(f"{x:.6f}")
                for idx in range(len(t11s_list)):
                    if i < len(t11s_list[idx]):
                        f.write(f" {t11s_list[idx][i]:.6f}")
                    else:
                        f.write(" NaN")
                f.write("\n")
        
        # Save errors data with x in first column
        with open(f"{result_dir}/errors_data{suffix}.txt", "w") as f:
            f.write("# Error values\n")
            for idx in range(len(errors_list)):
                work_idx, (hole, class_idx) = rank_work_items[idx]
                f.write(f"# hole{hole}_class{class_idx}")
            f.write("\n")
            
            for i, x in enumerate(x_list):
                f.write(f"{x:.6f}")
                for idx in range(len(errors_list)):
                    if i < len(errors_list[idx]):
                        f.write(f" {errors_list[idx][i]*100:.6f}")
                    else:
                        f.write(" NaN")
                f.write("\n")
        
        print(f"Data saved to {result_dir}{suffix}")

    # Extract real parts and find global min/max
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    max_y = np.max(np.array(t11s_list))
    min_y = np.min(np.array(t11s_list))
    delta=max_y-min_y
    max_y = max_y+0.1*delta
    min_y = min_y-0.1*delta
    for idx in range(len(errors_list)):
        work_idx, (hole, class_idx)=rank_work_items[idx]
        ax1.plot(x_list, t11s_list[idx], label=f"hole{hole}_class{class_idx}", linewidth=1.0)
        ax1.scatter(x_list, t11s_list[idx], linewidth=1.0)

    ax1.set_title("T11-1 of all clusters")
    ax1.set_xlabel("t/U")
    ax1.set_ylabel("T11-1")
    ax1.set_xlim(0.2000, 0.6000)
    ax1.set_ylim(0.0, max_y+0.1*delta)
    
    for idx in range(len(errors_list)):
        work_idx, (hole, class_idx)=rank_work_items[idx]
        ax2.plot(x_list, errors_list[idx], label=f"hole{hole}_class{class_idx}", linewidth=1.0)
        ax2.scatter(x_list, errors_list[idx], linewidth=1.0)

    ax2.set_title("Error of all clusters")
    ax2.set_xlabel("t/U")
    ax2.set_ylabel("Error")
    ax2.set_xlim(0.2000, 0.6000)
    ax2.set_ylim(0.0, 1.10)
    fig.savefig(f"{result_dir}/t11{suffix}.png", dpi=300, bbox_inches='tight')

    
    # Create a single legend for both plots
    #handles, labels = ax1.get_legend_handles_labels()
    # Calculate how many columns to fit the legend width to match the subplots
    #n_cols = max(1, len(handles) // 2)  # Adjust this number to control legend width
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=n_cols)
    
    # Adjust layout to ensure legend width matches subplot width
    #plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9)  # Leave space for legend at bottom
    plt.close(fig)
                    



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        if rank == 0:
            print("Usage: python main.py [N=n] [U=u] [t=t]")
        sys.exit(0)
    
    params = parse_arguments()
    if rank == 0:
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using parameters: N={params['N']}")
    main(params)
    if rank==0:
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


