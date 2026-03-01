'''
This program is used to calculate the figure of double occupation
'''

import os
import sys
import time
import numpy as np
from mpi4py import MPI
from Clusters_Square import *
from utils_main import *

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Monitoring configuration
MONITOR_MEMORY = True  # Set to False to disable detailed memory monitoring
MONITOR_TIME = True    # Set to False to disable detailed timing monitoring


def setup_work_environment_coupling(params, rank=0):
    """Setup the working environment and parameters"""
    result_dir = f"Block_Plot/N{params['N']}"

    suffix1 = f""
    if params['sz'] is not None:
        suffix1 = f"{suffix1}_sz{params['sz']:.4f}"
    if params['s2'] is not None:
        suffix1 = f"{suffix1}_s{params['s2']:.0f}"

    suffix2 = f""
    if params['type'] is not None:
        suffix2 = f"{suffix2}_{params['type']}"
    if params['restart']:
        suffix2 = f"{suffix2}_restart"

    if rank == 0:
        os.makedirs(f"{result_dir}{suffix1}", exist_ok=True)
        os.makedirs(f"{result_dir}{suffix1}/data", exist_ok=True)
        print(f"Working directory: {result_dir}{suffix1} with suffix: {suffix2}")

    return result_dir, suffix1, suffix2

def main(params):
    # Setup environment
    result_dir, suffix1, suffix2 = setup_work_environment_coupling(params, rank)

    
    # Initialize variables for all ranks
    clusters = None
    work_items = None
    U = params['U']
    t_list = list(np.linspace(0.2050, 0.6000, 80))
    #t_list = list(np.linspace(0.0200, 0.6000, 30))
    x_list = [ t/U for t in t_list[1:]]
    
    # Compute clusters on rank 0
    if rank == 0:
        print("Computing clusters...")
        clusters = Clusters_Square(params['N'])
        clusters.compute_clsuters(if_print_time=True)
        clusters.classify_clusters(if_print_time=True)
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
    
    
    overlaps_list=[]
    for work_idx, (hole, class_idx) in rank_work_items:
        # Process work item
        t0 = time.time()
        overlaps=[]
        eigvecs_previous=None
        for idx in range(len(t_list)):
            t=float(t_list[idx])
            eig_filename = f"Block_dense/Block_U{U:.4f}_t{t:.4f}/N{params['N']}{suffix1}/hole{hole}_class{class_idx}_eigvecs.npy"
            occ_filename = f"Block_dense/Block_U{U:.4f}_t{t:.4f}/N{params['N']}{suffix1}{suffix2}/hole{hole}_class{class_idx}_t11_selected_indices.npy"
            try:
                eigvecs = np.load(eig_filename, allow_pickle=False)
            except:
                eigvecs = np.loadtxt(eig_filename, dtype=complex)
            occ_indices = np.atleast_1d(np.loadtxt(occ_filename, dtype=int))

            eigvecs_now = eigvecs[:, occ_indices]
            if eigvecs_previous is None:
                eigvecs_previous = eigvecs_now.copy()
                continue
            
            overlap = np.sum(np.abs(eigvecs_previous.conj().T @ eigvecs_now)**2)/len(occ_indices)
            eigvecs_previous = eigvecs_now.copy()
            overlaps.append(overlap)
        overlaps_list.append(overlaps)

        # Print progress for rank 0
        if rank == 0:
            print(f"Rank 0 completed task {work_idx+1} / {len(rank_work_items)}, time: {time.time()-t0:.1f}s")

    # Ensure all ranks have finished their work before gathering results
    comm.Barrier()
    gathered_overlaps = comm.gather((rank, overlaps_list), root=0)

    if rank == 0:
        gathered_overlaps.sort(key=lambda item: item[0])
        combined_overlaps = []
        for _, rank_overlaps in gathered_overlaps:
            combined_overlaps.extend(rank_overlaps)
        write_data(x_list, combined_overlaps, filename=f"{result_dir}{suffix1}/data/data_coupling{suffix2}.txt")
        plot_data(x_list, combined_overlaps, filename=f"{result_dir}{suffix1}/coupling{suffix2}.png", labels=None, xlims=[0.2000, 0.600], ylims=[0.0, 1.1])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        if rank == 0:
            print("Usage: python main.py [N=n] [U=u] [t=t]")
        sys.exit(0)
    
    params = parse_arguments()
    if rank == 0:
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using parameters: N={params['N']}, U={params['U']}, t={params['t']}")
    main(params)
    
