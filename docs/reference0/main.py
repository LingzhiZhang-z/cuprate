from datetime import datetime
from mpi_config import comm, rank, size, is_root, barrier, broadcast, gather
from Hubbard_SingleBand import *
from Clusters_Square import *
from spin_operators import *
from utils_state import *
from utils_math import *
from utils_main import *
from utils_file import filter_work_items, check_and_print_adiabatic_info


# Monitoring configuration
FORCE_DISTRIBUTE = True

def main(params):
    # Setup environment
    # 使用新的 PathSpec 机制
    spec = setup_work_environment(params, rank=rank)
    params['path_spec'] = spec
    params['result_dir'] = spec.output_dir
    
    # Initialize timing and memory tracking
    monitor = Monitor(rank)
    t0_total = time.time()
    
    # Initialize variables for all ranks
    clusters = None
    work_items = None
    
    # Compute clusters on rank 0
    if is_root():
        analyze_space(params['N'])
        clusters, work_items_all = prepare_clusters_and_tasks(params)
        
        # 过滤需要 restart 的工作项，并将已完成的结果复制到 restart 目录
        work_items = filter_work_items(work_items_all, params, FORCE_DISTRIBUTE)
        
        monitor.record_memory("after_cluster_computation")
        monitor.record_time("after_cluster_computation", time.time() - t0_total)
        print("Broadcasting clusters...")
    
    # Ensure all ranks are ready to receive data
    barrier()
    
    # Broadcast data to all ranks
    # Rank 0 sends the data, other ranks receive it
    clusters = broadcast(clusters, root=0)
    work_items = broadcast(work_items, root=0)
    
    monitor.record_memory("after_broadcast")
    monitor.record_time("after_broadcast", time.time() - t0_total)
    
    # Process work items
    rank_work_items = distribute_work(work_items, rank, size)
    if is_root(size-1):
        print(f"Restart from previous computation: {params['restart']}") if params['restart'] else print("Starting computation...")
        print(f"We adopt the \"{params['type']}\" scheme to calculate T11")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}")
        print(f"Up to now, time cost: {time.time()-t0_total:.1f}s\n")
        check_and_print_adiabatic_info(params)
    
    for work_idx, (hole, class_idx) in rank_work_items:
        # Process work item
        t0 = time.time()
        cluster_process_work_item(hole, class_idx, clusters, params, rank)
        
        monitor.record_memory(f"class_{hole}_{class_idx}")
        monitor.record_time(f"class_{hole}_{class_idx}", time.time()-t0)
        
        # Print progress for rank 0
        if is_root(size-1):
            print(f"Rank {size-1} completed task {work_idx+1} / {len(rank_work_items)}, time: {time.time()-t0:.1f}s", flush=True)


    # Ensure all ranks have finished their work before gathering results
    barrier()
                    
    # Record final memory usage
    monitor.record_memory("final")
    monitor.record_time("final", time.time() - t0_total)
    
    # Save results (MPI gather must be called on ALL ranks)
    if is_root():
        print(f"\nTotal execution time: {time.time() - t0_total:.6f} seconds")

    monitor.save_with_total_time(params['result_dir'], time.time() - t0_total, comm)

    # Final barrier to ensure all ranks exit together
    barrier()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        sys.exit(0)
    
    params = parse_arguments(rank=rank)
    
    if is_root():
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using parameters: N={params['N']}, U={params['U']}, t={params['t']}, t2={params['t2']}, type={params['type']}, sz={params['sz']}, s={params['s']}, restart={params['restart']}")
    
    main(params)
    
    if is_root():
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


