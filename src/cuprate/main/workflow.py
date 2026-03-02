from typing import List, Tuple

from cuprate.clusters.square import Clusters_Square
from cuprate.main.params import Params


def prepare_clusters_and_tasks(params: Params) -> Tuple[Clusters_Square, List[Tuple[int, int]]]:
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


def distribute_work(work_items: list, rank: int, size: int) -> list:
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
