import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import networkx as nx

POINT_GROUP_OPERATIONS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "mirror_x",
    "mirror_y",
    "mirror_diag",
    "mirror_anti",
)


def get_neighbors(pos):
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _is_connected(sites, indices):
    """True if sites[i] for i in indices form a connected NN subset."""
    subset = {sites[i] for i in indices}
    start = sites[indices[0]]
    visited = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in get_neighbors(current):
            if neighbor in subset and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(indices)


def _perfect_matchings(items):
    """Enumerate all (2n-1)!! perfect matchings of a 2n-length sequence."""
    if len(items) == 0:
        return [[]]
    first = items[0]
    result = []
    for k in range(1, len(items)):
        rest = items[1:k] + items[k+1:]
        for sub in _perfect_matchings(rest):
            result.append([(first, items[k])] + sub)
    return result


def cluster_label(hole: int, class_idx: int, cluster_idx: int | None = None) -> str:
    label = f"hole{hole}_class{class_idx}"
    if cluster_idx is not None:
        label += f"_idx{cluster_idx}"
    return label


@dataclass(frozen=True)
class Cluster:
    """Single cluster: geometry (sites, bonds) plus enumeration tags (hole, class_idx)."""
    sites: tuple[tuple[int, int], ...]
    bonds: tuple[tuple[int, int], ...]
    hole: int = None
    class_idx: int = None
    cluster_idx: int = None

    @property
    def N(self) -> int:
        return len(self.sites)

    def label(self) -> str:
        return cluster_label(self.hole, self.class_idx, self.cluster_idx)

    def generate_bonds(self, N: int, is_connected: bool) -> list[list[Sequence[int]]]:
        """Bond groups of N-site operators. One singleton group per (subset, pairing)."""
        sites = list(self.sites)
        subsets = list(combinations(range(len(sites)), N))
        if is_connected:
            subsets = [s for s in subsets if _is_connected(sites, s)]
        return [
            [[s for pair in matching for s in pair]]
            for subset in subsets
            for matching in _perfect_matchings(subset)
        ]


class ClusterSets:
    """Collection of Cluster instances: enumerates and classifies by isomorphism."""

    def __init__(self, N: int):
        self.N = N
        self.clusters: list[Cluster] = []

    def generate(self) -> "ClusterSets":
        all_coords = generate_clusters(self.N)
        clusters_by_hole = clusters_sort_hole(all_coords)
        for hole, same_hole_clusters in enumerate(clusters_by_hole):
            iso_classes = classify_isomorphic_clusters(same_hole_clusters)
            for class_idx, iso_class in enumerate(iso_classes):
                for cluster_idx, coords in enumerate(iso_class):
                    sites = tuple(tuple(map(int, s)) for s in coords)
                    bond_classes = classify_two_site_bonds(list(sites))
                    bonds = tuple(tuple(b) for b in bond_classes[0][1]) if bond_classes else ()
                    self.clusters.append(Cluster(
                        sites=sites, bonds=bonds,
                        hole=hole, class_idx=class_idx, cluster_idx=cluster_idx,
                    ))
        return self

    def representatives(self) -> list[Cluster]:
        """First cluster of each (hole, class_idx) equivalence class."""
        seen: set[tuple[int, int]] = set()
        reps: list[Cluster] = []
        for c in self.clusters:
            key = (c.hole, c.class_idx)
            if key not in seen:
                seen.add(key)
                reps.append(c)
        return reps

    def print_info(self) -> None:
        counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for c in self.clusters:
            counts[c.hole][c.class_idx] += 1

        print(f"Size of clusters: {len(self.clusters)} for {self.N} sites.")
        total_classes = 0
        for hole in sorted(counts):
            k = len(counts[hole])
            total_classes += k
            print(f"Holes: {hole} , classification number: {k}")

        if total_classes == 0:
            print("No classification has been done!")
        else:
            print(f"Total number of classification: {total_classes}\n")

        for hole in sorted(counts):
            for class_idx in sorted(counts[hole]):
                print(f"Holes: {hole} , classification: {class_idx} , number: {counts[hole][class_idx]}")

    def write(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        for c in self.clusters:
            with open(os.path.join(dir_path, f"{c.label()}.txt"), "w") as f:
                for i, (x, y) in enumerate(c.sites):
                    f.write(f"{i}: {x} {y}\n")

    def plot(self, dir_path: str) -> None:
        import matplotlib.pyplot as plt

        os.makedirs(dir_path, exist_ok=True)
        for c in self.clusters:
            sites = c.sites
            N = len(sites)
            min_range, max_range = -1, N

            fig, ax = plt.subplots()
            ax.set_xticks(range(min_range, max_range))
            ax.set_yticks(range(min_range, max_range))
            ax.grid(True, which="both", axis="both", color="gray", linestyle="--", linewidth=0.5)
            ax.set_xlim(min_range, max_range)
            ax.set_ylim(min_range, max_range)
            ax.axhline(0, color="gray", linewidth=1.0)
            ax.axvline(0, color="gray", linewidth=1.0)
            ax.set_aspect("equal")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", which="both", length=0)

            for i, (x, y) in enumerate(sites):
                ax.plot(x, y, "o", color="black")
                ax.text(x + 0.05, y + 0.05, f"{i}", fontsize=12)

            for i, j in c.bonds:
                x1, y1 = sites[i]
                x2, y2 = sites[j]
                ax.plot([x1, x2], [y1, y2], "-", color="black", linewidth=2.0)

            fig.savefig(os.path.join(dir_path, f"{c.label()}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig)


def count_holes(cluster):
    cluster_set = set(cluster)
    hole_count = 0
    for x, y in cluster:
        square = [
            (x, y),
            (x + 1, y),
            (x, y + 1),
            (x + 1, y + 1),
        ]
        if all(pt in cluster_set for pt in square):
            hole_count += 1
    return hole_count


def transform(cluster, op):
    try:
        transform_point = {
            "id": lambda x, y: (x, y),
            "rot90": lambda x, y: (-y, x),
            "rot180": lambda x, y: (-x, -y),
            "rot270": lambda x, y: (y, -x),
            "mirror_x": lambda x, y: (x, -y),
            "mirror_y": lambda x, y: (-x, y),
            "mirror_diag": lambda x, y: (y, x),
            "mirror_anti": lambda x, y: (-y, -x),
        }[op]
    except KeyError as exc:
        raise ValueError("Unknown transformation operation")
    return [transform_point(x, y) for x, y in cluster]


def _normalize_cluster(cluster):
    min_x = min(x for x, y in cluster)
    min_y = min(y for x, y in cluster)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cluster))


def canonical_form(cluster, is_canon=True):
    operations = POINT_GROUP_OPERATIONS if is_canon else ("id",)
    return min(_normalize_cluster(transform(cluster, op)) for op in operations)


def generate_clusters(N, is_canon=True):
    if N <= 0:
        return []

    cluster_seen = set()
    clusters = []
    stack = [([(0, 0)], set(get_neighbors((0, 0))))]
    stack_seen = {canonical_form([(0, 0)], True)}

    while stack:
        cluster, boundary = stack.pop()
        if len(cluster) == N:
            canon = canonical_form(cluster, is_canon)
            if canon not in cluster_seen:
                cluster_seen.add(canon)
                clusters.append(list(canon))
            continue

        for point in boundary:
            new_cluster = cluster + [point]
            new_boundary = boundary | set(get_neighbors(point))
            new_boundary -= set(new_cluster)

            new_cluster_canon = canonical_form(new_cluster, True)
            if new_cluster_canon not in stack_seen:
                stack_seen.add(new_cluster_canon)
                stack.append((new_cluster, new_boundary))

    return clusters


def clusters_sort_hole(clusters):
    hole_max = max(count_holes(cluster) for cluster in clusters)
    clusters_by_hole = [[] for _ in range(hole_max + 1)]
    for cluster in clusters:
        clusters_by_hole[count_holes(cluster)].append(cluster)
    return clusters_by_hole


def graph_from_sites(sites):
    """Build NN graph: nodes 0..N-1, edges for nearest-neighbor site pairs."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sites)))
    for i, j in combinations(range(len(sites)), 2):
        if sites[j] in get_neighbors(sites[i]):
            graph.add_edge(i, j)
    return graph


def classify_isomorphic_clusters(clusters):
    graphs = [graph_from_sites(cluster) for cluster in clusters]

    class_assignments = [-1 for _ in clusters]
    node_mappings = [None for _ in graphs]
    representative_graphs = []

    for graph_idx, current_graph in enumerate(graphs):
        is_classified = False

        for class_idx, representative in enumerate(representative_graphs):
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(representative, current_graph)

            if graph_matcher.is_isomorphic():
                node_mappings[graph_idx] = graph_matcher.mapping
                class_assignments[graph_idx] = class_idx
                is_classified = True
                break

        if not is_classified:
            class_assignments[graph_idx] = len(representative_graphs)
            representative_graphs.append(current_graph)

            num_nodes = current_graph.number_of_nodes()
            node_mappings[graph_idx] = {node: node for node in range(num_nodes)}

    classified_clusters = defaultdict(list)
    for cluster_idx, class_idx in enumerate(class_assignments):
        mapping = node_mappings[cluster_idx]
        cluster = clusters[cluster_idx]
        reordered = [None] * len(cluster)
        for candidate, rep in mapping.items():
            reordered[candidate] = cluster[rep]
        classified_clusters[class_idx].append(reordered)

    return list(classified_clusters.values())

def classify_two_site_bonds(cluster):
    bond_types = defaultdict(list)
    for i, j in combinations(range(len(cluster)), 2):
        x1, y1 = cluster[i]
        x2, y2 = cluster[j]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        bond_types[tuple(sorted((dx, dy), reverse=True))].append((i, j))

    return sorted(
        bond_types.items(),
        key=lambda item: item[0][0] ** 2 + item[0][1] ** 2,
    )

