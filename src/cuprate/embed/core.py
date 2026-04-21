"""Core embedding and topology logic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from cuprate.clusters import POINT_GROUP_OPERATIONS, transform
from cuprate.io import (
    k4s,
    k6s,
    k8s,
    load_consolidated_results,
    spin_coupling_terms_from_artifact,
)


@dataclass
class EmbeddedCouplings:
    constant: complex = 0.0
    two_site: np.ndarray = field(default=None)
    four_site: dict = field(default_factory=dict)
    six_site: dict = field(default_factory=dict)
    eight_site: dict = field(default_factory=dict)


FOUR_SITE_TEMPLATES = (
    (0, ((0, 0), (1, 0), (1, 1), (0, 1))),
    (1, ((1, 0), (0, 0), (1, 1), (2, 0))),
    (2, ((0, 0), (1, 0), (2, 0), (3, 0))),
    (3, ((0, 0), (1, 0), (2, 0), (2, 1))),
    (4, ((0, 0), (1, 0), (1, 1), (2, 1))),
)


def unique_cluster_transformed(cluster_input):
    unique_clusters = []
    seen = set()
    for op in POINT_GROUP_OPERATIONS:
        transformed = transform(cluster_input, op)
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        normalized = [(x - min_x, y - min_y) for x, y in transformed]
        key = tuple(sorted(normalized))
        if key not in seen:
            seen.add(key)
            unique_clusters.append(normalized)
    return unique_clusters


def embed_cluster_to_squarecell_pbc(cluster_input, size):
    min_x = min(x for x, y in cluster_input)
    min_y = min(y for x, y in cluster_input)
    max_x = max(x for x, y in cluster_input)
    max_y = max(y for x, y in cluster_input)

    if max_x - min_x >= size or max_y - min_y >= size:
        raise ValueError("The size of the cluster is larger than the size of the square cell")
    cluster = [(x - min_x, y - min_y) for x, y in cluster_input]

    indices_list = []
    for delta_x in range(size):
        for delta_y in range(size):
            indices_list.append(
                [
                    ((x + delta_x) % size) * size + ((y + delta_y) % size)
                    for x, y in cluster
                ]
            )
    return indices_list


def embed_operator(couplings, terms, indices):
    couplings.constant += terms.constant

    for group_terms in terms.two_site.values():
        for idx1, idx2, value in group_terms:
            couplings.two_site[indices[idx1], indices[idx2]] += value
            couplings.two_site[indices[idx2], indices[idx1]] += value

    for idx1, idx2, idx3, idx4, value in terms.four_site:
        key = k4s(indices[idx1], indices[idx2], indices[idx3], indices[idx4])
        if key not in couplings.four_site:
            couplings.four_site[key] = 0.0
        couplings.four_site[key] += value

    for idx1, idx2, idx3, idx4, idx5, idx6, value in terms.six_site:
        key = k6s(indices[idx1], indices[idx2], indices[idx3], indices[idx4], indices[idx5], indices[idx6])
        if key not in couplings.six_site:
            couplings.six_site[key] = 0.0
        couplings.six_site[key] += value

    for idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, value in terms.eight_site:
        key = k8s(
            indices[idx1],
            indices[idx2],
            indices[idx3],
            indices[idx4],
            indices[idx5],
            indices[idx6],
            indices[idx7],
            indices[idx8],
        )
        if key not in couplings.eight_site:
            couplings.eight_site[key] = 0.0
        couplings.eight_site[key] += value


def populate_embedded_couplings(clusters, lce_root: str, run_dir: str, ncell: int, couplings_pbc) -> None:
    for nsites in clusters:
        if nsites < 2:
            continue
        lce_run_dir = f"{lce_root}/N{nsites}{run_dir}"
        if not os.path.exists(lce_run_dir):
            raise FileNotFoundError(f"Missing upstream LCE run directory: {lce_run_dir}")
        consolidated = load_consolidated_results(lce_run_dir)
        entry_by_id = {
            (entry["hole"], entry["class_idx"], entry["cluster_idx"]): entry
            for entry in consolidated["clusters"]
        }
        for hole in clusters[nsites]:
            for class_idx in clusters[nsites][hole]:
                for rank_idx, cluster in clusters[nsites][hole][class_idx].items():
                    terms = spin_coupling_terms_from_artifact(entry_by_id[(hole, class_idx, rank_idx)])
                    for unique_cluster in unique_cluster_transformed(cluster):
                        for indices_pbc in embed_cluster_to_squarecell_pbc(unique_cluster, ncell):
                            embed_operator(couplings_pbc, terms, indices_pbc)


def normalize_four_sites(indices, cluster):
    """Normalize a connected four-site set into one of five canonical tetromino types."""
    points = [cluster[idx] for idx in indices]
    best_match = None

    for op in POINT_GROUP_OPERATIONS:
        transformed = transform(points, op)
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        normalized = [(x - min_x, y - min_y) for x, y in transformed]
        normalized_set = set(normalized)
        coord_to_position = {coord: pos for pos, coord in enumerate(normalized)}

        for type_num, template in FOUR_SITE_TEMPLATES:
            if normalized_set != set(template):
                continue
            ordered_indices = tuple(indices[coord_to_position[coord]] for coord in template)
            candidate = (type_num, ordered_indices)
            if best_match is None or candidate < best_match:
                best_match = candidate

    if best_match is None:
        raise ValueError(f"Cannot identify the type of the four-site bond: {points}")

    return best_match[0], list(best_match[1])
