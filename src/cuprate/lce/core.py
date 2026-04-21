"""Core linked-cluster expansion logic."""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from cuprate.io import match_cluster_in_catalog, k4s, k6s, k8s


def get_connected_subgraphs(cluster, min_size=2):
    G = nx.Graph()
    n = len(cluster)

    for i in range(n):
        G.add_node(i)

    for i, (x1, y1) in enumerate(cluster):
        for j, (x2, y2) in enumerate(cluster):
            if i != j and abs(x1 - x2) + abs(y1 - y2) == 1:
                G.add_edge(i, j)

    connected_subgraphs = []
    connected_subgraphs_indices = []

    for size in range(min_size, n):
        for nodes in combinations(range(n), size):
            subgraph = G.subgraph(nodes)
            if nx.is_connected(subgraph):
                connected_subgraphs.append([cluster[i] for i in nodes])
                connected_subgraphs_indices.append(list(nodes))

    return connected_subgraphs, connected_subgraphs_indices


def subtract_spin_coupling_terms(target_terms, source_terms, source_to_target) -> None:
    target_terms.constant -= source_terms.constant

    two_site_lookup = {}
    for terms in target_terms.two_site.values():
        for term in terms:
            two_site_lookup[tuple(sorted((term[0], term[1])))] = term

    for terms in source_terms.two_site.values():
        for idx1, idx2, coeff in terms:
            key = tuple(sorted((source_to_target[idx1], source_to_target[idx2])))
            two_site_lookup[key][2] -= coeff

    four_site_lookup = {k4s(*term[:4]): term for term in target_terms.four_site}
    for idx1, idx2, idx3, idx4, coeff in source_terms.four_site:
        key = k4s(
            source_to_target[idx1],
            source_to_target[idx2],
            source_to_target[idx3],
            source_to_target[idx4],
        )
        four_site_lookup[key][4] -= coeff

    six_site_lookup = {k6s(*term[:6]): term for term in target_terms.six_site}
    for idx1, idx2, idx3, idx4, idx5, idx6, coeff in source_terms.six_site:
        key = k6s(
            source_to_target[idx1],
            source_to_target[idx2],
            source_to_target[idx3],
            source_to_target[idx4],
            source_to_target[idx5],
            source_to_target[idx6],
        )
        six_site_lookup[key][6] -= coeff

    eight_site_lookup = {k8s(*term[:8]): term for term in target_terms.eight_site}
    for idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, coeff in source_terms.eight_site:
        key = k8s(
            source_to_target[idx1],
            source_to_target[idx2],
            source_to_target[idx3],
            source_to_target[idx4],
            source_to_target[idx5],
            source_to_target[idx6],
            source_to_target[idx7],
            source_to_target[idx8],
        )
        eight_site_lookup[key][8] -= coeff


def collect_subgraph_data(clusters):
    data = {}
    for nsites in clusters:
        data[nsites] = {}
        for hole in clusters[nsites]:
            data[nsites][hole] = {}
            for class_idx in clusters[nsites][hole]:
                data[nsites][hole][class_idx] = {}
                for rank_idx in clusters[nsites][hole][class_idx]:
                    data[nsites][hole][class_idx][rank_idx] = {
                        "subgraph": [],
                        "indices": [],
                        "match": [],
                        "terms": None,
                    }
                    cluster = clusters[nsites][hole][class_idx][rank_idx]
                    subgraphs, indices = get_connected_subgraphs(cluster)
                    state = data[nsites][hole][class_idx][rank_idx]
                    for subgraph, subgraph_indices in zip(subgraphs, indices):
                        state["subgraph"].append(subgraph)
                        state["indices"].append(subgraph_indices)
                        state["match"].append(match_cluster_in_catalog(subgraph, clusters))
    return data


def subtract_subgraph_contributions(cluster_data, data) -> None:
    for subgraph_idx, match in enumerate(cluster_data["match"]):
        indices = cluster_data["indices"][subgraph_idx]
        idx_n, idx_h, idx_c, idx_r, mapping = match
        map_new = [indices[mapping[i]] for i in range(len(mapping))]
        subtract_spin_coupling_terms(
            cluster_data["terms"],
            data[idx_n][idx_h][idx_c][idx_r]["terms"],
            map_new,
        )
