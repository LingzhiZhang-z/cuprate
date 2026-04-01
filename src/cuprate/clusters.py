import os
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

# Edge matcher to consider 'weight' attribute during isomorphism checks
EDGE_MATCHER = nx.algorithms.isomorphism.numerical_edge_match('weight', 0)

def get_neighbors(pos, max_bond=1):
    x, y = pos
    if max_bond == 1:
        return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    elif max_bond == 2:
        return [(x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)]
    elif max_bond == 3:
        return [(x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1), (x+2,y), (x-2,y), (x,y+2), (x,y-2)]
    else:
        print("Error in getting neighbors!")
        exit(0)

def count_holes(cluster):
    cluster_set = set(cluster)
    hole_count = 0
    for (x, y) in cluster:
        square = [
            (x, y),
            (x + 1, y),
            (x, y + 1),
            (x + 1, y + 1)
        ]
        if all(pt in cluster_set for pt in square):
            hole_count += 1
    return hole_count

def transform(cluster, op):
    if op == 'id':
        return cluster
    elif op == 'rot90':
        return [(-y, x) for x, y in cluster]
    elif op == 'rot180':
        return [(-x, -y) for x, y in cluster]
    elif op == 'rot270':
        return [(y, -x) for x, y in cluster]
    elif op == 'mirror_x':
        return [(x, -y) for x, y in cluster]
    elif op == 'mirror_y':
        return [(-x, y) for x, y in cluster]
    elif op == 'mirror_diag':
        return [(y, x) for x, y in cluster]
    elif op == 'mirror_anti':
        return [(-y, -x) for x, y in cluster]
    else:
        print("Error operation in transformation function!")
        exit(0)

def canonical_form(cluster, is_canon=True):
    if is_canon:
        ops = ['id', 'rot90', 'rot180', 'rot270', 'mirror_x', 'mirror_y', 'mirror_diag', 'mirror_anti']
    else:
        ops = ['id']
    forms = []
    for op in ops:
        transformed = transform(cluster, op)
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        normalized = sorted((x - min_x, y - min_y) for x, y in transformed)
        forms.append(tuple(normalized))
    return min(forms)

def generate_clusters(N, max_bond=1, is_canon=True):
    stack_seen = set()
    cluster_seen = set()
    clusters = []
    stack = [([(0, 0)], set(get_neighbors((0, 0), max_bond)))]

    while stack:
        cluster, boundary = stack.pop()
        if len(cluster) == N:
            canon = canonical_form(cluster, is_canon)
            if canon not in cluster_seen:
                cluster_seen.add(canon)
                clusters.append(list(canon))
            continue

        for p in list(boundary):
            if p in cluster:
                continue
            new_cluster = cluster + [p]
            new_boundary = boundary | set(get_neighbors(p, max_bond))
            new_boundary -= set(new_cluster)

            new_cluster_canon = canonical_form(new_cluster, True)
            if new_cluster_canon not in stack_seen:
                stack_seen.add(new_cluster_canon)
                stack.append((new_cluster, new_boundary))
            continue

    return clusters


def clusters_sort_hole(clusters):
    hole_counts = [count_holes(cluster) for cluster in clusters]
    hole_max = max(hole_counts)
    clusters_by_hole = [[] for _ in range(hole_max + 1)]
    for cluster, hole in zip(clusters, hole_counts):
        clusters_by_hole[hole].append(cluster)

    clusters_resorted=[None for _ in range(hole_max+1)]
    for i, cluster in enumerate(clusters_by_hole):
        clusters_resorted[i]=cluster

    return clusters_resorted

def generate_adjacency_matrix(sites, max_bond=1):
    """Generate adjacency matrix distinguishing between nearest neighbor (NN) and next-nearest neighbor (NNN) bonds."""
    n = len(sites)
    adj = [[0 for _ in range(n)] for _ in range(n)]
    site_dict = {(x, y): i for i, (x, y) in enumerate(sites)}

    bond_list = [
        [(1,0), (-1,0), (0,1), (0,-1)],
        [(1,1), (1,-1), (-1,1), (-1,-1)],
        [(2,0), (-2,0), (0,2), (0,-2)]
    ]

    for bond_type in range(min(max_bond, len(bond_list))):
        for i, (x, y) in enumerate(sites):
            for dx, dy in bond_list[bond_type]:
                neighbor = (x + dx, y + dy)
                if neighbor in site_dict:
                    j = site_dict[neighbor]
                    adj[i][j] = bond_type + 1
    return adj

def generate_bonds(cluster, max_bond=1):
    """Generate bonds distinguishing between nearest neighbor (NN) and next-nearest neighbor (NNN) bonds."""
    adj = generate_adjacency_matrix(cluster, max_bond)
    bonds = [ [] for _ in range(max_bond) ]
    n = len(adj)

    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                continue
            bonds[adj[i][j]-1].append([i, j])

    return bonds

def graph_from_adj(adj):
    G = nx.Graph()
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                if adj[i][j] > 0:
                    G.add_edge(i, j, weight=adj[i][j])
    return G

def reorder_cluster(cluster, mapping):
    cluster_new=[None for _ in cluster]
    for candidate, rep in mapping.items():
        cluster_new[candidate]=cluster[rep]
    return cluster_new

def check_adjacency(adj1, adj2):
    return adj1 == adj2

def check_clusters(clusters, max_bond=1):
    """Check if clusters have the same adjacency matrices including both NN and NNN bonds."""
    adj_list = [ generate_adjacency_matrix(cluster, max_bond) for cluster in clusters ]

    rep = adj_list[0]
    for i, adj in enumerate(adj_list):
        FLAG = check_adjacency(rep, adj)
        if not FLAG:
            print("Error in checking the NN adjacency matrix!")
            print(f"Represented cluster: {clusters[0]}")
            print(f"Unmatched cluster:   {clusters[i]}")
            print(f"Represented adjacency matrix: {rep}")
            print(f"Unmatched adjacency matrix:   {adj}")
            match_graph = nx.algorithms.isomorphism.GraphMatcher(
                graph_from_adj(rep),
                graph_from_adj(adj),
                edge_match=EDGE_MATCHER
            )
            print(f"Isomorphic: {match_graph.is_isomorphic()}")
            raise ValueError("Error in checking the NN adjacency matrix!")

def classify_isomorphic_clusters(clusters, max_bond=1):
    graphs = [ graph_from_adj(generate_adjacency_matrix(cluster, max_bond)) for cluster in clusters ]

    class_assignments = [ -1 for _ in clusters ]
    node_mappings = [ None for _ in graphs ]
    representative_graphs = []

    for graph_idx, current_graph in enumerate(graphs):
        is_classified = False

        for class_idx, representative in enumerate(representative_graphs):
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
                representative,
                current_graph,
                edge_match=EDGE_MATCHER
            )

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
        reordered_cluster = reorder_cluster(clusters[cluster_idx], node_mappings[cluster_idx])
        classified_clusters[class_idx].append(reordered_cluster)

    return list(classified_clusters.values())


def plot_cluster(cluster, filename="cluster.png", max_bond=1, grid_size=1):
    N=len(cluster)
    cluster_set = set(cluster)
    fig, ax = plt.subplots()

    min_range = -1
    max_range = N

    ax.set_xticks(range(min_range, max_range, grid_size))
    ax.set_yticks(range(min_range, max_range, grid_size))
    ax.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)

    ax.axhline(0, color='gray',linewidth=1.0)
    ax.axvline(0, color='gray',linewidth=1.0)

    ax.set_aspect('equal')
    ax.axis('on')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(axis='both', which='both', length=0)

    for i, (x, y) in enumerate(cluster):
        ax.plot(x, y, 'o', color='black')
        plt.text(x + 0.05, y + 0.05, f"{i}", fontsize=12)

    bond_list = [
        [[(1,0),  (0,1)], "-", 'black'],
        [[(1,1), (1,-1)], "--", 'black'],
        [[(2,0), (0,2)], "--", 'orange']
    ]

    for bond_type in range(min(max_bond, len(bond_list))):
        for (x, y) in cluster:
            for dx, dy in bond_list[bond_type][0]:
                neighbor = (x + dx, y + dy)
                if neighbor in cluster_set:
                    ax.plot([x, neighbor[0]], [y, neighbor[1]], bond_list[bond_type][1], color=bond_list[bond_type][2], linewidth=2.0)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        print("No existing path! ", filename)

def write_cluster(cluster, filename="cluster.txt"):
    with open(filename, 'w') as file:
        for index, (x, y) in enumerate(cluster):
            file.write(f"{index}: {x} {y}\n")


def rotate_90(x, y):
    """Rotate a point (x, y) 90 degrees counterclockwise around the origin."""
    return -y, x

def get_min_lexicographic_bond(dx, dy):
    """Get the lexicographically smallest bond vector considering C4 symmetry."""
    bond_vectors = [(dx, dy)]
    for _ in range(3):
        dx, dy = rotate_90(dx, dy)
        bond_vectors.append((dx, dy))
    return min(bond_vectors)

def canonical_bond(site1, site2):
    x1, y1 = site1
    x2, y2 = site2
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    return sorted([dx, dy], reverse=True)

def canonical_bond_type(cluster, pair):
    idx1, idx2 = pair
    bond = canonical_bond(cluster[idx1], cluster[idx2])
    if bond == [1, 0]:
        return 1
    elif bond == [1, 1]:
        return 2
    elif bond == [2, 0]:
        return 3
    else:
        return -1


def find_bonds_twosites(cluster):
    """Generate and classify bonds in a cluster considering C4 symmetry, sorted by bond vector length."""
    bond_types = defaultdict(list)
    for i, (x1, y1) in enumerate(cluster):
        for j, (x2, y2) in enumerate(cluster):
            if i < j:
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                bond_vector = sorted([dx, dy], reverse=True)
                bond_types[tuple(bond_vector)].append((i, j))

    bonds = dict(sorted(bond_types.items(),
                        key=lambda item: sum(x*x for x in item[0])**0.5)).values()
    return bonds


def find_squares(cluster):
    """Find all squares in the cluster."""
    cluster_dict = {pt: idx for idx, pt in enumerate(cluster)}
    squares = []

    for (x, y) in cluster:
        square = [
            (x, y),
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1)
        ]

        if all(pt in cluster_dict for pt in square):
            indices = [cluster_dict[pt] for pt in square]
            squares.append(tuple(indices))

    return squares if squares else None


def find_connected_sites(cluster, N):
    def are_connected(site1, site2):
        x1, y1 = site1
        x2, y2 = site2
        return abs(x1 - x2) + abs(y1 - y2) == 1

    def is_connected_group(sites):
        """Judge if a group of sites are connected"""
        if not sites:
            return False

        visited = {sites[0]}
        queue = [sites[0]]

        while queue:
            current = queue.pop(0)
            for site in sites:
                if site not in visited and are_connected(current, site):
                    visited.add(site)
                    queue.append(site)

        return len(visited) == len(sites)

    many_sites = []
    for sites in combinations(cluster, N):
        if is_connected_group(sites):
            many_sites.append(list(cluster.index(s) for s in sites))
    return many_sites


def find_bonds_threesites(cluster):
    """Find all possible connected foursites in the cluster."""
    foursites = find_connected_sites(cluster, 3)
    foursites_bonds = []
    for foursite in foursites:
        foursites_bonds.append([foursite[0], foursite[1], foursite[1], foursite[2]])
        foursites_bonds.append([foursite[1], foursite[0], foursite[0], foursite[2]])
        foursites_bonds.append([foursite[0], foursite[2], foursite[2], foursite[1]])
    return foursites_bonds

def find_bonds_foursites(cluster):
    """Find all possible connected foursites in the cluster."""
    foursites = find_connected_sites(cluster, 4)
    foursites_bonds = []
    for foursite in foursites:
        foursites_bonds.append([foursite[0], foursite[1], foursite[2], foursite[3]])
        foursites_bonds.append([foursite[0], foursite[2], foursite[1], foursite[3]])
        foursites_bonds.append([foursite[0], foursite[3], foursite[1], foursite[2]])
    return foursites_bonds


def find_bonds_sixsites(cluster):
    """Find all possible connected sixsites in the cluster."""
    sixsites = find_connected_sites(cluster, 6)
    sixsites_bonds = []
    for sixsite in sixsites:
        sixsites_bonds.append([sixsite[0], sixsite[1], sixsite[2], sixsite[3], sixsite[4], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[1], sixsite[4], sixsite[2], sixsite[3], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[1], sixsite[5], sixsite[2], sixsite[3], sixsite[4]])
        sixsites_bonds.append([sixsite[0], sixsite[2], sixsite[1], sixsite[3], sixsite[4], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[2], sixsite[4], sixsite[1], sixsite[3], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[2], sixsite[5], sixsite[1], sixsite[3], sixsite[4]])
        sixsites_bonds.append([sixsite[0], sixsite[3], sixsite[1], sixsite[2], sixsite[4], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[3], sixsite[4], sixsite[1], sixsite[2], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[3], sixsite[5], sixsite[1], sixsite[2], sixsite[4]])
        sixsites_bonds.append([sixsite[0], sixsite[4], sixsite[1], sixsite[2], sixsite[3], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[4], sixsite[3], sixsite[1], sixsite[2], sixsite[5]])
        sixsites_bonds.append([sixsite[0], sixsite[4], sixsite[5], sixsite[1], sixsite[2], sixsite[3]])
        sixsites_bonds.append([sixsite[0], sixsite[5], sixsite[1], sixsite[2], sixsite[3], sixsite[4]])
        sixsites_bonds.append([sixsite[0], sixsite[5], sixsite[3], sixsite[1], sixsite[2], sixsite[4]])
        sixsites_bonds.append([sixsite[0], sixsite[5], sixsite[4], sixsite[1], sixsite[2], sixsite[3]])
    return sixsites_bonds

def find_bonds_multisites(cluster, N):
    """Find all possible connected multisites in the cluster."""
    multisites = find_connected_sites(cluster, N)
    multisites_bonds = []
    from itertools import permutations

    for multisite in multisites:
        for perm in permutations(multisite):
            multisites_bonds.append(list(perm))
    return multisites_bonds

def bond_analysis_spin_class(cluster):
    all_bond_classes = []
    bond_types = find_bonds_twosites(cluster)
    for bond_list in bond_types:
        all_bond_classes.append(bond_list)

    squares = find_squares(cluster)
    if squares:
        all_bond_classes.append(squares)

    return all_bond_classes


def bond_analysis_spin(cluster):
    all_bonds = []

    bonds_twosites = find_bonds_twosites(cluster)
    if bonds_twosites:
        all_bonds.extend(bonds_twosites)

    foursites_bonds = find_bonds_foursites(cluster)
    if foursites_bonds:
        all_bonds.append(foursites_bonds)

    sixsites_bonds = find_bonds_sixsites(cluster)
    if sixsites_bonds:
        all_bonds.append(sixsites_bonds)

    return all_bonds


class Clusters_Square:
    def __init__(self, Nsites=-1):
        self.Nsites = Nsites
        self.clusters=[]
        self.clusters_classified=[]
        self.bonds=[]
        self.bonds_classified=[]

    def clear(self):
        self.Nsites = -1
        self.clusters=[]
        self.clusters_classified=[]
        self.bonds=[]
        self.bonds_classified=[]

    def set_sites(self, Nsites):
        self.Nsites=Nsites

    def compute_clsuters(self, t2=None, if_print_time=False):
        if(self.Nsites<0):
            raise ValueError("Please set a positive number for the number of sites!")

        t0 = time.time()
        max_bond = 2 if t2 is not None else 1
        self.clusters=generate_clusters(self.Nsites, max_bond=max_bond)
        self.bonds=[None for _ in self.clusters]
        for idx, cluster in enumerate(self.clusters):
            self.bonds[idx]=generate_bonds(cluster, max_bond)

        print(f"Time of cluster calculation: {(time.time()-t0):.6f} s") if if_print_time else None

    def classify_clusters(self, t2=None, if_print_time=False):
        if not self.clusters:
            raise ValueError("Please run compute_clusters() first to obtain clusters!")

        t0 = time.time()
        max_bond = 2 if t2 is not None else 1
        clusters_by_holes = clusters_sort_hole(self.clusters)

        self.bonds_classified = [ None for _ in clusters_by_holes ]
        self.clusters_classified = [ None for _ in clusters_by_holes ]

        for hole_count, clusters_with_holes in enumerate(clusters_by_holes):
            isomorphic_groups = classify_isomorphic_clusters(clusters_with_holes, max_bond)
            self.clusters_classified[hole_count] = isomorphic_groups

            self.bonds_classified[hole_count] = [None for _ in isomorphic_groups]
            for group_idx, group in enumerate(isomorphic_groups):
                self.bonds_classified[hole_count][group_idx] = [ generate_bonds(cluster, max_bond) for cluster in group ]
                check_clusters(group, max_bond)

        print(f"Time of classification: {(time.time()-t0):.6f} s") if if_print_time else None
        print("All clusters have the same adjacency matrix in class!")

    def plot(self, main_path="./clusters", if_print_time=False):
        os.makedirs(main_path, exist_ok=True)
        t0 = time.time()
        for i, cluster in enumerate(self.clusters):
            plot_cluster(cluster, main_path+"/hole_"+str(i)+".png")

        print(f"Time of plotting: {(time.time()-t0):.6f} s") if if_print_time else None


    def plot_classified(self, main_path="./clusters_classified", max_bond=1, if_print_time=False):
        os.makedirs(main_path, exist_ok=True)
        t0 = time.time()
        for hole, clusters_classified in enumerate(self.clusters_classified):
            for classification, clusters in enumerate(clusters_classified):
                for i, cluster in enumerate(clusters):
                    plot_cluster(cluster, main_path+"/hole_"+str(hole)+"_"+str(classification)+"_"+str(i)+".png", max_bond=max_bond)

        print(f"Time of plotting: {(time.time()-t0):.6f} s") if if_print_time else None

    def write_classified(self, main_path="./clusters_classified", if_print_time=False):
        os.makedirs(main_path, exist_ok=True)
        t0 = time.time()
        for hole, clusters_classified in enumerate(self.clusters_classified):
            for classification, clusters in enumerate(clusters_classified):
                for i, cluster in enumerate(clusters):
                    write_cluster(cluster, main_path+"/hole_"+str(hole)+"_"+str(classification)+"_"+str(i)+".txt")

        print(f"Time of writting: {(time.time()-t0):.6f} s") if if_print_time else None


    def print_info(self):
        total_classification=0
        print("Size of clusters: ", len(self.clusters)," for ", self.Nsites, "sites.")
        for hole, clusters_classified in enumerate(self.clusters_classified):
            print("Holes: ", hole, " , classification number: ", len(clusters_classified))
            total_classification+=len(clusters_classified)

        if(total_classification==0):
            print("No classification has been done!")
        else:
            print("Total number of classification: ", total_classification, "\n")

        for hole, clusters_classified in enumerate(self.clusters_classified):
            for classification, clusters in enumerate(clusters_classified):
                print("Holes: ", hole, " , classification: ", classification, " , number: ", len(clusters))
