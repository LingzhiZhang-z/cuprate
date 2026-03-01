import os
import sys
from typing import List, Dict, Tuple, Optional
import networkx as nx
from itertools import combinations
from Clusters_Square import canonical_form, count_holes, graph_from_adj, generate_adjacency_matrix
from utils_main import find_bond_vector, get_all_possible_vectors, write_cluster_points
from utils_file import write_file

DEFAULT_N = 3
DEFAULT_U = 6.0
DEFAULT_T = 1.0

EDGE_MATCHER = nx.algorithms.isomorphism.numerical_edge_match('weight', 0)

def setup_work_environment(params):
    """Setup the working environment and parameters"""
    result_dir1 = f"U{params['U']:.4f}_t{params['t']:.4f}"
    if params['t2'] is not None:
        result_dir1 = f"{result_dir1}_tp{params['t2']:.4f}"

    result_dir2 = f""
    result_dir3 = f""
    if params['sz'] is not None:
        result_dir2 = f"{result_dir2}_sz{params['sz']:.4f}"
        result_dir3 = f"{result_dir3}_sz{0.5-params['sz']:.4f}"
    if params['s2'] is not None:
        result_dir2 = f"{result_dir2}_s{params['s2']:.0f}"
        result_dir3 = f"{result_dir3}_s{params['s2']:.0f}"
    if params['type'] is not None:
        result_dir2 = f"{result_dir2}_{params['type']}"
        result_dir3 = f"{result_dir3}_{params['type']}"
    if params['restart']:
        result_dir2 = f"{result_dir2}_restart"
        result_dir3 = f"{result_dir3}_restart"
    return result_dir1, result_dir2, result_dir3

def k2s(i, j):
    return tuple(sorted([i, j]))

def k4s(i, j, k, l):
    pairs = [tuple(sorted([i, j])), tuple(sorted([k, l]))]
    return tuple(sorted(pairs))

def k6s(i, j, k, l, m, n):
    pairs = [tuple(sorted([i, j])), tuple(sorted([k, l])), tuple(sorted([m, n]))]
    return tuple(sorted(pairs))

def k8s(i, j, k, l, m, n, a, b):
    pairs = [tuple(sorted([i, j])), tuple(sorted([k, l])), tuple(sorted([m, n])), tuple(sorted([a, b]))]
    return tuple(sorted(pairs))

def k6s_all(i, j, k, l, m, n):
    return tuple(sorted([i, j, k, l, m, n]))

def create_dict(ref_dict):
    data = {}
    for nsites in ref_dict:
        data[nsites] = {}
        for hole in ref_dict[nsites]:
            data[nsites][hole] = {}
            for class_idx in ref_dict[nsites][hole]:
                data[nsites][hole][class_idx] = {}
                for rank in ref_dict[nsites][hole][class_idx]:
                    data[nsites][hole][class_idx][rank] = {}
                    data[nsites][hole][class_idx][rank]["subgraph"]=[]
                    data[nsites][hole][class_idx][rank]["match"]=[]
                    data[nsites][hole][class_idx][rank]["indices"]=[]
                    data[nsites][hole][class_idx][rank]["coupling_original"]=[]
                    data[nsites][hole][class_idx][rank]["coupling_net"]=[]
    return data


def get_connected_subgraphs(cluster: List[Tuple[int, int]], min_size: int = 2) -> Tuple[List[List[Tuple[int, int]]], List[List[int]]]:
    # Create graph
    G = nx.Graph()
    n = len(cluster)
    
    # Add nodes
    for i in range(n):
        G.add_node(i)
    
    # Add edges based on adjacency
    for i, (x1, y1) in enumerate(cluster):
        for j, (x2, y2) in enumerate(cluster):
            if i != j and abs(x1 - x2) + abs(y1 - y2) == 1:  # Manhattan distance = 1
                G.add_edge(i, j)
    
    # Generate all possible subgraphs
    connected_subgraphs = []
    connected_subgraphs_indices = []
    
    for size in range(min_size, n):  # Note: n instead of n+1 to exclude the original cluster
        for nodes in combinations(range(n), size):
            subgraph = G.subgraph(nodes)
            if nx.is_connected(subgraph):
                # Convert node indices back to coordinates
                subgraph_coords = [cluster[i] for i in nodes]
                
                connected_subgraphs.append(subgraph_coords)
                connected_subgraphs_indices.append(list(nodes))  # Store the indices
    
    return connected_subgraphs, connected_subgraphs_indices


def print_subgraph(subgraph: List[Tuple[int, int]], indices: Optional[List[int]] = None):
    print("Subgraph:", subgraph)
    if indices is not None:
        print("Indices in original cluster:", indices)


def read_cluster_file(file_path: str) -> List[Tuple[int, int]]:
    cluster = []
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                _, coords = line.split(':')
                x, y = map(int, coords.strip().split())
                cluster.append((x, y))
    return cluster


def parse_filename(filename: str) -> Tuple[int, int, int]:
    parts = filename.replace('hole_', '').replace('.txt', '').split('_')
    return tuple(map(int, parts))


def read_coords_file(coord_dir, n_max):
    # 使用字典存储数据
    clusters = {}
    
    # 读取每个位点数的数据
    for n in range(1, n_max+1):
        n_dir = os.path.join(coord_dir, str(n))
        if not os.path.exists(n_dir):
            continue
            
        print(f"Reading clusters with {n} sites...")
        clusters[n] = {}
        
        # 读取每个团簇文件
        for file in os.listdir(n_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(n_dir, file)
                hole, class_idx, rank = parse_filename(file)
                
                # 初始化字典结构
                if hole not in clusters[n]:
                    clusters[n][hole] = {}
                if class_idx not in clusters[n][hole]:
                    clusters[n][hole][class_idx] = {}
                
                # 读取并存储团簇
                cluster = read_cluster_file(file_path)
                clusters[n][hole][class_idx][rank] = cluster
    
    # Print the structure of the data
    print("\nData structure summary:")
    for n in sorted(clusters.keys()):
        print(f"\n{n} sites:")
        for hole in sorted(clusters[n].keys()):
            print(f"  Hole {hole}:")
            for class_idx in sorted(clusters[n][hole].keys()):
                count = len(clusters[n][hole][class_idx])
                print(f"    Class {class_idx}: {count} clusters")
    
    return clusters


def read_cluster_file_Block(file_path: str) -> List[Tuple[int, int]]:
    cluster = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Point'):
                coords = line.split(':')[1].strip()
                x, y = map(int, coords.strip('()').split(','))
                cluster.append((x, y))
    return cluster

def parse_filename_Block(filename: str) -> Tuple[int, int, int]:
    parts = filename.replace('hole', '').replace('class', '').replace('cluster', '').replace('results.txt', '').rstrip('_').split('_')
    hole, class_idx, cluster = map(int, parts)
    return hole, class_idx, cluster

def read_coords_file_Block(prefix_path, suffix, suffix1, n_max):
    # 使用字典存储数据
    clusters = {}
    print(f"prefix_path: {prefix_path}")
    print(f"suffix: {suffix}")
    print(f"suffix1: {suffix1}")
    print(f"n_max: {n_max}")
    # 读取每个位点数的数据
    for n in range(1, n_max+1):
        n_dir = f"{prefix_path}/N{n}{suffix}"
        if not os.path.exists(n_dir):
            n_dir = f"{prefix_path}/N{n}{suffix1}"
            if not os.path.exists(n_dir):
                continue
        print(f"Reading clusters with {n} sites from {n_dir}...")
        clusters[n] = {}
        
        # 读取每个团簇文件
        for file in os.listdir(n_dir):
            if file.endswith('_results.txt'):
                file_path = os.path.join(n_dir, file)
                hole, class_idx, rank = parse_filename_Block(file)
                
                # 初始化字典结构
                if hole not in clusters[n]:
                    clusters[n][hole] = {}
                if class_idx not in clusters[n][hole]:
                    clusters[n][hole][class_idx] = {}
                
                # 读取并存储团簇
                cluster = read_cluster_file_Block(file_path)
                clusters[n][hole][class_idx][rank] = cluster
    
    # Print the structure of the data
    print("\nData structure summary:")
    for n in sorted(clusters.keys()):
        print(f"\n{n} sites:")
        for hole in sorted(clusters[n].keys()):
            print(f"  Hole {hole}:")
            for class_idx in sorted(clusters[n][hole].keys()):
                count = len(clusters[n][hole][class_idx])
                print(f"    Class {class_idx}: {count} clusters")
    
    return clusters

def create_graph_from_cluster(cluster, t2=None):
    max_bond = 2 if t2 is not None else 1
    adj = generate_adjacency_matrix(cluster, max_bond)
    G = graph_from_adj(adj)
    return G

def find_cluster_match(target_cluster, ref_clusters, t2=None):
    matches = []
    
    # 创建目标团簇的图
    target_canon = canonical_form(target_cluster)
    target_graph = create_graph_from_cluster(target_cluster, t2)
    target_graph_canon = create_graph_from_cluster(target_canon, t2)
    nsite = len(target_canon)
    hole = count_holes(target_canon)
    
    # 判断ref_clusters是2D还是4D
    is_4d = False
    for idx in ref_clusters:
        if isinstance(ref_clusters[idx], dict) and any(isinstance(ref_clusters[idx][hole], dict) for hole in ref_clusters[idx]):
            is_4d = True
            break
    
    if is_4d:
        clusters = ref_clusters[nsite][hole]
    else:
        clusters = ref_clusters
    

    # 对每个class，先用第一个团簇判断是否同构
    for class_idx in clusters:
        # 获取class的第一个团簇
        first_cluster = clusters[class_idx][0]
        
        # 创建参考团簇的图
        ref_graph = create_graph_from_cluster(first_cluster, t2)
        # 使用GraphMatcher判断是否同构

        graph_matcher = nx.algorithms.isomorphism.GraphMatcher(ref_graph, target_graph, edge_match=EDGE_MATCHER)
        graph_matcher_canon = nx.algorithms.isomorphism.GraphMatcher(ref_graph, target_graph_canon, edge_match=EDGE_MATCHER)

        if graph_matcher.is_isomorphic() and graph_matcher_canon.is_isomorphic():
            # 获取同构映射
            mapping = graph_matcher.mapping
            mapping_canon = graph_matcher_canon.mapping
            # 使用同构映射找到坐标对应关系
            coord_mapping = []
            coord_mapping_canon = []
            for i in range(len(first_cluster)):
                coord_mapping.append(mapping[i])
                coord_mapping_canon.append(mapping_canon[i])

            # 遍历class中的所有团簇
            for rank in clusters[class_idx]:
                ref_cluster = clusters[class_idx][rank]
                
                # 验证坐标是否匹配
                is_valid = True
                for i, ref_coord in enumerate(ref_cluster):
                    target_coord = target_canon[coord_mapping_canon[i]]
                    if ref_coord != target_coord:
                        is_valid = False
                        break
                
                if is_valid:
                    if is_4d:
                        matches.append((nsite, hole, class_idx, rank, coord_mapping))
                    else:
                        matches.append((class_idx, rank, coord_mapping))
        
    
    if not matches:
        raise ValueError(f"No matching cluster found for cluster: {target_cluster}, {target_canon}")
    elif len(matches)>1:
        raise ValueError(f"Multiple matching clusters found for cluster: {target_cluster}, {target_canon}")
    
    return matches[0]


def parse_coupling_params(operator_lines):
    flag=False
    line_start="=== Individual Bond Coefficients ===\n"
    line_end="=== Individual Fit Error ===\n"
    coupling_params = []
    coupling_params_foursite=[]
    coupling_params_sixsite=[]
    coupling_params_eightsite=[]
    count=0
    for line in operator_lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        if line==line_start:
            flag=True
            continue
        elif line==line_end:
            break
        if not flag:
            continue

        parts=line.strip().split()
        if parts[0]=="Constant" and parts[1]=="term:" and len(parts)==3:
            # Constant term: -0.4606417115
            coupling_params.append(float(parts[2]))
        elif parts[0]=="Bond" and parts[1]=="vector" and len(parts)==5:
            # Bond vector (dir1, dir2), Jx:
            count+=1
            coupling_params.append([])
        elif len(parts)==6 and parts[1]=="Sites":
            #     idx: Sites idx1-idx2: coupling_r  +  0.0000000i
            idx1, idx2 = map(int, parts[2].rstrip(':').split('-')) 
            coupling=float(parts[3])+1.0j*float(parts[5].rstrip('i'))
            coupling_params[count].append([idx1, idx2, coupling])
        elif parts[0]=="class" and parts[5]=="Four-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('():').split('-'))
            coupling=float(parts[9])+1.0j*float(parts[11].rstrip('i'))
            coupling_params_foursite.append([idx1, idx2, idx3, idx4, coupling])
        elif parts[0]=="class" and parts[5]=="Six-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('()').split('-'))
            idx5, idx6 = map(int, parts[10].strip('():').split('-'))
            coupling=float(parts[11])+1.0j*float(parts[13].rstrip('i'))
            coupling_params_sixsite.append([idx1, idx2, idx3, idx4, idx5, idx6, coupling])
        elif parts[0]=="class" and parts[5]=="Eight-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('()').split('-'))
            idx5, idx6 = map(int, parts[10].strip('()').split('-'))
            idx7, idx8 = map(int, parts[12].strip('():').split('-'))
            coupling=float(parts[13])+1.0j*float(parts[15].rstrip('i'))
            coupling_params_eightsite.append([idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, coupling])
        
    coupling_params.append(coupling_params_foursite)
    coupling_params.append(coupling_params_sixsite)
    coupling_params.append(coupling_params_eightsite)
    return coupling_params


def parse_coupling_params2(operator_lines):
    flag=False
    line_start="=== Individual Bond Coefficients ===\n"
    line_end="=== Individual Fit Error ===\n"
    coupling_params = []
    coupling_params_foursite=[]
    coupling_params_sixsite=[]
    coupling_params_eightsite=[]
    count=0
    for line in operator_lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        if line==line_start:
            flag=True
            continue
        elif line==line_end:
            break
        if not flag:
            continue

        parts=line.strip().split()
        if parts[0]=="Constant" and parts[1]=="term:" and len(parts)==3:
            # Constant term: -0.4606417115
            coupling_params.append(float(parts[2]))
        elif parts[0]=="Bond" and parts[1]=="vector" and len(parts)==5:
            # Bond vector (dir1, dir2), Jx:
            count+=1
            coupling_params.append([])
        elif parts[0]=="Bond" and parts[1]=="vector" and len(parts)==9:
            # Bond vector (dir1, dir2), Jx: (not present in cluster)
            count+=1
            coupling_params.append([])
        elif len(parts)==6 and parts[1]=="Sites":
            #     idx: Sites idx1-idx2: coupling_r  +  0.0000000i
            idx1, idx2 = map(int, parts[2].rstrip(':').split('-')) 
            coupling=float(parts[3])+1.0j*float(parts[5].rstrip('i'))
            coupling_params[count].append([idx1, idx2, coupling])
        elif parts[0]=="class" and parts[5]=="Four-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('():').split('-'))
            coupling=float(parts[9])+1.0j*float(parts[11].rstrip('i'))
            coupling_params_foursite.append([idx1, idx2, idx3, idx4, coupling])
        elif parts[0]=="class" and parts[5]=="Six-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('()').split('-'))
            idx5, idx6 = map(int, parts[10].strip('():').split('-'))
            coupling=float(parts[11])+1.0j*float(parts[13].rstrip('i'))
            coupling_params_sixsite.append([idx1, idx2, idx3, idx4, idx5, idx6, coupling])
        elif parts[0]=="class" and parts[5]=="Eight-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('()').split('-'))
            idx5, idx6 = map(int, parts[10].strip('()').split('-'))
            idx7, idx8 = map(int, parts[12].strip('():').split('-'))
            coupling=float(parts[13])+1.0j*float(parts[15].rstrip('i'))
            coupling_params_eightsite.append([idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, coupling])
        
    coupling_params.append(coupling_params_foursite)
    coupling_params.append(coupling_params_sixsite)
    coupling_params.append(coupling_params_eightsite)
    return coupling_params



def read_operators(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        coupling_params = parse_coupling_params(lines)
    return coupling_params

def read_operators2(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        coupling_params = parse_coupling_params2(lines)
    return coupling_params

def operator_minus(operators1, operators2, map_from2to1):
    operators1[0]-=operators2[0]
    for i, operator_group in enumerate(operators2):
        if i==0: continue
        for j, operator in enumerate(operator_group):
            if len(operator)==3:
                idx1, idx2, coeff = operator
                idx1_new, idx2_new =find_operator([map_from2to1[idx1], map_from2to1[idx2]], operators1)
                operators1[idx1_new][idx2_new][2]-=coeff
            elif len(operator)==5:
                idx1, idx2, idx3, idx4, coeff = operator
                idx1_new, idx2_new =find_operator([map_from2to1[idx1], map_from2to1[idx2], map_from2to1[idx3], map_from2to1[idx4]], operators1)
                operators1[idx1_new][idx2_new][4]-=coeff
            elif len(operator)==7:
                idx1, idx2, idx3, idx4, idx5, idx6, coeff = operator
                idx1_new, idx2_new =find_operator([map_from2to1[idx1], map_from2to1[idx2], map_from2to1[idx3], map_from2to1[idx4], map_from2to1[idx5], map_from2to1[idx6]], operators1)
                operators1[idx1_new][idx2_new][6]-=coeff
            elif len(operator)==9:
                idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, coeff = operator
                idx1_new, idx2_new =find_operator([map_from2to1[idx1], map_from2to1[idx2], map_from2to1[idx3], map_from2to1[idx4], map_from2to1[idx5], map_from2to1[idx6], map_from2to1[idx7], map_from2to1[idx8]], operators1)
                operators1[idx1_new][idx2_new][8]-=coeff


def find_operator(idx_list, operators):
    for i, operator_group in enumerate(operators):
        if i==0: continue
        for j, operator in enumerate(operator_group):
            if len(operator)==3 and len(idx_list)==2:
                idx1, idx2 = idx_list
                idx1_op, idx2_op, coeff = operator
                if ((idx1 == idx1_op and idx2 == idx2_op) or 
                    (idx1 == idx2_op and idx2 == idx1_op)):
                    return i, j
            elif len(operator)==5 and len(idx_list)==4:
                idx1, idx2, idx3, idx4 = idx_list
                idx1_op, idx2_op, idx3_op, idx4_op, coeff = operator
                if ((idx1 == idx1_op and idx2 == idx2_op and idx3 == idx3_op and idx4 == idx4_op) or
                    (idx1 == idx1_op and idx2 == idx2_op and idx3 == idx4_op and idx4 == idx3_op) or
                    (idx1 == idx2_op and idx2 == idx1_op and idx3 == idx3_op and idx4 == idx4_op) or
                    (idx1 == idx2_op and idx2 == idx1_op and idx3 == idx4_op and idx4 == idx3_op) or
                    (idx1 == idx3_op and idx2 == idx4_op and idx3 == idx1_op and idx4 == idx2_op) or
                    (idx1 == idx3_op and idx2 == idx4_op and idx3 == idx2_op and idx4 == idx1_op) or
                    (idx1 == idx4_op and idx2 == idx3_op and idx3 == idx1_op and idx4 == idx2_op) or
                    (idx1 == idx4_op and idx2 == idx3_op and idx3 == idx2_op and idx4 == idx1_op)):
                    return i, j
            elif len(operator)==7 and len(idx_list)==6:
                idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op, coeff = operator
                if k6s(idx_list[0], idx_list[1], idx_list[2], idx_list[3], idx_list[4], idx_list[5]) == k6s(idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op):
                    return i, j
            elif len(operator)==9 and len(idx_list)==8:
                idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op, idx7_op, idx8_op, coeff = operator
                if k8s(idx_list[0], idx_list[1], idx_list[2], idx_list[3], idx_list[4], idx_list[5], idx_list[6], idx_list[7]) == k8s(idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op, idx7_op, idx8_op):
                    return i, j


    raise ValueError(f"Operator not found for indices: {idx_list}")

def write_operators(file_path, operators, cluster, bonds):
    with open(file_path, 'w') as f:
        write_cluster_points(f, cluster)
        #write_bond_structure(f, cluster, bonds)
        write_operators_LCE(f, cluster, bonds, operators)


def write_operators_LCE(f, cluster, bonds, operators):
    """Write individual bond coefficients."""
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {operators[0].real:.10f}\n")
    
    # Create a dictionary to store coefficients by vector
    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in coeffs_by_vector:
                coeffs_by_vector[(dx, dy)] = []
            coeffs_by_vector[(dx, dy)].append((class_idx, bond_group, operators[class_idx+1]))
    
    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if (dx, dy) in coeffs_by_vector:
            for class_idx, bond_group, coeffs in coeffs_by_vector[(dx, dy)]:
                f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:\n")
                for idx, (bond, coef) in enumerate(zip(bond_group, coeffs)):
                    site1, site2 = bond
                    f.write(f"    {idx}: Sites {site1}-{site2}: {coef[2].real:.10f}  +  {coef[2].imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}: (not present in cluster)\n")
    
    f.write(f"\nFour-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 4:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//3+1} type {idx%3+1} : Four-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}): {operators[class_idx+1][idx][4].real:.10f}  +  {operators[class_idx+1][idx][4].imag:.10f}i\n")
                if (idx+1)%3==0:
                    f.write("\n")
    
    f.write(f"\nSix-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 6:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//15+1} type {idx%15+1} : Six-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}) * ({bond[4]}-{bond[5]}): {operators[class_idx+1][idx][6].real:.10f}  +  {operators[class_idx+1][idx][6].imag:.10f}i\n")
                if (idx+1)%15==0:
                    f.write("\n")
