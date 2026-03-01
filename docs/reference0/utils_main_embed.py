import sys
import numpy as np
import networkx as nx
from math import sqrt
from Clusters_Square import transform, find_connected_sites
from utils_main import write_cluster_points, get_all_possible_vectors
from utils_main_lce import k2s, k4s, k6s, k8s, k6s_all

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


def coord_to_index(x, y, size):
    return x*size+y

def index_to_coord(index, size):
    return index//size, index%size

# Here, size is the number of sites at the edge of the square cell.
def generate_square_cell(size):
    return [(x, y) for x in range(size) for y in range(size)]

def unique_cluster_transformed(cluster_input):
    ops = ['id', 'rot90', 'rot180', 'rot270', 'mirror_x', 'mirror_y', 'mirror_diag', 'mirror_anti']

    clusters_trans=[]
    for op in ops:
        transformed = transform(cluster_input, op)
        min_x = min(x for x, y in transformed)
        min_y = min(y for x, y in transformed)
        normalized = [(x - min_x, y - min_y) for x, y in transformed]
        clusters_trans.append(normalized)
        
    # 使用字典去除重复，保留第一次出现的顺序
    unique_dict = {}
    for cluster in clusters_trans:
        # 使用排序后的cluster作为键
        key = tuple(sorted(cluster))
        if key not in unique_dict:
            unique_dict[key] = cluster
    
    # 获取唯一的cluster（保持原始顺序）
    unique_clusters = list(unique_dict.values())

    return unique_clusters

# Embed the cluster into the supercell
# Here we request the size of cluster should be smaller than the size of supercell, and 
# the supercell should be a square in the first quadrant with the origin at (0,0).
# the index of the cell (i,j) is (i*size+j)
# Here, size is the number of sites at the edge of the square cell.
def embed_cluster_to_squarecell_obc(cluster_input, size):
    min_x = min(x for x, y in cluster_input)
    min_y = min(y for x, y in cluster_input)
    max_x = max(x for x, y in cluster_input)
    max_y = max(y for x, y in cluster_input)

    if max_x - min_x > size or max_y - min_y > size:
        raise ValueError("The size of the cluster is larger than the size of the square cell")
    cluster=[(x - min_x, y - min_y) for x, y in cluster_input]

    max_delta_x=size-(max_x-min_x)
    max_delta_y=size-(max_y-min_y)
    
    count=0
    indices_list = [[None for _ in range(len(cluster))] for _ in range(max_delta_x*max_delta_y)]

    for delta_x in range(max_delta_x):
        for delta_y in range(max_delta_y):
            for i, (x, y) in enumerate(cluster):
                index=coord_to_index(x+delta_x, y+delta_y, size)
                indices_list[count][i]=index
            count+=1

    return indices_list
    

def embed_cluster_to_squarecell_pbc(cluster_input, size):
    def idx_pbc(x, size):
        return x%size
    
    min_x = min(x for x, y in cluster_input)
    min_y = min(y for x, y in cluster_input)
    max_x = max(x for x, y in cluster_input)
    max_y = max(y for x, y in cluster_input)

    if max_x - min_x > size or max_y - min_y > size:
        raise ValueError("The size of the cluster is larger than the size of the square cell")
    cluster=[(x - min_x, y - min_y) for x, y in cluster_input]

    max_delta_x=size
    max_delta_y=size
    
    count=0
    indices_list = [[None for _ in range(len(cluster))] for _ in range(max_delta_x*max_delta_y)]

    for delta_x in range(max_delta_x):
        for delta_y in range(max_delta_y):
            for i, (x, y) in enumerate(cluster):
                index=coord_to_index(idx_pbc(x+delta_x, size), idx_pbc(y+delta_y, size), size)
                indices_list[count][i]=index
            count+=1

    return indices_list

def embed_operator(couplings, coupling_net, indices):
    couplings[0]+=coupling_net[0]
    for i, operators_group in enumerate(coupling_net):
        if i==0: continue
        for operator in operators_group:
            if len(operator)==3:
                idx1, idx2, value=operator
                couplings[1][indices[idx1], indices[idx2]]+=value
                couplings[1][indices[idx2], indices[idx1]]+=value
            elif len(operator)==5:
                idx1, idx2, idx3, idx4, value=operator
                key=k4s(indices[idx1], indices[idx2], indices[idx3], indices[idx4])
                if key not in couplings[2]:
                    couplings[2][key]=0.0
                couplings[2][key]+=value
            elif len(operator)==7:
                idx1, idx2, idx3, idx4, idx5, idx6, value=operator
                key=k6s(indices[idx1], indices[idx2], indices[idx3], indices[idx4], indices[idx5], indices[idx6])
                if key not in couplings[3]:
                    couplings[3][key]=0.0
                couplings[3][key]+=value
            elif len(operator)==9:
                idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, value=operator
                key=k8s(indices[idx1], indices[idx2], indices[idx3], indices[idx4], indices[idx5], indices[idx6], indices[idx7], indices[idx8])
                if key not in couplings[4]:
                    couplings[4][key]=0.0
                couplings[4][key]+=value



def bond_analysis_foursites(cluster, status="obc"):
    dict_indices=[[] for _ in range(5)]
    foursites_indices_list = find_connected_sites(cluster, 4)
    for foursites_indices in foursites_indices_list:
        idx_type, idx_sites=normalize_four_sites(foursites_indices, cluster)
        dict_indices[idx_type].append(idx_sites)
    return dict_indices

def bond_analysis_sixsites(cluster, status="obc"):
    dict_indices=[[] for _ in range(15)]
    print("Computing all possible six-site.")
    sixsites_indices_list = find_connected_sites(cluster, 6)
    print("Finish all possible six-site.")
    for sixsites_indices in sixsites_indices_list:
        idx_type, idx_sites=normalize_six_sites(sixsites_indices, cluster)
        dict_indices[idx_type].append(idx_sites)
    return dict_indices


def coupling_twosites_vector_from_matrix(couplings_twosites, cluster, status="obc"):
    size=int(sqrt(len(cluster)))
    all_vectors=get_all_possible_vectors(cluster, size)
    v2i = {vec: i for i, vec in enumerate(all_vectors)}
    dict_couplings=[[] for _ in range(len(all_vectors))]
    
    for i in range(len(cluster)):
        for j in range(i+1, len(cluster)):
            dx=cluster[j][0]-cluster[i][0]
            dy=cluster[j][1]-cluster[i][1]
            
            if status=="obc":
                key=tuple(sorted((abs(dx), abs(dy)), reverse=True))
            elif status=="pbc":
                key = min([
                    tuple(sorted((abs(dx), abs(dy)), reverse=True)),
                    tuple(sorted((abs(dx-size), abs(dy-size)), reverse=True)),
                    tuple(sorted((abs(dx-size), abs(dy)), reverse=True)),
                    tuple(sorted((abs(dx), abs(dy-size)), reverse=True))
                ], key=lambda k: k[0]**2 + k[1]**2)
            else:
                raise ValueError("The status is not valid")
                
            if key not in all_vectors:
                raise ValueError("The vector is not in the all_vectors ", key)
            dict_couplings[v2i[key]].append([i, j, couplings_twosites[i][j]])
    
    for i in range(len(dict_couplings)):
        dict_couplings[i]=sorted(dict_couplings[i], key=lambda x: x[2]*np.conj(x[2]), reverse=True)
    
    return dict_couplings


def coupling_foursites_vector_from_dict(couplings_foursites, cluster, status="obc"):
    dict_indices=bond_analysis_foursites(cluster, status)
    dict_couplings=[[] for _ in range(5)]

    #print(couplings_foursites)
    #print(dict_indices)
    for idx_type, sites_group in enumerate(dict_indices):
        for sites in sites_group:
            key1=k4s(sites[0], sites[1], sites[2], sites[3])
            key2=k4s(sites[0], sites[3], sites[1], sites[2])
            key3=k4s(sites[0], sites[2], sites[1], sites[3])
            if key1 in couplings_foursites and key2 in couplings_foursites and key3 in couplings_foursites:
                dict_couplings[idx_type].append([sites[0], sites[1], sites[2], sites[3], couplings_foursites[key1], couplings_foursites[key2], couplings_foursites[key3]])
            elif key1 not in couplings_foursites and key2 not in couplings_foursites and key3 not in couplings_foursites:
                #print(f"The four-site bond {sites} is not included in the couplings_foursites, it's ignored")
                a_tmp=1
            else:
                #if key1 not in couplings_foursites:
                #    print(f"The key {key1} is not included in the couplings_foursites.")
                #if key2 not in couplings_foursites:
                #    print(f"The key {key2} is not included in the couplings_foursites.")
                #if key3 not in couplings_foursites:
                #    print(f"The key {key3} is not included in the couplings_foursites.")
                raise ValueError(f"Missing key in couplings_foursites")
        dict_couplings[idx_type]=sorted(dict_couplings[idx_type], key=lambda x: x[4]*np.conj(x[4])+x[5]*np.conj(x[5])+x[6]*np.conj(x[6]), reverse=True)

    return dict_couplings



def coupling_foursites_vector_from_dict2(couplings_foursites, cluster, status="obc"):
    dict_indices=bond_analysis_foursites(cluster, status)
    dict_couplings=[[], [], [], [], []]

    #print(couplings_foursites)
    #print(dict_indices)
    key_list=[]
    value_list=[]
    for coupling in couplings_foursites:
        key_site=sorted((coupling[0], coupling[1], coupling[2], coupling[3]))
        key_list.append(key_site)
        value_list.append(coupling[-1])
    for idx_type, sites_group in enumerate(dict_indices):
        #print(f"idx_type: {idx_type}")
        #print(f"sites_group: {sites_group}")
        for sites in sites_group:
            print(f"sites: {sites}")
            key_site=sorted((sites[0], sites[1], sites[2], sites[3]))
            if key_site in key_list:
                idx=key_list.index(key_site)
                dict_couplings[idx_type].append([sites[0], sites[1], sites[2], sites[3], value_list[idx]])
            else:
                raise ValueError(f"Missing key in couplings_foursites")

 
    return dict_couplings


def coupling_sixsites_vector_from_dict(couplings_sixsites, cluster, status="obc"):
    print("Starting bond analysis")
    dict_indices=bond_analysis_sixsites(cluster, status)
    print("Finish bond analysis")
    dict_couplings=[[] for _ in range(15)]
    for idx_type, sites_group in enumerate(dict_indices):
        for sites in sites_group:
            # Generate all possible six-site coupling keys
            keys = [
                k6s(sites[0], sites[1], sites[2], sites[3], sites[4], sites[5]),
                k6s(sites[0], sites[1], sites[4], sites[2], sites[3], sites[5]),
                k6s(sites[0], sites[1], sites[5], sites[2], sites[3], sites[4]),
                k6s(sites[0], sites[2], sites[1], sites[3], sites[4], sites[5]),
                k6s(sites[0], sites[2], sites[4], sites[1], sites[3], sites[5]),
                k6s(sites[0], sites[2], sites[5], sites[1], sites[3], sites[4]),
                k6s(sites[0], sites[3], sites[1], sites[2], sites[4], sites[5]),
                k6s(sites[0], sites[3], sites[4], sites[1], sites[2], sites[5]),
                k6s(sites[0], sites[3], sites[5], sites[1], sites[2], sites[4]),
                k6s(sites[0], sites[4], sites[1], sites[2], sites[3], sites[5]),
                k6s(sites[0], sites[4], sites[3], sites[1], sites[2], sites[5]),
                k6s(sites[0], sites[4], sites[5], sites[1], sites[2], sites[3]),
                k6s(sites[0], sites[5], sites[1], sites[2], sites[3], sites[4]),
                k6s(sites[0], sites[5], sites[3], sites[1], sites[2], sites[4]),
                k6s(sites[0], sites[5], sites[4], sites[1], sites[2], sites[3])
            ]
            
            # Check if all keys are present in couplings_sixsites
            missing_keys = [key for key in keys if key not in couplings_sixsites]
            
            if not missing_keys:
                # All keys are present, add the coupling with all 15 coupling values
                coupling_values = [couplings_sixsites[key] for key in keys]
                dict_couplings[idx_type].append([sites[0], sites[1], sites[2], sites[3], sites[4], sites[5]] + coupling_values)
            elif len(missing_keys) == len(keys):
                # No keys are present, ignore this bond
                #print(f"The six-site bond {sites} is not included in the couplings_sixsites, it's ignored")
                a_tmp=1
            else:
                # Some keys are missing, report and raise error
                for key in missing_keys:
                    print(f"The key {key} is not included in the couplings_sixsites.")
                raise ValueError(f"Missing keys in couplings_sixsites")
        dict_couplings[idx_type]=sorted(dict_couplings[idx_type], key=lambda x: sum(x[i]*np.conj(x[i]) for i in range(6, 21)), reverse=True)

    return dict_couplings

def coupling_sixsites_vector_from_dict_simple(couplings_sixsites, cluster):
    dict_couplings=[[]]
    keys=couplings_sixsites.keys()

    return dict_couplings

def save_data_embed(filename, couplings, cluster, status="obc"):
    with open(filename, 'w') as f:
        write_cluster_points(f, cluster)
        f.write(f"\n=== Grid ===\n")
        f.write("\n"+print_grid(int(sqrt(len(cluster))), status)+"\n")
        write_couplings_embed(f, couplings, cluster, status)


def write_couplings_embed(f, couplings, cluster, status="obc"):
    square_cell_str=["\n4--3\n|  |\n1--2\n", 
                     "\n   3\n   |\n2--1--4\n", 
                     "\n1--2--3--4\n", 
                     "\n      4\n      |\n1--2--3\n", 
                     "\n   3--4\n   |\n1--2\n"]

    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {couplings[0].real:.10f}\n")

    all_vectors = get_all_possible_vectors(cluster, int(sqrt(len(cluster))))
    couplings_twosites_vector=coupling_twosites_vector_from_matrix(couplings[1], cluster, status)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if(len(couplings_twosites_vector[bond_idx])>0):
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:\n")
            for idx, (site1, site2, coef) in enumerate(couplings_twosites_vector[bond_idx]):
                f.write(f"    {idx}: Sites {site1}-{site2}: {coef.real:.10f}  +  {coef.imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}: (not present in cluster)\n")
    
    f.write(f"\nFour-site Bond:\n")
    couplings_foursites_vector=coupling_foursites_vector_from_dict(couplings[2], cluster)
    for idx_type, sites_group in enumerate(couplings_foursites_vector):
        if len(sites_group)>0:
            f.write(f"\nBond type {idx_type}:\n")
            f.write(square_cell_str[idx_type])
            for idx, sites in enumerate(sites_group):
                f.write(f"    class {idx} type 1 : Four-site ({sites[0]}-{sites[1]}) * ({sites[2]}-{sites[3]}): {sites[4].real:.10f}  +  {sites[4].imag:.10f}i\n")
                f.write(f"    class {idx} type 2 : Four-site ({sites[0]}-{sites[3]}) * ({sites[1]}-{sites[2]}): {sites[5].real:.10f}  +  {sites[5].imag:.10f}i\n")
                f.write(f"    class {idx} type 3 : Four-site ({sites[0]}-{sites[2]}) * ({sites[1]}-{sites[3]}): {sites[6].real:.10f}  +  {sites[6].imag:.10f}i\n")
                f.write("\n")
        else:
            f.write(square_cell_str[idx_type])
            f.write(f"\nBond type {idx_type+1}: (not present in cluster)\n")

    f.write(f"\nSix-site Bond:\n")
    key_list=[]
    count_class=0
    count_bond=0
    for key, value in couplings[3].items():
        '''
        key_all=k6s_all(key[0][0], key[0][1], key[1][0], key[1][1], key[2][0], key[2][1])
        if key_all not in key_list:
            key_list.append(key_all)
            count_class+=1
            count_bond=0
        count_bond+=1
        '''
        f.write(f"    class {count_class} type {count_bond} : Six-site ({key[0][0]}-{key[0][1]}) * ({key[1][0]}-{key[1][1]}) * ({key[2][0]}-{key[2][1]}): {value.real:.10f}  +  {value.imag:.10f}i\n")

    '''
    f.write(f"\nSix-site Bond:\n")
    if len(couplings[3])==0:
        couplings_sixsites_vector=[]
    else:
        couplings_sixsites_vector=coupling_sixsites_vector_from_dict(couplings[3], cluster)
    for idx_type, sites_group in enumerate(couplings_sixsites_vector):
        if len(sites_group)>0:
            f.write(f"\nBond type {idx_type}:\n")
            for idx, sites in enumerate(sites_group):
                f.write(f"    class {idx} type 1  : Six-site ({sites[0]}-{sites[1]}) * ({sites[2]}-{sites[3]}) * ({sites[4]}-{sites[5]}): {sites[6].real:.10f}  +  {sites[6].imag:.10f}i\n")
                f.write(f"    class {idx} type 2  : Six-site ({sites[0]}-{sites[1]}) * ({sites[4]}-{sites[2]}) * ({sites[3]}-{sites[5]}): {sites[7].real:.10f}  +  {sites[7].imag:.10f}i\n")
                f.write(f"    class {idx} type 3  : Six-site ({sites[0]}-{sites[1]}) * ({sites[5]}-{sites[2]}) * ({sites[3]}-{sites[4]}): {sites[8].real:.10f}  +  {sites[8].imag:.10f}i\n")
                f.write(f"    class {idx} type 4  : Six-site ({sites[0]}-{sites[2]}) * ({sites[1]}-{sites[3]}) * ({sites[4]}-{sites[5]}): {sites[9].real:.10f}  +  {sites[9].imag:.10f}i\n")
                f.write(f"    class {idx} type 5  : Six-site ({sites[0]}-{sites[2]}) * ({sites[4]}-{sites[1]}) * ({sites[3]}-{sites[5]}): {sites[10].real:.10f}  +  {sites[10].imag:.10f}i\n")
                f.write(f"    class {idx} type 6  : Six-site ({sites[0]}-{sites[2]}) * ({sites[5]}-{sites[1]}) * ({sites[3]}-{sites[4]}): {sites[11].real:.10f}  +  {sites[11].imag:.10f}i\n")
                f.write(f"    class {idx} type 7  : Six-site ({sites[0]}-{sites[3]}) * ({sites[1]}-{sites[2]}) * ({sites[4]}-{sites[5]}): {sites[12].real:.10f}  +  {sites[12].imag:.10f}i\n")
                f.write(f"    class {idx} type 8  : Six-site ({sites[0]}-{sites[3]}) * ({sites[4]}-{sites[1]}) * ({sites[2]}-{sites[5]}): {sites[13].real:.10f}  +  {sites[13].imag:.10f}i\n")
                f.write(f"    class {idx} type 9  : Six-site ({sites[0]}-{sites[3]}) * ({sites[5]}-{sites[1]}) * ({sites[2]}-{sites[4]}): {sites[14].real:.10f}  +  {sites[14].imag:.10f}i\n")
                f.write(f"    class {idx} type 10 : Six-site ({sites[0]}-{sites[4]}) * ({sites[1]}-{sites[2]}) * ({sites[3]}-{sites[5]}): {sites[15].real:.10f}  +  {sites[15].imag:.10f}i\n")
                f.write(f"    class {idx} type 11 : Six-site ({sites[0]}-{sites[4]}) * ({sites[3]}-{sites[1]}) * ({sites[2]}-{sites[5]}): {sites[16].real:.10f}  +  {sites[16].imag:.10f}i\n")
                f.write(f"    class {idx} type 12 : Six-site ({sites[0]}-{sites[4]}) * ({sites[5]}-{sites[1]}) * ({sites[2]}-{sites[3]}): {sites[17].real:.10f}  +  {sites[17].imag:.10f}i\n")
                f.write(f"    class {idx} type 13 : Six-site ({sites[0]}-{sites[5]}) * ({sites[1]}-{sites[2]}) * ({sites[3]}-{sites[4]}): {sites[18].real:.10f}  +  {sites[18].imag:.10f}i\n")
                f.write(f"    class {idx} type 14 : Six-site ({sites[0]}-{sites[5]}) * ({sites[3]}-{sites[1]}) * ({sites[2]}-{sites[4]}): {sites[19].real:.10f}  +  {sites[19].imag:.10f}i\n")
                f.write(f"    class {idx} type 15 : Six-site ({sites[0]}-{sites[5]}) * ({sites[4]}-{sites[1]}) * ({sites[2]}-{sites[3]}): {sites[20].real:.10f}  +  {sites[20].imag:.10f}i\n")
                f.write("\n")
        else:
            f.write(f"\nBond type {idx_type+1}: (not present in cluster)\n")
        '''



def normalize_four_sites(indices, cluster):
    """将四个点的连接关系标准化为五种基本形式之一。
    
    Args:
        points: 四个点的坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        connections: 连接关系列表，每个元素是 (i,j) 表示点i和点j相连
        
    Returns:
        tuple: (标准化后的点列表, 类型编号(1-5))

        类型1: 4--3
              |  |
              1--2

        类型2:    3
                 |
              2--1--4

        类型3: 1--2--3--4

        类型4:       4
                    |
              1--2--3
              
        类型5:    3--4
                 |
              1--2
    """
    def get_adjacency_matrix(sites):
        """根据连接关系创建邻接矩阵"""
        adj = np.zeros([len(sites), len(sites)], dtype=int)
        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                if abs(sites[i][0]-sites[j][0])+abs(sites[i][1]-sites[j][1])==1:
                    adj[i][j] = adj[j][i] = 1
        return adj
    
    def get_degrees(adj):
        """获取每个点的度数"""
        return [sum(row) for row in adj]
    
    def is_square(adj):
        """判断是否为方形结构（类型1）"""
        degrees = get_degrees(adj)
        return all(d == 2 for d in degrees)
    
    def is_star(adj):
        """判断是否为星形结构（类型2）"""
        degrees = get_degrees(adj)
        return max(degrees) == 3 and degrees.count(1) == 3
        
    def is_linear_structure(adj):
        """判断是否为线性结构（类型3、4、5）"""
        degrees = get_degrees(adj)
        return degrees.count(1) == 2 and degrees.count(2) == 2
    
    def count_turns(points, adj):
        """计算线性结构中的拐弯数量"""
        # 找到端点
        ends = [i for i, d in enumerate(get_degrees(adj)) if d == 1]
        start = ends[0]
        
        # 按顺序遍历所有点
        current = start
        path = [start]
        
        while len(path) < 4:
            for j in range(4):
                if adj[current][j] and j not in path:
                    path.append(j)
                    current = j
                    break
        
        # 计算拐弯数量
        turns = 0
        for i in range(1, len(path)-1):
            p1 = points[path[i-1]]
            p2 = points[path[i]]
            p3 = points[path[i+1]]
            
            # 计算两个向量的方向
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # 如果方向不同，说明有拐弯
            if v1[0] * v2[1] != v1[1] * v2[0]:  # 向量叉积不为0表示方向不同
                turns += 1
        
        return turns
    
    def reorder_points(points, adj, type_num):
        if type_num == 0:  # Square
            start = 0
            # Sort in clockwise order
            ordered = [start]
            current = start
            for _ in range(3):
                for j in range(4):
                    if adj[current][j] and j not in ordered:
                        ordered.append(j)
                        current = j
                        break
            return ordered
            
        elif type_num == 1:  # Star
            # Center point has maximum degree
            center = get_degrees(adj).index(3)
            others = [i for i in range(4) if i != center]
            
            # Find points on opposite sides of center
            center_point = points[center]
            other_points = [points[i] for i in others]
            
            # Calculate vectors from center to other points
            vectors = [(p[0] - center_point[0], p[1] - center_point[1]) for p in other_points]
            
            # Find points that are on opposite sides (dot product < 0)
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    if vectors[i][0]*vectors[j][0] + vectors[i][1]*vectors[j][1] < 0:
                        # Found opposite points, put them in positions 2 and 3
                        opposite1, opposite2 = others[i], others[j]
                        remaining = [x for x in others if x not in [opposite1, opposite2]][0]
                        return [center, opposite1, remaining, opposite2]
            
            # If no opposite points found, return original order
            return [center] + others
            
        elif type_num < 5:  # Linear structures (type 2,3,4)
            # Reorder points for linear structures
            ends = [i for i, d in enumerate(get_degrees(adj)) if d == 1]
            start = ends[0]
            current = start
            ordered = [start]
            
            # First find the path
            path = [start]
            current = start
            while len(path) < 4:
                for j in range(4):
                    if adj[current][j] and j not in path:
                        path.append(j)
                        current = j
                        break
            
            if type_num == 2 or type_num == 4:
                return path
            elif type_num == 3:
                # Find the turning point by checking direction changes
                p1 = points[path[0]]
                p2 = points[path[1]]
                p3 = points[path[2]]
                    
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    
                if v1[0] * v2[1] != v1[1] * v2[0]:  # Direction changes
                    return [path[3], path[2], path[1], path[0]]
                else:
                    return path
        else:
            raise ValueError("The type number is not valid")
            
    sites=[cluster[idx] for idx in indices]
    adj = get_adjacency_matrix(sites)
    degrees = get_degrees(adj)
    
    # 判断类型
    if all(d == 2 for d in degrees):  # 所有点度数为2
        type_num = 0  # 方形
    elif max(degrees) == 3 and degrees.count(1) == 3:
        type_num = 1  # 星形
    elif is_linear_structure(adj):
        # 计算拐弯数量
        turns = count_turns(sites, adj)
        if turns == 0:
            type_num = 2  # 直线
        elif turns == 1:
            type_num = 3  # 一个拐弯
        else:
            type_num = 4  # 两个拐弯
    else:
        raise ValueError("Cannot identify the type of the four-site bond.")
    
    # Get reordered indices
    ordered_indices = reorder_points(sites, adj, type_num)
    
    # Map indices back to original indices list
    ordered_original_indices = [indices[i] for i in ordered_indices]
    
    return type_num, ordered_original_indices


def normalize_six_sites(indices, cluster):
    return 0, indices


def print_grid(size, status="obc"):
    """Print an NxN grid with cell indices at grid points.
    
    Args:
        size: Size of the grid (NxN)
    """
    def repeat_str(str, n):
        return str*n
    
    def int2str(num, width=3):
        s = str(num)
        # Calculate padding on each side
        padding = width - len(s)
        left = padding // 2
        right = padding - left
        return ' ' * left + s + ' ' * right
    
    def str_space(width):
        # Calculate padding on each side
        padding = width - 1
        left = padding // 2
        right = padding - left
        return ' ' * left + "|" + ' ' * right
    
    
    digit=len(str(size*size-1))
    width=2*digit-1

    indices=np.zeros([size, size], dtype=int)
    for i in range(size):
        for j in range(size):
            indices[i][j]=i*size+j

    line=repeat_str("-", 2*digit)
    line_pbc=repeat_str("- ", digit)
    line_space=repeat_str(" ", 2*digit)
    line_space_row=repeat_str(str_space(width)+line_space, size)
    if status=="pbc":
        line_space_row=repeat_str(" ", 4*digit-1)+line_space_row
    line_space_row=repeat_str(line_space_row+"\n", digit)

    str_grid=""
    for i in range(size):
        str_grid_row=""
        if status=="pbc":
            str_grid_row=str_grid_row+int2str(indices[i][size-1], width)+line_pbc

        for j in range(size):
            idx=indices[i][j]
            if j<size-1:
                str_grid_row=str_grid_row+int2str(idx, width)+line
            else:
                str_grid_row=str_grid_row+int2str(idx, width)

            if status=="pbc" and j==size-1:
                str_grid_row=str_grid_row+line_pbc+int2str(indices[i][0], width)

        if i<size-1:
            str_grid=line_space_row+str_grid_row+"\n"+str_grid
        else:
            str_grid=str_grid_row+"\n"+str_grid

    
    str1=repeat_str(" ", width)
    str2=repeat_str(" ", width)
    str3=repeat_str(" ", width)
    if status=="pbc":
        for j in range(size):
            str1=str1+line_space+int2str(indices[0][j], width)
            str2=str2+line_space+int2str(indices[size-1][j], width)
            str3=str3+line_space+str_space(width)
        str_grid=str1+"\n"+(str3+"\n")*digit+str_grid+(str3+"\n")*digit+str2+"\n"
    return str_grid
