import os
import sys
import time
from datetime import datetime
import networkx as nx
from itertools import combinations

from cuprate.io import (
    iter_cli_assignments,
    parse_bool_arg,
    tune_sz,
    build_legacy_result_dirs,
    find_existing_cluster_report,
    find_existing_run_dir,
    find_cluster_match,
    read_key,
    write_file,
    read_fit_metric,
    read_coords_file_Block,
    create_dict,
    read_operators,
    write_result_artifact,
    build_spin_coupling_artifact_from_operator_list,
    write_cluster_points,
    find_bond_vector,
    get_all_possible_vectors,
    k2s,
    k4s,
    k6s,
    k8s,
    ANALYSIS_ONLY_ERROR,
)

from cuprate.clusters import bond_analysis_spin


DEFAULT_N = 3
DEFAULT_U = 6.0
DEFAULT_T = 1.0

work_dir_block = "data_transfer/Block/Block_"
work_dir_lce = "data_transfer/LCE/LCE_"


def parse_arguments(rank=0):
    """Parse command-line arguments for LCE module. Returns a dict."""
    params = {}
    params["N"] = DEFAULT_N
    params["U"] = DEFAULT_U
    params["t"] = DEFAULT_T
    params["t2"] = None
    params["t3"] = None
    params["sz"] = None
    params["s2"] = None
    params["s2_fix"] = False
    params["type"] = None
    params["delta"] = None
    params["delta2"] = None
    params["restart"] = False
    params["type_delta"] = None

    params["Ncell"] = None
    params["Ncut"] = None
    params["ratio"] = None
    params["match_spin_sectors"] = False

    for key, value in iter_cli_assignments(sys.argv):
        if key == "N":
            params["N"] = int(value)
        elif key == "U":
            params["U"] = float(value)
        elif key == "T":
            params["t"] = float(value)
        elif key == "T2":
            params["t2"] = float(value)
        elif key == "T3":
            params["t3"] = float(value)
        elif key == "SZ":
            params["sz"] = float(value)
        elif key == "S2":
            params["s2"] = float(value)
        elif key == "S2_FIX":
            params["s2_fix"] = parse_bool_arg(value)
        elif key == "TYPE":
            params["type"] = value
        elif key == "DELTA":
            params["delta"] = float(value)
        elif key == "DELTA2":
            params["delta2"] = float(value)
        elif key == "RESTART":
            params["restart"] = parse_bool_arg(value)
        elif key == "TYPE_DELTA":
            params["type_delta"] = value
        elif key == "NCELL":
            params["Ncell"] = int(value)
        elif key == "NCUT":
            params["Ncut"] = int(value)
        elif key == "RATIO":
            params["ratio"] = float(value)
        elif key == "MATCH_SPIN_SECTORS":
            params["match_spin_sectors"] = parse_bool_arg(value)

    params["sz"] = tune_sz(params["sz"], params["N"], rank=rank)
    return params


def setup_work_environment(params):
    """Setup the working environment and parameters."""
    return build_legacy_result_dirs(params)


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


def operator_minus(operators1, operators2, map_from2to1):
    operators1[0] -= operators2[0]
    for i, operator_group in enumerate(operators2):
        if i == 0:
            continue
        for operator in operator_group:
            if len(operator) == 3:
                idx1, idx2, coeff = operator
                idx1_new, idx2_new = find_operator(
                    [map_from2to1[idx1], map_from2to1[idx2]], operators1
                )
                operators1[idx1_new][idx2_new][2] -= coeff
            elif len(operator) == 5:
                idx1, idx2, idx3, idx4, coeff = operator
                idx1_new, idx2_new = find_operator(
                    [
                        map_from2to1[idx1],
                        map_from2to1[idx2],
                        map_from2to1[idx3],
                        map_from2to1[idx4],
                    ],
                    operators1,
                )
                operators1[idx1_new][idx2_new][4] -= coeff
            elif len(operator) == 7:
                idx1, idx2, idx3, idx4, idx5, idx6, coeff = operator
                idx1_new, idx2_new = find_operator(
                    [
                        map_from2to1[idx1],
                        map_from2to1[idx2],
                        map_from2to1[idx3],
                        map_from2to1[idx4],
                        map_from2to1[idx5],
                        map_from2to1[idx6],
                    ],
                    operators1,
                )
                operators1[idx1_new][idx2_new][6] -= coeff
            elif len(operator) == 9:
                idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, coeff = operator
                idx1_new, idx2_new = find_operator(
                    [
                        map_from2to1[idx1],
                        map_from2to1[idx2],
                        map_from2to1[idx3],
                        map_from2to1[idx4],
                        map_from2to1[idx5],
                        map_from2to1[idx6],
                        map_from2to1[idx7],
                        map_from2to1[idx8],
                    ],
                    operators1,
                )
                operators1[idx1_new][idx2_new][8] -= coeff


def find_operator(idx_list, operators):
    for i, operator_group in enumerate(operators):
        if i == 0:
            continue
        for j, operator in enumerate(operator_group):
            if len(operator) == 3 and len(idx_list) == 2:
                idx1, idx2 = idx_list
                idx1_op, idx2_op, coeff = operator
                if (
                    (idx1 == idx1_op and idx2 == idx2_op)
                    or (idx1 == idx2_op and idx2 == idx1_op)
                ):
                    return i, j
            elif len(operator) == 5 and len(idx_list) == 4:
                idx1, idx2, idx3, idx4 = idx_list
                idx1_op, idx2_op, idx3_op, idx4_op, coeff = operator
                if (
                    (idx1 == idx1_op and idx2 == idx2_op and idx3 == idx3_op and idx4 == idx4_op)
                    or (idx1 == idx1_op and idx2 == idx2_op and idx3 == idx4_op and idx4 == idx3_op)
                    or (idx1 == idx2_op and idx2 == idx1_op and idx3 == idx3_op and idx4 == idx4_op)
                    or (idx1 == idx2_op and idx2 == idx1_op and idx3 == idx4_op and idx4 == idx3_op)
                    or (idx1 == idx3_op and idx2 == idx4_op and idx3 == idx1_op and idx4 == idx2_op)
                    or (idx1 == idx3_op and idx2 == idx4_op and idx3 == idx2_op and idx4 == idx1_op)
                    or (idx1 == idx4_op and idx2 == idx3_op and idx3 == idx1_op and idx4 == idx2_op)
                    or (idx1 == idx4_op and idx2 == idx3_op and idx3 == idx2_op and idx4 == idx1_op)
                ):
                    return i, j
            elif len(operator) == 7 and len(idx_list) == 6:
                idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op, coeff = operator
                if k6s(
                    idx_list[0],
                    idx_list[1],
                    idx_list[2],
                    idx_list[3],
                    idx_list[4],
                    idx_list[5],
                ) == k6s(idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op):
                    return i, j
            elif len(operator) == 9 and len(idx_list) == 8:
                idx1_op, idx2_op, idx3_op, idx4_op, idx5_op, idx6_op, idx7_op, idx8_op, coeff = operator
                if k8s(
                    idx_list[0],
                    idx_list[1],
                    idx_list[2],
                    idx_list[3],
                    idx_list[4],
                    idx_list[5],
                    idx_list[6],
                    idx_list[7],
                ) == k8s(
                    idx1_op,
                    idx2_op,
                    idx3_op,
                    idx4_op,
                    idx5_op,
                    idx6_op,
                    idx7_op,
                    idx8_op,
                ):
                    return i, j

    raise ValueError(f"Operator not found for indices: {idx_list}")


def write_operators(file_path, operators, cluster, bonds):
    with open(file_path, "w") as f:
        write_cluster_points(f, cluster)
        write_operators_LCE(f, cluster, bonds, operators)
    write_result_artifact(
        file_path,
        build_spin_coupling_artifact_from_operator_list(cluster, operators),
    )


def write_operators_LCE(f, cluster, bonds, operators):
    """Write individual bond coefficients."""
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {operators[0].real:.10f}\n")

    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in coeffs_by_vector:
                coeffs_by_vector[(dx, dy)] = []
            coeffs_by_vector[(dx, dy)].append((class_idx, bond_group, operators[class_idx + 1]))

    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if (dx, dy) in coeffs_by_vector:
            for class_idx, bond_group, coeffs in coeffs_by_vector[(dx, dy)]:
                f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}:\n")
                for idx, (bond, coef) in enumerate(zip(bond_group, coeffs)):
                    site1, site2 = bond
                    f.write(
                        f"    {idx}: Sites {site1}-{site2}: {coef[2].real:.10f}  +  {coef[2].imag:.10f}i\n"
                    )
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}: (not present in cluster)\n")

    f.write("\nFour-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 4:
            for idx, bond in enumerate(bond_group):
                f.write(
                    f"    class {idx // 3 + 1} type {idx % 3 + 1} : Four-site "
                    f"({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}): "
                    f"{operators[class_idx + 1][idx][4].real:.10f}  +  "
                    f"{operators[class_idx + 1][idx][4].imag:.10f}i\n"
                )
                if (idx + 1) % 3 == 0:
                    f.write("\n")

    f.write("\nSix-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 6:
            for idx, bond in enumerate(bond_group):
                f.write(
                    f"    class {idx // 15 + 1} type {idx % 15 + 1} : Six-site "
                    f"({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}) * ({bond[4]}-{bond[5]}): "
                    f"{operators[class_idx + 1][idx][6].real:.10f}  +  "
                    f"{operators[class_idx + 1][idx][6].imag:.10f}i\n"
                )
                if (idx + 1) % 15 == 0:
                    f.write("\n")


def main(params):
    result_dir1, result_dir2, result_dir3 = setup_work_environment(params)
    print(f"Working directory: {work_dir_lce}{result_dir1}/N{params['N']}{result_dir2}")
    block_root = f"{work_dir_block}{result_dir1}"
    lce_root = f"{work_dir_lce}{result_dir1}"

    clusters = read_coords_file_Block(block_root, result_dir2, result_dir3, params["N"])
    data = create_dict(clusters)

    for nsites in clusters:
        print(f"\n\nnsites: {nsites}")
        for hole in clusters[nsites]:
            print(f"hole: {hole}")
            for class_idx in clusters[nsites][hole]:
                print(f"class_idx: {class_idx}")
                for rank_idx in clusters[nsites][hole][class_idx]:
                    cluster = clusters[nsites][hole][class_idx][rank_idx]
                    subgraphs, indices = get_connected_subgraphs(cluster)
                    print(
                        f"\nsites: {nsites}, hole: {hole}, class_idx: {class_idx}, rank_idx: {rank_idx}"
                    )
                    print(f"cluster: {cluster}")
                    for i, subgraph in enumerate(subgraphs):
                        print(f"subgraph: {subgraph}")
                        match = find_cluster_match(subgraph, clusters, params["t2"])
                        data[nsites][hole][class_idx][rank_idx]["subgraph"].append(subgraph)
                        data[nsites][hole][class_idx][rank_idx]["indices"].append(indices[i])
                        data[nsites][hole][class_idx][rank_idx]["match"].append(match)

    for nsites in clusters:
        if nsites < 2:
            continue
        for hole in clusters[nsites]:
            for class_idx in clusters[nsites][hole]:
                for rank_idx in clusters[nsites][hole][class_idx]:
                    bonds = bond_analysis_spin(clusters[nsites][hole][class_idx][rank_idx])
                    file_path, _ = find_existing_cluster_report(
                        block_root,
                        nsites,
                        result_dir2,
                        result_dir3,
                        hole=hole,
                        class_idx=class_idx,
                        cluster_idx=rank_idx,
                    )
                    error = read_fit_metric(file_path, "relative_error")
                    if error is None:
                        error = read_key(file_path, "Relative")
                    t11 = read_fit_metric(file_path, "t11_minus_1_norm")
                    if t11 is None:
                        t11 = read_key(file_path, "T11")

                    data[nsites][hole][class_idx][rank_idx]["coupling_original"] = read_operators(file_path)
                    data[nsites][hole][class_idx][rank_idx]["coupling_net"] = read_operators(file_path)
                    for subgraph_idx, subgraph in enumerate(
                        data[nsites][hole][class_idx][rank_idx]["subgraph"]
                    ):
                        indices = data[nsites][hole][class_idx][rank_idx]["indices"][subgraph_idx]
                        match = data[nsites][hole][class_idx][rank_idx]["match"][subgraph_idx]
                        idx_n, idx_h, idx_c, idx_r, mapping = match
                        map_new = [indices[mapping[i]] for i in range(len(mapping))]
                        operator_minus(
                            data[nsites][hole][class_idx][rank_idx]["coupling_net"],
                            data[idx_n][idx_h][idx_c][idx_r]["coupling_net"],
                            map_new,
                        )

                    _, output_suffix = find_existing_run_dir(block_root, nsites, result_dir2, result_dir3)
                    output_dir = f"{lce_root}/N{nsites}{output_suffix}"
                    os.makedirs(output_dir, exist_ok=True)
                    file_path_new = (
                        f"{output_dir}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                    )
                    write_operators(
                        file_path_new,
                        data[nsites][hole][class_idx][rank_idx]["coupling_net"],
                        clusters[nsites][hole][class_idx][rank_idx],
                        bonds,
                    )

                    write_file(file_path_new, "")
                    write_file(file_path_new, f"This file is read from {file_path}.")
                    write_file(
                        file_path_new,
                        "Please check the following values, we did not check the fitting accuracy!",
                    )
                    write_file(file_path_new, f"Relative Error: {error}")
                    write_file(file_path_new, f"T11-1 Norm: {t11}")

        print(f"Now is saving the results in {lce_root}/N{nsites}{result_dir2}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python -m cuprate.lce [N=n] [U=u] [t=t]")
        print("  N: number of max sites (default: 3)")
        print("  U: Hubbard U parameter (default: 6.0)")
        print("  t: hopping parameter (default: 1.0)")
        print("\nExamples:")
        print("  python -m cuprate.lce              # Use default values")
        print("  python -m cuprate.lce N=4 U=3 t=2  # Custom values")
        sys.exit(0)
    params = parse_arguments()
    t0 = time.time()
    print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main(params)
    t1 = time.time()
    print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task is finished in {t1 - t0} seconds")
