import os
import sys
import time
from datetime import datetime

from cuprate.lce.cli import parse_arguments
from cuprate.clusters.square import bond_analysis_spin
from cuprate.lce.utils import (
    setup_work_environment, create_dict, get_connected_subgraphs,
    find_cluster_match, read_coords_file_Block, read_operators,
    operator_minus, write_operators, write_operators_LCE,
)
from cuprate.shared.io import read_key, write_file

work_dir_block = "data_transfer/Block/Block_"
work_dir_lce = "data_transfer/LCE/LCE_"

def main(params):
    result_dir1, result_dir2, result_dir3 = setup_work_environment(params)
    print(f"Working directory: {work_dir_lce}{result_dir1}/N{params['N']}{result_dir2}")

    clusters = read_coords_file_Block(f"{work_dir_block}{result_dir1}", result_dir2, result_dir3, params['N'])
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
                    print(f"\nsites: {nsites}, hole: {hole}, class_idx: {class_idx}, rank_idx: {rank_idx}")
                    print(f"cluster: {cluster}")
                    for i, subgraph in enumerate(subgraphs):
                        print(f"subgraph: {subgraph}")
                        match = find_cluster_match(subgraph, clusters, params['t2'])
                        data[nsites][hole][class_idx][rank_idx]["subgraph"].append(subgraph)
                        data[nsites][hole][class_idx][rank_idx]["indices"].append(indices[i])
                        data[nsites][hole][class_idx][rank_idx]["match"].append(match)

    for nsites in clusters:
        if nsites<2: continue
        for hole in clusters[nsites]:
            for class_idx in clusters[nsites][hole]:
                for rank_idx in clusters[nsites][hole][class_idx]:
                    bonds=bond_analysis_spin(clusters[nsites][hole][class_idx][rank_idx])
                    file_path=f"{work_dir_block}{result_dir1}/N{nsites}{result_dir2}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                    if not os.path.exists(file_path):
                        file_path=f"{work_dir_block}{result_dir1}/N{nsites}{result_dir3}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                        if not os.path.exists(file_path):
                            print(f"File {file_path} does not exist")
                            exit(1)
                    error=read_key(file_path, "Relative")
                    t11=read_key(file_path, "T11")

                    data[nsites][hole][class_idx][rank_idx]["coupling_original"]=read_operators(file_path)
                    data[nsites][hole][class_idx][rank_idx]["coupling_net"]=read_operators(file_path)
                    for subgraph_idx, subgraph in enumerate(data[nsites][hole][class_idx][rank_idx]["subgraph"]):
                        indices=data[nsites][hole][class_idx][rank_idx]["indices"][subgraph_idx]
                        match=data[nsites][hole][class_idx][rank_idx]["match"][subgraph_idx]
                        idx_n, idx_h, idx_c, idx_r, map = match
                        map_new=[indices[map[i]] for i in range(len(map))]
                        operator_minus(data[nsites][hole][class_idx][rank_idx]["coupling_net"], data[idx_n][idx_h][idx_c][idx_r]["coupling_net"], map_new)

                    if os.path.exists(f"{work_dir_block}{result_dir1}/N{nsites}{result_dir2}"):
                        os.makedirs(f"{work_dir_lce}{result_dir1}/N{nsites}{result_dir2}", exist_ok=True)
                        file_path_new=f"{work_dir_lce}{result_dir1}/N{nsites}{result_dir2}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                    else:
                        os.makedirs(f"{work_dir_lce}{result_dir1}/N{nsites}{result_dir3}", exist_ok=True)
                        file_path_new=f"{work_dir_lce}{result_dir1}/N{nsites}{result_dir3}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                    write_operators(file_path_new, data[nsites][hole][class_idx][rank_idx]["coupling_net"], clusters[nsites][hole][class_idx][rank_idx], bonds)

                    write_file(file_path_new, f"")
                    write_file(file_path_new, f"This file is read from {file_path}.")
                    write_file(file_path_new, f"Please check the following values, we did not check the fitting accuracy!")
                    write_file(file_path_new, f"Relative Error: {error}")
                    write_file(file_path_new, f"T11-1 Norm: {t11}")

        print(f"Now is saving the results in {work_dir_lce}{result_dir1}/N{nsites}{result_dir2}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python -m cuprate.lce [N=n] [U=u] [t=t]")
        print("  N: number of max sites (default: 3)")
        print("  U: Hubbard U parameter (default: 6.0)")
        print("  t: hopping parameter (default: 1.0)")
        print("\nExamples:")
        print("  python -m cuprate.lce              # Use default values")
        print("  python -m cuprate.lce N=4 U=3 t=2  # Custom values")
        sys.exit(0)
    params = parse_arguments()
    t0=time.time()
    print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main(params)
    t1=time.time()
    print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task is finished in {t1-t0} seconds")
