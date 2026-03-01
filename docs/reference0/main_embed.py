import sys
import time
import numpy as np
from datetime import datetime
from Clusters_Square import *
from utils_main import parse_arguments, tune_sz
from utils_main_embed import *
from utils_main_lce import read_coords_file_Block, create_dict, read_operators



work_dir_block = "data_transfer/Block/Block_"
work_dir_lce   = "data_transfer/LCE/LCE_"
work_dir_embed = "data_transfer/Embed/Embed_"
def main(params):
    result_dir1, result_dir2, result_dir3 = setup_work_environment(params)
    print(f"Working directory: {work_dir_embed}{result_dir1}/Ncell{params['Ncell']}_Ncut{params['Ncut']}{result_dir2}")

    clusters = read_coords_file_Block(f"{work_dir_lce}{result_dir1}", result_dir2, result_dir3, params['Ncut'])
    data = create_dict(clusters)
    
    square_cell=generate_square_cell(params['Ncell'])
    #couplings_obc=[0.0, np.zeros([params['Ncell']*params['Ncell'], params['Ncell']*params['Ncell']], dtype=complex), {}, {}, {}]
    couplings_pbc=[0.0, np.zeros([params['Ncell']*params['Ncell'], params['Ncell']*params['Ncell']], dtype=complex), {}, {}, {}]
    for nsites in clusters:
        if nsites<2: continue
        for hole in clusters[nsites]:
            for class_idx in clusters[nsites][hole]:
                for rank_idx in clusters[nsites][hole][class_idx]:
                    file_path=f"{work_dir_lce}{result_dir1}/N{nsites}{result_dir2}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                    if not os.path.exists(file_path):
                        file_path=f"{work_dir_lce}{result_dir1}/N{nsites}{result_dir3}/hole{hole}_class{class_idx}_cluster{rank_idx}_results.txt"
                        if not os.path.exists(file_path):
                            print(f"File {file_path} does not exist")
                            exit(1)
                    data[nsites][hole][class_idx][rank_idx]["coupling_net"]=read_operators(file_path)
                    
                    indices_obc_list=[]
                    indices_pbc_list=[]
                    cluster=clusters[nsites][hole][class_idx][rank_idx]
                    unique_clusters=unique_cluster_transformed(cluster)
                    for unique_cluster in unique_clusters:
                        #indices_obc=embed_cluster_to_squarecell_obc(unique_cluster, params['Ncell'])
                        #indices_obc_list.extend(indices_obc)
                        indices_pbc=embed_cluster_to_squarecell_pbc(unique_cluster, params['Ncell'])
                        indices_pbc_list.extend(indices_pbc)
                    #for indices_obc in indices_obc_list:
                    #    embed_operator(couplings_obc, data[nsites][hole][class_idx][rank_idx]["coupling_net"], indices_obc)
                    for indices_pbc in indices_pbc_list:
                        embed_operator(couplings_pbc, data[nsites][hole][class_idx][rank_idx]["coupling_net"], indices_pbc)


    prefix_output=f"{work_dir_embed}{result_dir1}/Ncell{params['Ncell']}_Ncut{params['Ncut']}{result_dir2}"
    os.makedirs(prefix_output, exist_ok=True)
    #save_data_embed(f"{prefix_output}/couplings_obc.txt", couplings_obc, square_cell, status="obc")
    save_data_embed(f"{prefix_output}/couplings_pbc.txt", couplings_pbc, square_cell, status="pbc")
                    
        
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python main.py [N=n] [U=u] [t=t]")
        print("  N: number of max sites (default: 3)")
        print("  U: Hubbard U parameter (default: 6.0)")
        print("  t: hopping parameter (default: 1.0)")
        print("  Ncell: number of cells in the square cell (default: 8)")
        print("\nExamples:")
        print("  python main_embed.py              # Use default values")
        print("  python main_embed.py N=4 U=3 t=2  # Custom values")
        sys.exit(0)
    params = parse_arguments()
    params['N'] = params['Ncut']
    params['sz'] = tune_sz(params['sz'], params['N'], rank=1)
    t0=time.time()
    print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main(params)
    t1=time.time()
    print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task is finished in {t1-t0} seconds")