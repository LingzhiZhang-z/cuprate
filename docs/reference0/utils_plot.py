import os
import math
import numpy as np
import matplotlib.pyplot as plt
from utils_main_embed import normalize_four_sites
from utils_main_lce import read_operators

def parse_coupling_params_embed(operator_lines):
    flag=False
    line_start="=== Individual Bond Coefficients ===\n"
    line_end="=== Individual Fit Error ===\n"
    coupling_params = []
    coupling_params_foursite=[]
    coupling_params_sixsite=[]
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
        elif len(parts)==3 and parts[0]=="Bond":
            #Four-site Bond:
            #Bond type count:
            count+=1
            coupling_params.append([])
        elif parts[0]=="class" and parts[5]=="Four-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('():').split('-'))
            coupling=float(parts[9])+1.0j*float(parts[11].rstrip('i'))
            coupling_params[count].append([idx1, idx2, idx3, idx4, coupling])
            coupling_params_foursite.append([idx1, idx2, idx3, idx4, coupling])
        elif parts[0]=="class" and parts[5]=="Six-site":
            idx1, idx2 = map(int, parts[6].strip('()').split('-'))
            idx3, idx4 = map(int, parts[8].strip('()').split('-'))
            idx5, idx6 = map(int, parts[10].strip('():').split('-'))
            coupling=float(parts[11])+1.0j*float(parts[13].rstrip('i'))
            coupling_params_sixsite.append([idx1, idx2, idx3, idx4, idx5, idx6, coupling])
        
    coupling_params.append(coupling_params_sixsite)
    return coupling_params

def read_operators_embed(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        coupling_params = parse_coupling_params_embed(lines)
    return coupling_params

def read_cluster_file_embed(file_path):
    cluster = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts=line.strip().split()
            if(len(parts)==0): continue
            if parts[0]=="Point" and len(parts)==4:
                x, y = map(int, (parts[2]+" "+parts[3]).strip('()').split(','))
                cluster.append((x, y))
    return cluster

def plot_embed_twosite(couplings_twosite, cluster, J, filename="test.png"):
    size=int(math.sqrt(len(cluster)))
    fig, ax = plt.subplots(figsize=(6, 6))
    xs=[x for (x,y) in cluster]
    ys=[y for (x,y) in cluster]
    ax.scatter(xs, ys, color='black', s=10)
    for i, site1 in enumerate(cluster):
        for j, site2 in enumerate(cluster):
            if i>=j: continue
            if math.fabs(site1[0]-site2[0])+math.fabs(site1[1]-site2[1])==1:
                ax.plot([site1[0], site2[0]], [site1[1], site2[1]], color='black', linestyle='--', linewidth=1)

    couplings=[coupling.real*1000 for (i,j,coupling) in couplings_twosite[J-1]]
    norm=plt.Normalize(vmin=min(couplings), vmax=max(couplings))
    distances=[math.sqrt((cluster[i][0]-cluster[j][0])**2+(cluster[i][1]-cluster[j][1])**2) for i, j, coupling in couplings_twosite[J-1]]
    min_distance=min(distances)
    for i, j, coupling in couplings_twosite[J-1]:
        if math.sqrt((cluster[i][0]-cluster[j][0])**2+(cluster[i][1]-cluster[j][1])**2)>min_distance+0.2:
            continue
        if J==1 or J==2:
            ax.plot([cluster[i][0], cluster[j][0]], [cluster[i][1], cluster[j][1]], linestyle='-', color=plt.cm.viridis(norm(coupling.real*1000)), linewidth=1)
        else:
            curvature=0.2
            n_points = 8
            t = np.linspace(0, 1, n_points)
            x = cluster[i][0] + (cluster[j][0]-cluster[i][0])*t
            y = cluster[i][1] + (cluster[j][1]-cluster[i][1])*t
            perp = np.array([-(cluster[j][1]-cluster[i][1]), (cluster[j][0]-cluster[i][0])])
            perp = perp/np.linalg.norm(perp)
            offset = np.sin(np.pi*t) * curvature
            xc = x + perp[0]*offset
            yc = y + perp[1]*offset
            ax.plot(xc, yc, color=plt.cm.viridis(norm(coupling.real*1000)), linewidth=0.8)


    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis');
    sm.set_array([]); 
    plt.colorbar(sm, ax=ax, label='Coupling Strength/meV', shrink=0.6, aspect=10)
    ax.set_aspect('equal')
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.axis('off')
    #plt.title("Square Lattice with Highlighted Cluster")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)


def plot_embed_foursite(couplings_foursite, cluster, J, filename="test.png"):
    size=int(math.sqrt(len(cluster)))
    xs=[x for (x,y) in cluster]
    ys=[y for (x,y) in cluster]

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax1.scatter(xs, ys, color='black', s=10)
    ax2.scatter(xs, ys, color='black', s=10)
    ax3.scatter(xs, ys, color='black', s=10)
    for i, site1 in enumerate(cluster):
        for j, site2 in enumerate(cluster):
            if i>=j: continue
            if math.fabs(site1[0]-site2[0])+math.fabs(site1[1]-site2[1])==1:
                ax1.plot([site1[0], site2[0]], [site1[1], site2[1]], color='black', linestyle='--', linewidth=1)
                ax2.plot([site1[0], site2[0]], [site1[1], site2[1]], color='black', linestyle='--', linewidth=1)
                ax3.plot([site1[0], site2[0]], [site1[1], site2[1]], color='black', linestyle='--', linewidth=1)

    couplings1=[coupling.real*1000 for idx, (i,j,k,l, coupling) in enumerate(couplings_foursite[J-1]) if idx%3==0]
    couplings2=[coupling.real*1000 for idx, (i,j,k,l, coupling) in enumerate(couplings_foursite[J-1]) if idx%3==1]
    couplings3=[coupling.real*1000 for idx, (i,j,k,l, coupling) in enumerate(couplings_foursite[J-1]) if idx%3==2]

    curvature = 0.2
    n_points = 8
    t = np.linspace(0, 1, n_points)
    offset = np.sin(np.pi * t) * curvature
    norm1=plt.Normalize(vmin=min(couplings1), vmax=max(couplings1))
    norm2=plt.Normalize(vmin=min(couplings2), vmax=max(couplings2))
    norm3=plt.Normalize(vmin=min(couplings3), vmax=max(couplings3))
    #cmap = plt.cm.seismic  # 蓝到红（中性为白）
    #norm = TwoSlopeNorm(vmin=min(couplings1+couplings2+couplings3), vcenter=0.0, vmax=max(couplings1+couplings2+couplings3))
            
    for idx in range(int(len(couplings_foursite[J-1])/3)):
        if J==1:
            for idx_four in range(3):
                i,j,k,l, coupling=couplings_foursite[J-1][idx*3+idx_four]
                site1, site2, site3, site4 = np.array(cluster[i]), np.array(cluster[j]), np.array(cluster[k]), np.array(cluster[l])
                center=np.array([(site1[0]+site2[0]+site3[0]+site4[0])/4, (site1[1]+site2[1]+site3[1]+site4[1])/4])

                distances=[np.linalg.norm(site1-center), np.linalg.norm(site2-center), np.linalg.norm(site3-center), np.linalg.norm(site4-center)]
                max_distance=max(distances)
                if max_distance > 3.1:
                    continue

                if idx_four==0:
                    p1, q1 = site1, site2
                    x1 = p1[0] + (q1[0] - p1[0]) * t
                    y1 = p1[1] + (q1[1] - p1[1]) * t
                    mid1 = (p1 + q1) / 2
                    dir_to_center1 = (center - mid1) / np.linalg.norm(center - mid1)
                    xc1 = x1 + dir_to_center1[0] * offset
                    yc1 = y1 + dir_to_center1[1] * offset
                    ax1.plot(xc1, yc1, color=plt.cm.viridis(norm1(coupling.real*1000)), linewidth=0.8)

                    p2, q2 = site3, site4
                    x2 = p2[0] + (q2[0] - p2[0]) * t
                    y2 = p2[1] + (q2[1] - p2[1]) * t
                    mid2 = (p2 + q2) / 2
                    dir_to_center2 = (center - mid2) / np.linalg.norm(center - mid2)
                    xc2 = x2 + dir_to_center2[0] * offset
                    yc2 = y2 + dir_to_center2[1] * offset
                    ax1.plot(xc2, yc2, color=plt.cm.viridis(norm1(coupling.real*1000)), linewidth=0.8)
                elif idx_four==1:
                    p1, q1 = site1, site2
                    x1 = p1[0] + (q1[0] - p1[0]) * t
                    y1 = p1[1] + (q1[1] - p1[1]) * t
                    mid1 = (p1 + q1) / 2
                    dir_to_center1 = (center - mid1) / np.linalg.norm(center - mid1)
                    xc1 = x1 + dir_to_center1[0] * offset
                    yc1 = y1 + dir_to_center1[1] * offset
                    ax2.plot(xc1, yc1, color=plt.cm.viridis(norm2(coupling.real*1000)), linewidth=0.8)

                    p2, q2 = site3, site4
                    x2 = p2[0] + (q2[0] - p2[0]) * t
                    y2 = p2[1] + (q2[1] - p2[1]) * t
                    mid2 = (p2 + q2) / 2
                    dir_to_center2 = (center - mid2) / np.linalg.norm(center - mid2)
                    xc2 = x2 + dir_to_center2[0] * offset
                    yc2 = y2 + dir_to_center2[1] * offset
                    ax2.plot(xc2, yc2, color=plt.cm.viridis(norm2(coupling.real*1000)), linewidth=0.8)
                else:
                    ax3.plot([site1[0], site2[0]], [site1[1], site2[1]], linestyle='-', color=plt.cm.viridis(norm3(coupling.real*1000)), linewidth=1)
                    ax3.plot([site3[0], site4[0]], [site3[1], site4[1]], linestyle='-', color=plt.cm.viridis(norm3(coupling.real*1000)), linewidth=1)
                    

    sm = plt.cm.ScalarMappable(norm=norm1, cmap='viridis');
    sm.set_array([]); 
    plt.colorbar(sm, ax=ax1, label='Coupling Strength/meV', shrink=0.6, aspect=10)
    ax1.set_aspect('equal')
    ax1.set_xlim(-1, size)
    ax1.set_ylim(-1, size)
    ax1.axis('off')
    fig1.savefig(filename+"_1.png", bbox_inches='tight', pad_inches=0)

    sm = plt.cm.ScalarMappable(norm=norm2, cmap='viridis');
    sm.set_array([]); 
    plt.colorbar(sm, ax=ax2, label='Coupling Strength/meV', shrink=0.6, aspect=10)
    ax2.set_aspect('equal')
    ax2.set_xlim(-1, size)
    ax2.set_ylim(-1, size)
    ax2.axis('off')
    fig2.savefig(filename+"_2.png", bbox_inches='tight', pad_inches=0)

    sm = plt.cm.ScalarMappable(norm=norm3, cmap='viridis');
    sm.set_array([]); 
    plt.colorbar(sm, ax=ax3, label='Coupling Strength/meV', shrink=0.6, aspect=10)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1, size)
    ax3.set_ylim(-1, size)
    ax3.axis('off')
    fig3.savefig(filename+"_3.png", bbox_inches='tight', pad_inches=0)

def plot_embed(couplings, cluster, prefix="test"):
    if len(couplings[-1][0])==5:
        ncouplings=len(couplings)-6
    elif len(couplings[-1][0])==7:
        ncouplings=len(couplings)-7
    else:
        print("Error: couplings_foursite is not a list of length 5 or 7")
        exit()
    couplings_twosite=couplings[1:ncouplings+1]
    couplings_foursite=couplings[ncouplings+1:]

    for i in range(10,20):
        plot_embed_twosite(couplings_twosite, cluster, i, f"{prefix}_two_J{i}.png")
    #for i in range(1,5):
    #    plot_embed_foursite(couplings_foursite, cluster, i, f"{prefix}_four_type{i}.png")
    
    

def resort_bond_foursite(couplings_foursite, cluster):
    couplings_foursite_new=[[] for _ in range(5)]
    for idx, _ in enumerate(couplings_foursite):
        i,j,k,l, coupling=couplings_foursite[idx]
        type, (i1,j1,k1,l1)=normalize_four_sites([i,j,k,l], cluster)
        couplings_foursite_new[type].append([i1,j1,k1,l1, coupling])
    return couplings_foursite_new


def out_data_cluster(couplings, filename, cluster):
    ncouplings=len(couplings)-2
    couplings_twosite=couplings[1:ncouplings+1]
    couplings_foursite=resort_bond_foursite(couplings[-1],cluster)
    out_data(couplings_twosite, couplings_foursite, filename)

def out_data_square(couplings, filename, cluster=None):
    ncouplings=len(couplings)-6
    couplings_twosite=couplings[1:ncouplings+1]
    couplings_foursite=couplings[ncouplings+1:]
    out_data(couplings_twosite, couplings_foursite, filename)

def out_data(couplings_twosite, couplings_foursite, filename):
    xs_two=[]
    ys_two=[]
    xs_four=[]
    ys_four=[]
    for idx in range(len(couplings_twosite)):
        for idx_C, (i, j, coupling) in enumerate(couplings_twosite[idx]):
            xs_two.append(idx+1)
            ys_two.append(coupling.real*1000)
    for idx in range(len(couplings_foursite)):
        for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[idx]):
            xs_four.append(-(idx+1))
            ys_four.append(math.fabs(coupling.real*1000))
    
    xs=xs_four+xs_two
    ys=ys_four+ys_two
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, color='black', s=20)
    #ax.set_aspect('equal')
    ax.set_xlim(-6, min(10, len(couplings_twosite)+1))
    ax.set_xticks(range(-6, min(10, len(couplings_twosite)+1), 1))
    ax.set_ylim(0.001, max(ys)*10)
    ax.axis('on')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axhline(y=1, linestyle='--', color='black', linewidth=1)
    ax.set_yscale('log')
    #plt.title("Square Lattice with Highlighted Cluster")
    plt.savefig(filename)



def data_pp_Block(data, cluster):
    ncouplings=len(data)-2 if len(data[-1][0])==5 else len(data)-1
    couplings_twosite=data[1:ncouplings+1]
    couplings_foursite=resort_bond_foursite(data[-1], cluster) if len(data[-1][0])==5 else []

    xs_two=[]
    ys_two=[]
    xs_four=[]
    ys_four=[]
    for idx in range(len(couplings_twosite)):
        for idx_C, (i, j, coupling) in enumerate(couplings_twosite[idx]):
            xs_two.append(idx+1)
            ys_two.append(coupling.real*1000)
    for idx in range(len(couplings_foursite)):
        for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[idx]):
            xs_four.append(-(idx+1))
            ys_four.append(math.fabs(coupling.real*1000))
    
    return xs_two, ys_two, xs_four, ys_four
    
def plot_Block(U, t, N_list, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    max_ys=0.0
    for N in N_list:
        filepath_input=f"Block_U{U}.0_t{t}.0/N{N}/"
        for file in os.listdir(filepath_input):
            if file.endswith('_results.txt'):
                file_path = os.path.join(filepath_input, file)
                data=read_operators(file_path)
                cluster=read_cluster_file_embed(file_path)
                xs_two, ys_two, xs_four, ys_four=data_pp_Block(data, cluster)
                ax.scatter(xs_four+xs_two, ys_four+ys_two, color='black', s=20)

                max_ys=max(max_ys, max(ys_four+ys_two))
    
    
    ax.set_xlim(-6, 10)
    ax.set_xticks(range(-6, 10, 1))
    ax.set_ylim(0.001, max_ys*10)
    ax.axis('on')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axhline(y=1, linestyle='--', color='black', linewidth=1)
    ax.set_yscale('log')
    plt.savefig(filename)



def data_pp_LCE(data, cluster):
    ncouplings=len(data)-2 if len(cluster)>=4 else len(data)-1
    couplings_twosite=data[1:ncouplings+1]
    couplings_foursite=resort_bond_foursite(data[-1], cluster) if len(cluster)>=4 else []

    xs_two=[]
    ys_two=[]
    xs_four=[]
    ys_four=[]
    for idx in range(len(couplings_twosite)):
        for idx_C, (i, j, coupling) in enumerate(couplings_twosite[idx]):
            xs_two.append(idx+1)
            ys_two.append(coupling.real*1000)
    for idx in range(len(couplings_foursite)):
        for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[idx]):
            xs_four.append(-(idx+1))
            ys_four.append(math.fabs(coupling.real*1000))
    
    return xs_two, ys_two, xs_four, ys_four
    

def plot_LCE(U, t, N_list, filename):
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    figc1, axc1 = plt.subplots(figsize=(6, 6))
    figc2, axc2 = plt.subplots(figsize=(6, 6))
    figc3, axc3 = plt.subplots(figsize=(6, 6))
    figc4, axc4 = plt.subplots(figsize=(6, 6))
    figc5, axc5 = plt.subplots(figsize=(6, 6))

    x1_list=[]
    x2_list=[]
    x3_list=[]
    x4_list=[]
    x5_list=[]
    xc1_list=[]
    xc2_list=[]
    xc3_list=[]
    xc4_list=[]
    xc5_list=[]

    J1_list=[]
    J2_list=[]
    J3_list=[]
    J4_list=[]
    J5_list=[]
    Jc1_list=[]
    Jc2_list=[]
    Jc3_list=[]
    Jc4_list=[]
    Jc5_list=[]
    for N in N_list:
        filepath_input=f"LCE_U{U}.0_t{t}.0/N{N}/"
        for file in os.listdir(filepath_input):
            if file.endswith('_results.txt'):
                file_path = os.path.join(filepath_input, file)
                data=read_operators(file_path)
                cluster=read_cluster_file_embed(file_path)
                
                ncouplings=len(data)-2 if len(cluster)>=4 else len(data)-1
                couplings_twosite=data[1:ncouplings+1]
                couplings_foursite=resort_bond_foursite(data[-1], cluster) if len(cluster)>=4 else []

                if len(couplings_twosite)>=1:
                    J1_list.extend([coupling.real*1000 for idx_C, (i, j, coupling) in enumerate(couplings_twosite[0])])
                    x1_list.extend([N for _ in range(len(couplings_twosite[0]))])
                if len(couplings_twosite)>=2:
                    J2_list.extend([coupling.real*1000 for idx_C, (i, j, coupling) in enumerate(couplings_twosite[1])])
                    x2_list.extend([N for _ in range(len(couplings_twosite[1]))])
                if len(couplings_twosite)>=3:
                    J3_list.extend([coupling.real*1000 for idx_C, (i, j, coupling) in enumerate(couplings_twosite[2])])
                    x3_list.extend([N for _ in range(len(couplings_twosite[2]))])
                if len(couplings_twosite)>=4:
                    J4_list.extend([coupling.real*1000 for idx_C, (i, j, coupling) in enumerate(couplings_twosite[3])])
                    x4_list.extend([N for _ in range(len(couplings_twosite[3]))])
                if len(couplings_twosite)>=5:
                    J5_list.extend([coupling.real*1000 for idx_C, (i, j, coupling) in enumerate(couplings_twosite[4])])
                    x5_list.extend([N for _ in range(len(couplings_twosite[4]))])
                
                if len(couplings_foursite)>=1:
                    Jc1_list.extend([math.fabs(coupling.real)*1000 for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[0])])
                    xc1_list.extend([N for _ in range(len(couplings_foursite[0]))])
                if len(couplings_foursite)>=2:
                    Jc2_list.extend([math.fabs(coupling.real)*1000 for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[1])])
                    xc2_list.extend([N for _ in range(len(couplings_foursite[1]))])
                if len(couplings_foursite)>=3:
                    Jc3_list.extend([math.fabs(coupling.real)*1000 for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[2])])
                    xc3_list.extend([N for _ in range(len(couplings_foursite[2]))])
                if len(couplings_foursite)>=4:
                    Jc4_list.extend([math.fabs(coupling.real)*1000 for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[3])])
                    xc4_list.extend([N for _ in range(len(couplings_foursite[3]))])
                if len(couplings_foursite)>=5:
                    Jc5_list.extend([math.fabs(coupling.real)*1000 for idx_C, (i, j, k, l, coupling) in enumerate(couplings_foursite[4])])
                    xc5_list.extend([N for _ in range(len(couplings_foursite[4]))])
                


    ax1.scatter(x1_list, J1_list, color='black', s=20)
    ax2.scatter(x2_list, J2_list, color='black', s=20)
    ax3.scatter(x3_list, J3_list, color='black', s=20)
    ax4.scatter(x4_list, J4_list, color='black', s=20)
    ax5.scatter(x5_list, J5_list, color='black', s=20)
    axc1.scatter(xc1_list, Jc1_list, color='black', s=20)
    axc2.scatter(xc2_list, Jc2_list, color='black', s=20)
    axc3.scatter(xc3_list, Jc3_list, color='black', s=20)
    axc4.scatter(xc4_list, Jc4_list, color='black', s=20)
    axc5.scatter(xc5_list, Jc5_list, color='black', s=20)


    ax1.set_xlim(min(N_list)-1, max(N_list)+1)
    ax1.set_xticks(range(min(N_list)-1, max(N_list)+1, 1))
    ax2.set_xlim(min(N_list)-1, max(N_list)+1)
    ax2.set_xticks(range(min(N_list)-1, max(N_list)+1, 1))
    ax3.set_xlim(min(N_list)-1, max(N_list)+1)
    ax3.set_xticks(range(min(N_list)-1, max(N_list)+1, 1))
    ax4.set_xlim(min(N_list)-1, max(N_list)+1)
    ax4.set_xticks(range(min(N_list)-1, max(N_list)+1, 1))
    ax5.set_xlim(min(N_list)-1, max(N_list)+1)
    ax5.set_xticks(range(min(N_list)-1, max(N_list)+1, 1))

    axc1.set_xlim(3, max(N_list)+1)
    axc1.set_xticks(range(3, max(N_list)+1, 1))
    axc2.set_xlim(3, max(N_list)+1)
    axc2.set_xticks(range(3, max(N_list)+1, 1))
    axc3.set_xlim(3, max(N_list)+1)
    axc3.set_xticks(range(3, max(N_list)+1, 1))
    axc4.set_xlim(3, max(N_list)+1)
    axc4.set_xticks(range(3, max(N_list)+1, 1))
    axc5.set_xlim(3, max(N_list)+1)
    axc5.set_xticks(range(3, max(N_list)+1, 1))
    
    
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')
    ax5.set_yscale('log')
    axc1.set_yscale('log')
    axc2.set_yscale('log')
    axc3.set_yscale('log')
    axc4.set_yscale('log')
    axc5.set_yscale('log')

    ax1.set_ylim(0.001, max(J1_list)*10)
    ax2.set_ylim(0.001, max(J2_list)*10)
    ax3.set_ylim(0.001, max(J3_list)*10)
    ax4.set_ylim(0.001, max(J4_list)*10)
    ax5.set_ylim(0.001, max(J5_list)*10)
    axc1.set_ylim(0.001, max(Jc1_list)*10)
    axc2.set_ylim(0.001, max(Jc2_list)*10)
    axc3.set_ylim(0.001, max(Jc3_list)*10)
    axc4.set_ylim(0.001, max(Jc4_list)*10)
    axc5.set_ylim(0.001, max(Jc5_list)*10)

    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.axhline(y=1, linestyle='--', color='black', linewidth=1)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.axhline(y=1, linestyle='--', color='black', linewidth=1)
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.axhline(y=1, linestyle='--', color='black', linewidth=1)
    ax4.axvline(x=0, color='black', linewidth=1)
    ax4.axhline(y=1, linestyle='--', color='black', linewidth=1)
    ax5.axvline(x=0, color='black', linewidth=1)
    ax5.axhline(y=1, linestyle='--', color='black', linewidth=1)
    axc1.axvline(x=0, color='black', linewidth=1)
    axc1.axhline(y=1, linestyle='--', color='black', linewidth=1)
    axc2.axvline(x=0, color='black', linewidth=1)
    axc2.axhline(y=1, linestyle='--', color='black', linewidth=1)
    axc3.axvline(x=0, color='black', linewidth=1)
    axc3.axhline(y=1, linestyle='--', color='black', linewidth=1)
    axc4.axvline(x=0, color='black', linewidth=1)
    axc4.axhline(y=1, linestyle='--', color='black', linewidth=1)
    axc5.axvline(x=0, color='black', linewidth=1)
    axc5.axhline(y=1, linestyle='--', color='black', linewidth=1)

    fig1.savefig(filename+"_J1.png", bbox_inches='tight', pad_inches=0)
    fig2.savefig(filename+"_J2.png", bbox_inches='tight', pad_inches=0)
    fig3.savefig(filename+"_J3.png", bbox_inches='tight', pad_inches=0)
    fig4.savefig(filename+"_J4.png", bbox_inches='tight', pad_inches=0)
    fig5.savefig(filename+"_J5.png", bbox_inches='tight', pad_inches=0)
    figc1.savefig(filename+"_Jc1.png", bbox_inches='tight', pad_inches=0)
    figc2.savefig(filename+"_Jc2.png", bbox_inches='tight', pad_inches=0)
    figc3.savefig(filename+"_Jc3.png", bbox_inches='tight', pad_inches=0)
    figc4.savefig(filename+"_Jc4.png", bbox_inches='tight', pad_inches=0)
    figc5.savefig(filename+"_Jc5.png", bbox_inches='tight', pad_inches=0)

def plot(xs, ys, filename, y_ref=None):
    fig, ax=plt.subplots(figsize=(6, 6))
    # Sort the points by x value to ensure proper line connection
    sorted_indices = np.argsort(xs)
    xs_sorted = np.array(xs)[sorted_indices]
    ys_sorted = np.array(ys)[sorted_indices]

    
    # Plot lines first (so they appear behind the scatter points)
    ax.plot(xs_sorted, ys_sorted, color='red', linestyle='-', linewidth=1, alpha=0.5, label='Block-ED')
    # Plot scatter points on top
    ax.scatter(xs, ys, color='black', s=20)


    if y_ref is not None:
        y_ref_sorted = np.array(y_ref)[sorted_indices]
        ax.plot(xs_sorted, y_ref_sorted, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Second-order perturbation')
        ax.scatter(xs, y_ref, color='gray', s=20)
    
    ax.set_xlim(min(xs), max(xs))
    #ax.set_xticks(range(min(xs)-1, max(xs)+1, 1))
    ax.axis('on')
    ax.legend()
    #ax.set_yscale('log')
    plt.savefig(filename)



def plot2(xs, ys1, ys2, ys3, filename, y_ref1=None, y_ref2=None):
    fig, ax=plt.subplots(figsize=(6, 6))
    # Sort the points by x value to ensure proper line connection
    sorted_indices = np.argsort(xs)
    xs_sorted = np.array(xs)[sorted_indices]
    ys1_sorted = np.array(ys1)[sorted_indices]
    ys2_sorted = np.array(ys2)[sorted_indices]
    ys3_sorted = np.array(ys3)[sorted_indices]
    
    # Plot lines first (so they appear behind the scatter points)
    ax.plot(xs_sorted, ys1_sorted, color='red', linestyle='-', linewidth=1, alpha=0.5, label='Series 1')
    ax.plot(xs_sorted, ys2_sorted, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Series 2')
    ax.plot(xs_sorted, ys3_sorted, color='blue', linestyle='-', linewidth=1, alpha=0.5, label='Series 3')
    # Plot scatter points on top with different markers
    ax.scatter(xs, ys1, color='black', s=20, marker='o', label='Series 1')
    ax.scatter(xs, ys2, color='black', s=20, marker='s', label='Series 2')
    ax.scatter(xs, ys3, color='black', s=20, marker='^', label='Series 3')

    if y_ref1 is not None and y_ref2 is not None:
        y_ref1_sorted = np.array(y_ref1)[sorted_indices]
        y_ref2_sorted = np.array(y_ref2)[sorted_indices]
        ax.plot(xs_sorted, y_ref1_sorted, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Second-order perturbation')
        ax.plot(xs_sorted, y_ref2_sorted, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Second-order perturbation')
        ax.scatter(xs, y_ref1, color='gray', s=20, marker='o', label='Second-order perturbation')
        ax.scatter(xs, y_ref2, color='gray', s=20, marker='s', label='Second-order perturbation')
    
    ax.set_xlim(min(xs), max(xs))
    #ax.set_xticks(range(min(xs)-1, max(xs)+1, 1))
    ax.axis('on')
    #ax.set_yscale('log')
    # Add legend
    ax.legend()
    plt.savefig(filename)

def plot_array(xs, ys_list, filename, y_ref=None):
    ys_list_plot=np.array(ys_list)

    fig, ax=plt.subplots(figsize=(6, 6))
    for idx in range(ys_list_plot.shape[0]):
        ax.plot(xs, ys_list_plot[:,idx].real*1000, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.savefig(filename)