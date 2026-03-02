import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from cuprate.shared.io import write_file
from cuprate.shared.bonds import find_bond_vector, write_cluster_points, get_all_possible_vectors


def write_basic_info(f, hole: int, class_idx: int, cluster_idx: int, rank: int, cluster_time: float) -> None:
    """Write basic information about the calculation."""
    f.write(f"Hole: {hole}\n")
    f.write(f"Class: {class_idx}\n")
    f.write(f"Cluster: {cluster_idx}\n")
    f.write(f"Rank: {rank}\n")
    f.write(f"Computation time: {cluster_time:.6f} seconds\n")


def write_coefficients(f, cluster, bonds, coeffs_list, error, T11m1_norm: float, overlap) -> None:
    """Write individual bond coefficients."""
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {coeffs_list[0].real:.10f}\n")

    # Create a dictionary to store coefficients by vector
    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in coeffs_by_vector:
                coeffs_by_vector[(dx, dy)] = []
            coeffs_by_vector[(dx, dy)].append((class_idx, bond_group, coeffs_list[class_idx+1]))

    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if (dx, dy) in coeffs_by_vector:
            for class_idx, bond_group, coeffs in coeffs_by_vector[(dx, dy)]:
                f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:\n")
                for idx, (bond, coef) in enumerate(zip(bond_group, coeffs)):
                    site1, site2 = bond
                    f.write(f"    {idx}: Sites {site1}-{site2}: {coef.real:.10f}  +  {coef.imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}: (not present in cluster)\n")

    f.write(f"\nFour-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 4:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//3+1} type {idx%3+1} : Four-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}): {coeffs_list[class_idx+1][idx].real:.10f}  +  {coeffs_list[class_idx+1][idx].imag:.10f}i\n")
                if (idx+1)%3==0:
                    f.write("\n")

    f.write(f"\nSix-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 6:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//15+1} type {idx%15+1} : Six-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}) * ({bond[4]}-{bond[5]}): {coeffs_list[class_idx+1][idx].real:.10f}  +  {coeffs_list[class_idx+1][idx].imag:.10f}i\n")
                if (idx+1)%15==0:
                    f.write("\n")

    """Write error information."""
    f.write("\n=== Individual Fit Error ===\n")
    f.write(f"Relative Error: {error[0]:.10f}\n")
    f.write(f"Residual: {error[1]:.10f}\n")
    f.write(f"R^2: {error[2]:.10f}\n")
    f.write(f"T11-1 norm: {T11m1_norm:.10f}\n")
    if overlap is not None:
        f.write(f"Overlap: {overlap:.10f}\n")


def write_errors(f, individual_error, class_error) -> None:
    """Write error information."""
    f.write("\n=== Individual Fit Error ===\n")
    np.savetxt(f, [individual_error])

    f.write("\n=== Class-Averaged Fit Error ===\n")
    np.savetxt(f, [class_error])


def write_bond_structure(f, cluster, bonds) -> None:
    """Write bond structure information."""
    f.write("\n=== Bond Structure ===")

    # Create a dictionary to store bonds by their vector
    bonds_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in bonds_by_vector:
                bonds_by_vector[(dx, dy)] = []
            bonds_by_vector[(dx, dy)].append((class_idx, bond_group))

    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:")
        if (dx, dy) in bonds_by_vector:
            f.write("\n")
            for class_idx, bond_group in bonds_by_vector[(dx, dy)]:
                for idx, bond in enumerate(bond_group):
                    site1, site2 = bond
                    x1, y1 = cluster[site1]
                    x2, y2 = cluster[site2]
                    f.write(f"    {idx}: Sites {site1}-{site2} ({x1},{y1})-({x2},{y2})\n")
        else:
            f.write(" (not present in cluster)\n")

    # Output square bonds if any
    square_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 4]
    if square_bonds:
        f.write("\nFour-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            square_bonds_in_class = [bond for bond in bond_group if len(bond) == 4]
            if square_bonds_in_class:
                for idx, bond in enumerate(square_bonds_in_class):
                    f.write(f"    {idx//3+1} type {idx%3+1}: Four-site bond: {bond}\n")
                    if (idx+1)%3==0:
                        f.write("\n")

    # Output six-site bonds if any
    six_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 6]
    if six_bonds:
        f.write("\nSix-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            six_bonds_in_class = [bond for bond in bond_group if len(bond) == 6]
            if six_bonds_in_class:
                for idx, bond in enumerate(six_bonds_in_class):
                    f.write(f"    {idx//15+1} type {idx%15+1}: Six-site bond: {bond}\n")
                    if (idx+1)%15==0:
                        f.write("\n")


def write_results_to_file(f, hole, class_idx, cluster_idx, rank, cluster_time,
                         cluster, bonds, coeffs, error, T11m1_norm, overlap) -> None:
    """Write all results to a file in a structured format."""
    write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time)
    write_cluster_points(f, cluster)
    write_bond_structure(f, cluster, bonds)
    write_coefficients(f, cluster, bonds, coeffs, error, T11m1_norm, overlap)


def cluster_save_results(result_dir: str, hole: int, class_idx: int, cluster_idx: int,
                        rank: int, cluster_time: float,
                        cluster, bonds,
                        coeffs, error, T11m1_norm: float, overlap) -> None:
    try:
        base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
        with open(f"{base_filename}_results.txt", 'w') as f:
            write_results_to_file(f, hole, class_idx, cluster_idx, rank, cluster_time,
                                cluster, bonds, coeffs, error, T11m1_norm, overlap)
        write_file(f"{base_filename}_results.txt", f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        raise IOError(f"Failed to save cluster results: {str(e)}")


def write_data(x_list, data, filename: str) -> None:
    nx = len(x_list)
    ny = len(data)
    with open(filename, "w") as f:
        for i in range(nx):
            f.write(f"{x_list[i]:.6f}")
            for j in range(ny):
                f.write(f" {data[j][i]:.12f}")
            f.write("\n")


def plot_data(x_list, data, filename: str, labels=None, ylims=None, xlims=None) -> None:
    if labels is None:
        labels = [f"hole{idx}" for idx in range(len(data))]
    if xlims is None:
        xlims = [0.0000, 1.000]
    if ylims is None:
        ylims = [0.0, 1.0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for idx in range(len(data)):
        ax.plot(x_list, data[idx], label=labels[idx], linewidth=1.0)
        ax.scatter(x_list, data[idx], linewidth=1.0)
    ax.set_title("Overlap of all clusters")
    ax.set_xlabel("t/U")
    ax.set_ylabel("Overlap")
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
