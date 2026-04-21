"""Human-readable reporting for embedded couplings."""

from __future__ import annotations

from itertools import combinations
from math import sqrt

from cuprate.io import get_all_possible_vectors, k4s

from .core import normalize_four_sites


FOUR_SITE_SHAPES = [
    "\n4--3\n|  |\n1--2\n",
    "\n   3\n   |\n2--1--4\n",
    "\n1--2--3--4\n",
    "\n      4\n      |\n1--2--3\n",
    "\n   3--4\n   |\n1--2\n",
]


def _neighbors(point):
    x, y = point
    return {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}


def _connected_four_site_indices(cluster):
    for indices in combinations(range(len(cluster)), 4):
        subset = {cluster[idx] for idx in indices}
        stack = [cluster[indices[0]]]
        visited = {stack[0]}
        while stack:
            current = stack.pop()
            for neighbor in _neighbors(current):
                if neighbor in subset and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(visited) == 4:
            yield list(indices)


def write_couplings_embed(f, couplings, cluster):
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {couplings.constant.real:.10f}\n")

    size = int(sqrt(len(cluster)))
    all_vectors = get_all_possible_vectors(cluster, size)
    vector_to_index = {vector: idx for idx, vector in enumerate(all_vectors)}
    grouped_two_site = [[] for _ in range(len(all_vectors))]
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dx = cluster[j][0] - cluster[i][0]
            dy = cluster[j][1] - cluster[i][1]
            vector = min(
                [
                    tuple(sorted((abs(dx), abs(dy)), reverse=True)),
                    tuple(sorted((abs(dx - size), abs(dy - size)), reverse=True)),
                    tuple(sorted((abs(dx - size), abs(dy)), reverse=True)),
                    tuple(sorted((abs(dx), abs(dy - size)), reverse=True)),
                ],
                key=lambda candidate: candidate[0] ** 2 + candidate[1] ** 2,
            )
            grouped_two_site[vector_to_index[vector]].append([i, j, couplings.two_site[i][j]])

    for bond_group in grouped_two_site:
        bond_group.sort(key=lambda entry: abs(entry[2]) ** 2, reverse=True)

    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if grouped_two_site[bond_idx]:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}:\n")
            for idx, (site1, site2, coef) in enumerate(grouped_two_site[bond_idx]):
                f.write(f"    {idx}: Sites {site1}-{site2}: {coef.real:.10f}  +  {coef.imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx + 1}: (not present in cluster)\n")

    f.write("\nFour-site Bond:\n")
    grouped_four_site = [[] for _ in range(5)]
    for indices in _connected_four_site_indices(cluster):
        idx_type, sites = normalize_four_sites(indices, cluster)
        key1 = k4s(sites[0], sites[1], sites[2], sites[3])
        key2 = k4s(sites[0], sites[3], sites[1], sites[2])
        key3 = k4s(sites[0], sites[2], sites[1], sites[3])
        if key1 not in couplings.four_site and key2 not in couplings.four_site and key3 not in couplings.four_site:
            continue
        if key1 not in couplings.four_site or key2 not in couplings.four_site or key3 not in couplings.four_site:
            raise ValueError("Missing key in couplings_foursites")
        grouped_four_site[idx_type].append(
            [
                sites[0],
                sites[1],
                sites[2],
                sites[3],
                couplings.four_site[key1],
                couplings.four_site[key2],
                couplings.four_site[key3],
            ]
        )

    for sites_group in grouped_four_site:
        sites_group.sort(
            key=lambda entry: abs(entry[4]) ** 2 + abs(entry[5]) ** 2 + abs(entry[6]) ** 2,
            reverse=True,
        )

    for idx_type, sites_group in enumerate(grouped_four_site):
        if sites_group:
            f.write(f"\nBond type {idx_type + 1}:\n")
            f.write(FOUR_SITE_SHAPES[idx_type])
            for idx, sites in enumerate(sites_group):
                f.write(
                    f"    class {idx} type 1 : Four-site ({sites[0]}-{sites[1]}) * ({sites[2]}-{sites[3]}): {sites[4].real:.10f}  +  {sites[4].imag:.10f}i\n"
                )
                f.write(
                    f"    class {idx} type 2 : Four-site ({sites[0]}-{sites[3]}) * ({sites[1]}-{sites[2]}): {sites[5].real:.10f}  +  {sites[5].imag:.10f}i\n"
                )
                f.write(
                    f"    class {idx} type 3 : Four-site ({sites[0]}-{sites[2]}) * ({sites[1]}-{sites[3]}): {sites[6].real:.10f}  +  {sites[6].imag:.10f}i\n"
                )
                f.write("\n")
        else:
            f.write(FOUR_SITE_SHAPES[idx_type])
            f.write(f"\nBond type {idx_type + 1}: (not present in cluster)\n")

    f.write("\nSix-site Bond:\n")
    for bond_type, (key, value) in enumerate(couplings.six_site.items(), start=1):
        f.write(
            f"    class 0 type {bond_type} : Six-site ({key[0][0]}-{key[0][1]}) * ({key[1][0]}-{key[1][1]}) * ({key[2][0]}-{key[2][1]}): {value.real:.10f}  +  {value.imag:.10f}i\n"
        )


def print_grid(size):
    """Print an NxN periodic grid with cell indices at the grid points."""
    width = max(3, len(str(size * size - 1)))
    blank = " " * width
    vertical_row = blank + "".join("  " + "|".center(width) for _ in range(size))
    rows = [
        blank + "".join(f"  {str(col).center(width)}" for col in range(size)),
        vertical_row,
    ]

    for row in range(size - 1, -1, -1):
        row_values = [str(row * size + col).center(width) for col in range(size)]
        row_text = f"{str(row * size + size - 1).center(width)}- "
        row_text += "--".join(row_values)
        row_text += f"- {str(row * size).center(width)}"
        rows.append(row_text)
        if row > 0:
            rows.append(vertical_row)

    rows.extend(
        [
            vertical_row,
            blank + "".join(f"  {str((size - 1) * size + col).center(width)}" for col in range(size)),
        ]
    )
    return "\n".join(rows) + "\n"
