def find_bond_vector(bond, cluster):
    """Find the bond vector of a bond"""
    site1, site2 = bond
    x1, y1 = cluster[site1]
    x2, y2 = cluster[site2]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    return sorted([dx, dy], reverse=True)

def write_cluster_points(f, cluster):
    """Write cluster points information."""
    f.write("\n=== Cluster Points ===\n")
    for i, point in enumerate(cluster):
        f.write(f"Point {i}: {point}\n")

def get_all_possible_vectors(cluster, max_distance=None):
    """Get all possible bond vectors in the first quadrant below y=x."""
    if max_distance is None:
        max_distance = len(cluster)
    vectors = []
    for dx in range(max_distance):
        for dy in range(dx + 1):
            if dx == 0 and dy == 0:
                continue
            vectors.append((dx, dy))
    return sorted(vectors, key=lambda v: v[0]**2 + v[1]**2)
