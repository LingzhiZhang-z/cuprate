# 01-CLUSTERS

Square-lattice cluster enumeration, nearest-neighbor graph classification,
and multi-site operator support patterns.

## 1) Cluster Definition (MUST)

MUST:
- A cluster is a connected set of sites on the 2D square lattice $\mathbb{Z}^2$.
- Connectivity is determined by nearest-neighbor adjacency only
  (Manhattan distance $= 1$).

Code form:
```python
cluster = [(x0,y0), (x1,y1), ..., (x_{N-1}, y_{N-1})]
```

## 2) Canonical Form and Enumeration (MUST)

MUST:
- Canonical form is obtained by applying all 8 operations of the C4v point group
  (`id, rot90, rot180, rot270, mirror_x, mirror_y, mirror_diag, mirror_anti`),
  translating each result so `min(x) = min(y) = 0`, sorting the site list,
  and taking the lexicographic minimum.
- Clusters are enumerated by DFS growth from a seed site, deduplicating via canonical form.

Code form:
```python
canonical_form(cluster, is_canon=True)  # C4v + translation
generate_clusters(N)                    # DFS enumeration
```

Validation:
- Two clusters represent the same physical geometry iff they share the same canonical form.

## 3) Hole Count (MUST)

MUST:
- A "hole" is a completed $2 \times 2$ plaquette entirely contained in the cluster.
- Clusters are first sorted by hole count before graph classification.

Code form:
```python
count_holes(cluster)  # counts all (x,y) where {(x,y),(x+1,y),(x,y+1),(x+1,y+1)} ⊂ cluster
```

## 4) Adjacency Matrix and Weighted Graph (MUST)

MUST:
- The adjacency matrix $A_{ij}$ encodes nearest-neighbor connectivity:
  - `1` = NN bond
  - `0` = no bond
- The graph $G(C)$ is constructed from this matrix with edge attribute `weight=1`.

Code form:
```python
adj = generate_adjacency_matrix(sites)
G   = graph_from_adj(adj)  # nx.Graph with weight attribute
```

## 5) Isomorphic Classification (MUST)

MUST:
- Two clusters are equivalent iff their graphs are isomorphic
  (checked via `nx.GraphMatcher` with `numerical_edge_match('weight', 0)`).
- Classification proceeds: group by hole count, then by graph isomorphism within each hole group.
- Within each equivalence class, the first cluster is the **representative**.
  All others are reordered via the isomorphism mapping so their adjacency matrices match the representative exactly.

Code form:
```python
classify_isomorphic_clusters(clusters)
# Returns: list of groups, each group = list of site-reordered clusters
```

Validation:
- All clusters in the same class must have identical adjacency matrices after reordering.
- The representative solves the Hamiltonian once; other members reuse the eigensystem.

## 6) Bond Generation (MUST)

MUST:
- `Cluster.bonds` stores all unique nearest-neighbor pairs `(i, j)` with
  Manhattan distance `1` and `i < j`. This is the bond list used by the Hubbard
  hopping matrix.
- `Cluster.generate_bonds(N=2, is_connected=False)` returns fit-operator groups:
  one singleton group `[[i, j]]` per two-site operator.
- `Cluster.generate_bonds(N=4 or 6, is_connected=True)` returns singleton
  groups for connected multi-site spin-operator pairings.

Code form:
```python
nn_bonds = cluster.bonds
bond_groups = (
    cluster.generate_bonds(N=2, is_connected=False)
    + cluster.generate_bonds(N=4, is_connected=True)
    + cluster.generate_bonds(N=6, is_connected=True)
)
```

## 7) Two-Site Bond Classification (MUST)

MUST:
- `find_bonds_twosites(cluster)` groups all pairs `(i, j)` by their canonical bond vector
  $(\max(|dx|, |dy|), \min(|dx|, |dy|))$, sorted by Euclidean distance.
- This produces bond classes for the spin-coupling fit (e.g. all NN bonds, all NNN bonds, etc.).

Code form:
```python
bond_types = find_bonds_twosites(cluster)
# Returns: dict_values of lists of (i,j) pairs, one list per canonical bond vector
```

## 8) Multi-Site Operator Patterns (MUST)

MUST:
- 4-site patterns: all connected 4-site subsets of the cluster.
- 6-site patterns: all connected 6-site subsets.
- Square plaquette patterns: all $2 \times 2$ squares, used for ring-exchange operators.
- Pairing counts, canonical ordering, and output conventions are defined in
  `08-OPERATOR_OUTPUT.md`.

Code form:
```python
find_bonds_foursites(cluster)   # 4-site connected subsets × 3 pairings (canonical order)
find_bonds_sixsites(cluster)    # 6-site connected subsets × 15 pairings (canonical order)
find_squares(cluster)           # completed 2×2 plaquettes as index tuples
```

## 9) Extensibility (MUST)

MUST:
- The committed baseline lattice family is `square`.
- Extension to other lattice families (triangular, honeycomb, etc.) requires:
  1. A new `get_neighbors` function defining connectivity.
  2. A new `canonical_form` respecting the lattice symmetry group.
  3. New bond-type definitions in the adjacency matrix.
- Existing square-lattice standards must not be modified to accommodate extensions.
