# 05-LCE_AND_EMBEDDING

Linked-cluster expansion (subcluster subtraction) and supercell embedding.

## 1) LCE Goal (MUST)

MUST:
- For each cluster $C$, the raw spin couplings $O(C)$ contain contributions from all subclusters.
- The net (connected, irreducible) contribution is obtained by subtracting all proper connected subclusters:

Math:
$$
W(C) = O(C) - \sum_{\substack{C' \subsetneq C \\ C'\text{ connected}}} W(C').
$$

- This is computed bottom-up: smallest clusters first, then progressively larger ones.

## 2) Subgraph Enumeration (MUST)

MUST:
- For a cluster of $N$ sites, all connected subgraphs of size $2$ to $N-1$ are enumerated.
- Connectivity is by NN adjacency (Manhattan distance $= 1$).
- Each subgraph is matched to a previously computed cluster result via weighted-graph isomorphism.

Code form:
```python
subgraphs, indices = get_connected_subgraphs(cluster, min_size=2)
match = find_cluster_match(subgraph, clusters)
# match = (nsites, hole, class_idx, variant_idx, mapping)
```

## 3) Operator Subtraction (MUST)

MUST:
- The operator list has the structure:
  `[constant, [2-site terms], [4-site terms], [6-site terms], ...]`
  where each term is `[site_indices..., coefficient]`.
- Subtraction maps subcluster site indices to parent cluster indices via the isomorphism mapping,
  then subtracts coefficients term by term.

Code form:
```python
operator_minus(parent_operators, subcluster_operators, index_mapping)
```

- Operator matching uses canonical key functions that sort site-pair tuples:

Code form:
```python
k2s(i, j)          = sorted([i, j])
k4s(i, j, k, l)    = sorted([(sorted([i,j]), sorted([k,l]))])
k6s(i,j,k,l,m,n)   = sorted([(sorted([i,j]), sorted([k,l]), sorted([m,n]))])
```

Validation:
- After LCE, the net contribution of a 1-site cluster is zero (no bonds).
- Summing all $W(C')$ for $C' \subseteq C$ must recover $O(C)$.

## 4) Supercell Embedding (MUST)

MUST:
- LCE net couplings are placed onto a periodic square supercell of size $N_{\text{cell}} \times N_{\text{cell}}$.
- For each cluster, all symmetry-distinct orientations are generated (up to 8 via C4v),
  then each orientation is translated to all $N_{\text{cell}}^2$ positions under PBC.
- Site indices in the supercell: `index = x * N_cell + y`, with PBC: `x_pbc = x % N_cell`.

Code form:
```python
unique_clusters = unique_cluster_transformed(cluster)  # up to 8 orientations
indices_pbc = embed_cluster_to_squarecell_pbc(unique_cluster, N_cell)
embed_operator(couplings_pbc, coupling_net, indices_pbc)
```

## 5) Accumulated Coupling Structure (MUST)

MUST:
- The supercell coupling data structure is:
  - `couplings[0]`: constant term (scalar).
  - `couplings[1]`: two-site coupling matrix ($N_{\text{cell}}^2 \times N_{\text{cell}}^2$, complex).
  - `couplings[2]`: four-site couplings (dict keyed by `k4s` tuples).
  - `couplings[3]`: six-site couplings (dict keyed by `k6s` tuples).
  - `couplings[4]`: eight-site couplings (dict keyed by `k8s` tuples).
- Two-site couplings are symmetrised: both `[i,j]` and `[j,i]` are incremented.

## 6) Output Analysis (MUST)

MUST:
- Two-site couplings are grouped by canonical bond vector and sorted by magnitude.
- Four-site couplings are classified into 5 topological types (square, T-shape, line variants)
  via `normalize_four_sites`.
- Results are written with both human-readable text and machine-readable JSON artifacts.
