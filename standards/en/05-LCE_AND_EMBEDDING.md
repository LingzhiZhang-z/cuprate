# 05-LCE_AND_EMBEDDING

Linked-cluster expansion (subcluster subtraction) and embed output.

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
- Each subgraph is matched to a previously computed cluster result via NN-graph isomorphism.
- LCE input must contain consecutive `N=2..Nmax` raw spin-coupling results.

Code form:
```python
subgraphs, indices = get_connected_subgraphs(cluster, min_size=2)
match = find_cluster_match(subgraph, clusters)
# match = (nsites, hole, class_idx, variant_idx, mapping)
```

## 3) Operator Subtraction (MUST)

MUST:
- LCE reads fitted operators from the `results.json` schema in
  `08-OPERATOR_OUTPUT.md`; it does not read text files or projection NPZ files.
- The operator data contains a constant term plus keyed spin-coupling terms.
- Subtraction maps subcluster site indices to parent cluster indices via the isomorphism mapping,
  then subtracts coefficients term by term.
- Operator identity is the canonical `key` field, not the display labels `J*`,
  `K*`, or `L*`.

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

## 4) LCE Output (MUST)

MUST:
- The LCE entry point writes `lce_results.json`, per-cluster weight files under
  `weights/`, and a minimal `lce_summary.txt`.
- `lce_results.json` has `result_kind = "lce_spin_couplings"`.
- `lce_results.json` is a manifest. It records run parameters and points to one
  weight file per concrete cluster.
- Weight files live under `weights/` and are named
  `hole{h}_class{c}_idx{v}.json`.
- Each weight file has `result_kind = "lce_cluster_weight"`.
- Each weight file records `sites`, `indices`, the net operators `W(C)` using
  the same operator group schema as raw spin couplings, and reconstruction
  diagnostics.

Manifest form:
```json
{
  "schema_version": 1,
  "result_kind": "lce_spin_couplings",
  "weights": [
    {
      "N": 4,
      "hole": 0,
      "class_idx": 1,
      "cluster_idx": 0,
      "weight_file": "weights/hole0_class1_idx0.json"
    }
  ]
}
```

Weight-file form:
```json
{
  "schema_version": 1,
  "result_kind": "lce_cluster_weight",
  "N": 4,
  "hole": 0,
  "class_idx": 1,
  "cluster_idx": 0,
  "sites": [[0, 0], [1, 0], [0, 1], [1, 1]],
  "indices": [0, 1, 2, 3],
  "operators": {"constant_term": {"real": 0.0, "imag": 0.0}, "groups": []},
  "raw_summary": {},
  "diagnostics": {
    "subcluster_count": 0,
    "reconstruction_error": 0.0,
    "net_term_count": 0,
    "max_abs_net_coefficient": 0.0
  }
}
```

Validation:
- For every cluster, summing all connected subcluster net contributions
  $W(C')$ for $C' \subseteq C$ must recover the raw fitted operators $O(C)$
  within `ATOL["loose"]`.
- For `N=2`, the net operators equal the raw operators.

## 5) Embed Input And Output (MUST)

MUST:
- The embed stage must read the `lce_results.json` manifest and its
  referenced weight files, not raw `results.json`.
- Output lives under
  `ROOT/block_embed/N_{Nmax}_nelec_{nelec}_U_{U:.4f}_t_{T:.4f}/mode_*/workflow_*/`.
- `embed_results.json` is a manifest with
  `result_kind = "embedded_spin_couplings"`.
- `embed_summary.txt` records the source LCE file and output counts.
- `two_site.txt` contains all candidate two-site bond vectors within `Nmax`;
  vectors not present in accumulated LCE output are written as `None`.
- Multi-site cluster files are named `N{N}_hole{h}_class{c}_idx{i}.txt`.
- Production code uses the name `embed`, not `periodize`.

Code form:
```text
ROOT/block_embed/N_6_nelec_6_U_1.0000_t_0.0200/mode_full/workflow_occ/two_site.txt
ROOT/block_embed/N_6_nelec_6_U_1.0000_t_0.0200/mode_full/workflow_occ/clusters/N4_hole0_class0_idx0.txt
```

## 6) Embed Matching Algorithm (MUST)

MUST:
- The numeric algorithm is target-driven.
- C4v operations are used only to generate distinct parent-cluster
  orientations.
- The target is concrete. Matching allows translation only; do not rotate,
  mirror, canonicalize, or divide by the target multiplicity.
- Do not compute or output `m(candidate)`.
- Do not do L2 orbit averaging or build a `Wtilde` numeric path.
- Do not do another Möbius subtraction in embed.

Code form:
```text
for each output target P:
    value(P) = 0
    seen(P) = false
    for each LCE weight W(C):
        for each distinct C4v orientation g(C):
            for each term t in W(C) with arity(P):
                if g(t) matches P by translation only:
                    value(P) += coeff(t)
                    seen(P) = true
```

Validation:
- Plaquette distinct orientations = 1.
- Four-site line distinct orientations = 2.
- L-shape distinct orientations = 8.
- Synthetic plaquette with each NN two-site term coefficient set to `1.0`
  gives `two_site.txt[(1,0)] = 2.0`.
- Synthetic L-tetromino with each NN two-site term coefficient set to `1.0`
  gives `two_site.txt[(1,0)] = 12.0`.

## 7) Embed Two-Site Candidate Order (MUST)

MUST:
- Two-site candidates use canonical vectors `(dx, dy)` with `dx >= dy >= 0`
  and `dx > 0`.
- Candidate vectors satisfy `dx + dy + 1 <= Nmax`.
- Sorting is by Manhattan shell, squared distance, then lexicographic vector.
- This makes increasing `Nmax` append new candidate shells without reordering
  earlier entries.

Code form:
```python
sorted(candidates, key=lambda v: (v[0] + v[1], v[0]**2 + v[1]**2, v[0], v[1]))
```

## 8) Embed Multi-Site Text Files (MUST)

MUST:
- Four-site and six-site embed files list the cluster sites, the site
  indices, all perfect pairings, and the coefficient for each pairing.
- Missing pairings are written as `None`.
- Files are written under `clusters/`.
- A simple text graph section may be added, but is not required.
