# 08-OPERATOR_OUTPUT

Spin-coupling operator definitions, canonical pairing order for multi-site
operators, and result output formats.

## 1) Operator Types (MUST)

MUST:
- The spin-coupling fit decomposes $H_{\text{eff}}$ into a linear combination of
  spin operators, grouped by the number of sites involved:

| Arity | Operator form | Label prefix |
|-------|---------------|--------------|
| 1 | Identity $I$ (constant $E_0$) | — |
| 2 | $\mathbf{S}_i \cdot \mathbf{S}_j$ | J |
| 4 | $(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)$ | K |
| 6 | $(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)(\mathbf{S}_m \cdot \mathbf{S}_n)$ | L |

- Two-site operators are classified by **bond vector** (see `01-CLUSTERS.md` §7)
  and labelled `J1`, `J2`, `J3`, ... in order of increasing Euclidean distance.
- Multi-site operators (arity $\geq 4$) are generated from all connected subsets
  of $2n$ sites (see `01-CLUSTERS.md` §8), then canonicalized for output.

## 2) Multi-Site Pairing Count (MUST)

MUST:
- For a connected group of $2n$ sites, the number of ways to partition into $n$
  Heisenberg pairs is:

Math:
$$
(2n-1)!! = \frac{(2n)!}{2^n \, n!}
$$

| Sites | Pairs | Pairings |
|-------|-------|----------|
| 4 | 2 | 3 |
| 6 | 3 | 15 |
| 8 | 4 | 105 |

## 3) Canonical Multi-Site Output Order (MUST)

MUST:
- Multi-site ordering is defined at the **serialization layer**. Internal
  enumeration order is not part of the contract.
- Each multi-site term is canonicalized as follows:
  1. Each pair is written as `(min(i,j), max(i,j))`.
  2. Each pair receives a descriptor
     `(d^2, dx_canon, dy_canon, i, j)`, where `(dx_canon, dy_canon)` is the
     canonical bond direction with `dx > 0`, or `dx == 0 and dy > 0`.
  3. Pairs within the same term are sorted by that descriptor.
- Terms are grouped by their support set, i.e. the sorted set of involved sites.
  Group order is the lexicographic order of support-site coordinates
  `(x, y, site_index)`.
- Inside one support set, terms are sorted lexicographically by the sorted tuple
  of pair descriptors.

- This guarantees:
  - deterministic output for 4-site and 6-site operators;
  - more local pairings appear first;
  - text and JSON outputs share the same K/L ordering.

- LCE and embed must continue to match operators by canonical keys
  (`k4s`, `k6s`, `k8s`), not by K/L label position.
- The canonical key is serialized on every term as sorted pair lists:
  `[[i,j]]`, `[[i,j],[k,l]]`, or `[[i,j],[k,l],[m,n]]`.

Code form:
```python
def canonical_pair_descriptor(cluster, i, j):
    i, j = sorted((i, j))
    dx = cluster[j][0] - cluster[i][0]
    dy = cluster[j][1] - cluster[i][1]
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx*dx + dy*dy, dx, dy, i, j)
```

Validation:
- Re-running the same case must reproduce the same multi-site group order and
  the same term order within each group.

## 4) Coefficient Structure (MUST)

MUST:
- The fit produces one coefficient per operator term. The coefficient vector is
  ordered as:
  1. Constant term $c_0$.
  2. Two-site terms, grouped by bond vector (J1, J2, J3, ...), each bond individually.
  3. Four-site terms, grouped by connected site group, each pairing in canonical order.
  4. Six-site terms, same structure.
- Each coefficient is a complex number. For a Hermitian $H_{\text{eff}}$ the imaginary
  part must be negligible; this is validated in code, not in the output format.

## 5) Human-Readable Output (MUST)

MUST:
- Human-readable text files are sidecars for inspection only. LCE/embed and
  adiabatic seed loading must continue to read the JSON/NPZ machine outputs.
- Exchange text file name: `exchanges/hole{h}_class{c}_exchange.txt`.
- Cluster text file name: `clusters/hole{h}_class{c}_clusters.txt`.
- LCE weight text file name:
  `weights/hole{h}_class{c}_idx{v}.txt`.
- Embed text files are `two_site.txt` plus optional files under
  `clusters/`.
- Exchange text structure:
  ```
  Family: hole={h} class={c} representative={v}
  N={N}
  Sites: 0:(x0,y0)  1:(x1,y1) ...

  Projection:
    method={workflow} artifact={projection_npz}
    block={block} twoSz={twoSz|all} twoS={twoS|all} eta={eta|all} spin_dim={d} selected={d}
      selected_indices=...

  Fit:
    R2={r_squared} relative_error={rel_err} residual={residual}
    |T11-I|={t11m1_norm} overlap={overlap}
    constant={c0}

  Couplings:
    J1  vector=(dx,dy)
      (S0.S1) sites=0-1 coords=(x0,y0)-(x1,y1) coefficient={coefficient}
    K1  support=0,1,2,3 coords=0:(x0,y0) ...
      (S0.S1)(S2.S3) sites=0-1 2-3 coords=... coefficient={coefficient}
  ```
- Cluster text structure:
  ```
  Family: hole={h} class={c} representative={v}
  N={N}
  Representative sites: 0:(x0,y0)  1:(x1,y1) ...

  Clusters:
    cluster {cluster_idx}:
      {operator_index} -> ({x},{y})
  ```
- LCE weight text structure:
  ```
  LCE weight: N={N} hole={h} class={c} cluster={v}
  Sites: 0:(x0,y0)  1:(x1,y1) ...

  Diagnostics:
    subclusters={count}
    reconstruction_error={error}

  Raw summary:
    term_count={count}
    constant={raw_c0}

  Net couplings:
    constant={net_c0}

  Couplings:
    J1  vector=(dx,dy)
      (S0.S1) sites=0-1 coords=(x0,y0)-(x1,y1) coefficient={coefficient}
  ```
- Formatting rules:
  - Summary lines appear first so key metrics are visible without scrolling.
  - Coefficients are printed as real floats when the imaginary part is negligible.
  - Bond directions not present in the cluster are omitted (no placeholder lines).
  - Multi-site groups are labelled `K1`, `K2`, ... and `L1`, `L2`, ... in canonical group order.
  - Multi-site pairings are listed in canonical order (§3) with explicit
    operator notation `(Si.Sj)(Sk.Sl)`.
  - Small quantities use scientific notation (`1.23e-7` not `0.0000001230`).
  - `projection_analysis` text output reuses the same operator-group order for
    bond-structure listings, but omits coupling coefficients.

## 6) Machine-Readable Output (MUST)

MUST:
- File name: `results.json` in the main workflow output directory:
  `ROOT/block_main/N_{N}_nelec_{nelec}_U_{U:.4f}_t_{T:.4f}/mode_*/workflow_*/results.json`.
- `results.json` is a manifest. It records run parameters and points to one
  exchange file plus one cluster-geometry file per `(hole, class_idx)` family.
- Current production `cuprate.main` output represents the complete family set
  for the requested `N`.
- A future partial-family optimization must mark incompleteness explicitly in
  the manifest before LCE is allowed to consume it. The intended future shape is:
  ```json
  {
    "complete_family_set": false,
    "family_selection": {
      "mode": "explicit",
      "families": [[0, 0], [0, 1]]
    }
  }
  ```
- Exchange files live under `exchanges/` and are named
  `hole{h}_class{c}_exchange.json`.
- Cluster-geometry files live under `clusters/` and are named
  `hole{h}_class{c}_clusters.json`.
- No per-cluster sidecar JSON is written.
- `projection` is required in each exchange file because it is the durable
  location for selected eigenstate indices and per-block projection diagnostics.
- Manifest structure:
  ```json
  {
    "schema_version": 5,
    "result_kind": "spin_couplings",
    "run_params": {
      "U": 1.0,
      "T": 0.24,
      "N": 4,
      "nelec": 4,
      "MODE": "full",
      "twoSz": null,
      "twoS": null,
      "eta": null,
      "SCOPE": "nonnegative",
      "workflow": "greedy",
      "parameter_token": "N_4_nelec_4_U_1.0000_t_0.2400",
      "mode_token": "mode_full",
      "workflow_token": "workflow_greedy"
    },
    "families": [
      {
        "hole": 0,
        "class_idx": 0,
        "exchange_file": "exchanges/hole0_class0_exchange.json",
        "clusters_file": "clusters/hole0_class0_clusters.json"
      }
    ]
  }
  ```
- Exchange file structure:
  ```json
  {
    "schema_version": 5,
    "result_kind": "spin_coupling_exchange",
    "N": 4,
    "hole": 0,
    "class_idx": 0,
    "representative_cluster_idx": 0,
    "representative_sites": [[0,0], [1,0], [0,1], [1,1]],
    "projection": {
      "method": "greedy",
      "artifact": "artifacts/hole0_class0_projection.npz",
      "blocks": [
        {
          "block": "full",
          "twoSz": null,
          "twoS": null,
          "eta": null,
          "selected_indices": [0, 1, 2, 3],
          "t11_minus_1_norm": 0.0,
          "overlap": null,
          "selection_info": {}
        }
      ]
    },
    "operators": {
      "constant_term": {"real": ..., "imag": ...},
      "groups": [...]
    },
    "fit": {
      "relative_error": ...,
      "residual": ...,
      "r_squared": ...,
      "t11_minus_1_norm": ...,
      "overlap": ...
    },
    "metadata": {
      "rank": 0,
      "computation_time_s": 1.23
    }
  }
  ```
- Cluster-geometry file structure:
  ```json
  {
    "schema_version": 5,
    "result_kind": "cluster_family_geometry",
    "N": 4,
    "hole": 0,
    "class_idx": 0,
    "representative_cluster_idx": 0,
    "representative_sites": [[0,0], [1,0], [0,1], [1,1]],
    "clusters": [
      {
        "cluster_idx": 0,
        "sites": [[0,0], [1,0], [0,1], [1,1]],
        "indices": [0, 1, 2, 3]
      }
    ]
  }
  ```
- Each entry in `operators.groups` must have a consistent structure regardless of arity:
  ```json
  {
    "arity": 2,
    "vector": [1, 0],
    "label": "J1",
    "terms": [
      {
        "sites": [0, 1],
        "key": [[0, 1]],
        "coefficient": {"real": ..., "imag": ...}
      }
    ]
  }
  ```
  For multi-site groups (arity 4, 6, ...), `vector` is `null` and `label` follows
  the prefix convention in §1 (`K1`, `K2`, ...; `L1`, `L2`, ...).
- Multi-site groups also record `support`, the sorted set of involved sites.
- Every term must include `key`. LCE/embed must use `key` for operator identity;
  `sites` and `label` are serialization/display aids.
- Family exchange metadata (`rank`, `computation_time_s`) lives in a
  `metadata` sub-object, not at the top level.
- In cluster-geometry files, `indices[k]` is the family/operator site index for
  `sites[k]`. The current representative-reordered cluster enumeration normally
  writes `[0, 1, ..., N-1]`.
- Each entry in `projection.blocks` records selected eigenvector column indices
  in that block's solved eigenvector frame. `eta` is null outside
  `MODE=SzS2eta2` and is `0` for the first eta-refined implementation.
- `run_params.SCOPE` is required and must match across main inputs consumed by
  LCE and embed workflows.
- For `workflow=adiabatic`, `run_params.adiabatic_seed` records the seed
  `results.json` path, seed schema version, seed run parameters, and seed
  workflow. Each entry in `projection.blocks` also records its seed block label
  and seed selected indices.
- Projection artifacts must contain enough data to seed a later adiabatic run:
  block labels, basis states, Fock-coordinate eigenvectors, selected indices,
  `H_eff`, and `T11` metrics.
- `selection_info` stores method-specific diagnostics. It may contain compact
  summary fields for `greedy_multi`; detailed trial logs may also be written as
  JSONL by the selector, but the final `results.json` is the durable output.
- `projection_analysis` results use the same family manifest structure with
  `projection` and without `operators` or `fit`.
- LCE workchains must read from the `results.json` manifest plus the referenced
  exchange and cluster-geometry JSON files, not from text files.
- LCE output writes an `lce_results.json` manifest plus referenced
  `weights/hole{h}_class{c}_idx{v}.json` files. Each weight file uses this
  same `operators` schema for net couplings.
- Each LCE weight JSON may have a text sidecar with the same stem. The text
  file is not an input to downstream stages.
- Embed workchains must read the LCE manifest and its referenced weight
  files.
