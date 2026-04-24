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
- File name: `hole{h}_class{c}_cluster{v}_results.txt`.
- Structure:
  ```
  === Summary ===
  Hole: {h}  Class: {c}  Cluster: {v}
  N={N}  Sites: (x0,y0) (x1,y1) ...
  R²: {r_squared}  |T11-I|: {t11m1_norm}  Overlap: {overlap}

  === Two-site couplings ===
  J1  vector (dx,dy):
      Sites i-j  (xi,yi)-(xj,yj):  {coefficient}
      ...

  === Four-site couplings ===
  Group K1: sites {a, b, c, d}
      (Sa·Sb)(Sc·Sd):  {coefficient}
      (Sa·Sc)(Sb·Sd):  {coefficient}
      (Sa·Sd)(Sb·Sc):  {coefficient}

  === Six-site couplings ===
  Group L1: sites {a, b, c, d, e, f}
      (Sa·Sb)(Sc·Sd)(Se·Sf):  {coefficient}
      ...

  === Fit quality ===
  Constant term:   {c0}
  Relative error:  {rel_err}
  Residual:        {residual}
  R²:              {r_squared}
  |T11-I|:         {t11m1_norm}
  Overlap:         {overlap}
  ```
- Formatting rules:
  - Summary block appears first so key metrics are visible without scrolling.
  - Coefficients are real floats; imaginary parts are validated in code, not printed.
  - Bond directions not present in the cluster are omitted (no placeholder lines).
  - Multi-site groups are labelled `K1`, `K2`, ... and `L1`, `L2`, ... in canonical group order.
  - Multi-site pairings are listed in canonical order (§3) with explicit
    operator notation `(Si·Sj)(Sk·Sl)`.
  - Small quantities use scientific notation (`1.23e-7` not `0.0000001230`).
  - `projection_analysis` text output reuses the same operator-group order for
    bond-structure listings, but omits coupling coefficients.

## 6) Machine-Readable Output (MUST)

MUST:
- Per-cluster sidecar JSON files are written as `hole{h}_class{c}_cluster{v}_results.json`.
  Each sidecar contains the same payload shape as the corresponding entry in the consolidated results.
- File name: `results.json` in the run output directory.
- Contains all cluster results for the run in a single file.
- `projection` is required for spin-coupling results because it is the durable
  location for selected eigenstate indices and per-block projection diagnostics.
- Structure:
  ```json
  {
    "schema_version": 2,
    "result_kind": "spin_couplings",
    "run_params": {"U": 1.0, "t": 0.24, "N": 4, "MODE": "full", "workflow": "greedy"},
    "clusters": [
      {
        "hole": 0,
        "class_idx": 0,
        "cluster_idx": 0,
        "sites": [[0,0], [1,0], [0,1], [1,1]],
        "projection": {
          "method": "greedy",
          "blocks": [
            {
              "block": "twoSz_all_twoS_all",
              "twoSz": null,
              "twoS": null,
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
    ]
  }
  ```
- Each entry in `operators.groups` must have a consistent structure regardless of arity:
  ```json
  {
    "arity": 2,
    "vector": [1, 0],
    "label": "J1",
    "terms": [{"sites": [0, 1], "coefficient": {"real": ..., "imag": ...}}]
  }
  ```
  For multi-site groups (arity 4, 6, ...), `vector` is `null` and `label` follows
  the prefix convention in §1 (`K1`, `K2`, ...; `L1`, `L2`, ...).
- Per-cluster metadata (`rank`, `computation_time_s`) lives in a `metadata`
  sub-object, not at the top level.
- Each entry in `projection.blocks` records selected eigenvector column indices
  in that block's solved eigenvector frame.
- `selection_info` stores method-specific diagnostics. It may contain compact
  summary fields for `greedy_multi`; detailed trial logs may also be written as
  JSONL by the selector, but the final `results.json` is the durable output.
- `projection_analysis` results use the same consolidated structure with
  `projection` and without `operators` or `fit`.
- Future LCE/embed workchains must read from this consolidated JSON, not from
  text files.
