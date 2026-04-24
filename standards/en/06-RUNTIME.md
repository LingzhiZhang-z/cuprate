# 06-RUNTIME

Active workchain, cache, and I/O contract for the current `HubbardModel` +
`Block` core.

## 1) Active Runtime Boundary (MUST)

MUST:
- The current active source tree has no production `cuprate.main`,
  `cuprate.lce`, or `cuprate.embed` entry point.
- Runtime code must be rebuilt on top of the active modules:
  `clusters.py`, `states.py`, `sectors.py`, `manifold.py`, `hubbard.py`, and
  `mpi.py`.
- `src/cuprate/back/` is reference material only. Do not add compatibility
  paths that execute or preserve the old workchain structure.

## 2) Single-Cluster Lifecycle (MUST)

MUST:
- A single-cluster calculation follows this staged lifecycle:

Code form:
```python
model = HubbardModel(cluster, U, t)
model.set_symmetry(mode, twoSz=twoSz, twoS=twoS)
model.build_hamiltonians()
model.solve(cache_mode=cache_mode, cache_dir=cache_dir)
model.project(method=workflow, **select_kwargs)
model.fit(bond_groups=bond_groups)
```

- Optional reconstruction is explicit:

Code form:
```python
model.merge_by_s2()  # merge twoS sectors inside each twoSz
model.merge_by_sz()  # merge fixed-twoSz blocks into one full block
```

- Public canonical modes are `full`, `fixed_sz`, `block_sz_full`,
  `fixed_sz_s2`, and `block_sz_s2_full`.
- Internal mode constants and cache buckets are owned by `hubbard.py`.

## 3) Eigensystem Cache (MUST)

MUST:
- The solve cache stores only solved blocks: basis states, eigenvalues,
  eigenvectors, Hamiltonian, and optional `basis_transform`.
- The solve cache does not store selected indices, `H_eff`, `T11`, or fit
  results.
- `HubbardModel.solve()` supports exactly four cache modes:

| cache_mode | Behavior |
|------------|----------|
| `none` | Solve every current block in memory; no disk cache |
| `load` | Load every current block from disk; fail if any block is missing |
| `save` | Solve every current block, then save every block |
| `partial` | Load existing blocks and solve/save missing blocks |

- `cache_dir` is required for `load`, `save`, and `partial`, and forbidden for
  `none`.
- `HubbardModel.save(cache_dir)` and `Block.save(cache)` must write the same
  solved block format.

Code form:
```python
cache = Path(cache_dir) / model.label() / cluster.label() / model._bucket
block.save(cache)
block = Block.load(cache, twoSz, twoS)
```

## 4) Workchain Computation Model (MUST)

MUST:
- Cluster enumeration uses `ClusterSets(N)`.
- ED, projection, and downfolding are computed once per isomorphic family
  representative, identified by `(hole, class_idx)`.
- Other `cluster_idx` members in the same family reuse the representative
  eigensystem/projection data.
- Fitting and reporting may still be emitted for every `cluster_idx`, using the
  member cluster's own operator groups.
- The workchain must not solve every isomorphic member independently unless the
  user explicitly disables representative reuse.

## 5) Workflow Selection (MUST)

MUST:
- The canonical workflow key is `workflow`.
- Supported workflow values are `occ`, `energy`, `greedy`, `greedy_multi`, and
  `adiabatic`.
- Old names such as `single`, `multi`, `multi_restart`, and
  `adiabatic_restart` are reference-data names only and are not production
  workflow values.
- `adiabatic` seeds are located from `DELTA`: the previous point is
  `t_previous = t - DELTA` under the same cache/output root convention.
- The adiabatic seed consists of previous block eigenvectors and previous
  selected indices. The seed must come from the previous point's projection
  output, not from the solve cache alone.
- Missing adiabatic seed data is an error. Do not fall back to a baseline
  same-parameter run.

## 6) Derived Output Ownership (MUST)

MUST:
- Derived projection output includes at least selected indices, per-block
  selection diagnostics, `H_eff`, and `T11` metrics.
- Fit output includes coefficients and fit metrics.
- The workchain/result-output layer owns writing derived outputs.
- `Block` owns local computation and optional selection JSONL logging, but not
  the durable result schema.
- The canonical JSON shape is defined in `08-OPERATOR_OUTPUT.md`.

## 7) Future CLI Parameters (MUST)

MUST:
- CLI parameters, once reintroduced, are passed as `KEY=VALUE`.
- CLI keys are case-insensitive.
- Canonical keys include:
  - `N`, `U`, `T`: physical parameters.
  - `MODE`: one of the five public modes.
  - `twoSz`, `twoS`: fixed-sector labels when required by `MODE`.
  - `workflow`: one of `occ`, `energy`, `greedy`, `greedy_multi`, `adiabatic`.
  - `CACHE_MODE`: one of `none`, `load`, `save`, `partial`.
  - `CACHE_DIR`: root directory for the solve cache.
  - `DELTA`: adiabatic step size.
- The standard does not permit production keys `TYPE`, `SZ`, `S`, `S2`,
  `SZ_IDX`, `S_IDX`, `SELECT`, or `MATCH_SPIN_SECTORS`.
