# 06-RUNTIME

Active workchain, cache, and I/O contract for the current `HubbardModel` +
`Block` core.

## 1) Active Runtime Boundary (MUST)

MUST:
- The current active source tree has production `cuprate.main` for raw
  spin-coupling generation and production `cuprate.lce` for linked-cluster
  subtraction.
- The current active source tree has production `cuprate.embed` for target
  embedding of LCE weights.
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
model.set_symmetry(mode, twoSz=twoSz, twoS=twoS, scope=scope)
model.build_hamiltonians()
model.solve(cache_mode=cache_mode, cache_dir=cache_dir)
if merge == "Sz":
    model.merge_to_sz(merge_basis)
model.project(method=workflow, **select_kwargs)
model.fit(bond_groups=bond_groups)
```

- Optional merge is explicit:

Code form:
```python
model.merge_to_sz("fock")   # merge twoS sectors into fixed-twoSz Fock rows
model.merge_to_sz("block")  # merge twoS sectors in merged block coordinates
```

- Public canonical CLI modes are `full`, `Sz`, `SzS2`, and `SzS2eta2`.
- Fixed `twoSz` and `twoS` values are separate optional selector parameters.
- `SCOPE` controls the sector range for all-`twoSz` `MODE=Sz`, `MODE=SzS2`,
  and `MODE=SzS2eta2` runs; default `SCOPE=nonnegative` builds `twoSz >= 0`,
  while `SCOPE=pm` builds both positive and negative `twoSz`.
- `MODE=SzS2eta2` first follows the `MODE=SzS2` block selection rules, then
  refines each selected `(twoSz,twoS)` block to `eta=0`.
- `MERGE=Sz` is accepted only for fixed-`twoSz` `MODE=SzS2` and
  `MODE=SzS2eta2` runs without fixed `twoS`. `MERGE_BASIS` is `fock` or
  `block`.

## 3) Eigensystem Cache (MUST)

MUST:
- The solve cache stores only solved blocks: basis states, eigenvalues,
  eigenvectors, and optional `basis_transform`; it does not store the
  Hamiltonian.
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
- `EIGH=lowmem|fast` selects the dense diagonalisation driver for newly solved
  blocks. It has no effect for blocks loaded from cache.
- Runtime/workchain callers default `CACHE_MODE` to `save`.
- Runtime/workchain callers default `EIGH` to `lowmem`.
- `HubbardModel.save(cache_dir)` and `Block.save(cache)` must write the same
  solved block format.

Code form:
```python
cache = Path(cache_dir) / cluster.label()
block.save(cache)
block = Block.load(cache, twoSz, twoS, eta)
```

- Runtime path construction is centralized in `cuprate.paths`.
- Eigensystem data directories are:
  - `DATA` for `MODE=full`.
  - `DATA_twoSz` for `MODE=Sz`, shared by default and `SCOPE=pm`.
  - `DATA_twoSz_twoS` for `MODE=SzS2`, shared by default and `SCOPE=pm`.
  - `DATA_twoSz_twoS_eta_0` for `MODE=SzS2eta2`, shared by default and
    `SCOPE=pm`.
- `SCOPE=pm` does not create a separate eigensystem cache directory. It reuses
  the same block-keyed cache and may extend it via `CACHE_MODE=partial`.

Code form:
```text
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/DATA_twoSz/hole0_class0_idx0/twoSz_0_data.npz
```

## 4) Workchain Computation Model (MUST)

MUST:
- Cluster enumeration uses `ClusterSets(N)`.
- ED, projection, downfolding, fitting, and operator output are computed once
  per isomorphic family representative, identified by `(hole, class_idx)`.
- Other `cluster_idx` members in the same family reuse the representative
  eigensystem, projection data, and fitted exchange coefficients.
- Per-`cluster_idx` output stores only geometry metadata (`sites` and
  `indices`) needed to place the family exchange data on that member.
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
- `adiabatic` seeds are explicit. The caller must pass `SEED_RESULTS`, a path
  to a previously completed `results.json`.
- The seed selection scheme is read from the seed file's
  `run_params.workflow`.
- The adiabatic seed consists of previous block eigenvectors and previous
  selected indices loaded through that seed file's projection artifact. The
  seed must come from projection output, not from the solve cache alone.
- Missing adiabatic seed data is an error. Do not fall back to a baseline
  same-parameter run.

## 6) Derived Output Ownership (MUST)

MUST:
- Derived projection output includes at least selected indices, per-block
  selection diagnostics, `H_eff`, and `T11` metrics.
- Fit output includes family-level coefficients and fit metrics.
- The workchain/result-output layer owns writing derived outputs.
- `cuprate.io` owns durable output payload construction, schema-version
  constants, JSON/NPZ writes, and human-readable text sidecars for main, LCE,
  and embed outputs.
- `Block` owns local computation and optional selection JSONL logging, but not
  the durable result schema.
- The canonical JSON shape is defined in `08-OPERATOR_OUTPUT.md`.

## 7) Runtime CLI Parameters (MUST)

MUST:
- CLI parameters are passed as `KEY=VALUE`.
- `cuprate.cli` owns shared `KEY=VALUE` parsing, main-stage defaults, main-stage
  `MODE`/`twoSz`/`twoS` validation, and seed-set parsing for `cuprate.lce` and
  `cuprate.embed`.
- CLI keys are case-insensitive.
- Canonical keys include:
  - `N`, `U`, `T`: physical parameters.
  - `MODE`: one of `full`, `Sz`, `SzS2`, or `SzS2eta2`; defaults to `full`.
  - `twoSz`, `twoS`: optional fixed-block selectors. `twoS` requires
    `MODE=SzS2` or `MODE=SzS2eta2` and a fixed `twoSz`.
  - `SCOPE`: one of `nonnegative` or `pm`; defaults to `nonnegative` and applies
    only to all-`twoSz` `MODE=Sz` / `MODE=SzS2` / `MODE=SzS2eta2` runs.
  - `eta` is not a production CLI key. `MODE=SzS2eta2` currently calculates
    `eta=0` blocks only.
  - `workflow`: one of `occ`, `energy`, `greedy`, `greedy_multi`, `adiabatic`;
    defaults to `occ`.
  - `CACHE_MODE`: one of `none`, `load`, `save`, `partial`; defaults to
    `save`.
  - `ROOT`: root directory for all `block_main`, `block_lce`, and
    `block_embed` outputs; defaults to `results`.
  - `SEED_RESULTS`: previous `results.json`, required only for
    `workflow=adiabatic`.
  - `SEED_SET`: LCE/embed text file under `ROOT`; each non-comment line is a
    main `results.json` path relative to `ROOT`.
- `CACHE_DIR`, `OUTPUT_DIR`, and `INPUTS` are not production CLI keys.
- `cuprate.lce` takes exactly `ROOT`, `N`, `U`, `T`, and `SEED_SET`; for this
  stage `N` means `Nmax`.
- `cuprate.lce` reads main results from `SEED_SET`, validates that they cover
  `N=2..Nmax`, and validates their `U/T` against the CLI values.
- `cuprate.lce` writes `lce_results.json` as a manifest plus concrete cluster
  weight JSON files and inspection-only text sidecars under `weights/N_<N>/`.
- `cuprate.embed` takes exactly `ROOT`, `N`, `U`, `T`, and the same `SEED_SET`;
  it must read the corresponding `lce_results.json` manifest and its referenced
  weight files from the matching `seed_<stem>` directory.
- The standard does not permit production keys `TYPE`, `SZ`, `S`, `S2`,
  `SZ_IDX`, `S_IDX`, `BLOCKS`, `SELECT`, or `MATCH_SPIN_SECTORS`.

## 8) Runtime Directory Contract (MUST)

MUST:
- The three runtime stages differ only by the stage prefix:
  - `block_main`
  - `block_lce`
  - `block_embed`
- The common parameter directory token is
  `N_{N}_nelec_{nelec}_U_{U:.4f}_t_{T:.4f}`.
- Main workflow outputs live under `mode_* / workflow_*`.
- Default `SCOPE=nonnegative` uses the short all-`twoSz` path tokens
  `mode_twoSz`, `mode_twoSz_twoS`, and `mode_twoSz_twoS_eta_0`.
- Explicit `SCOPE=pm` uses `mode_twoSz_pm`, `mode_twoSz_pm_twoS`, and
  `mode_twoSz_pm_twoS_eta_0`.
- LCE and embed outputs live under `seed_<stem>`, where `<stem>` is the
  `SEED_SET` file stem.

Code form:
```text
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_0/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_pm/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_0_twoS_2/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_twoS_eta_0/workflow_occ/results.json
ROOT/seed_sets/block.txt
ROOT/block_lce/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/lce_results.json
ROOT/block_lce/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/weights/N_2/hole0_class0_idx0.json
ROOT/block_lce/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/weights/N_2/hole0_class0_idx0.txt
ROOT/block_embed/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/
```

## 9) Future Partial-Family Runs (MAY)

MAY:
- The first production `cuprate.main` workchain computes the complete
  `(hole, class_idx)` family set for the requested `N`.
- A future optimization may add an explicit family-selection parameter, for
  example `FAMILIES=0:0,0:1,1:0`.
- The selection unit is a family `(hole, class_idx)`, not an individual
  `cluster_idx`, unless representative reuse is explicitly redesigned.
- Family selection must be applied to the global family task list before MPI
  rank distribution. Do not filter independently inside each rank after
  slicing.

Code form:
```python
families = _enumerate_families(N)
families = _filter_families(families, selected_families)
for family in families[rank::size]:
    process_family(family)
```

- Partial-family output must be marked explicitly in `results.json`; downstream
  LCE must reject partial main outputs unless a future debug/partial mode is
  explicitly requested.
- For `workflow=adiabatic`, the seed `results.json` only needs to contain the
  selected families that the current partial run will process.
