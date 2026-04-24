# Agent Instructions

## Before modifying code

1. Read the relevant standard in `standards/en/` first.
2. Follow the naming conventions in `00-CONVENTIONS.md` §6-7. Never introduce forbidden names.
3. Check `CLAUDE.md` for the code ↔ standards mapping to find the right standard.

## Naming rules (non-negotiable)

- `twoSz` (not `two_sz`, `sz`, `Sz`)
- `twoS` (not `two_s`, `s`, `S`)
- `S2` for S² (not `s_squared`, `ssq`, `SSquared`)
- Mode strings: `fixed_sz_s2`, `block_sz_s2_full` (not `fixed_sz_ssq`, `block_sz_ssq_full`)
- Path tokens: `twoSz_<value>`, `twoS_<value>`, negative uses `n` prefix (`twoSz_n1`)
- CLI: `workflow` (not `TYPE`)

## Standards structure

```
00-CONVENTIONS       Naming, tolerances
01-CLUSTERS          Geometry (independent of physics)
02-HAMILTONIAN       Fock basis (§1-7) + Hubbard model (§8-11)
03-SYMMETRY_SECTORS  Sz/S² block diagonalization, five modes
04-DOWNFOLDING       Selection → T₁₁ → H_eff → spin fit
05-LCE_AND_EMBEDDING Möbius inversion + supercell embedding contract
06-RUNTIME           Workchain/cache/output contract
07-TESTING           Regression/reference-data policy
08-OPERATOR_OUTPUT   Spin-coupling and projection output schema
```

## How to read the code

- First pass for the single-cluster physics line:
  `hubbard.py` → `states.py` → `sectors.py` → `manifold.py`
- Read `hubbard.py` first:
  it is the active single-cluster coordinator. `HubbardModel` owns basis generation, symmetry blocking, Hamiltonian construction, solving, merging, projection, and fitting state. Its lifecycle is `set_symmetry()` → `build_hamiltonians()` → `solve()` → optional `merge_by_s2()` / `merge_by_sz()` → `project()` → `fit()`.
- `hubbard.py` owns the public mode aliases and internal `MODE_*` constants. The public canonical modes remain `full`, `fixed_sz`, `block_sz_full`, `fixed_sz_s2`, and `block_sz_s2_full`.
- Read `states.py`:
  it defines the Fock basis, sorting, double occupation, state-space `Sz`/`S2` operators, and fermionic signs — the primitives consumed by `hubbard.py`.
- Read `sectors.py` after that:
  this is where algebraic `S2` sector blocks (`S2SectorBlock`) and fixed-`twoSz` → `(twoSz,twoS)` transforms live.
- Read `manifold.py` last in the core line:
  it holds the `Block` container, per-block selection methods (`occ`, `energy`, `greedy`, `greedy_multi`, `adiabatic`), `Block.downfold`, `Block.t11_norm`, spin-operator construction, and least-squares fit helpers.
- Treat `mpi.py` as runtime plumbing only.
- Treat `src/cuprate/back/` as old reference material. It may be useful for physics comparison, but it is not active structure and must not be copied as compatibility scaffolding.

## Where the physics core lives

- `clusters.py`:
  square-lattice geometry, bond generation, cluster classification.
- `states.py`:
  Hubbard basis encoding, ordering, state-space `Sz`/`S2` operators, and the single-state hopping primitive `apply_hop`; pure-spin rows are identified with `pure_spin_state_indices(...)`.
- `hubbard.py`:
  the `HubbardModel` single-cluster coordinator; owns the Hubbard Hamiltonian matrix-element methods (`_build_hamiltonian_t`, `_build_hamiltonian_U`, composed as `build_hamiltonian`) and wires basis generation, symmetry blocking, and per-block diagonalisation (`np.linalg.eigh`).
- `sectors.py`:
  `Sz` grouping, highest-weight `S2` multiplets, and `Sz -> (Sz,S)` transforms.
- `manifold.py`:
  `Block`, eigenstate selection, `T11`, `H_eff`, and spin-operator fitting helpers.

## Key dataclasses

- `Cluster` (in `clusters.py`) — geometry plus enumeration tags (`hole`, `class_idx`, `cluster_idx`); representatives are the first member of each `(hole, class_idx)` family.
- `S2SectorBlock` (in `sectors.py`) — one algebraic `(twoSz, twoS, D)` transform block used to assemble full `(twoSz, twoS)` transforms.
- `Block` (in `manifold.py`) — one symmetry block with `basis_states`, optional `ham`, `eigvals`, `eigvecs`, optional `basis_transform`, `twoSz`, and `twoS`; owns `spin_fock_rows`, `spin_sector_columns`, `spin_dim`, `t11_norm`, `selected_*`, `downfold`, and `_spin_operators`.
- `HubbardModel` (in `hubbard.py`) — the active single-cluster coordinator. It stores solved `blocks` and in-memory projection/fit results (`selected_indices`, `selection_info`, `heff`, `t11m1_norms`, `coupling_coeffs`, `fit_metrics`).

## Five diagonalization modes

| Mode | Blocks before optional merge | Optional reconstruction |
|------|------------------------------|-------------------------|
| `full` | One full Fock block | N/A |
| `fixed_sz` | One fixed-`twoSz` block | No |
| `block_sz_full` | All fixed-`twoSz` blocks | `merge_by_sz()` |
| `fixed_sz_s2` | One fixed-`(twoSz,twoS)` block | No |
| `block_sz_s2_full` | All fixed-`(twoSz,twoS)` blocks | `merge_by_s2()` then `merge_by_sz()` |

## Do not modify tests without explicit request

Test files under `tests/` are separately maintained. Do not modify them unless the user explicitly asks.

## No Compatibility Layer

- Do not add compatibility shims, fallback paths, translation layers, or dual-format support unless the user explicitly asks.
- Remove obsolete transition code instead of preserving `legacy`, `old`, `deprecated`, or compatibility helpers.
- Do not keep renamed or superseded interfaces alive alongside the current one.
- Production code should use the current canonical representation directly, including persisted `_states.npy` output.
- Old files under `src/cuprate/back/` may be read for reference, but production code must be based on the active modules above.
