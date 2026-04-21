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
05-LCE_AND_EMBEDDING Möbius inversion + supercell embedding
06-RUNTIME           CLI, paths, I/O
```

## How to read the code

- First pass for the single-cluster physics line:
  `main/solver.py` → `hubbard.py` → `states.py` → `hamiltonian.py` → `sectors.py` → `downfolding.py`
- Read `main/solver.py` first to see the execution order:
  set mode and states, build `S2` transforms when needed, solve the eigensystem, then run selection/downfolding.
- Treat `hubbard.py` as the coordinator:
  it does not contain most formulas, but it owns the data flow between blocks, eigensystems, selection, and outputs.
- Read `states.py` and `hamiltonian.py` together:
  they define the Fock basis, sorting, double occupation, `Sz`/`S2` state-space operators, fermionic signs, and the Hubbard matrix elements.
- Read `sectors.py` after that:
  this is where `Sz` / `S2` block structure, `S2` transforms, and full-spectrum reconstruction actually live.
- Read `downfolding.py` last in the core line:
  this is where state selection, `T11`, `H_eff`, and spin-coupling fitting are implemented.
- Only after the single-cluster line is clear, read the thermodynamic post-processing line:
  `lce/workflow.py` → `lce/core.py` → `embed/workflow.py` → `embed/core.py` → `embed/reporting.py`
- On a first pass, treat `io.py`, `main/workflow.py`, `main/reporting.py`, and `mpi.py` as orchestration/I/O rather than physics kernels.

## Where the physics core lives

- `clusters.py`:
  square-lattice geometry, bond generation, cluster classification.
- `states.py`:
  Hubbard basis encoding, ordering, and state-space `Sz`/`S2` operators; pure-spin states are tracked explicitly via
  `pure_spin_state_indices(...)`, `states_spin`, and `spin_basis`.
- `hamiltonian.py`:
  the single-band Hubbard Hamiltonian and diagonalisation.
- `sectors.py`:
  `Sz` sectors, `Sz -> (Sz,S)` transforms, and reconstruction.
- `downfolding.py`:
  eigenstate selection, `T11`, `H_eff`, and spin-operator fitting.
- `lce/core.py`:
  Möbius-style subcluster subtraction.
- `embed/core.py`:
  periodic supercell embedding and coupling accumulation.
- `hubbard.py`:
  the physics coordinator; it wires together basis generation, symmetry blocking, solving, and downfolding, but most formulas live in the modules above.

## Key dataclasses in hubbard.py

- `SzSectors` — Sz sector decomposition data
- `S2Sectors` — S² sector decomposition data (transforms, sector_list, etc.)
- `S2Diagnostics` — S² expectation values (diag, error)
- `DownfoldResult` — SVD downfolding results (heff, t11m1, selected_indices, etc.)

## Five diagonalization modes

| Mode | Blocks | Reconstruct full spectrum |
|------|--------|--------------------------|
| `full` | None | N/A |
| `fixed_sz` | Single Sz | No |
| `block_sz_full` | All Sz | Yes |
| `fixed_sz_s2` | Single (Sz,S) | No |
| `block_sz_s2_full` | All (Sz,S) | Yes |

## Do not modify tests without explicit request

Test files under `tests/` are separately maintained. Do not modify them unless the user explicitly asks.

## No Compatibility Layer

- Do not add compatibility shims, fallback paths, translation layers, or dual-format support unless the user explicitly asks.
- Remove obsolete transition code instead of preserving `legacy`, `old`, `deprecated`, or compatibility helpers.
- Do not keep renamed or superseded interfaces alive alongside the current one.
- Production code should use the current canonical representation directly, including persisted `_states.npy` output.
