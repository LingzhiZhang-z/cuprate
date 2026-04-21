# 06-RUNTIME

MPI work distribution, I/O conventions, and path layout.

## 1) MPI Execution Model (MUST)

MUST:
- The code uses `mpi4py` with `MPI.COMM_WORLD`.
- Rank 0 is root for broadcasting and printing.
- Work distribution: round-robin with extra tasks assigned to later ranks.

Code form:
```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# distribute_work: items_per_rank = total // size, remainder to last ranks
```

## 2) Execution Pipeline (MUST)

MUST:
- The three entry points execute in sequence (not in the same process):
  1. `python -m cuprate.main` — cluster enumeration, Hamiltonian solve, selection, downfolding, fit.
  2. `python -m cuprate.lce` — reads results from step 1, performs subcluster subtraction.
  3. `python -m cuprate.embed` — reads results from step 2, embeds into supercell.
- Only `cuprate.main` uses MPI. LCE and embed are single-process.

## 3) Directory Layout (MUST)

MUST:
- `cuprate.main` writes to:
  ```
  ./Block/{base_dir}/{run_dir}/
  ```
  where `base_dir = U{U:.4f}_t{t:.4f}`
  and `run_dir` follows the canonical grammar
  ```
  N{N}[ _twoSz_<value> | _twoSz_all ][ _twoS_<value> | _twoS_all ][_match_spin_sectors][_{workflow}][_restart]
  ```
  with these mode-specific cases:
  - `full` -> `N{N}`
  - `fixed_sz` -> `N{N}_twoSz_<value>`
  - `fixed_sz_s2` -> `N{N}_twoSz_<value>_twoS_<value>`
  - `block_sz_full` -> `N{N}_twoSz_all`
  - `block_sz_s2_full` -> `N{N}_twoSz_all_twoS_all`
- `all` and `n` are the only non-numeric tokens permitted in `twoSz` / `twoS` path slots.
- Negative fixed values use `n` as the minus sign, e.g. `twoSz_n1` for $2S_z = -1$.

- `cuprate.lce` reads from `Block/{base_dir}/` and writes to `data_transfer/LCE/LCE_{base_dir}/`.

- `cuprate.embed` reads from `data_transfer/LCE/LCE_{base_dir}/` and writes to `data_transfer/Embed/Embed_{base_dir}/`.

## 4) File Naming (MUST)

MUST:
- Per-cluster results: `hole{h}_class{c}_cluster{v}_results.txt` and `.json`.
- Eigensystem: `hole{h}_class{c}_eigvals.npy`, `hole{h}_class{c}_eigvecs.npy`.
- Derived data: `_Heff.npy`, `_T11m1.npy`, `_t11_selected_indices.npy`,
  `_double_occupation_expectation.npy`, `_states.npy`, `_S2_diagonal.npy`.

## 5) Result Output (MUST)

MUST:
- Operator definitions, canonical ordering, and output file formats (`.txt` and `.json`)
  are defined in `08-OPERATOR_OUTPUT.md`.

## 6) Restart Protocol (MUST)

MUST:
- When `restart=True`, the code reads precomputed eigensystems from disk instead of rediagonalising.
- Only the selection + downfolding + fit steps are re-executed.
- The restart directory contains `_eigvals.npy` and `_eigvecs.npy` for each cluster.
- The `adiabatic` workflow reads eigensystems and selected indices from the **previous** parameter point,
  computed from `t_previous = t - delta`.
- At the first adiabatic point of a sweep, if the previous-point adiabatic result is absent,
  the code seeds adiabatic selection from the same-parameter baseline run (`workflow=None`).

## 7) CLI Parameters (MUST)

MUST:
- All parameters are passed as `KEY=VALUE` on the command line.
- CLI keys are case-insensitive.
- Key parameters:
  - `N`, `U`, `T`: physical parameters.
  - `MODE`: one of `full`, `fixed_sz`, `block_sz_full`, `fixed_sz_s2`, `block_sz_s2_full`.
  - `MODE=all` is accepted as an input alias for `MODE=full`.
  - `twoSz`, `twoS`: physical symmetry labels for fixed-sector runs.
    The implementation may accept any capitalization of these keys.
  - `workflow`: workflow (`occ`, `energy`, `single`, `multi`, `adiabatic`).
    The implementation may accept any capitalization of this key.
  - `SELECT`: selection mode (`block` or default).
  - `RESTART`: `true`/`false`.
  - `DELTA`: adiabatic step size.
  - `NCELL`, `NCUT`: embedding parameters.
  - `MATCH_SPIN_SECTORS`: `true`/`false`.
- The standard does not permit `TYPE`, `SZ`, `S`, `S2`, `SZ_IDX`, or `S_IDX`.
- `twoSz` and `twoS` must satisfy the physical constraints defined in `00-CONVENTIONS.md`.
