# 2026-04-06 Adversarial Review Summary

Purpose: concise review notes for follow-up work by Opus.

## Scope Agreed With User

Out of scope for now:

- `t2` / `t3` support regression
- embed-stage machine-readable JSON output

In scope for this round:

- fix the shared `S2` cache race / shared write path
- fix silent dropping of contradictory `twoSz` / `twoS` inputs
- fix rank selection for root-only progress printing
- keep the remaining correctness concerns documented for later follow-up

## Fixed In This Round

### 1. Mode / sector argument contradictions now fail explicitly

File: `src/cuprate/io.py`

Before:

- `MODE=full twoS=2`
- `MODE=block_sz_full twoSz=2`
- `MODE=fixed_sz twoSz=0 twoS=2`

were parsed and then silently normalized away by `resolve_mode_spec()`.

Now:

- contradictory fixed-sector inputs raise `ValueError`
- no silent correction remains for these cases

Relevant code:

- `src/cuprate/io.py`, helper `_sector_arg_labels()`
- `src/cuprate/io.py`, `resolve_mode_spec()`

### 2. `S2` basis cache is no longer written to shared repo-root `S2/`

Files:

- `src/cuprate/hubbard.py`
- `src/cuprate/main/__main__.py`

Before:

- every work item called `save_blocks()` then `load_blocks()`
- `save_blocks()` wrote to a shared relative path `S2/`
- multi-rank runs could race on the same `.npy` files

Now:

- main flow no longer performs the eager shared write step
- `load_blocks()` loads or builds basis data through `_load_or_build_S2_basis()`
- if `tmp_dir` is available, cache files are written under:

```text
<run tmp dir>/s2_rank{rank}/
```

- cache is therefore rank-local and run-local instead of global to the repository root

Observed verification result:

```text
root_entries= ['tmp']
has_shared_S2= False
has_rank_cache= True
cache_entries= ['basis_eigvals_N2_twoSz_0.npy', 'basis_eigvecs_N2_twoSz_0.npy']
```

### 3. Root-only progress output now uses rank 0 again

File: `src/cuprate/main/__main__.py`

Before:

- progress / run-summary printing used `is_root(size - 1)`

Now:

- root-only printing uses `is_root()`
- progress messages are emitted by rank 0, matching the runtime standard

## Deferred By Explicit User Choice

### A. `t2` / `t3` support

This remains deferred intentionally.

Important note:

- this is still a real standards gap versus `02-HAMILTONIAN.md` and `06-RUNTIME.md`
- it is not treated as a bug for the current round only because the user explicitly deprioritized it

### B. Embed JSON / machine-readable output

This remains deferred intentionally.

## Still Outstanding After This Round

### 1. New core modules are still untracked in git

These files are imported by the refactored code path but were not tracked in git at review time:

- `src/cuprate/states.py`
- `src/cuprate/hamiltonian.py`
- `src/cuprate/sectors.py`
- `src/cuprate/downfolding.py`

Risk:

- if someone uses partial staging such as `git add -u`, the refactor can be committed without these required files
- that would produce an immediately broken tree after checkout

Recommended action:

- verify final intended contents of the four new modules
- stage them explicitly before any merge or handoff

### 2. Degenerate-eigenspace canonicalization is still missing in the Hamiltonian path

File:

- `src/cuprate/hamiltonian.py`

Current state:

- `diagonalise()` and `diagonalise_blocked()` call `np.linalg.eigh` directly
- they do not apply the canonicalization required by the standard

Why this matters:

- degenerate eigenvectors can rotate arbitrarily
- this can affect reproducibility, restart consistency, and reference-data comparisons

Recommended action:

- wire `canonicalize_eigenpairs()` into the Hamiltonian diagonalization path
- do this for both single-matrix and blocked diagonalization flows

## Verification Run For This Round

Executed:

```bash
python -m pytest tests/ -x -q
```

Result:

```text
42 passed in 6.07s
```

Also verified with targeted repro scripts:

- contradictory mode / sector inputs now raise explicit `ValueError`
- `S2` basis cache no longer creates a shared repo-root `S2/` directory

## Recommended Next Fix Order

1. Git hygiene / safety:
   explicitly add and review the new core modules before any merge
2. Determinism:
   restore canonicalized eigenpair handling in `src/cuprate/hamiltonian.py`
3. Deferred feature track:
   reintroduce `t2` / `t3` support across CLI, paths, clusters, and hopping assignment

