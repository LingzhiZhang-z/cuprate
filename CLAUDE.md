# Cuprate Project

Finite-cluster exact diagonalization of the single-band Hubbard model, with SVD downfolding to effective spin Hamiltonians and linked-cluster expansion (LCE) to the thermodynamic limit.

## Standards

Authoritative scientific standards live in `standards/en/`. Code MUST follow these standards.

| File | Scope |
|------|-------|
| `00-CONVENTIONS.md` | Naming conventions (`twoSz`, `twoS`, `S2`), forbidden names, tolerances |
| `01-CLUSTERS.md` | Square-lattice cluster enumeration, bond classification |
| `02-HAMILTONIAN.md` | Fock encoding, Hubbard Hamiltonian, diagonalization |
| `03-SYMMETRY_SECTORS.md` | Sz/S² sector decomposition, five diagonalization modes |
| `04-DOWNFOLDING.md` | Eigenstate selection, T₁₁ metric, H_eff via SVD, spin-coupling fit |
| `05-LCE_AND_EMBEDDING.md` | Möbius inversion (LCE), supercell embedding |
| `06-RUNTIME.md` | CLI parameters, path grammar, I/O layout |
| `07-TESTING.md` | Regression testing strategy, reference data, naming translation |
| `08-OPERATOR_OUTPUT.md` | Spin-coupling operators, canonical pairing order, output formats |

## Naming Conventions

Canonical names (from `00-CONVENTIONS.md` §6):
- `twoSz` = 2Sz (integer), `twoS` = 2S (integer), `S2` = S² = S(S+1)
- Forbidden: `ssq`, `s_squared`, `two_sz`, `two_s`, `SSquared`
- Modes: `full`, `fixed_sz`, `block_sz_full`, `fixed_sz_s2`, `block_sz_s2_full`
- Path tokens: `twoSz_<value>`, `twoS_<value>`, `twoSz_all`, `twoSz_all_twoS_all`
- Negative path values use `n` prefix: `twoSz_n1` for -1
- CLI keys are case-insensitive. `TYPE`, `SZ`, `S`, `SZ_IDX`, `S_IDX` are forbidden.

## Code ↔ Standards Mapping

| Code module | Primary standard | Role |
|-------------|-----------------|------|
| `clusters.py` | 01-CLUSTERS | Cluster geometry, enumeration, bond classification |
| `states.py` | 02-HAMILTONIAN §1-7 | Fock-state encoding, basis generation, sorting, and state-space `Sz`/`S2` operators |
| `hamiltonian.py` | 02-HAMILTONIAN §8-11 | Hamiltonian construction, diagonalization |
| `sectors.py` | 03-SYMMETRY_SECTORS | Sz/S² blocking and reconstruction |
| `downfolding.py` | 04-DOWNFOLDING | Selection, SVD, H_eff, spin-coupling fit |
| `io.py` | 06-RUNTIME | CLI parsing, path construction, file I/O |
| `hubbard.py` | — | HubbardModel thin coordinator |
| `main/__main__.py` | — | Main entry point (MPI) |
| `lce/__main__.py` | 05-LCE_AND_EMBEDDING | LCE subcluster subtraction |
| `embed/__main__.py` | 05-LCE_AND_EMBEDDING | Supercell embedding |

## Physics Pipeline

### Main line 1: Single-cluster projection
```
Cluster geometry (01) → Fock basis + Hamiltonian (02) → Sz/S² blocking (03) → Select → SVD → H_eff → Spin fit (04)
```

### Main line 2: LCE to thermodynamic limit
```
All cluster H_eff results → Möbius inversion → Supercell embedding (05)
```

## How to Read the Code

### First pass: single-cluster line
```
main/solver.py
  -> hubbard.py
     -> states.py
     -> hamiltonian.py
     -> sectors.py
     -> downfolding.py
```

- Start with `main/solver.py` to see the runtime order for one cluster.
- Then read `hubbard.py` as the coordinator that owns the model state and calls into the real physics modules.
- Read `states.py` and `hamiltonian.py` together:
  basis generation/sorting, state-space spin operators, and Hubbard matrix elements are one conceptual unit.
- Read `sectors.py` next if you care about `Sz` / `S2` blocking, `S2` transforms, or reconstructed full spectra.
- Read `downfolding.py` after that for the low-energy mapping:
  state selection, `T11`, `H_eff`, and spin-coupling fitting.

### Second pass: thermodynamic line
```
main/workflow.py
  -> lce/workflow.py + lce/core.py
  -> embed/workflow.py + embed/core.py + embed/reporting.py
```

- `lce/core.py` is the connected-subcluster subtraction logic.
- `embed/core.py` is the periodic supercell embedding logic.
- `embed/reporting.py` is presentation/output formatting, not the embedding kernel itself.

### Files to defer on a first pass

- `io.py`:
  important for CLI/path rules, but not where the main physics formulas live.
- `main/workflow.py`, `main/reporting.py`, `mpi.py`:
  orchestration, persistence, and MPI plumbing.

## Where the Physical Core Lives

| Module | Physical role |
|--------|---------------|
| `clusters.py` | Cluster geometry, nearest-neighbor bonds, graph classification |
| `states.py` | Fock-state encoding, electron counting, double occupation, basis ordering, `Sz`/`S2` state-space operators |
| `hamiltonian.py` | Hubbard matrix elements and diagonalisation |
| `sectors.py` | `Sz` sectors, `Sz -> (Sz,S)` transforms, reconstruction |
| `downfolding.py` | Eigenstate selection, `T11`, `H_eff`, spin-operator fit |
| `lce/core.py` | LCE / Möbius subtraction of connected subclusters |
| `embed/core.py` | Embedding LCE couplings into the periodic square supercell |

The most important conceptual boundary is:
- `hubbard.py` coordinates.
- `states.py`, `hamiltonian.py`, `sectors.py`, and `downfolding.py` implement the core single-cluster physics.
- `lce/core.py` and `embed/core.py` are the thermodynamic post-processing layer.

## Running

```bash
mpirun -np N python -m cuprate.main KEY=VALUE ...   # Step 1: diagonalize + fit
python -m cuprate.lce KEY=VALUE ...                  # Step 2: LCE subtraction
python -m cuprate.embed KEY=VALUE ...                # Step 3: embed to supercell
```

## Testing

```bash
python -m pytest tests/ -x -q
```
