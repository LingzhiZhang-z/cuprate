# Review Context

Reply to the user in Simplified Chinese.

This review targets the eigenstate-selection part of the cuprate workflow.

Goal of the workflow:

- Solve the finite-cluster single-band Hubbard model at half filling.
- Select a `dimspin`-dimensional low-energy manifold that best matches the
  pure-spin subspace.
- Use the selected manifold to build `H_eff`.
- Fit `H_eff` with a spin-operator basis.

Why this review matters:

- The selection algorithm directly affects `T11`, the extracted `H_eff`, and the
  final fit error.
- A low fit residual alone is not enough; a bad selected manifold can still
  produce misleading couplings.
- Conversely, a small `T11` norm may still coexist with a poor fit if the issue
  is the operator basis rather than the selected manifold.

Main modules:

- `src/cuprate/downfolding.py`
- `src/cuprate/hubbard.py`
- `src/cuprate/sectors.py`
- `src/cuprate/states.py`

Relevant standards:

- `standards/en/04-DOWNFOLDING.md`
- `standards/en/03-SYMMETRY_SECTORS.md`
- `standards/en/02-HAMILTONIAN.md`

Implemented selection methods:

- `occ`:
  sort by expected double occupation and select the first `dimspin` states per
  block.
- `energy`:
  select the lowest-energy states per block, then re-order the selected subset
  by expected double occupation.
- `single`:
  start from the occupation-based seed and use greedy swaps to reduce the `T11`
  objective.
- `multi`:
  use repeated greedy improvement with random perturbations of the high-double-
  occupation tail.
- `adiabatic`:
  select states that maximize overlap with the previous parameter point; if the
  previous adiabatic point is absent, seed from the same-parameter baseline run
  with `workflow=None`.

Important structural details:

- Selection may run on one global block or on multiple blocks depending on
  `match_spin_sectors`, `select_mode == "block"`, and the diagonalization mode.
- The pure-spin subspace sits in the first `dimspin` rows because the basis is
  sorted by double occupation.
- `S2` block structure and reconstructed full spectra live in `sectors.py`.
- `hubbard.py` is the coordinator that chooses whether selection operates on
  reconstructed global eigensystems or on block-structured ones.

Questions to force:

1. Does each method optimize the same objective that is later reported?
2. Are block-wise and global selection semantics consistent?
3. Can a method look good under `T11` but still harm fit error?
4. In a bad-fit regime, is the likely cause wrong-state selection, an
   underpowered operator basis, or both?
5. Are there mode-dependent edge cases involving `match_spin_sectors`,
   `block_sz_full`, or `block_sz_s2_full`?

Specific suspicion worth checking:

- Verify carefully whether `multi` still optimizes the correct global `T11`
  objective when selection is split across multiple blocks.
