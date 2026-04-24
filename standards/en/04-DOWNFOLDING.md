# 04-DOWNFOLDING

Eigenstate selection, the $T_{11}$ projection metric, SVD downfolding to
$H_{\text{eff}}$, and spin-coupling fitting.

## 1) Selection Goal (MUST)

MUST:
- Selection operates on one solved `Block` at a time.
- For each block, select exactly `block.spin_dim` eigenstates.
- `block.spin_dim` is the dimension of the `D = 0` spin subspace in the
  block's current coordinate frame.
- `HubbardModel.project(method=...)` applies the same per-block selection
  method to every current block and stores `selected_indices`, `selection_info`,
  `heff`, and `t11m1_norms` on the `HubbardModel`.

Code form:
```python
selected, info = block.selected(method=method, return_info=True, **kwargs)
heff, t11m1_norm = block.downfold(selected)
```

## 2) $T_{11}$ Projection Metric (MUST)

MUST:
- Let `selected` be eigenvector column indices in the block coordinate frame.
- Let `spin_sector_columns = block.spin_sector_columns()` be the `D = 0`
  columns in that same block frame.
- The spin-projected matrix is:

Code form:
```python
S_BD = block.eigvecs[np.ix_(spin_sector_columns, selected)]
U, sigma, VH = np.linalg.svd(S_BD, full_matrices=False)
T11m1 = U @ np.diag(sigma) @ U.conj().T - np.eye(S_BD.shape[0])
norm = np.linalg.norm(T11m1.flatten())
```

Validation:
- Small `norm` means the selected manifold closely spans the spin subspace.
- `norm == 0` means perfect projection within numerical tolerance.

## 3) Selection Strategies (MUST)

MUST:
- The implemented method keys are:

| Method | Key | Rule |
|--------|-----|------|
| By occupation | `occ` | Sort by lowest `<D>`, then energy, then column index |
| By energy | `energy` | Sort by lowest energy, then `<D>`, then column index |
| Greedy swap | `greedy` | Start from the occupation-sorted pool and perform one deterministic swap pass to reduce `block.t11_norm` |
| Greedy multi | `greedy_multi` | Start from greedy, randomize the high-`D` tail over multiple trials, and rerun greedy |
| Adiabatic | `adiabatic` | Select current states with largest overlap against previously selected eigenvectors |

- `HubbardModel.project()` defaults to `method="occ"`.
- Unknown methods raise an error. There is no `None` fallback from occupation to
  greedy.
- `greedy` and `greedy_multi` form their candidate pool from the first
  `ratio * block.spin_dim` eigenstates sorted by occupation.
- `greedy_multi` may write one JSONL record per block/trial through
  `selection_info_path` or `info_callback`.
- `adiabatic` requires explicit `eigvecs_previous` and `selected_previous`
  arrays for the same block frame. Loading those arrays from a previous
  parameter point is a workchain responsibility, not a `Block` responsibility.
- If an adiabatic seed is absent, the workchain must fail directly unless the
  user requested another selection method. It must not silently fall back to a
  baseline run.

Code form:
```python
selected = block.selected_occ()
selected = block.selected_energy()
selected = block.selected_greedy(ratio=5)
selected = block.selected_greedy_multi(n_trials=40, max_failures=4)
selected = block.selected_adiabatic(eigvecs_previous, selected_previous)
```

## 4) Double-Occupation Expectation (MUST)

MUST:
- Before occupation-based selection, compute diagonal expectation values of the
  double-occupation operator in the Fock-coordinate eigenvectors:

Code form:
```python
dom = calc_double_occupation_matrix(block.basis_states, block.N)
eigvecs = block.eigvecs_fock
double_occ = np.real(np.diag(eigvecs.conj().T @ dom @ eigvecs))
```

## 5) Effective Hamiltonian $H_{\text{eff}}$ (MUST)

MUST:
- Given the SVD from Rule 2 and the selected eigenvalues
  `Lambda = diag(block.eigvals[selected])`, compute:

Code form:
```python
Lambda = np.diag(block.eigvals[selected])
Heff = U @ VH @ Lambda @ VH.conj().T @ U.conj().T
```

Validation:
- `H_eff` must be Hermitian within tolerance.
- The spectrum of `H_eff` must equal the selected eigenvalues.

## 6) Spin-Operator Basis (MUST)

MUST:
- The spin-operator basis for fitting consists of:
  1. Identity `I` (constant energy offset).
  2. Two-site Heisenberg: `S_i . S_j`.
  3. Four-site products: `(S_i . S_j)(S_k . S_l)`.
  4. Six-site products: `(S_i . S_j)(S_k . S_l)(S_m . S_n)`.
- Operators are constructed on the explicit `D = 0` Fock rows
  `block.spin_fock_rows()`.
- In an `S2` block, operators are transformed into the current block's spin
  sector using `block.spin_sector_columns()` and `block.basis_transform`.
- Multi-site operator ordering for output follows `08-OPERATOR_OUTPUT.md`.

Code form:
```python
spin_fock_rows = block.spin_fock_rows()
spin_sector_columns = block.spin_sector_columns()
operators = block._spin_operators(bonds)
```

## 7) Least-Squares Fit (MUST)

MUST:
- `HubbardModel.fit()` flattens every current block's `H_eff` into the target
  vector `b`.
- It stacks the corresponding per-block spin-operator columns into `A`.
- It solves `A x = b` with `np.linalg.lstsq`.
- It reports relative error, residual norm, and $R^2$.
- If no `bond_groups` are supplied, it fits all current two-site, four-site, and
  six-site groups generated from the model's `Cluster`.

Code form:
```python
A = np.vstack([block._spin_operators(bonds) for block in model.blocks])
b = np.concatenate([heff.flatten() for heff in model.heff])
x = np.linalg.lstsq(A, b, rcond=None)[0]
```

## 8) Persistence Boundary (MUST)

MUST:
- `Block.save()` / `Block.load()` persist the solved eigensystem cache only.
- Selected indices, selection diagnostics, `H_eff`, and fit metrics are derived
  outputs and must be written by the workchain/result-output layer.
- The canonical machine-readable shape for derived projection and fit results
  is defined in `08-OPERATOR_OUTPUT.md`.

## 9) Interpretation Rule (MUST)

MUST:
- A small fit residual does not by itself validate the spin mapping.
- The `T11` metric and fit diagnostics must be interpreted together.
- If `|T11-I|` is large, `H_eff` may not faithfully represent the low-energy
  physics regardless of the fit residual.
