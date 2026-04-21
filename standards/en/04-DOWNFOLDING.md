# 04-DOWNFOLDING

Eigenstate selection, $T_{11}$ projection metric, SVD downfolding to $H_{\text{eff}}$,
and spin-coupling fitting.

## 1) Selection Goal (MUST)

MUST:
- From the full eigensystem, select $d_{\text{spin}}$ eigenstates whose projection onto the
  pure-spin model space is as close to a unitary map as possible.
- In block modes, selection operates per-block with per-block $d_{\text{spin}}$ values.

## 2) $T_{11}$ Projection Metric (MUST)

MUST:
- Let $\Psi_{\text{sel}}$ be the matrix of selected eigenvectors (full-space rows, $d_{\text{spin}}$ columns).
- The spin-projected matrix is $S_{BD} = P_{\text{spin}} \Psi_{\text{sel}}$
  where $P_{\text{spin}}$ is the explicit pure-spin basis matrix.
- SVD: $S_{BD} = U \Sigma V^\dagger$.
- The projection quality metric is:

Math:
$$
T_{11} = U \Sigma U^\dagger, \qquad
\Delta_{T_{11}} = \|T_{11} - I\|_F.
$$

Code form:
```python
S_BD = spin_basis @ eigvecs_selected
U, Sigma, VH = np.linalg.svd(S_BD, full_matrices=False)
T11m1 = U @ np.diag(Sigma) @ U.conj().T - np.eye(dimspin)
norm  = np.linalg.norm(T11m1.flatten())
```

Implementation rule:
- For `full`, `fixed_sz`, and `block_sz_full`, `spin_basis` is the row-selector matrix built from
  `pure_spin_state_indices(states, N)`.
- For `fixed_sz_s2`, `spin_basis` is the explicit basis of the $D = 0$ model space inside the
  selected $(S_z, S)$ block.

Validation:
- Small $\Delta_{T_{11}}$ means the selected manifold closely spans the spin subspace.
- $\Delta_{T_{11}} = 0$ means perfect projection.

## 3) Selection Strategies (MUST)

MUST:
- Five strategies are implemented:

| Method | Key | Description |
|--------|-----|-------------|
| By occupation | `occ` | Select $d_{\text{spin}}$ states with lowest $\langle D \rangle$ per block |
| By energy | `energy` | Select $d_{\text{spin}}$ lowest-energy states per block |
| Greedy swap | `single` | Start from occupation-sorted guess, try all pairwise swaps to minimise $\Delta_{T_{11}}$ |
| Multi-restart | `multi` | Greedy + random perturbation of high-$D$ tail, multiple restarts |
| Adiabatic | `adiabatic` | Maximise overlap $|\langle\psi_{\text{prev}}|\psi_{\text{curr}}\rangle|^2$ with previous parameter point |

- Default (`None`): try occupation first, fall back to greedy if $\Delta_{T_{11}} = \infty$.
- The candidate pool for greedy/multi is `ratio * dimspin` states sorted by $\langle D \rangle$.
- At the first adiabatic point of a sweep, if the previous-point adiabatic result is absent,
  initialise from the same-parameter baseline run (`workflow=None`).

Code form:
```python
selected_indices, best_norm, overlap = select_eigenstates(method, ...)
```

## 4) Double-Occupation Expectation (MUST)

MUST:
- Before selection, compute diagonal expectation values of the double-occupation operator
  in the eigenbasis:

Math:
$$
\langle D \rangle_a = \langle \psi_a | \hat{D} | \psi_a \rangle,
\qquad
\hat{D} = \text{diag}(D(|s_0\rangle), D(|s_1\rangle), \ldots).
$$

Code form:
```python
dom = calc_double_occupation_matrix(states)  # diagonal matrix
double_occ = np.diag(eigvecs.conj().T @ dom @ eigvecs)
```

## 5) Effective Hamiltonian $H_{\text{eff}}$ (MUST)

MUST:
- Given the SVD from Rule 2 and the selected eigenvalues $\Lambda = \text{diag}(E_{\text{sel}})$:

Math:
$$
H_{\text{eff}} = U V^\dagger \Lambda (U V^\dagger)^\dagger = U V^\dagger \Lambda V U^\dagger.
$$

Code form:
```python
Lambda = np.diag(eigvals[selected_indices])
Heff   = U @ VH @ Lambda @ VH.conj().T @ U.conj().T
```

Validation:
- $H_{\text{eff}}$ must be Hermitian within tolerance.
- The spectrum of $H_{\text{eff}}$ must equal the selected eigenvalues.

## 6) Spin-Operator Basis (MUST)

MUST:
- The spin-operator basis for fitting consists of:
  1. Identity $I$ (constant energy offset $E_0$).
  2. Two-site Heisenberg: $\mathbf{S}_i \cdot \mathbf{S}_j = S_z^i S_z^j + \frac{1}{2}(S_+^i S_-^j + S_-^i S_+^j)$.
  3. Four-site biquadratic: $(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)$.
  4. Six-site: $(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)(\mathbf{S}_m \cdot \mathbf{S}_n)$.
- All operators are constructed on `states_spin`, the explicit list of pure-spin basis states.
- $S_+$, $S_-$ are defined on the site encoding:
  $S_+(|-1\rangle) = |+1\rangle$, $S_-(|+1\rangle) = |-1\rangle$, zero otherwise
  (sites with $s = 0$ or $s = 2$ are annihilated).
- Multi-site operators (4-site, 6-site, ...) must be enumerated in the **canonical
  pairing order** defined in `08-OPERATOR_OUTPUT.md` §3. This ensures the fit coefficient
  vector has a geometry-determined meaning: the $k$-th coefficient for a given site group
  always corresponds to the same physical pairing pattern, regardless of site labelling.

Code form:
```python
spin_matrix_J_ij(states, [i, j])       # S_i · S_j
spin_matrix_JJ_ij(states, [i,j,k,l])   # (S_i·S_j)(S_k·S_l)
spin_matrix_JJJ_ij(states, [i,j,k,l,m,n])  # triple product
```

## 7) Least-Squares Fit (MUST)

MUST:
- Flatten $H_{\text{eff}}$ into vector $b$ and each spin operator into column vectors of matrix $A$.
- Solve $A x = b$ by normal equations: $x = (A^T A)^{-1} A^T b$, with `lstsq` fallback.
- Report fit quality: relative error, residual norm, and $R^2$.

Math:
$$
\min_x \|Ax - b\|, \qquad
R^2 = 1 - \frac{\|Ax - b\|^2}{\|b - \bar{b}\|^2}.
$$

Code form:
```python
coeffs, error = model.calc_spin_coeff(bonds)
# error = (relative_error, residual_norm, R_squared)
```

## 8) Interpretation Rule (MUST)

MUST:
- A small fit residual does not by itself validate the spin mapping.
- $\Delta_{T_{11}}$ and the fit diagnostics must be interpreted together.
- If $\Delta_{T_{11}}$ is large, $H_{\text{eff}}$ may not faithfully represent the low-energy physics
  regardless of how well the spin operators fit $H_{\text{eff}}$.
