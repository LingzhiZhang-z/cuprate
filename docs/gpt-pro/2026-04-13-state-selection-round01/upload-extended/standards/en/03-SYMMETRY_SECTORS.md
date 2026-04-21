# 03-SYMMETRY_SECTORS

$S_z$ blocking, $S^2$ basis transform, spectrum reconstruction, and the five diagonalisation modes.

## 1) $S_z$ Sector Decomposition (MUST)

MUST:
- At half filling on $N$ sites, $S_z$ takes values from $-N/2$ to $N/2$ in integer steps
  (half-integer steps if $N$ is odd).
- Only non-negative $S_z$ values are computed; negative sectors are obtained by spin-flip symmetry.
- Each $S_z$ sector has dimension $\binom{N}{N_\uparrow} \cdot \binom{N}{N_\downarrow}$
  where $N_\uparrow = N/2 + S_z$, $N_\downarrow = N/2 - S_z$.
- Runtime inputs and path names use the canonical integer labels `twoSz = 2 S_z`
  and `twoS = 2 S`.

Code form:
```python
twoSz_list = nonneg_twoSz_values(N)  # [0, 2, ..., N] or [1, 3, ..., N]
sz_states[idx] = generate_states_twoSz(N, N, twoSz, s_values=[0,1,-1,2])
```

## 2) $S^2$ Matrix in Fock Basis (MUST)

MUST:
- The $\hat{S}^2$ matrix is computed in the Fock basis within each $S_z$ sector.
- Diagonal elements: $\langle s | \hat{S}^2 | s \rangle = \sum_i s_i(s_i + 1) + 2\sum_{i<j} m_i m_j$
  where $m_i = s_i/2$ for singly-occupied sites, $0$ otherwise.
- Off-diagonal elements: nonzero only when $|s_1\rangle$ and $|s_2\rangle$ differ by exactly
  one spin-flip pair $(i, j)$ with $s_1[i] = 1, s_1[j] = -1, s_2[i] = -1, s_2[j] = 1$ (or vice versa),
  contributing $+1/2$ per such pair (times 2 for $S_+ S_- + S_- S_+$).

Math:
$$
\hat{S}^2 = \sum_i \hat{s}_i(\hat{s}_i + 1) + 2\sum_{i<j}\left(S_z^i S_z^j + \frac{1}{2}S_+^i S_-^j + \frac{1}{2}S_-^i S_+^j\right).
$$

Code form:
```python
S2_matrix = compute_S2_matrix(basis)  # basis = states in one Sz sector
```

## 3) Block Diagonalisation of $S^2$ (MUST)

MUST:
- Within each $S_z$ sector, $\hat{S}^2$ is further block-diagonalised by double-occupation count $D$.
  States are grouped by $D$, sorted by magnetisation within each group, then each block is
  diagonalised independently.
- Block results are assembled via `scipy.linalg.block_diag`.
- Eigenvalues are classified into $S$ values via $S = (-1 + \sqrt{1 + 4\lambda})/2$,
  rounded to the nearest half-integer.

Code form:
```python
eigvals, eigvecs = solve_S2_blocks(basis)
s_list, index_groups, bad = sort_S2(eigvals)
```

Validation:
- `bad` must be `None` (all eigenvalues must round to valid $S(S+1)$).
- Each $(S_z, S)$ sector dimension must equal $d(S) = \binom{N}{N/2-S} - \binom{N}{N/2-S-1}$.

## 4) Unitary Transform to $(S_z, S)$ Basis (MUST)

MUST:
- For each $(S_z, S)$ sector, the eigenvectors of $\hat{S}^2$ form a unitary transform $U_{S^2}$
  from the $S_z$ Fock basis to the $(S_z, S)$ eigenbasis.
- The Hamiltonian in the $(S_z, S)$ basis is $H_{S^2} = U_{S^2}^\dagger H_{S_z} U_{S^2}$.

Code form:
```python
model.construct_transform_matrix(N)
# Produces sectors [(twoSz, twoS, idx_sz), ...] and per-sector transforms U
# H_S2[idx] = U† @ H_Sz[idx_sz] @ U
```

## 5) Full Spectrum Reconstruction (MUST)

MUST:
- In `block_sz_full` or `block_sz_s2_full` modes, sector eigensystems are assembled into the full spectrum.
- For $S_z > 0$, the negative-$S_z$ sector is obtained by spin-flip:
  $|s'\rangle = $ flip all singly-occupied spins, eigenvectors pick up sign $(-1)^D$ per basis state.
- Global indices track which column of the reconstructed eigenvector matrix belongs to which sector block.

Code form:
```python
_reconstruct_from_sz()    # for block_sz_full
_reconstruct_from_S2()    # for block_sz_s2_full
```

Validation:
- Reconstructed eigenvalue count must equal full Hilbert-space dimension $\binom{2N}{N}$.

## 6) Five Diagonalisation Modes (MUST)

MUST:
- The code supports exactly five modes:

| Mode | Blocks | Reconstruct | Spin couplings |
|------|--------|-------------|----------------|
| `full` | None | N/A | Yes |
| `fixed_sz` | Single $S_z$ | No | Yes |
| `block_sz_full` | All $S_z$ | Yes | Yes |
| `fixed_sz_s2` | Single $(S_z, S)$ | No | No (projection analysis only) |
| `block_sz_s2_full` | All $(S_z, S)$ | Yes | Yes |

- `fixed_sz_s2` does NOT produce spin couplings because a single total-spin sector
  does not determine unique SU(2)-invariant couplings. It reports projection diagnostics only.

Code form:
```python
ModeSpec = resolve_mode_spec(params)
# ModeSpec.result_kind == "projection_analysis" only for fixed_sz_s2
```
