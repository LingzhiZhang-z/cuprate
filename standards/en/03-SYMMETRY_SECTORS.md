# 03-SYMMETRY_SECTORS

$S_z$ blocking, fixed-$S_z$ diagonalisation of $S^2$, and optional spectrum reconstruction.

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
sz_states[idx] = generate_states(N, N, twoSz=twoSz)
```

## 2) `fourS2 = 4 S^2` Matrix in Fock Basis (MUST)

MUST:
- The implementation constructs the `fourS2 = 4 * S^2` matrix in the Fock basis within each $S_z$ sector.
- Diagonal elements: each singly occupied site contributes `3`; the pair term uses
  local `twoSz_i = +1` for code `1`, `twoSz_i = -1` for code `2`, and `0` otherwise.
- Off-diagonal elements: nonzero only when $|s_1\rangle$ and $|s_2\rangle$ differ by exactly
  one spin-flip pair $(i, j)$ where the local site codes swap `1 ↔ 2` across those two sites
  and every other site is unchanged, contributing `4` per such pair.

Math:
$$
4\hat{S}^2 = 4\sum_i \hat{s}_i(\hat{s}_i + 1) + 8\sum_{i<j} S_z^i S_z^j + 4\sum_{i<j}\left(\frac{1}{2}S_+^i S_-^j + \frac{1}{2}S_-^i S_+^j\right).
$$

Code form:
```python
fourS2_matrix = calc_fourS2_matrix(basis, N)  # basis = states in one Sz sector
```

## 3) Fixed-$S_z$ Construction of $(S_z, S)$ Sectors (MUST)

MUST:
- The primary implementation builds $(S_z, S)$ sectors from the total-spin ladder operators
  `S_plus` and `S_minus`, not by diagonalising the full `fourS2` matrix in every sector.
- The implementation directly consumes the fixed-`twoSz` bases produced by `generate_states(...)`;
  it does not regenerate or re-validate them inside the sector builder.
- Within each fixed `twoSz` basis, states are first grouped by double occupation `D`.
- For each non-negative `twoS` and each fixed `D`, the highest-weight space is the right null
  space of `S_plus` inside the `(twoSz = twoS, D)` block.
- The remaining columns of the same `(twoS, D)` multiplet are obtained by repeatedly applying
  `S_minus` inside the same `D` block with the standard SU(2) lowering coefficient.
- For each `(twoSz, twoS)` sector, columns from different `D` blocks are concatenated in
  ascending `D`, so the `D=0` pure-spin columns appear first.
- The implementation returns explicit $(twoSz, twoS)$ sector metadata, per-sector transforms,
  explicit double-occupation eigenvalues for each sector column, and the pure-spin dimension
  for each sector.

Code form:
```python
sector_list, transforms, double_occ_eigvals, dimspin = build_S2_sectors(
    twoSz_list, sz_states, N
)
```

Validation:
- Highest-weight columns must satisfy `S_plus @ U_hw = 0`.
- Each transform must satisfy `U.conj().T @ fourS2 @ U = twoS * (twoS + 2) * I`.

## 4) Unitary Transform to $(S_z, S)$ Basis (MUST)

MUST:
- For each $(S_z, S)$ sector, the highest-weight basis together with repeated lowering
  forms a unitary transform $U_{S^2}$ from the $S_z$ Fock basis to the $(S_z, S)$ eigenbasis.
- The Hamiltonian in the $(S_z, S)$ basis is $H_{S^2} = U_{S^2}^\dagger H_{S_z} U_{S^2}$.

Code form:
```python
sector_list, transforms, dimspin = build_S2_sectors(twoSz_list, sz_states, N)
for idx, (_twoSz, _twoS, sz_sector_index) in enumerate(sector_list):
    U = transforms[idx]
    H_S2[idx] = U.conj().T @ H_Sz[sz_sector_index] @ U
```

## 5) Full Spectrum Reconstruction (MUST)

MUST:
- In `block_sz_full` or `block_sz_s2_full` modes, sector eigensystems are assembled into the full spectrum.
- For $S_z > 0$, the negative-$S_z$ sector is obtained by spin-flip:
  $|s'\rangle = $ flip all singly-occupied spins, eigenvectors pick up sign $(-1)^D$ per basis state.
- Global indices track which column of the reconstructed eigenvector matrix belongs to which sector block.

Code form:
```python
reconstruct_from_sz(...)  # for block_sz_full
reconstruct_from_S2(...)  # for block_sz_s2_full
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
