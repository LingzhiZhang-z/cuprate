# 03-SYMMETRY_SECTORS

$S_z$ blocking, fixed-$S_z$ diagonalisation of $S^2$, and optional spectrum reconstruction.

## 1) $S_z$ Sector Decomposition (MUST)

MUST:
- At half filling on $N$ sites, $S_z$ takes values from $-N/2$ to $N/2$ in integer steps
  (half-integer steps if $N$ is odd).
- All valid `twoSz` sectors are built directly when an all-`Sz` mode is requested.
- Each $S_z$ sector has dimension $\binom{N}{N_\uparrow} \cdot \binom{N}{N_\downarrow}$
  where $N_\uparrow = N/2 + S_z$, $N_\downarrow = N/2 - S_z$.
- Runtime inputs and path names use the canonical integer labels `twoSz = 2 S_z`
  and `twoS = 2 S`.

Code form:
```python
twoSz_values = range(-N, N + 1, 2)
states = generate_states(N, N, twoSz=twoSz)
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
- The implementation returns explicit `S2SectorBlock` records. Each record is one
  `(twoSz, twoS, D)` transform block and contains the fixed-`D` basis states plus
  the transform columns for that block.

Code form:
```python
grouped_states = group_states(generate_states(N, N), N)
_hw, multiplets = build_S2_multiplets(grouped_states, N)
sector_blocks = build_S2_sectors(grouped_states, multiplets)
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
transforms = build_S2_transforms(grouped_states, N, sector_blocks)
U = transforms[(twoSz, twoS)]
H_S2 = U.conj().T @ H_Sz @ U
```

## 5) Full Spectrum Reconstruction (MUST)

MUST:
- Sector eigensystems stay as `Block` objects until the caller explicitly merges them.
- `HubbardModel.merge_by_s2()` merges all `twoS` sectors with the same `twoSz`
  by building block-diagonal sector matrices and transforming them back to the
  fixed-`twoSz` Fock basis.
- `HubbardModel.merge_by_sz()` merges all fixed-`twoSz` Fock-coordinate blocks
  into one full Fock-coordinate block.

Code form:
```python
model.merge_by_s2()  # block_sz_s2_full -> block_sz_full frame
model.merge_by_sz()  # block_sz_full -> full frame
```

Validation:
- Reconstructed eigenvalue count must equal full Hilbert-space dimension $\binom{2N}{N}$.

## 6) Five Diagonalisation Modes (MUST)

MUST:
- The code supports exactly five modes:

| Mode | Blocks before optional merge | Optional reconstruction |
|------|------------------------------|-------------------------|
| `full` | one full Fock block | N/A |
| `fixed_sz` | one fixed-`twoSz` block | No |
| `block_sz_full` | all fixed-`twoSz` blocks | `merge_by_sz()` |
| `fixed_sz_s2` | one fixed-`(twoSz,twoS)` block | No |
| `block_sz_s2_full` | all fixed-`(twoSz,twoS)` blocks | `merge_by_s2()` then `merge_by_sz()` |

- Projection and spin fitting are per current `Block` frame. They are not
  prohibited by `fixed_sz_s2`; the caller is responsible for choosing the block
  frame whose fitted operators answer the intended physics question.

Code form:
```python
model.set_symmetry(mode, twoSz=twoSz, twoS=twoS)
```
