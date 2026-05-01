# 03-SYMMETRY_SECTORS

$S_z$ blocking, fixed-$S_z$ diagonalisation of $S^2$, and optional spectrum reconstruction.

## 1) $S_z$ Sector Decomposition (MUST)

MUST:
- At half filling on $N$ sites, $S_z$ takes values from $-N/2$ to $N/2$ in integer steps
  (half-integer steps if $N$ is odd).
- All-`twoSz` modes use `SCOPE` to choose the sector range:
  `SCOPE=nonnegative` builds valid `twoSz >= 0` sectors, while `SCOPE=pm`
  builds all valid positive and negative `twoSz` sectors.
- Each $S_z$ sector has dimension $\binom{N}{N_\uparrow} \cdot \binom{N}{N_\downarrow}$
  where $N_\uparrow = N/2 + S_z$, $N_\downarrow = N/2 - S_z$.
- Runtime inputs and path names use the canonical integer labels `twoSz = 2 S_z`
  and `twoS = 2 S`.

Code form:
```python
twoSz_values = [twoSz for twoSz in range(-N, N + 1, 2) if scope == "pm" or twoSz >= 0]
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
- Runtime `Block` objects store the transform as explicit fixed-`D` pieces
  rather than as one dense long-lived `basis_transform`.
- `build_hamiltonians()` consumes all fixed-`D` transform pieces to assemble the
  complete symmetry-sector Hamiltonian, then keeps only the `D=0` transform
  matrix plus `D` column metadata needed by projection and fitting.

Code form:
```python
blocks = blocks_by_sector[(twoSz, twoS)]  # one S2SectorBlock per D
H_S2[D1, D2] = U_D1.conj().T @ H_Sz[D1, D2] @ U_D2
U_spin = blocks_by_D[0].transform
```

## 5) Half-Filled Eta-Pseudospin Refinement (MUST)

MUST:
- `MODE=SzS2eta2` constructs `eta=0` blocks from the rectangular `eta_plus`
  operator and validates them with the fixed-sector `eta2` operator.
- The rectangular `eta_plus` matrix maps
  `H(Ne=N, twoSz) -> H(Ne=N+2, twoSz)`. The `twoSz` label is unchanged because
  one spin-up and one spin-down electron are added together.
- The square-lattice sublattice signs are `epsilon_i = (-1) ** (x_i + y_i)`.
  The eta diagnostic is valid only when every hopping bond satisfies
  `epsilon_i * epsilon_j == -1`.
- This sign check is a geometry check. The production eta diagnostic assumes
  the current real nearest-neighbor hopping convention; complex hopping phases
  must be treated as unsupported unless the commutator validation below passes.
- In the fixed half-filled basis, `eta2` may be built directly: each basis
  state receives a diagonal contribution equal to the number of empty sites,
  and each empty/doublon site pair contributes a doublon move with coefficient
  `epsilon_i * epsilon_j`.
- The direct fixed-sector `eta2` construction is the validation construction.
  It must be tested by Hermiticity, eigenvalue quantisation, commutator,
  pure-spin, and projected zero-block checks.
- `MODE=SzS2eta2` is a refinement of `MODE=SzS2`: first build fixed
  `(twoSz, twoS, D)` sector blocks, apply `eta_plus` to the S2 columns, then
  keep its kernel.
- In the first runtime implementation, `MODE=SzS2eta2` keeps only `eta=0`
  blocks for calculation. It does not classify or output nonzero eta sectors.
- `twoSz` and `twoS` remain optional selectors exactly as in `MODE=SzS2`.
  With neither selector set, the mode builds all default-scope `(twoSz,twoS)`
  blocks and refines each one to `eta=0`.
- A refined block label appends `eta_<value>` after the `(twoSz,twoS)` label,
  for example `twoSz_0_twoS_0_eta_0`.
- Production block construction consumes `build_S2eta0_sectors(...)` directly
  as fixed-`D` transform pieces. Dense `build_S2eta0_transforms(...)` output is
  retained only as a reference/helper representation.

Math:
$$
\eta^+ = \sum_i \epsilon_i c^\dagger_{i\uparrow}c^\dagger_{i\downarrow},
\qquad
\eta^- = (\eta^+)^\dagger,
\qquad
\eta^z = \tfrac{1}{2}(N_e - N).
$$
At half filling, $\eta^z = 0$, and the fixed-sector operator is:
$$
\eta^2 = \eta^-\eta^+.
$$
For a half-filled Fock state $|s\rangle$:
$$
\eta^2|s\rangle =
|E(s)|\,|s\rangle +
\sum_{i\in E(s)}\sum_{j\in D(s)}
\epsilon_i\epsilon_j
|s_{i:0\to3,\ j:3\to0}\rangle.
$$

Code form:
```python
grouped_states = group_states(generate_states(N, N), N)
_hw, multiplets = build_S2_multiplets(grouped_states, N)
sector_blocks = build_S2_sectors(grouped_states, multiplets)
eta0_sector_blocks = build_S2eta0_sectors(N, sector_blocks, cluster)
blocks = blocks_by_sector[(twoSz, twoS, 0)]  # one eta=0 S2SectorBlock per D
H_S2eta0[D1, D2] = U_D1.conj().T @ H_Sz[D1, D2] @ U_D2
```

Validation:
- `eta2` must be Hermitian and positive semidefinite.
- `eta2` eigenvalues must match `eta * (eta + 1)` within tolerance.
- For valid bipartite nearest-neighbor Hubbard hopping,
  `H @ eta2 - eta2 @ H` must vanish within tolerance in both the full
  half-filled basis and each fixed-`twoSz` basis.
- Pure-spin `D = 0` basis rows and columns must be annihilated by `eta2`.
- After projection with any valid `S2` transform, eta cross blocks between
  different `twoS` sectors must vanish within tolerance.
- In `MODE=SzS2eta2`, each produced block must have `eta == 0` and satisfy
  `U.conj().T @ eta2 @ U == 0` within tolerance.

## 6) Merge To Fixed Sz Blocks (MUST)

MUST:
- Sector eigensystems stay as `Block` objects until the caller explicitly merges them.
- `HubbardModel.merge_to_sz(merge_basis)` merges all selected `twoS` sectors
  at one fixed `twoSz` into one fixed-`twoSz` block.
- `merge_basis="fock"` stores eigenvectors in fixed-`twoSz` Fock row
  coordinates and sets `basis_transform=None`.
- `merge_basis="block"` keeps block-coordinate eigenvectors and stores a
  column-concatenated `basis_transform` from the merged block coordinates to
  the fixed-`twoSz` Fock rows.
- The production runtime supports `MERGE=Sz` only for `MODE=SzS2` and
  `MODE=SzS2eta2` runs with fixed `twoSz` and without fixed `twoS`.
- The production runtime does not provide Sz-to-full reconstruction.

Code form:
```python
model.merge_to_sz("fock")   # eigvecs are in fixed-twoSz Fock rows
model.merge_to_sz("block")  # eigvecs remain in merged block coordinates
```

Validation:
- Merged `SzS2` block labels omit `twoS`.
- Merged `SzS2eta2` block labels omit `twoS` and keep `eta=0`.
- `MERGE=Sz` with fixed `twoS`, `MODE=Sz`, or `MODE=full` must fail.

## 7) Diagonalisation Modes (MUST)

MUST:
- The code supports exactly four `MODE` values plus optional block selectors:

| CLI input | Blocks before optional merge | Optional merge |
|-----------|------------------------------|-------------------------|
| `MODE=full` | one full Fock block | N/A |
| `MODE=Sz twoSz=<value>` | one fixed-`twoSz` block | No |
| `MODE=Sz` | fixed-`twoSz` blocks with `twoSz >= 0` | No |
| `MODE=Sz SCOPE=pm` | all fixed-`twoSz` blocks | No |
| `MODE=SzS2 twoSz=<value> twoS=<value>` | one fixed-`(twoSz,twoS)` block | No |
| `MODE=SzS2 twoSz=<value>` | all `twoS` blocks at one fixed `twoSz` | optional `MERGE=Sz` |
| `MODE=SzS2` | fixed-`(twoSz,twoS)` blocks with `twoSz >= 0` | No full reconstruction |
| `MODE=SzS2 SCOPE=pm` | all fixed-`(twoSz,twoS)` blocks | No full reconstruction |
| `MODE=SzS2eta2 twoSz=<value> twoS=<value>` | one fixed-`(twoSz,twoS,eta=0)` block | No |
| `MODE=SzS2eta2 twoSz=<value>` | all `twoS` blocks at one fixed `twoSz`, each refined to `eta=0` | optional `MERGE=Sz` |
| `MODE=SzS2eta2` | default-scope fixed-`(twoSz,twoS,eta=0)` blocks | No full reconstruction |
| `MODE=SzS2eta2 SCOPE=pm` | all fixed-`(twoSz,twoS,eta=0)` blocks for all positive and negative `twoSz` | No full reconstruction until nonzero eta sectors are supported |

- Projection and spin fitting are per current `Block` frame. They are not
  prohibited by S2-resolved modes; the caller is responsible for choosing the block
  frame whose fitted operators answer the intended physics question.
- Fixed `twoSz` and `twoS` values are separate optional selectors, not part of
  `MODE`.
- `SCOPE` applies only when `twoSz` is not explicitly specified.

Code form:
```python
model.set_symmetry("SzS2", twoSz=0, twoS=0, scope="nonnegative")
model.set_symmetry("SzS2eta2", scope="nonnegative")
```
