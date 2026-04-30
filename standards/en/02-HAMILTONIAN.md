# 02-HAMILTONIAN

Fock-state encoding, basis construction, and the single-band Hubbard Hamiltonian.

## 1) Local Site Alphabet (MUST)

MUST:
- Each site is encoded by the two bits `(n_{i\uparrow}, n_{i\downarrow})`.
- The derived local site code is fixed:
  - `0` = empty
  - `1` = spin-up (single occupancy)
  - `2` = spin-down (single occupancy)
  - `3` = doubly occupied (up + down)

Code form:
```
code = (state >> (2 * i)) & 3   # 0=empty, 1=up, 2=down, 3=double
```

Validation:
- Any other local symbol is invalid.

## 2) Global Many-Body State (MUST)

MUST:
- A many-body basis state on $N$ sites is one Python `int`.
- Bits are interleaved in the canonical fermionic order
  $c^\dagger_{0\uparrow} c^\dagger_{0\downarrow} c^\dagger_{1\uparrow} c^\dagger_{1\downarrow} \cdots
  c^\dagger_{(N-1)\uparrow} c^\dagger_{(N-1)\downarrow}$.

Code form:
```
state: int
bit 2i   = n_{i\uparrow}
bit 2i+1 = n_{i\downarrow}
```

Validation:
- No occupied bit may lie at or above position `2 * N`.

## 3) Electron Count and Magnetisation (MUST)

MUST:
- Electron number per site: $n_i = n_{i\uparrow} + n_{i\downarrow}$.
- Total electron number: $N_e = \sum_i (n_{i\uparrow} + n_{i\downarrow})$.
- Total magnetisation (twice $S_z$): `twoSz = N_up - N_down`.

Math:
$$
N_e = \sum_i (n_{i\uparrow} + n_{i\downarrow}), \qquad
S_z = \frac{1}{2}\left(\sum_i n_{i\uparrow} - \sum_i n_{i\downarrow}\right).
$$

Code form:
```python
count_electrons(state) = state.bit_count()
calc_twoSz(state, N) = (state & up_mask(N)).bit_count() - (state & down_mask(N)).bit_count()
```

Validation:
- At half filling: $N_e = N$.

## 4) Double Occupation (MUST)

MUST:
- Double occupation count is the number of sites with `n_{i\uparrow} = n_{i\downarrow} = 1`.

Math:
$$
D(|s\rangle) = \sum_i \mathbf{1}_{n_{i\uparrow}=1}\mathbf{1}_{n_{i\downarrow}=1}.
$$

Code form:
```python
count_double_occ(state, N) = (state & ((state >> 1) & up_mask(N))).bit_count()
```

## 5) Fermionic Sign Convention (MUST)

MUST:
- The fermionic sign for annihilating or creating at orbital bit index `bit_idx`
  is $(-1)^{n_{\text{below}}}$ where $n_{\text{below}}$ is the number of occupied
  orbitals at smaller bit indices.
- The spin-down sign on a doubly occupied site is not a special case in code:
  it is produced automatically because bit `2i` (up) lies below bit `2i+1` (down).

Math:
$$
\text{sign}(\text{state}, \text{bit\_idx}) =
(-1)^{\#\{\text{occupied orbitals with index} < \text{bit\_idx}\}}.
$$

Code form:
```python
sign_below(state, bit_idx) = 1 if (state & ((1 << bit_idx) - 1)).bit_count() % 2 == 0 else -1
# apply_hop uses sign_below(state, src_bit), flips src, then sign_below(new_state, dst_bit)
```

Validation:
- Hopping matrix elements must satisfy $H_{ij} = H_{ji}^*$ (Hermiticity).

## 6) Basis Sorting Order (MUST)

MUST:
- States are sorted canonically by the tuple
  `(twoSz(state), double_occ(state), state)` in ascending order.
- This canonical order is used for the full basis and for every restricted basis built by `generate_states(...)`.

Code form:
```python
states = sort_states(generate_states(N, N_e), N)
```

Validation:
- The sorted list is non-decreasing in `twoSz`, then in $D$, then in the integer value of `state`.
- `is_pure_spin_state(state, N)` returns `True` iff every site is singly occupied.

## 7) Half-Filled Pure-Spin Subspace (MUST)

MUST:
- The pure-spin subspace consists of states where every site is singly occupied
  (`site_code(state, i)` is `1` or `2`, i.e. $D = 0$).
- At half filling ($N_e = N$), this subspace has dimension $2^N$.
- In the canonical Fock basis, these states are identified explicitly by
  `pure_spin_state_indices(states, N)` rather than by a contiguous row prefix.

Math:
$$
\mathcal{H}_{\text{spin}}(C) =
\{|s\rangle : (n_{i\uparrow}, n_{i\downarrow}) \in \{(1,0),(0,1)\},\; \forall i\},
\qquad
\dim \mathcal{H}_{\text{spin}} = 2^N.
$$

Code form:
```python
pure_spin_idx = pure_spin_state_indices(states, N)
dimspin = len(pure_spin_idx)
states_spin = [states[i] for i in pure_spin_idx]
```

Validation:
- `dimspin == 2**N` at half filling.

## 8) Model Definition (MUST)

MUST:
- The code solves the single-band Hubbard model at half filling ($N_e = N$) on finite clusters.

Math:
$$
H = -\sum_{\langle ij\rangle, \sigma} t_{ij}
\left(c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.}\right)
+ U \sum_i n_{i\uparrow} n_{i\downarrow}.
$$

Index:
- $t_{ij}$: hopping amplitude between nearest-neighbor sites $i$ and $j$.
- $U$: on-site Coulomb repulsion.
- $\sigma \in \{\uparrow, \downarrow\}$.

## 9) Hopping Assignment (MUST)

MUST:
- The production workflow assigns hopping only on nearest-neighbor bonds.
- The bond list comes from `Cluster.bonds`.

Code form:
```python
model = HubbardModel(cluster, U, t)
# model.bonds is initialized from cluster.bonds
```

## 10) Hamiltonian Matrix Elements (MUST)

MUST:
- The full matrix element is $\langle s_1 | H | s_2 \rangle = H_t + H_U$.
- **On-site U term** (diagonal):
  $H_U = U \cdot D(|s\rangle)$ when $|s_1\rangle = |s_2\rangle$, else $0$.
- **Hopping term**: for each stored bond $(i, j)$ and each spin channel $\sigma$,
  apply $c^\dagger_{i\sigma} c_{j\sigma}$ with amplitude $t_{ij}$ and
  $c^\dagger_{j\sigma} c_{i\sigma}$ with amplitude $t_{ij}^*$ to $|s_2\rangle$.
  If the resulting state equals $|s_1\rangle$, accumulate the corresponding signed contribution.

Math:
$$
\langle s_1 | H_t | s_2 \rangle =
-\sum_{(i,j)\in\text{bonds}} \sum_{\sigma}
\left[
t_{ij}\langle s_1|c^\dagger_{i\sigma} c_{j\sigma}|s_2\rangle +
t_{ij}^*\langle s_1|c^\dagger_{j\sigma} c_{i\sigma}|s_2\rangle
\right].
$$

Code form:
```python
H_t = build_hamiltonian_t(states, bonds, hoppings)
H_U = build_hamiltonian_U(states, N, U)
H = H_t + H_U
```

Validation:
- $H$ must be Hermitian.
- Diagonal elements for pure-spin states ($D = 0$) must be zero (no double occupancy → no $U$ term, no hopping to self).

## 11) Diagonalisation (MUST)

MUST:
- The Hamiltonian is diagonalised with `scipy.linalg.eigh` (assumes Hermitian).
- `EIGH=lowmem` uses LAPACK driver `ev`.
- `EIGH=fast` uses LAPACK driver `evd`.
- The matrix passed to `eigh` is Fortran-contiguous and may be overwritten.

Code form:
```python
eigvals, eigvecs = scipy.linalg.eigh(
    H,
    driver="ev",
    overwrite_a=True,
    check_finite=False,
)
```

Validation:
- Eigenvalues must be real.
- Eigenvectors must be orthonormal.
- The ground-state energy must decrease as $U/t$ decreases (metallic limit).
