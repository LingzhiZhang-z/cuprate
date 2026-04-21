# 02-HAMILTONIAN

Fock-state encoding, basis construction, and the single-band Hubbard Hamiltonian.

## 1) Local Site Alphabet (MUST)

MUST:
- Each site is encoded by one value from `{0, 1, -1, 2}`.
- The meanings are fixed:
  - `0` = empty
  - `1` = spin-up (single occupancy)
  - `-1` = spin-down (single occupancy)
  - `2` = doubly occupied (up + down)

Code form:
```
site_state in {0, 1, -1, 2}
```

Validation:
- Any other local symbol is invalid.

## 2) Global Many-Body State (MUST)

MUST:
- A many-body basis state on $N$ sites is an ordered length-$N$ tuple.
- Site index order is the fermionic order used for sign computation.

Code form:
```
state = (s_0, s_1, ..., s_{N-1})
```

Validation:
- `len(state) == N` for the cluster in use.

## 3) Electron Count and Magnetisation (MUST)

MUST:
- Electron number per site: $n_i = |s_i|$ when $s_i \in \{0, \pm 1, 2\}$, i.e. $n_i \in \{0, 1, 1, 2\}$.
- Total electron number: $N_e = \sum_i n_i$.
- Total magnetisation (twice $S_z$): $M = \sum_i s_i$ restricted to singly-occupied sites.

Math:
$$
N_e = \sum_{i} |s_i|, \qquad
S_z = \frac{1}{2}\bigl(\#\{i : s_i = 1\} - \#\{i : s_i = -1\}\bigr).
$$

Code form:
```python
sum_elec(state, n) = sum(abs(state[i]) for i in range(n))
total_mag(state)   = sum(s for s in state if abs(s) == 1)   # = 2 * Sz
```

Validation:
- At half filling: $N_e = N$.

## 4) Double Occupation (MUST)

MUST:
- Double occupation count is the number of sites with local value `2`.

Math:
$$
D(|s\rangle) = \sum_i \mathbf{1}_{s_i = 2}.
$$

Code form:
```python
calc_double_occupation(state) = state.count(2)
```

## 5) Fermionic Sign Convention (MUST)

MUST:
- The fermionic sign for annihilating or creating at site $i$ is $(-1)^{n_{\text{left}}}$ where $n_{\text{left}} = \sum_{k < i} n_k$.
- For spin-down annihilation on a doubly-occupied site, an additional factor of $-1$ applies (the down electron sits "behind" the up electron in the ordering convention).

Math:
$$
\text{sign}(|s\rangle, i) = (-1)^{\sum_{k<i} |s_k|}.
$$

Code form:
```python
sign_fermi(n)        = 1 if n % 2 == 0 else -1
sign_state(state, i) = sign_fermi(sum_elec(state, i))
```

Validation:
- Hopping matrix elements must satisfy $H_{ij} = H_{ji}^*$ (Hermiticity).

## 6) Basis Sorting Order (MUST)

MUST:
- States are sorted first by double-occupation count $D$ (ascending), then within each $D$-block by $|M|$ (ascending), then by $M$ (descending).
- This places the pure-spin states ($D = 0$) as the first $2^N$ rows.

Code form:
```python
states = sort_by_double_occupation(generate_states(N, N_e))
# sort_by_double_occupation calls sort_by_magnetisation within each D-block
```

Validation:
- `states[:2**N]` are exactly all states with $D = 0$ (every site is $\pm 1$).
- `is_half_filled(state)` returns `True` iff all sites are singly occupied.

## 7) Half-Filled Pure-Spin Subspace (MUST)

MUST:
- The pure-spin subspace consists of states where every site is singly occupied ($s_i \in \{-1, 1\}$, i.e. $D = 0$).
- At half filling ($N_e = N$), this subspace has dimension $2^N$.
- After basis sorting (Rule 6), these states occupy rows $0$ through $2^N - 1$ of any matrix built on the full Fock basis.

Math:
$$
\mathcal{H}_{\text{spin}}(C) = \{|s\rangle : s_i \in \{-1, 1\},\; \forall i\},
\qquad
\dim \mathcal{H}_{\text{spin}} = 2^N.
$$

Code form:
```python
dimspin = 2 ** N
dimspin = len([s for s in states if is_half_filled(s)])
```

Validation:
- Both expressions must agree.

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
- The bond list comes from `generate_bonds(cluster)[0]`.

Code form:
```python
nn_bonds = generate_bonds(cluster)[0]
model.add_hopping_bonds(nn_bonds, params.t)
```

## 10) Hamiltonian Matrix Elements (MUST)

MUST:
- The full matrix element is $\langle s_1 | H | s_2 \rangle = H_t + H_U$.
- **On-site U term** (diagonal):
  $H_U = U \cdot D(|s\rangle)$ when $|s_1\rangle = |s_2\rangle$, else $0$.
- **Hopping term**: for each bonded pair $(m, n)$ and each spin channel $\sigma$,
  annihilate $\sigma$ at site $n$ in $|s_2\rangle$ and at site $m$ in $|s_1\rangle$.
  If the resulting states match, accumulate $-t_{mn} \cdot \text{sign}_m \cdot \text{sign}_n$.

Math:
$$
\langle s_1 | H_t | s_2 \rangle =
-\sum_{(m,n)\in\text{bonds}} \sum_{\sigma} t_{mn}\;
\text{sign}(s_1, m)\;\text{sign}(s_2, n)\;
\delta_{a_m(s_1),\, a_n(s_2)}
$$

where $a_k(s)$ denotes the state after annihilating spin $\sigma$ at site $k$.

Code form:
```python
H[i][j] = calc_ham_t_ij(states[i], states[j]) + calc_ham_U_ij(states[i], states[j])
```

Validation:
- $H$ must be Hermitian.
- Diagonal elements for pure-spin states ($D = 0$) must be zero (no double occupancy → no $U$ term, no hopping to self).

## 11) Diagonalisation (MUST)

MUST:
- The Hamiltonian is diagonalised with `numpy.linalg.eigh` (assumes Hermitian).
- Degenerate eigenstates are disambiguated via `canonicalize_eigenpairs`:
  within each degenerate subspace (tolerance `ATOL["tight"]`, see `00-CONVENTIONS.md` §8), project a tie-breaker matrix
  $T = \text{diag}(1, 2, \ldots, n)$ and diagonalise the projection to fix the rotation.
- Global phases are fixed by `canonicalize_vector_phases`: the largest-magnitude
  component is made real and positive.

Code form:
```python
eigvals, eigvecs = np.linalg.eigh(H)
eigvals, eigvecs = canonicalize_eigenpairs(eigvals, eigvecs)
```

Validation:
- Eigenvalues must be real.
- Eigenvectors must be orthonormal.
- The ground-state energy must decrease as $U/t$ decreases (metallic limit).
