# 00-CONVENTIONS

Writing conventions, symbol definitions, and numerical tolerances for all standards.

## 1) Scope (MUST)

MUST:
- This convention applies to all files under `./standards/en/` and `./standards/zh/`.
- English standards are authoritative; Chinese files are translations only.

## 2) Per-Rule Structure (MUST)

Each nontrivial rule should be written in the following order:
1. `MUST` bullets (hard constraints).
2. `Math:` block with LaTeX (`$$ ... $$`).
3. `Code form:` single-line ASCII expression or short block.
4. `Index:` explicit symbol meaning.
5. `Validation:` minimal checks.

## 3) Formula Style (MUST)

- Prefer display math blocks over long inline formulas.
- Redefine key symbols locally instead of relying on distant sections.

## 4) Code Mapping (MUST)

- Every core formula must include one `Code form` line.
- `Code form` must be implementation-oriented plain text.

## 5) Symbol Table (MUST)

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of lattice sites in a cluster |
| $N_e$ | Total electron number (at half filling $N_e = N$) |
| $U$ | On-site Coulomb repulsion |
| $t, t_2, t_3$ | NN, NNN, 3rd-neighbor hopping amplitudes |
| $s_i$ | Local site occupation $\in \{0, 1, -1, 2\}$ |
| $S_z$ | Total spin z-component $= \frac{1}{2}(N_\uparrow - N_\downarrow)$ |
| $S$ | Total spin quantum number from $\hat{S}^2 = S(S+1)$ |
| $D$ | Number of doubly-occupied sites |
| $d_{\text{spin}}$ | Dimension of pure-spin subspace $= 2^N$ |
| $d(S)$ | Dimension of $(S_z, S)$ sector $= \binom{N}{N/2-S} - \binom{N}{N/2-S-1}$ |
| $H_{\text{eff}}$ | Effective spin Hamiltonian from SVD downfolding |
| $T_{11}$ | Projection quality metric $= U\Sigma U^\dagger$ |
| $W(C)$ | Net LCE contribution of cluster $C$ |

## 6) Canonical Naming (MUST)

MUST:
- `twoSz` ($= 2S_z$) and `twoS` ($= 2S$) are **integer labels** used for
  CLI parameters, path names, file names, and index comparisons.
  Their purpose is to eliminate floating-point ambiguity: half-integer quantum
  numbers like $S_z = 1/2$ cannot be represented exactly in float, which would
  cause unpredictable behaviour in string formatting, file lookup, and equality checks.
- Actual physics computation still uses the half-integer values
  $S_z = twoSz / 2$ and $S = twoS / 2$. The integer form is only for labelling.
- `S2` ($= S(S+1)$) is a computed float, not an integer label.
- The canonical runtime workflow parameter name is `workflow`; the canonical
  all-`twoSz` sector-range parameter name is `SCOPE`.
- Standards and implementation must not introduce alternative canonical names such as
  `ssq`, `s_squared`, `sz`, or `s` for these quantities.
- The only canonical CLI `MODE` spellings are `full`, `Sz`, and `SzS2`.
- `MODE` chooses the diagonalization block layer. Optional `twoSz` and `twoS`
  CLI parameters choose a specific block subset inside that layer.
- `SCOPE` chooses the sector range only for all-`twoSz` `MODE=Sz` and `MODE=SzS2`
  runs. Valid values are `nonnegative` and `pm`.
- Canonical runtime path tokens are:
  - `N_<N>_nelec_<nelec>_U_<U:.4f>_t_<T:.4f>`
  - `twoSz_<value>` (negative values use `n` prefix: `twoSz_n1` for $-1$)
  - `twoS_<value>` (`twoS` is always non-negative, no `n` prefix needed)
  - `mode_full`
  - `mode_twoSz`
  - `mode_twoSz_pm`
  - `mode_twoSz_<value>`
  - `mode_twoSz_twoS`
  - `mode_twoSz_pm_twoS`
  - `mode_twoSz_<value>_twoS`
  - `mode_twoSz_<value>_twoS_<value>`
  - `seed_<stem>` for LCE/embed outputs selected by a `SEED_SET` text file
- Path `mode_*` tokens are output directory names, not accepted CLI `MODE`
  values.
- Path `seed_*` tokens are derived from the `SEED_SET` file stem. They are
  output directory names, not physics mode names.
- Legacy CLI names such as `fixed_sz`, `block_sz_full`, `fixed_sz_s2`,
  `fixed_sz_s2_all`, `block_sz_s2_full`, `fixed_sz_ssq`,
  `block_sz_ssq_full`, `_sz...`, `_s...`, `SZ`, `S`, `S2`, `SZ_IDX`,
  `S_IDX`, and `TYPE` are not part of the production CLI standard.

Code form:
```text
canonical physics names: twoSz, twoS, S2
canonical workflow key: workflow
canonical LCE/embed input key: SEED_SET
canonical all-twoSz scope key: SCOPE
canonical CLI modes: full, Sz, SzS2
canonical path tokens: N_<N>_nelec_<nelec>_U_<U:.4f>_t_<T:.4f>, twoSz_<value>, twoS_<value>, mode_*, seed_*
negative value encoding: n prefix (e.g. twoSz_n1 = twoSz = -1)
```

## 7) Quantum-Number Value Rules (MUST)

MUST:
- The code assumes half filling: $N_e = N$ (electron count equals site count).
  Under half filling $N_\uparrow + N_\downarrow = N$, so
  $S_z = \tfrac{1}{2}(N_\uparrow - N_\downarrow)$ ranges over
  $\{-N/2, -N/2+1, \dots, N/2\}$, i.e. `twoSz` ∈ $[-N, N]$ with step 2.
- `twoSz` and `twoS` are integers.
- `twoSz` may be negative.
- `twoS` must satisfy `0 <= twoS <= N`.
- `twoSz` and `twoS` must satisfy `|twoSz| <= twoS`.
- `twoSz` and `twoS` must have the same parity as `N`.
- Invalid inputs must raise an error; the standard does not permit silent correction.

Math:
$$
N_e = N, \qquad N_\uparrow + N_\downarrow = N, \qquad S_z = \tfrac{1}{2}(N_\uparrow - N_\downarrow).
$$
$$
N \bmod 2 = twoSz \bmod 2 = twoS \bmod 2, \qquad |twoSz| \le twoS \le N.
$$

Code form:
```python
assert isinstance(twoSz, int)
assert isinstance(twoS, int)
assert N % 2 == twoSz % 2 == twoS % 2
assert abs(twoSz) <= twoS <= N
```

Validation:
- Even `N` permits only even `twoSz` and even `twoS`.
- Odd `N` permits only odd `twoSz` and odd `twoS`.
- Examples:
  - `N=4`: valid `(twoSz, twoS)` includes `(0, 0)`, `(0, 2)`, `(-2, 2)`, `(2, 4)`.
  - `N=5`: valid `(twoSz, twoS)` includes `(-1, 1)`, `(1, 3)`, `(-3, 5)`.

## 8) Numerical Tolerances (MUST)

MUST:
- All tolerances must be defined in a single parameter table (`ATOL` dict in code)
  and referenced from there. Hard-coded magic numbers are forbidden.

| Key | Value | Usage |
|-----|-------|-------|
| `tight` | `1e-10` | Eigenvalue degeneracy grouping, phase canonicalization |
| `loose` | `1e-6` | $S^2$/$S_z$ validation, general floating-point comparison |

Code form:
```python
ATOL = {
    "tight": 1e-10,
    "loose": 1e-6,
}
```

## 9) Engineering Discipline (MUST)

This is scientific computing code, not a production service. Code MUST stay
minimal and trust its callers.

MUST:
- Validate once at the system boundary (CLI entry, file I/O, user input).
  Internal helper functions MUST trust the values their callers pass in.
- Do not repeat the same check in every layer. Once a precondition is
  established, downstream code treats it as given.
- Do not add `try/except` blocks to "be safe". Only catch an exception when
  there is a concrete recovery path.
- Do not add fallback branches for inputs that the scientific workflow will
  never produce. If a branch cannot be reached in the documented pipeline,
  delete it rather than leave it as defensive scaffolding.
- Do not introduce optional parameters, feature flags, or configuration knobs
  unless a current standard or workflow requires them.
- Do not wrap simple operations in helper classes/functions just for
  abstraction. One-time use code stays inline.
- Prefer a direct failure (uncaught exception, assertion) over silent
  correction, defaulting, or clamping of scientifically meaningful quantities.

Code form:
```text
boundary validation: CLI parsing, file readers, top-level entry points
internal functions: trust arguments, no re-validation
error handling: raise on invalid state, do not mask
new abstractions: only when a real second caller exists
```
