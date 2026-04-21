# 07-TESTING

Regression testing strategy, reference data conventions, and naming translation rules.

## 1) Test Tiers (MUST)

MUST:
- The test suite consists of three tiers:

| Tier | Marker | Scope | Trigger |
|------|--------|-------|---------|
| Unit | (none) | Pure-function logic: basis generation, sorting, matrix elements, sector counts | Every commit; `pytest tests/` |
| Integration | `@pytest.mark.integration` | End-to-end single-cluster pipeline: Hamiltonian → diagonalise → select → H_eff → fit | `pytest -m integration` |
| Regression | `@pytest.mark.regression` | Numerical comparison against `data_test/` reference data from pre-refactor code | `pytest -m regression` |

- Unit tests must complete in < 30 s on a single core.
- Integration tests may take up to 5 min.
- Regression tests may take longer; they read large `.npy` files from disk.

## 2) Reference Data Layout (MUST)

MUST:
- Reference data lives under `data_test/` with the **old naming convention** (pre-refactor).
  It must NOT be renamed or reformatted — it serves as a frozen ground truth.
- Production code does NOT provide backward compatibility with old naming or formats.
  All old-to-new translation is the responsibility of the test-side `ReferenceData` tool.
- Directory structure:
  ```
  data_test/
  └── Block_U{U:.4f}_t{t:.4f}/
      ├── N{N}/                                     # old MODE=full (TYPE=all)
      ├── N{N}_sz{Sz:.4f}/                          # old MODE=fixed_sz
      ├── N{N}_adiabatic_restart/                   # old workflow=adiabatic
      ├── N{N}_multi_restart/                       # old workflow=multi
      ├── N{N}_sz{Sz:.4f}_adiabatic_restart/
      ├── N{N}_sz{Sz:.4f}_multi_restart/
      └── log_*.txt
  ```
- Per-cluster files within each run directory:
  ```
  hole{h}_class{c}_eigvals.npy
  hole{h}_class{c}_states.npy
  hole{h}_class{c}_Heff.npy
  hole{h}_class{c}_T11m1.npy
  hole{h}_class{c}_t11_selected_indices.npy
  hole{h}_class{c}_t11_selected_occupation.npy
  hole{h}_class{c}_double_occupation_expectation.npy
  hole{h}_class{c}_s2_digonal.npy                   # NOTE: old typo "digonal"
  hole{h}_class{c}_s2_selected.npy
  hole{h}_class{c}_cluster{v}_results.txt
  ```

## 3) Naming Translation (MUST)

MUST:
- The test framework must translate between old and new naming conventions.
  The reference data retains old names; comparison code maps them to current semantics.

### 3.1) Directory names

| Old pattern | New equivalent | Translation rule |
|-------------|----------------|------------------|
| `Block_U{U}_t{t}` | `U{U}_t{t}` | Strip `Block_` prefix |
| `N{N}` | `N{N}` (MODE=full) | No change |
| `N{N}_sz{Sz:.4f}` | `N{N}_twoSz_{twoSz}` | `twoSz = int(2 * Sz)` |
| `N{N}_multi_restart` | `N{N}_multi_restart` | Workflow suffix unchanged |
| `N{N}_adiabatic_restart` | `N{N}_adiabatic_restart` | Workflow suffix unchanged |
| `N{N}_sz{Sz}_multi_restart` | `N{N}_twoSz_{twoSz}_multi_restart` | Combine Sz translation + suffix |

### 3.2) File names

| Old name | New name | Notes |
|----------|----------|-------|
| `*_s2_digonal.npy` | `*_S2_diagonal.npy` | Typo fix + capitalisation |
| `*_s2_selected.npy` | `*_S2_selected.npy` | Capitalisation |
| All others | Same | No change |

### 3.3) CLI parameter names

| Old key | New key | Translation |
|---------|---------|-------------|
| `TYPE` | `MODE` | `TYPE=all` → `MODE=full`, `TYPE=sz` → `MODE=fixed_sz` |
| `SZ` | `twoSz` | `twoSz = int(2 * SZ)` |
| `S` | `twoS` | `twoS = int(2 * S)` |
| `WORKFLOW` | `workflow` | Case change only |

Code form:
```python
def old_sz_to_twoSz(sz_str: str) -> int:
    """Convert old 'sz0.0000' or 'sz0.5000' to integer twoSz."""
    sz_float = float(sz_str.replace("sz", ""))
    return int(round(2 * sz_float))

def old_dirname_to_new(dirname: str) -> str:
    """Translate old run directory name to new convention."""
    # N4_sz0.0000_multi_restart → N4_twoSz_0_multi_restart
    ...
```

## 4) Comparison Rules (MUST)

MUST:
- Not all outputs are equally deterministic. Comparison strategy depends on the quantity:

### 4.1) Deterministic quantities (strict comparison)

| Quantity | File | Tolerance | Notes |
|----------|------|-----------|-------|
| Eigenvalues | `eigvals.npy` | `ATOL["tight"]` | Fully deterministic from Hamiltonian |
| States | `states.npy` | Exact (integer) | Basis ordering is canonical |
| Double-occupation expectation | `double_occupation_expectation.npy` | `ATOL["tight"]` | Derived from eigvecs, deterministic |
| S² diagonal | `s2_digonal.npy` / `S2_diagonal.npy` | `ATOL["loose"]` | Column 0: ⟨S²⟩, Column 1: variance |

### 4.2) Selection-dependent quantities (conditional comparison)

| Quantity | File | Condition | Notes |
|----------|------|-----------|-------|
| Selected indices | `t11_selected_indices.npy` | Only compare for `occ` / `energy` workflows | Greedy/multi/adiabatic involve randomness |
| H_eff | `Heff.npy` | Only when selected indices match | H_eff is deterministic given same selection |
| T11-I | `T11m1.npy` | Only when selected indices match | Same as above |
| Selected occupation | `t11_selected_occupation.npy` | Only when selected indices match | Derived from selection |

### 4.3) Non-deterministic quantities (norm-only comparison)

| Quantity | Comparison | Notes |
|----------|------------|-------|
| H_eff from `multi` workflow | `‖T11-I‖` should be ≤ old value + `ATOL["loose"]` | Random restarts may find different minima |
| Adiabatic overlap | Should be ≥ old value - `ATOL["loose"]` | May differ due to different selection |

## 5) Test Case Selection (MUST)

MUST:
- Regression tests must not compare all 126k files. Select representative cases:

### 5.1) Minimal regression set

| Case | Parameters | Covers |
|------|-----------|--------|
| N=2, full | U=1, t=0.24 | Smallest cluster, full diag, baseline |
| N=4, full | U=1, t=0.24 | 4-site with holes, isomorphic classes |
| N=4, fixed_sz | U=1, t=0.24, Sz=0 | Sz-blocked mode |
| N=6, fixed_sz | U=1, t=0.24, Sz=0 | Larger cluster, more sectors |
| N=4, adiabatic | U=1, t=0.24 | Adiabatic selection (needs t=0.22 as previous) |

### 5.2) Extended regression set

- All N from 2 to 7 at t=0.24 (one parameter point, all modes).
- Boundary parameter points: t=0.02 (strong coupling) and t=0.60 (weak coupling).
- N=8 fixed_sz (largest available, tests scalability).

## 6) Test Infrastructure (MUST)

MUST:
- A shared fixture provides the translation layer:

Code form:
```python
# tests/conftest.py

DATA_TEST_DIR = Path(__file__).parent.parent / "data_test"

@pytest.fixture
def ref_data():
    """Accessor for reference data with automatic naming translation."""
    return ReferenceData(DATA_TEST_DIR)

class ReferenceData:
    def load_npy(self, U, t, run_dir_new, filename_new) -> np.ndarray:
        """Load a reference .npy file, translating new names to old paths."""
        ...

    def has_case(self, U, t, run_dir_new) -> bool:
        """Check if reference data exists for this case."""
        ...
```

- Regression tests skip gracefully if `data_test/` is absent (e.g. in CI without data):

Code form:
```python
pytestmark = pytest.mark.regression

@pytest.fixture(autouse=True)
def skip_if_no_data():
    if not DATA_TEST_DIR.exists():
        pytest.skip("data_test/ not present")
```

## 7) Comparison Helpers (MUST)

MUST:
- Provide reusable assertion helpers:

Code form:
```python
def assert_eigvals_match(new, ref, label=""):
    """Strict eigenvalue comparison."""
    np.testing.assert_allclose(np.sort(new), np.sort(ref),
                                atol=ATOL["tight"], err_msg=f"eigvals mismatch {label}")

def assert_heff_match(new, ref, label=""):
    """H_eff comparison (when selection matches)."""
    np.testing.assert_allclose(new, ref, atol=ATOL["loose"], err_msg=f"Heff mismatch {label}")

def assert_indices_match(new, ref, label=""):
    """Selected indices comparison (sorted, for deterministic methods only)."""
    np.testing.assert_array_equal(np.sort(new), np.sort(ref), err_msg=f"indices mismatch {label}")

def assert_t11_no_worse(new_norm, ref_norm, label=""):
    """For non-deterministic selection: new T11 norm should not be significantly worse."""
    assert new_norm <= ref_norm + ATOL["loose"], \
        f"T11 norm regression {label}: new={new_norm}, ref={ref_norm}"
```

## 8) Integration Test Structure (MUST)

MUST:
- Integration tests run the full pipeline on small clusters (N=2, N=3) with known parameters,
  without reference data. They verify internal consistency:
  1. Hamiltonian is Hermitian.
  2. Eigenvalues are real and sorted.
  3. Eigenvector orthonormality.
  4. Sector dimensions sum to full Hilbert space.
  5. H_eff is Hermitian.
  6. Spin-coupling fit R² > 0.99 for small U/t.
  7. Selected indices count equals dimspin.

Code form:
```python
@pytest.mark.integration
def test_full_pipeline_N2():
    model = HubbardModel(2, 1.0, 0.3)
    # ... build, solve, downfold, fit ...
    assert model.downfold.t11m1_norm < 1e-6
    assert model.downfold.heff is not None
```

## 9) What NOT to Compare (MUST)

MUST:
- Do NOT compare:
  - `memory_status.txt`, `timing_status.txt` — machine-dependent.
  - `log_*.txt` — contains timestamps, MPI rank info.
  - `results.txt` text format — formatting may change; compare `.npy` data instead.
  - Eigenvectors directly — gauge-dependent (phase, degenerate rotation).
    Compare eigenvector-derived quantities (eigenvalues, ⟨S²⟩, double occupation) instead.
  - File existence or count — new code may produce additional files (e.g. `.json` artifacts).
