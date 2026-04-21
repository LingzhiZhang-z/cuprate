# Step B: End-to-End Pipeline Verification Test

## Purpose

Verify the complete cuprate pipeline (main → LCE → embed) produces physically correct results after t2/t3 removal. Validate J₁ ≈ 4t²/U in the weak-coupling limit.

## Test Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| U | 1.0 | Standard Hubbard U |
| t | 0.02 | Weak coupling (t/U = 0.02), perturbation theory highly accurate |
| N_max | 4 | Include N=2,3,4 clusters; captures corrections up to O(t⁴/U³) |
| Ncell | 8 | 8×8 PBC supercell for embedding |
| mode | full | Full Hilbert space diagonalization |
| workflow | None | Default eigenstate selection (occupation-based) |

**Expected physics**: J₁ ≈ 4t²/U = 0.0016. At t/U = 0.02, higher-order corrections are O(10⁻⁶), so J₁ should match the analytic value to ~0.2%. Reference data confirms: the N=2 cluster at t=0.02 gives J₁ = 0.0015974482.

## Architecture

**File**: `tests/test_pipeline_e2e.py`

**Strategy**: Use `chdir(tmp_path)` so all three pipeline stages read/write the natural directory tree without monkey-patching. Import library functions directly (no subprocess needed — mpi4py is installed and works at rank=0).

### Directory Layout (relative to tmp_path)

```
Block/U1.0000_t0.0200/
  N2/results.json          ← Stage 1 output
  N3/results.json
  N4/results.json
data_transfer/LCE/LCE_U1.0000_t0.0200/
  N2/results.json          ← Stage 2 output
  N3/results.json
  N4/results.json
data_transfer/Embed/Embed_U1.0000_t0.0200/
  Ncell8_Ncut4/
    couplings_pbc.txt      ← Stage 3 output (final result)
```

### Stage 1: Diagonalization + Spin Coupling Fit

Uses library-level calls — no MPI subprocess needed.

```
For each N in [2, 3, 4]:
  1. clusters = Clusters_Square(N)
     clusters.compute_clusters()
     clusters.classify_clusters()
  2. params = Params(N=N, U=1.0, t=0.02, mode="full")
     spec = resolve_mode_spec(params)
  3. For each (hole, class_idx) in clusters.clusters_classified:
     - model = cluster_process(cluster, params, params_cluster)
     - cluster_entries += _save_spin_coupling_results(model, clusters, params, output_dir, ...)
     - Save eigvals/eigvecs .npy files
  4. Write results.json via build_consolidated_results()
```

**Key imports from `cuprate.main.__main__`**:
- `cluster_process` — builds HubbardModel, diagonalizes, extracts Heff
- `_save_spin_coupling_results` — fits spin couplings for all cluster variants
- `build_consolidated_results`, `_serialize_run_params` — JSON assembly

**Note**: `build_path_spec(params)` constructs the output directory. The test must either use `build_path_spec` to get the correct path or replicate the path manually. Using `build_path_spec` is preferred since it's the canonical path constructor.

### Stage 2: LCE (Möbius Inversion)

```python
from cuprate.lce.__main__ import main as lce_main

params_lce = {
    "N": 4, "U": 1.0, "t": 0.02,
    "twoSz": None, "twoS": None,
    "workflow": None, "restart": False,
}
lce_main(params_lce)
```

LCE's `main()` calls `find_first_existing_dir("data_transfer/Block/Block_U1.0000_t0.0200", "Block/U1.0000_t0.0200")` — the second path matches our Stage 1 output.

### Stage 3: Embedding

```python
from cuprate.embed.__main__ import main as embed_main

params_embed = {
    "N": 4, "U": 1.0, "t": 0.02,
    "Ncell": 8, "Ncut": 4,
    "twoSz": None, "twoS": None,
    "workflow": None, "restart": False,
}
embed_main(params_embed)
```

### Stage 4: Verification

Parse `data_transfer/Embed/Embed_U1.0000_t0.0200/Ncell8_Ncut4/couplings_pbc.txt` for coupling values.

**Assertions**:
1. J₁ exists and is real: `|Im(J₁)| < 1e-10`
2. J₁ matches perturbation theory: `|J₁ - 4t²/U| / (4t²/U) < 0.01` (1% tolerance)
3. J₂ is much smaller: `|J₂| < |J₁| / 100`
4. LCE and embed did not crash — all intermediate `results.json` files exist

## Output Parsing

The embed output `couplings_pbc.txt` has the format:
```
Bond vector (1, 0), J1:
    0: Sites 0-1: <real>  +  <imag>i
    1: Sites 1-2: <real>  +  <imag>i
    ...
Bond vector (1, 1), J2:
    ...
```

J₁ is the average of all `(1,0)` bond couplings. On a translationally invariant PBC lattice, all NN bonds should have the same coupling value.

## Edge Cases and Risks

1. **S2 directory**: `cluster_process` may call `model.save_blocks()` which creates `S2/` under cwd. For mode=full, `use_S2_blocks=False`, so this is skipped.
2. **MPI at import time**: `cuprate.main.__init__.py` uses lazy `__getattr__` — MPI is only initialized when functions are accessed, not on import. Since mpi4py is installed, this works.
3. **Timing**: N=4 has dim=70 (full space), diagonalization is instant. The entire test should complete in <10 seconds.
