# Entrypoint Structure Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `cuprate.main`, `cuprate.lce`, and `cuprate.embed` so each package has a thin workflow layer plus small focused modules, without changing CLI behavior, runtime paths, output files, or JSON artifacts.

**Architecture:** Keep `__main__.py` as a compatibility entrypoint and keep package exports stable. Move computation and reporting out of the current large workflow files into a few focused sibling modules: `main/{solver,reporting}.py`, `lce/core.py`, and `embed/{core,reporting}.py`. Do not change tests unless explicitly requested.

**Tech Stack:** Python 3, NumPy, SciPy, mpi4py, networkx, pytest

---

### Task 1: Freeze Entrypoint Compatibility

**Files:**
- Modify: `src/cuprate/main/__main__.py`
- Modify: `src/cuprate/lce/__main__.py`
- Modify: `src/cuprate/embed/__main__.py`
- Modify: `src/cuprate/main/__init__.py`
- Modify: `src/cuprate/lce/__init__.py`
- Modify: `src/cuprate/embed/__init__.py`
- Test: `tests/test_pipeline_e2e.py`
- Test: `tests/test_mode_routing.py`

**Step 1: Write the failing compatibility tests**

Use existing tests as the contract:
- `tests/test_pipeline_e2e.py`
- `tests/test_mode_routing.py`

**Step 2: Run tests to capture the baseline**

Run:
```bash
python -m pytest tests/test_pipeline_e2e.py tests/test_mode_routing.py -q
```

Expected: current behavior is either already green or fails only because of known compatibility drift in `__main__` imports.

**Step 3: Keep `__main__` as the stable compatibility shell**

Requirements:
- `python -m cuprate.main`, `python -m cuprate.lce`, `python -m cuprate.embed` still work.
- Direct imports from `cuprate.main.__main__`, `cuprate.lce.__main__`, `cuprate.embed.__main__` still expose the names used by tests.
- `__init__.py` exports continue to work.

**Step 4: Re-run compatibility tests**

Run:
```bash
python -m pytest tests/test_pipeline_e2e.py tests/test_mode_routing.py -q
```

Expected: PASS.


### Task 2: Extract Main Solver and Reporting

**Files:**
- Create: `src/cuprate/main/solver.py`
- Create: `src/cuprate/main/reporting.py`
- Modify: `src/cuprate/main/workflow.py`
- Modify: `src/cuprate/main/__main__.py`
- Modify: `src/cuprate/main/__init__.py`
- Test: `tests/test_embed_lce_cli.py`
- Test: `tests/test_sz0_output_and_projection.py`
- Test: `tests/test_pipeline_e2e.py`

**Step 1: Write/identify the contract tests**

Use existing tests as the lock:
- `tests/test_embed_lce_cli.py`
- `tests/test_sz0_output_and_projection.py`
- `tests/test_pipeline_e2e.py`

Do not edit them.

**Step 2: Run the targeted main-stage tests**

Run:
```bash
python -m pytest tests/test_embed_lce_cli.py tests/test_sz0_output_and_projection.py tests/test_pipeline_e2e.py -q
```

Expected: PASS before refactor.

**Step 3: Move responsibilities**

Move into `src/cuprate/main/solver.py`:
- `find_restart_path`
- `cluster_process`
- `cluster_process_work_item`
- `_save_results_for_model`
- `_save_spin_coupling_results`
- `_save_projection_only_results`

Move into `src/cuprate/main/reporting.py`:
- result dataclasses
- text formatting helpers
- `cluster_save_results`
- `cluster_save_projection_results`
- artifact-building helpers

Leave in `src/cuprate/main/workflow.py` only:
- CLI parsing
- cluster preparation
- MPI distribution
- top-level `main`
- `run_cli`

**Step 4: Re-run targeted tests**

Run:
```bash
python -m pytest tests/test_embed_lce_cli.py tests/test_sz0_output_and_projection.py tests/test_pipeline_e2e.py -q
```

Expected: PASS.


### Task 3: Extract LCE Core

**Files:**
- Create: `src/cuprate/lce/core.py`
- Modify: `src/cuprate/lce/workflow.py`
- Modify: `src/cuprate/lce/__main__.py`
- Modify: `src/cuprate/lce/__init__.py`
- Test: `tests/test_embed_lce_cli.py`
- Test: `tests/test_pipeline_e2e.py`

**Step 1: Run the LCE contract tests**

Run:
```bash
python -m pytest tests/test_embed_lce_cli.py tests/test_pipeline_e2e.py -q
```

Expected: PASS before refactor.

**Step 2: Move pure LCE logic into `core.py`**

Move:
- `get_connected_subgraphs`
- `find_operator`
- `operator_minus`
- subgraph-match collection helpers
- subtraction helpers

Leave in `workflow.py` only:
- CLI parsing
- upstream/downstream directory resolution
- per-`nsites` orchestration
- consolidated JSON writing

Keep `write_operators` monkeypatch-compatible from `cuprate.lce.__main__`.

**Step 3: Re-run LCE tests**

Run:
```bash
python -m pytest tests/test_embed_lce_cli.py tests/test_pipeline_e2e.py -q
```

Expected: PASS.


### Task 4: Extract Embed Core and Reporting

**Files:**
- Create: `src/cuprate/embed/core.py`
- Create: `src/cuprate/embed/reporting.py`
- Modify: `src/cuprate/embed/workflow.py`
- Modify: `src/cuprate/embed/__main__.py`
- Modify: `src/cuprate/embed/__init__.py`
- Test: `tests/test_embed_lce_cli.py`
- Test: `tests/test_pipeline_e2e.py`

**Step 1: Run the embed contract tests**

Run:
```bash
python -m pytest tests/test_embed_lce_cli.py tests/test_pipeline_e2e.py -q
```

Expected: PASS before refactor.

**Step 2: Move pure embedding logic into `core.py`**

Move:
- transform/orientation helpers
- OBC/PBC placement helpers
- coupling accumulation helpers
- four-site and six-site normalization helpers

Move into `reporting.py`:
- `save_data_embed`
- `write_couplings_embed`
- `print_grid`

Leave in `workflow.py` only:
- CLI parsing
- upstream result loading
- call into embedding core
- output directory creation

**Step 3: Re-run embed tests**

Run:
```bash
python -m pytest tests/test_embed_lce_cli.py tests/test_pipeline_e2e.py -q
```

Expected: PASS.


### Task 5: Final Verification and Boundaries

**Files:**
- Modify: `docs/plans/2026-04-08-entrypoint-structure-plan-codex.md`
- Test: `tests/test_embed_lce_cli.py`
- Test: `tests/test_mode_paths.py`
- Test: `tests/test_mode_routing.py`
- Test: `tests/test_sz0_output_and_projection.py`
- Test: `tests/test_pipeline_e2e.py`
- Test: `tests/verify_data_test.py`

**Step 1: Run the full targeted verification set**

Run:
```bash
python -m pytest \
  tests/test_embed_lce_cli.py \
  tests/test_mode_paths.py \
  tests/test_mode_routing.py \
  tests/test_sz0_output_and_projection.py \
  tests/test_pipeline_e2e.py \
  tests/verify_data_test.py -q
```

Expected: PASS.

**Step 2: Structural acceptance check**

Confirm:
- `src/cuprate/main/workflow.py` is only orchestration and stays near 150-250 lines.
- `src/cuprate/lce/workflow.py` is only orchestration and stays near 120-220 lines.
- `src/cuprate/embed/workflow.py` is only orchestration and stays near 120-220 lines.
- Pure computation code does not call `open()` or do CLI parsing.
- Reporting code does not perform cluster matching or operator subtraction.
- No test files were modified.

**Step 3: Record any residual debt**

Residual debt that is explicitly out of scope for this plan:
- replacing list-shaped operator records with typed structures
- changing the consolidated JSON schema
- redesigning CLI parsing beyond current standard-compliant behavior
