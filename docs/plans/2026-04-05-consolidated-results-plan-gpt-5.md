# Consolidated Results Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-cluster JSON artifacts with one `results.json` per run and update Block/LCE/embed/test readers to use schema version 2.

**Architecture:** Keep Stage 1 text outputs per cluster, but move machine-readable output to a single run-level payload assembled on rank 0 from rank-local cluster entries. Centralize schema-v2 serialization/deserialization in `src/cuprate/io.py`, then have cluster discovery plus LCE/embed read from the consolidated file once per run directory.

**Tech Stack:** Python, mpi4py, pytest, JSON file I/O.

---

### Task 1: Update tests first

**Files:**
- Modify: `tests/test_embed_lce_cli.py`
- Modify: `tests/test_sz0_output_and_projection.py`
- Modify: `tests/test_shared_periphery.py`

1. Rewrite JSON helpers/fixtures to generate a run-level `results.json` payload with `schema_version = 2`.
2. Update assertions to expect `results.json` in the `N...` directory instead of per-cluster JSON files.
3. Keep Stage 1 `.txt` assertions intact.
4. Run targeted tests to confirm failures are due to the old production schema/paths.

### Task 2: Refactor schema helpers in `io.py`

**Files:**
- Modify: `src/cuprate/io.py`

1. Set `RESULT_SCHEMA_VERSION = 2`.
2. Change artifact builders to return a single cluster entry shaped for `clusters[]`, including nested `metadata`.
3. Add consolidated helpers: `build_consolidated_results`, `load_consolidated_results`, `find_cluster_entry`.
4. Update operator/fit readers and cluster discovery to use only consolidated JSON, without per-cluster JSON fallback.
5. Ensure group entries always expose `{arity, vector, label, terms}`, with `K*`/`L*` labels and `vector = null` for multi-site groups.

### Task 3: Refactor Block writer aggregation

**Files:**
- Modify: `src/cuprate/main/__main__.py`

1. Keep per-cluster `.txt` writing.
2. Change result-save helpers to return cluster entries instead of writing JSON files.
3. Accumulate rank-local entries during the work loop.
4. Gather entries with `mpi.comm.gather()` and write one sorted `results.json` on rank 0 using a stable serialized `run_params` subset.

### Task 4: Refactor LCE/embed readers

**Files:**
- Modify: `src/cuprate/lce/__main__.py`
- Modify: `src/cuprate/embed/__main__.py`

1. Load consolidated payload once per `N` directory.
2. Look up cluster entries by `(hole, class_idx, cluster_idx)`.
3. Feed operator lists from `artifact_to_operator_list(...)`.
4. Remove text-fit fallback logic in LCE so metrics come from JSON only.

### Task 5: Verify

**Files:**
- Modify only if required by failing tests in the scoped files above.

1. Run `python -m pytest tests/ -x -q`.
2. Fix regressions until green.
3. Spot-check one generated `results.json` to confirm schema-v2 top-level and per-group structure.
