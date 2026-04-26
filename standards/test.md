# Restartable data_test Regression Plan

## Goal

Build a restartable regression workflow that validates all `data_test/` reference
systems for `N=2,3,4,5`.

The test must cover:
- `U=1.0`, `T=0.02..0.60` in steps of `0.02`.
- `MODE=full`.
- `MODE=Sz` with `twoSz=0` for even `N`, `twoSz=1` for odd `N`.
- `workflow=occ`, `workflow=greedy_multi`, and `workflow=adiabatic`.
- All `(hole, class_idx)` families and all `cluster_idx` placements represented
  by `ClusterSets(N)`.

Do not rename or rewrite `data_test/`. Old-format handling belongs only in the
test-side reference reader.

## Reference Mapping

Old `data_test` names map to current runtime names as follows:

| Old reference | Current run |
| --- | --- |
| `N{N}` | `MODE=full workflow=occ` |
| `N{N}_sz{Sz:.4f}` | `MODE=Sz twoSz=int(2*Sz) workflow=occ` |
| `N{N}_multi_restart` | `MODE=full workflow=greedy_multi` |
| `N{N}_sz{Sz:.4f}_multi_restart` | `MODE=Sz twoSz=int(2*Sz) workflow=greedy_multi` |
| `N{N}_adiabatic_restart` | `MODE=full workflow=adiabatic` |
| `N{N}_sz{Sz:.4f}_adiabatic_restart` | `MODE=Sz twoSz=int(2*Sz) workflow=adiabatic` |

The old `states.npy` files are site-code text data, not current bit-encoded
states. The reference reader must convert:

| Old site code | Meaning | Current local code |
| --- | --- | --- |
| `0` | empty | `0` |
| `1` | up | `1` |
| `-1` | down | `2` |
| `2` | double | `3` |

Some old `.npy` files are true NumPy binaries and some are ASCII text with
complex numbers. The reference reader must try binary `np.load` first, then
fallback to text parsing.

## Restart State

Use one output root, for example:

```text
tmp/data_test_regression/
```

Inside it, maintain:

```text
tmp/data_test_regression/status.jsonl
tmp/data_test_regression/failures.jsonl
tmp/data_test_regression/summary.json
```

Each atomic task writes one JSONL record:

```json
{
  "case_id": "N4_T0.2400_MODE-full_workflow-occ",
  "stage": "main",
  "status": "passed",
  "started_at": "...",
  "finished_at": "...",
  "results_json": ".../results.json"
}
```

Restart rule:
- Before running a task, check `status.jsonl`.
- If the latest record for `case_id` is `passed`, skip it.
- If required output files exist but no `passed` record exists, run validation
  first; only recompute if validation fails.
- Write task output into a temporary directory or temporary status record first,
  then atomically mark the task `passed`.
- Never infer adiabatic progress from directory names alone; use the validated
  seed `results.json`.

## Execution Order

### 1. Enumerate cases

For every `N in [2,3,4,5]` and every `T in [0.02,0.04,...,0.60]`, create two
mode cases:

```text
MODE=full
MODE=Sz twoSz=(0 if N even else 1)
```

For each mode case, schedule workflows in this order:

```text
occ -> greedy_multi -> adiabatic
```

### 2. Warm the eigensystem cache with occ

Run `occ` first with `CACHE_MODE=save`. This computes and stores eigvals/eigvecs
for the current `(N,T,MODE)`:

```bash
python -m cuprate.main ROOT=tmp/data_test_regression N=4 U=1.0 T=0.2400 MODE=full workflow=occ CACHE_MODE=save
python -m cuprate.main ROOT=tmp/data_test_regression N=4 U=1.0 T=0.2400 MODE=Sz twoSz=0 workflow=occ CACHE_MODE=save
```

Validate the `occ` output before scheduling dependent workflows.

### 3. Run greedy_multi from cache

Run `greedy_multi` with `CACHE_MODE=load` so the eigensystem is reused:

```bash
python -m cuprate.main ROOT=tmp/data_test_regression N=4 U=1.0 T=0.2400 MODE=full workflow=greedy_multi CACHE_MODE=load
```

Because `greedy_multi` has random trials and the production CLI has no RNG seed
key, exact selected indices are not a hard pass condition. The pass condition is
norm/fit based unless the runner uses an in-process deterministic seed.

### 4. Run the adiabatic chain

For each fixed `(N,MODE)`, process `T` in increasing order.

At `T=0.0200`, use the same-parameter `occ` result as the seed. This matches the
old reference behavior: the first old `adiabatic_restart` result is effectively
the `occ` result.

```bash
python -m cuprate.main \
  ROOT=tmp/data_test_regression \
  N=4 U=1.0 T=0.0200 MODE=full workflow=adiabatic CACHE_MODE=load \
  SEED_RESULTS=tmp/data_test_regression/block_main/N_4_nelec_4_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json
```

For later `T`, seed from the previous `T` adiabatic result:

```text
T=0.0400 seed <- T=0.0200 workflow_adiabatic/results.json
T=0.0600 seed <- T=0.0400 workflow_adiabatic/results.json
...
T=0.6000 seed <- T=0.5800 workflow_adiabatic/results.json
```

Restart rule for adiabatic:
- If `T=0.3000` fails, restart resumes from that `T`.
- It must not rerun earlier passed `T` values.
- It must verify the immediately previous seed output before using it.

## Validation Layers

### Layer A: output schema

For every main run:
- `results.json` exists and has `result_kind="spin_couplings"`.
- `run_params` match `N`, `U`, `T`, `MODE`, `twoSz`, `twoS`, `workflow`.
- `complete_family_set` is true.
- Every family entry points to an exchange JSON and cluster JSON.
- Every exchange JSON has `projection`, `operators`, and `fit`.
- Every projection artifact contains block labels, basis states,
  `eigvecs_fock`, selected indices, `Heff`, and `T11` metrics.

For `workflow=adiabatic`:
- `run_params.SEED_RESULTS` exists.
- `run_params.adiabatic_seed.workflow` is `occ` at `T=0.0200`.
- `run_params.adiabatic_seed.workflow` is `adiabatic` for later `T`.
- Each projection block records its block-level `adiabatic_seed`.

### Layer B: family-level numerical comparison

For every `(N,T,MODE,workflow,hole,class_idx)` compare against `data_test`:

Strict:
- Eigenvalues, sorted.
- Converted state set and current basis set.
- Double-occupation expectation after basis/eigenstate alignment.
- S2 diagnostics, with loose tolerance.

Conditional:
- For `occ`, selected indices after alignment should match unless there is a
  true degenerate-gauge ambiguity.
- If selected indices match, compare `Heff`, `T11m1`, and selected occupation.
- If selected indices do not match but the selected subspace is degenerate,
  compare projector-level quantities and fitted coefficients.

Norm-based:
- For `greedy_multi`, require current `T11-I` norm to be no worse than the old
  reference within loose tolerance.
- For `adiabatic`, require overlap and `T11-I` to match or improve within loose
  tolerance; at `T=0.0200`, require equality with the `occ` result.

### Layer C: cluster placement and coefficients

For every `cluster_idx` member in each family:
- Verify cluster geometry count matches `ClusterSets(N)`.
- Verify `indices` is a permutation of `0..N-1`.
- Verify operator keys are present and canonical.
- Compare coefficient dictionaries by canonical key, not text label position.
- Do not compare old `*_cluster{idx}_results.txt` formatting directly.

### Layer D: LCE and embed smoke

After all main outputs pass for one `(MODE,workflow)` and `T`, run:

```bash
python -m cuprate.lce ROOT=tmp/data_test_regression N=5 U=1.0 T=0.2400 MODE=full workflow=occ
python -m cuprate.embed ROOT=tmp/data_test_regression N=5 U=1.0 T=0.2400 MODE=full workflow=occ
```

Then repeat at least for:
- `MODE=Sz twoSz=1 workflow=occ`
- `MODE=full workflow=adiabatic`
- `MODE=Sz twoSz=1 workflow=adiabatic`

Pass conditions:
- LCE reads consecutive `N=2..5` main outputs.
- All LCE reconstruction errors are within tolerance.
- Embed reads the LCE manifest and writes `embed_results.json`, `two_site.txt`,
  and expected cluster sidecars.

## Case Counts

For `N=2..5`:

| N | families | cluster members |
| --- | ---: | ---: |
| 2 | 1 | 1 |
| 3 | 1 | 2 |
| 4 | 3 | 5 |
| 5 | 4 | 12 |

Main family-level comparisons:

```text
30 T values * 9 families * 2 modes * 3 workflows = 1620 cases
```

Cluster-placement comparisons:

```text
30 T values * 20 cluster members * 2 modes * 3 workflows = 3600 cases
```

## Failure Policy

A failure record must include:
- `case_id`
- command or in-process params
- reference directory
- current output path
- failing quantity
- max absolute difference or norm
- seed path for adiabatic
- traceback or validation message

Stop policy:
- Default: continue after failures and write `failures.jsonl`.
- Debug mode: stop at first failure.
- Never delete passed outputs during retry.

## Not In Scope

- Do not modify production code to read old `data_test` names.
- Do not compare old text report formatting.
- Do not compare eigenvectors directly.
- Do not include `N=6+` in the required matrix; those are optional benchmarks.
- Do not treat `MODE=SzS2` as a `data_test` regression case unless new
  reference data is generated for it.
