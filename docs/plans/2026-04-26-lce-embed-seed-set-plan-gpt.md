# LCE/Embed Seed-Set Input Plan

## Summary

Replace LCE/embed input selection from global `MODE/workflow/twoSz/twoS`
path inference with a shared `SEED_SET` txt file.

New interface:

```bash
mpirun -np 4 env PYTHONPATH=src python -m cuprate.lce \
  ROOT=results_gpt N=5 U=1.0 T=0.2400 \
  SEED_SET=seed_sets/block.txt

mpirun -np 4 env PYTHONPATH=src python -m cuprate.embed \
  ROOT=results_gpt N=5 U=1.0 T=0.2400 \
  SEED_SET=seed_sets/block.txt
```

`ROOT/N/U/T` remain explicit. `SEED_SET` is shared by LCE and embed and
defines which main `results.json` files are used.

## Seed File

`SEED_SET` is a plain txt file under `ROOT`, one relative path per line:

```text
block_main/N_2_nelec_2_U_1.0000_t_0.2400/mode_twoSz_0/workflow_occ/results.json
block_main/N_3_nelec_3_U_1.0000_t_0.2400/mode_twoSz_1/workflow_occ/results.json
block_main/N_4_nelec_4_U_1.0000_t_0.2400/mode_twoSz_0/workflow_occ/results.json
block_main/N_5_nelec_5_U_1.0000_t_0.2400/mode_twoSz_1/workflow_occ/results.json
```

Rules:

- Empty lines and `#` comments are ignored.
- Paths are resolved as `ROOT / line`.
- Each seed must be a complete main `spin_couplings` output.
- Seed `run_params.N` must cover exactly `2..N`.
- Seed `run_params.U/T` must match CLI `U/T`.
- `MODE`, `workflow`, `twoSz`, `twoS`, and `SCOPE` do not need to match across
  seeds; they are recorded as provenance only.

## Output Layout

Use the seed filename stem directly as the seed label:

```text
seed_sets/block.txt -> seed_block
```

LCE output:

```text
ROOT/block_lce/N_5_nelec_5_U_1.0000_t_0.2400/seed_block/
  lce_results.json
  lce_summary.txt
  weights/
    N_2/
      hole0_class0_idx0.json
      hole0_class0_idx0.txt
    N_3/
      ...
    N_4/
      ...
    N_5/
      ...
```

Embed output:

```text
ROOT/block_embed/N_5_nelec_5_U_1.0000_t_0.2400/seed_block/
  embed_results.json
  embed_summary.txt
  two_site.txt
  clusters/
    N4_hole0_class0_idx0.txt
    ...
```

LCE outputs all weights for `N=2..Nmax`, because the `Nmax` calculation already
computes the full bottom-up expansion.

## Implementation Changes

- `src/cuprate/cli.py`: add `SEED_SET` parsing for LCE/embed; keep main
  unchanged.
- `src/cuprate/paths.py`: add helpers for `seed_token(seed_set)` and
  seed-based stage output directories.
- `src/cuprate/lce.py`: read seed txt, validate `N=2..Nmax` and `U/T`, remove
  global equality checks for `MODE/workflow/twoSz/twoS/SCOPE`, and write under
  `seed_*`.
- `src/cuprate/embed.py`: use the same seed txt to locate LCE output and write
  embed output under the same `seed_*` token.
- `src/cuprate/io.py`: bump LCE/embed manifest schemas, record seed-set
  provenance and per-source `run_params`, and write LCE weights under
  `weights/N_<N>/`.
- Update standards, README, and agent guidance to describe the seed-set
  runtime contract.

## Test Plan

Create seed files for `T=0.2400`:

```text
seed_sets/full_occ.txt
seed_sets/full_adiabatic.txt
seed_sets/block_occ.txt
seed_sets/block_adiabatic.txt
```

Run:

```bash
python -m compileall -q src/cuprate

mpirun -np 4 env PYTHONPATH=src python -m cuprate.lce ROOT=results_gpt N=5 U=1.0 T=0.2400 SEED_SET=seed_sets/full_occ.txt
mpirun -np 4 env PYTHONPATH=src python -m cuprate.embed ROOT=results_gpt N=5 U=1.0 T=0.2400 SEED_SET=seed_sets/full_occ.txt

mpirun -np 4 env PYTHONPATH=src python -m cuprate.lce ROOT=results_gpt N=5 U=1.0 T=0.2400 SEED_SET=seed_sets/block_occ.txt
mpirun -np 4 env PYTHONPATH=src python -m cuprate.embed ROOT=results_gpt N=5 U=1.0 T=0.2400 SEED_SET=seed_sets/block_occ.txt
```

Acceptance:

- `weights/N_2..N_5` exist.
- `lce_results.json` records all source inputs and per-source provenance.
- `embed_results.json` points to the matching seed-based LCE manifest.
- The old failure from `MODE=Sz twoSz=1` disappears because LCE no longer
  treats `twoSz` as a global parameter.

## Assumptions

- Do not keep the old LCE/embed `MODE/workflow/twoSz/twoS` path-inference
  interface.
- Do not normalize seed filenames beyond using their stem for `seed_<stem>`.
- Same seed filename with changed contents reuses the same output path;
  manifest hash is used for provenance and embed validation, not path
  disambiguation.
- Main output format remains unchanged.
