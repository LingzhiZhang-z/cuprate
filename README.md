# cuprate

`cuprate` computes effective spin couplings from single-band Hubbard clusters.
The active production pipeline is:

```text
main  ->  lce  ->  embed
```

## Experimental Memory-Optimized Version

This branch is an experimental memory-optimized version and is not recommended
for general scientific production calculations.

It was changed for rapid deployment under tight memory limits. The current
implementation applies aggressive memory release around diagonalization and
cache handling, which reduces runtime stability and downstream artifact
guarantees. In this version, the intended correctness guarantee is limited to
the cached eigensystem data: `eigvals`, `eigvecs`, and the saved symmetry
transforms.

- `main` runs ED, projection, downfolding, and spin-coupling fitting for the
  complete cluster-family set at a given `N, U, T`.
- `lce` reads a `SEED_SET` text file listing main outputs for consecutive
  `N=2..Nmax` and performs linked-cluster subtraction.
- `embed` uses the same `SEED_SET`, reads the matching LCE weights, and embeds
  the net couplings onto target two-site and multi-site outputs.

## How To Run

Install the package from the repository root:

```bash
python -m pip install -e .
```

If the package is not installed, prefix commands with `PYTHONPATH=src`.

Minimal three-stage run:

```bash
PYTHONPATH=src python -m cuprate.main N=2 U=1 T=0.02
mkdir -p results/seed_sets
printf "block_main/N_2_nelec_2_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json\n" > results/seed_sets/full_occ.txt
PYTHONPATH=src python -m cuprate.lce N=2 U=1 T=0.02 SEED_SET=seed_sets/full_occ.txt
PYTHONPATH=src python -m cuprate.embed N=2 U=1 T=0.02 SEED_SET=seed_sets/full_occ.txt
```

For `N=4`, LCE requires main outputs for `N=2,3,4`:

```bash
PYTHONPATH=src python -m cuprate.main N=2 U=1 T=0.02
PYTHONPATH=src python -m cuprate.main N=3 U=1 T=0.02
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.02
mkdir -p results/seed_sets
printf "%s\n" \
  "block_main/N_2_nelec_2_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json" \
  "block_main/N_3_nelec_3_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json" \
  "block_main/N_4_nelec_4_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json" \
  > results/seed_sets/full_occ.txt
PYTHONPATH=src python -m cuprate.lce N=4 U=1 T=0.02 SEED_SET=seed_sets/full_occ.txt
PYTHONPATH=src python -m cuprate.embed N=4 U=1 T=0.02 SEED_SET=seed_sets/full_occ.txt
```

Run the main stage with MPI:

```bash
mpirun -n 4 env PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.02
```

## CLI Reference

All entry points use `KEY=VALUE` arguments. CLI keys are case-insensitive,
but the canonical spellings below are used in output metadata.

### Shared Parameters

| Key | Applies to | Meaning |
|-----|------------|---------|
| `N` | all stages | Cluster size. In `main` it is the solved cluster size; in `lce` and `embed` it is `Nmax`. Required. |
| `U` | all stages | On-site Hubbard repulsion. Required. |
| `T` | all stages | Nearest-neighbor hopping amplitude used by the runtime path. Required. |
| `ROOT` | all stages | Output root directory. Default: `results`. |

### Main Stage Parameters

| Key | Meaning |
|-----|---------|
| `MODE` | Diagonalization block layer: `full`, `Sz`, `SzS2`, or `SzS2eta2`. Default: `full`. |
| `twoSz` | Optional fixed `2S_z` block selector. Must have the same parity as `N`. |
| `twoS` | Optional fixed `2S` selector. Requires `MODE=SzS2` or `MODE=SzS2eta2` and fixed `twoSz`. |
| `SCOPE` | Sector range for all-`twoSz` `MODE=Sz`, `MODE=SzS2`, and `MODE=SzS2eta2` runs. `nonnegative` builds `twoSz >= 0`; `pm` builds positive and negative sectors. Default: `nonnegative`. |
| `workflow` | Eigenstate selection method: `occ`, `energy`, `greedy`, `greedy_multi`, or `adiabatic`. Default: `occ`. |
| `CACHE_MODE` | Eigensystem cache policy: `none`, `load`, `save`, or `partial`. Default: `save`. |
| `SEED_RESULTS` | Previous main-stage `results.json`. Required only for `workflow=adiabatic`. |
| `RATIO` | Optional search-pool multiplier for `workflow=greedy` or `workflow=greedy_multi`. |
| `N_TRIALS` | Optional trial count for `workflow=greedy_multi`. |
| `MAX_FAILURES` | Optional early-stop failure limit for `workflow=greedy_multi`. |
| `MERGE` | Optional pre-projection merge target. Current supported value: `Sz`. Default: `none`. |
| `MERGE_BASIS` | Merge representation for `MERGE=Sz`: `fock` or `block`. Default when merging: `fock`. Do not set it when `MERGE=none`. |

`MERGE=Sz` is accepted only for fixed-`twoSz` `MODE=SzS2` or
`MODE=SzS2eta2` runs without fixed `twoS`. It merges all selected `twoS`
sectors at that fixed `twoSz` before projection:

- `MERGE_BASIS=fock`: stores merged eigenvectors directly in fixed-`twoSz`
  Fock row coordinates.
- `MERGE_BASIS=block`: keeps block-coordinate eigenvectors internally and uses
  a column-concatenated `basis_transform`; projection artifacts still store
  `eigvecs_fock`.

### LCE and Embed Parameters

`cuprate.lce` and `cuprate.embed` take exactly `N`, `U`, `T`, optional `ROOT`,
and `SEED_SET`.

`SEED_SET` is a text file under `ROOT`. Each non-comment line is a main-stage
`results.json` path relative to `ROOT`. For `N=4`, for example, the seed set
must list the `N=2`, `N=3`, and `N=4` main outputs with matching `U`, `T`, and
run configuration.

### Common Main-Stage Examples

```bash
# Full Fock-space ED, default occ selection.
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.04

# Fixed twoSz block.
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.04 MODE=Sz twoSz=0

# All S2 sectors at fixed twoSz, then merge to one fixed-Sz block before projection.
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.04 MODE=SzS2 twoSz=0 MERGE=Sz MERGE_BASIS=fock

# Eta-refined S2 sectors at fixed twoSz, merged in block coordinates.
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.04 MODE=SzS2eta2 twoSz=0 MERGE=Sz MERGE_BASIS=block

# Greedy multi selection with explicit search parameters.
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.04 MODE=SzS2eta2 workflow=greedy_multi RATIO=8 N_TRIALS=40 MAX_FAILURES=4

# Adiabatic selection from a previous completed main result.
PYTHONPATH=src python -m cuprate.main N=4 U=1 T=0.08 MODE=SzS2eta2 workflow=adiabatic SEED_RESULTS=results/.../results.json
```

Output directory shape:

```text
ROOT/block_main/N_{N}_nelec_{N}_U_{U:.4f}_t_{T:.4f}/mode_*/workflow_*/
ROOT/block_lce/N_{N}_nelec_{N}_U_{U:.4f}_t_{T:.4f}/seed_<stem>/
ROOT/block_embed/N_{N}_nelec_{N}_U_{U:.4f}_t_{T:.4f}/seed_<stem>/
```

Default all-`twoSz` output paths use `mode_twoSz` / `mode_twoSz_twoS`;
`SCOPE=pm` uses `mode_twoSz_pm` / `mode_twoSz_pm_twoS`. Fixed selectors add
the selected value, for example `mode_twoSz_0` or
`mode_twoSz_0_twoS_2`.

`MODE=SzS2eta2` adds an `_eta_0` suffix, for example
`mode_twoSz_twoS_eta_0`, `mode_twoSz_pm_twoS_eta_0`, or
`mode_twoSz_0_twoS_eta_0`. It refines each selected `(twoSz, twoS)` block to
the eta-pairing kernel.

When `MERGE=Sz` is enabled, the merge information is appended to the workflow
token:

```text
workflow_occ_merge_Sz_basis_fock
workflow_occ_merge_Sz_basis_block
workflow_greedy_multi_merge_Sz_basis_fock
```

The solve cache mirrors only the diagonalization layer:
`DATA`, `DATA_twoSz`, `DATA_twoSz_twoS`, or `DATA_twoSz_twoS_eta_0`. It stores
solved pre-merge blocks. Merged blocks are not written back to the solve cache.
`CACHE_MODE=partial` can extend an existing cache with missing concrete blocks.

## Output Meaning

### Main Stage

- `block_main/.../results.json`: main manifest.
- `block_main/.../exchanges/*.json`: fitted spin couplings per
  `(hole, class_idx)` family.
- `block_main/.../clusters/*.json`: geometry mapping for every `cluster_idx`
  in the same family.
- `block_main/.../artifacts/*.npz`: projection artifacts, selected indices,
  `H_eff`, and `T11` diagnostics.

`results.json` is a manifest, not the full data payload. Important fields:

- `schema_version`: output schema version.
- `result_kind`: `spin_couplings` for main-stage results.
- `complete_family_set`: true for current production main runs.
- `run_params`: parsed runtime parameters, path tokens, cache mode, and merge
  metadata.
- `families`: one entry per computed `(hole, class_idx)` family, with relative
  paths to its exchange and cluster-geometry JSON files.

Each `exchanges/*.json` file contains the durable fitted physics output for one
family:

- `projection.method`: the selection workflow used for this family.
- `projection.artifact`: relative path to the `.npz` projection artifact.
- `projection.blocks`: block labels, quantum numbers, selected eigenstate
  indices, spin dimension, `T11` norm, overlap, and per-block fit diagnostics.
- `operators.constant_term`: fitted constant term.
- `operators.groups`: spin-coupling terms grouped by arity and geometry. Two
  site groups are labelled `J1`, `J2`, ...; four-site groups are `K1`, `K2`,
  ...; six-site groups are `L1`, `L2`, ...
- `fit`: family-level least-squares fit metrics: relative error, residual,
  `r_squared`, max `T11` norm, and adiabatic overlap when available.
- `metadata`: runtime metadata such as MPI rank and family wall time.

Each `clusters/*.json` file maps the representative family output to every
cluster member in that family. It stores `cluster_idx`, site coordinates, and
the local site-index mapping. It does not duplicate the fitted couplings.

Each `artifacts/*.npz` file stores projection data that is useful for debugging
and adiabatic seeding:

- `block_labels`, `twoSz`, `twoS`, `eta`
- `block_<i>_basis_states`
- `block_<i>_selected_indices`
- `block_<i>_Heff`
- `block_<i>_eigvecs_fock`
- `t11_minus_1_norm`, `overlap`

For merged runs, `block_<i>_eigvecs_fock` is always in Fock row coordinates,
even when `MERGE_BASIS=block`.

### LCE Stage

- `block_lce/.../lce_results.json`: LCE manifest.
- `block_lce/.../weights/N_*/*.json`: net LCE weight for each concrete cluster.
- `block_lce/.../weights/N_*/*.txt`: human-readable sidecar with the same stem.

`lce_results.json` records the seed set, source main inputs, run parameters,
and the list of generated weight files. Each weight JSON contains the net
linked-cluster contribution for one concrete cluster, using the same operator
schema as main-stage exchange files.

### Embed Stage

- `block_embed/.../two_site.txt`: embedded two-site target couplings.
- `block_embed/.../clusters/*.txt`: embedded multi-site target couplings.
- `block_embed/.../embed_results.json`: embed manifest.
- `block_embed/.../embed_summary.txt`: summary of source LCE inputs and target
  output counts.

`embed_results.json` records the source LCE manifest, seed set metadata,
two-site output file, cluster output files, and embedding diagnostics. The
plain-text files are the main human-facing embed output.

Text files are inspection sidecars unless explicitly documented as the embed
target output. LCE and adiabatic seed loading read JSON/NPZ machine outputs.

## Diagnostics and Plotting

Helper scripts under `scripts/` consume the JSON/NPZ outputs only:

- `plot_regression.py` — per-(block, family) projection diagnostics
  (eigvals, ⟨D⟩, ‖T₁₁−I‖, relative error, optional adiabatic overlap) with
  per-combo total panels.
- `plot_jps_figures.py` — matplotlib renderer that mirrors the JPS-slide
  figure style (dominant J₁/Jc, subdominant J₂/J₃, comparison panels with the
  Cuprates t/U region marked).
- `plot_embed_vs_perturbation.py` — gnuplot renderer for embed couplings
  versus 4th-order perturbation theory: absolute meV panels for J₁/Jc, ratio
  panels for J₂/J₃ and a comparison-R panel; uses `_pert_formulas.py` as the
  user-editable reference.
- `gen_nmax_seeds.py` / `run_embed_convergence.sh` — drive an Nmax=2..6
  embed convergence series by truncating canonical seed-set files.

## Code Structure

The physics core lives in `src/cuprate/`:

- `clusters.py`: square-lattice cluster enumeration, isomorphic families,
  bonds, and multi-site patterns.
- `states.py`: Hubbard Fock basis, state ordering, double occupation,
  fermionic signs, and state-space `Sz`/`S2` operators.
- `sectors.py`: fixed-`twoSz` grouping, `S2` sector construction, and
  `twoSz -> (twoSz,twoS)` basis transforms.
- `hubbard.py`: single-cluster ED coordinator. It owns basis generation,
  symmetry blocking, Hamiltonian construction, diagonalization, optional sector
  merging, projection, and fit orchestration.
- `manifold.py`: `Block`, eigenstate selection, `T11`, `H_eff`, spin-operator
  matrices, and least-squares spin-coupling fits.
- `operators.py`: canonical spin-operator keys, JSON serialization, and
  conversion from fit coefficients to operator terms.
- `lce.py`: linked-cluster subtraction using canonical operator keys.
- `embed.py`: target-driven embedding from LCE weights to two-site and
  multi-site target outputs.

Runtime and I/O layers:

- `main.py`: CLI entry point for `python -m cuprate.main`.
- `workchain.py`: main-stage orchestration and MPI family distribution.
- `cli.py`: shared `KEY=VALUE` parsing, defaults, main-stage
  `MODE/twoSz/twoS` validation, and LCE/embed `SEED_SET` parsing.
- `paths.py`: runtime directories, filenames, mode tokens, and seed tokens.
- `io.py`: schema constants, JSON/NPZ writes, and human-readable text sidecars.
- `mpi.py`: MPI rank and communicator plumbing.

Standards live under `standards/en/`; the English standards are authoritative.
Chinese translations live under `standards/zh/`. Start with
`standards/en/00-CONVENTIONS.md` and the standard corresponding to the module
being changed.

`src/cuprate/back/` contains old reference code only. It is not active
structure and should not be used as a compatibility layer.
