# cuprate

`cuprate` computes effective spin couplings from single-band Hubbard clusters.
The active production pipeline is:

```text
main  ->  lce  ->  embed
```

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

Common CLI parameters:

```text
N, U, T        Required physical parameters
MODE           full | Sz | SzS2 | SzS2eta2; default: full
twoSz, twoS    Optional fixed-sector selectors; twoS requires MODE=SzS2 or SzS2eta2
SCOPE          nonnegative | pm; default: nonnegative for all-twoSz Sz/SzS2/SzS2eta2 runs
workflow       occ | energy | greedy | greedy_multi | adiabatic; default: occ
ROOT           Output root directory; default: results
CACHE_MODE     none | load | save | partial; main default: save
SEED_RESULTS   Previous results.json for workflow=adiabatic
SEED_SET       LCE/embed text file under ROOT; each row is a main results.json
```

For all-`twoSz` `MODE=Sz` / `MODE=SzS2` runs, default `SCOPE=nonnegative`
builds only `twoSz >= 0` sectors. Use `SCOPE=pm` to build both positive and
negative `twoSz` sectors. Explicit selectors such as `twoSz=-2` are unaffected.

Output directory shape:

```text
ROOT/block_main/N_{N}_nelec_{N}_U_{U:.4f}_t_{T:.4f}/mode_*/workflow_*/
ROOT/block_lce/N_{N}_nelec_{N}_U_{U:.4f}_t_{T:.4f}/seed_<stem>/
ROOT/block_embed/N_{N}_nelec_{N}_U_{U:.4f}_t_{T:.4f}/seed_<stem>/
```

Default all-`twoSz` output paths use `mode_twoSz` / `mode_twoSz_twoS`; `SCOPE=pm`
uses `mode_twoSz_pm` / `mode_twoSz_pm_twoS`. `MODE=SzS2eta2` adds an `_eta_0`
suffix (e.g. `mode_twoSz_twoS_eta_0`, `mode_twoSz_pm_twoS_eta_0`) and refines
each `(twoSz, twoS)` block to the eta-pairing kernel; the solve cache mirrors
this with `DATA_twoSz_twoS_eta_0`. The solve cache remains shared:
`DATA_twoSz` / `DATA_twoSz_twoS` are keyed by concrete block labels, so
`CACHE_MODE=partial` can extend a default cache with missing negative sectors.

Key outputs:

- `block_main/.../results.json`: main manifest.
- `block_main/.../exchanges/*.json`: fitted spin couplings per
  `(hole, class_idx)` family.
- `block_main/.../clusters/*.json`: geometry mapping for every `cluster_idx`
  in the same family.
- `block_main/.../artifacts/*.npz`: projection artifacts, selected indices,
  `H_eff`, and `T11` diagnostics.
- `block_lce/.../lce_results.json`: LCE manifest.
- `block_lce/.../weights/N_*/*.json`: net LCE weight for each concrete cluster.
- `block_lce/.../weights/N_*/*.txt`: human-readable sidecar with the same stem.
- `block_embed/.../two_site.txt`: embedded two-site target couplings.
- `block_embed/.../clusters/*.txt`: embedded multi-site target couplings.

Text files are inspection sidecars only. Downstream stages read JSON/NPZ.

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
