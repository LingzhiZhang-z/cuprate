#!/usr/bin/env python3
"""Generate truncated seed_set files for Nmax convergence experiments.

For each (workflow, Nmax, t), copies the first `Nmax-1` lines of the
existing canonical seed file `seed_sets/szs2eta2_<workflow>_T<t>.txt`
(which lists N=2..6 main results) into a new seed file
`seed_sets/szs2eta2_<workflow>_Nmax<Nmax>_T<t>.txt` (lists N=2..Nmax).

Möbius inversion in LCE makes per-N weights independent of Nmax, so the
resulting LCE+embed at Nmax<6 is mathematically self-consistent.

Nmax=6 reuses the existing untagged seed (no new file written).

Usage:
  python scripts/gen_nmax_seeds.py [--root results_szs2eta2]
  python scripts/gen_nmax_seeds.py --workflow occ --nmax 4 --t 0.04
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results_szs2eta2"

WORKFLOWS = ("occ", "greedy_multi", "adiabatic")
NMAX_VALUES = (2, 3, 4, 5)
T_VALUES = [round(0.02 * i, 2) for i in range(1, 11)]


def t_token(t: float) -> str:
    return f"{t:.4f}"


def source_seed(root: Path, workflow: str, t: float) -> Path:
    return root / "seed_sets" / f"szs2eta2_{workflow}_T{t_token(t)}.txt"


def target_seed(root: Path, workflow: str, nmax: int, t: float) -> Path:
    return root / "seed_sets" / f"szs2eta2_{workflow}_Nmax{nmax}_T{t_token(t)}.txt"


def truncate_seed(src: Path, nmax: int, dst: Path) -> bool:
    """Read src, keep N=2..nmax lines, write to dst.
    Returns True if a file was written, False if skipped (src missing)."""
    if not src.is_file():
        return False
    lines = [line.strip() for line in src.read_text().splitlines() if line.strip()]
    keep: list[str] = []
    for n in range(2, nmax + 1):
        prefix = f"block_main/N_{n}_"
        match = next((line for line in lines if line.startswith(prefix)), None)
        if match is None:
            return False
        keep.append(match)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(keep) + "\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workflow", nargs="*", default=None, choices=WORKFLOWS)
    parser.add_argument("--nmax", type=int, nargs="*", default=None)
    parser.add_argument("--t", type=float, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    workflows = tuple(args.workflow) if args.workflow else WORKFLOWS
    nmaxes = tuple(args.nmax) if args.nmax else NMAX_VALUES
    ts = args.t if args.t else T_VALUES

    written = skipped = 0
    for workflow in workflows:
        for t in ts:
            src = source_seed(args.root, workflow, t)
            for nmax in nmaxes:
                dst = target_seed(args.root, workflow, nmax, t)
                if args.dry_run:
                    exists = "exists" if src.is_file() else "MISSING"
                    print(f"[dry] {src.name} ({exists}) -> {dst.name} (N=2..{nmax})")
                    continue
                if truncate_seed(src, nmax, dst):
                    written += 1
                else:
                    print(f"[skip] cannot create {dst.relative_to(args.root)} (source missing or incomplete)")
                    skipped += 1

    if not args.dry_run:
        print(f"Wrote {written} seed files; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
