"""CLI entry point for `python -m cuprate.main`."""

from __future__ import annotations

import sys
from pathlib import Path

from cuprate.cli import COMMON_KEYS, parse_common_runtime, parse_key_values, parse_optional_int
from cuprate.workchain import WorkchainParams, run_workchain


MAIN_KEYS = {
    "cache_mode": "CACHE_MODE",
    "ratio": "RATIO",
    "n_trials": "N_TRIALS",
    "max_failures": "MAX_FAILURES",
    "seed_results": "SEED_RESULTS",
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        params = parse_args(argv)
        run_workchain(params)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def parse_args(argv: list[str]) -> WorkchainParams:
    raw = parse_key_values(argv, {**COMMON_KEYS, **MAIN_KEYS})
    common = parse_common_runtime(raw)

    cache_mode = raw.get("CACHE_MODE", "save").lower()
    if cache_mode not in {"none", "load", "save", "partial"}:
        raise ValueError(f"unsupported CACHE_MODE={cache_mode!r}")

    ratio = parse_optional_int(raw, "RATIO")
    n_trials = parse_optional_int(raw, "N_TRIALS")
    max_failures = parse_optional_int(raw, "MAX_FAILURES")
    for key, value in (("RATIO", ratio), ("N_TRIALS", n_trials), ("MAX_FAILURES", max_failures)):
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if common.workflow not in {"greedy", "greedy_multi"} and ratio is not None:
        raise ValueError("RATIO applies only to workflow=greedy or workflow=greedy_multi")
    if common.workflow != "greedy_multi" and (n_trials is not None or max_failures is not None):
        raise ValueError("N_TRIALS and MAX_FAILURES apply only to workflow=greedy_multi")

    seed_results = Path(raw["SEED_RESULTS"]) if "SEED_RESULTS" in raw else None
    if common.workflow == "adiabatic" and seed_results is None:
        raise ValueError("SEED_RESULTS is required when workflow=adiabatic")
    if common.workflow != "adiabatic" and seed_results is not None:
        raise ValueError("SEED_RESULTS applies only to workflow=adiabatic")

    return WorkchainParams(
        N=common.N,
        U=common.U,
        t=common.t,
        mode=common.mode,
        twoSz=common.twoSz,
        twoS=common.twoS,
        scope=common.scope,
        workflow=common.workflow,
        root=common.root,
        cache_mode=cache_mode,
        ratio=ratio,
        n_trials=n_trials,
        max_failures=max_failures,
        seed_results=seed_results,
    )


if __name__ == "__main__":
    raise SystemExit(main())
