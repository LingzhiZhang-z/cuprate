"""CLI entry point for `python -m cuprate.main`."""

from __future__ import annotations

import sys
from pathlib import Path

from cuprate.paths import mode_spec
from cuprate.workchain import WorkchainParams, run_workchain


CANONICAL_KEYS = {
    "n": "N",
    "u": "U",
    "t": "T",
    "mode": "MODE",
    "twosz": "twoSz",
    "twos": "twoS",
    "workflow": "workflow",
    "root": "ROOT",
    "cache_mode": "CACHE_MODE",
    "ratio": "RATIO",
    "n_trials": "N_TRIALS",
    "max_failures": "MAX_FAILURES",
    "seed_results": "SEED_RESULTS",
}

REQUIRED_KEYS = {"N", "U", "T", "MODE", "workflow", "ROOT"}
SUPPORTED_WORKFLOWS = {"occ", "energy", "greedy", "greedy_multi", "adiabatic"}


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
    raw: dict[str, str] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"expected KEY=VALUE argument, got {arg!r}")
        key, value = arg.split("=", 1)
        if not key:
            raise ValueError(f"empty key in argument {arg!r}")
        canonical = CANONICAL_KEYS.get(key.lower())
        if canonical is None:
            raise ValueError(f"unknown parameter {key!r}")
        if canonical in raw:
            raise ValueError(f"duplicate parameter {canonical}")
        raw[canonical] = value

    missing = sorted(REQUIRED_KEYS - raw.keys())
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}")

    N = _parse_int(raw["N"], "N")
    U = _parse_float(raw["U"], "U")
    t = _parse_float(raw["T"], "T")
    twoSz = _parse_optional_int(raw, "twoSz")
    twoS = _parse_optional_int(raw, "twoS")
    spec = mode_spec(raw["MODE"], twoSz=twoSz, twoS=twoS)
    mode = spec.mode

    workflow = raw["workflow"].lower()
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(f"unsupported workflow={raw['workflow']!r}")

    _validate_mode_args(N, spec)

    cache_mode = raw.get("CACHE_MODE", "none").lower()
    if cache_mode not in {"none", "load", "save", "partial"}:
        raise ValueError(f"unsupported CACHE_MODE={cache_mode!r}")

    ratio = _parse_optional_int(raw, "RATIO")
    n_trials = _parse_optional_int(raw, "N_TRIALS")
    max_failures = _parse_optional_int(raw, "MAX_FAILURES")
    for key, value in (("RATIO", ratio), ("N_TRIALS", n_trials), ("MAX_FAILURES", max_failures)):
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if workflow not in {"greedy", "greedy_multi"} and ratio is not None:
        raise ValueError("RATIO applies only to workflow=greedy or workflow=greedy_multi")
    if workflow != "greedy_multi" and (n_trials is not None or max_failures is not None):
        raise ValueError("N_TRIALS and MAX_FAILURES apply only to workflow=greedy_multi")

    seed_results = Path(raw["SEED_RESULTS"]) if "SEED_RESULTS" in raw else None
    if workflow == "adiabatic" and seed_results is None:
        raise ValueError("SEED_RESULTS is required when workflow=adiabatic")
    if workflow != "adiabatic" and seed_results is not None:
        raise ValueError("SEED_RESULTS applies only to workflow=adiabatic")

    return WorkchainParams(
        N=N,
        U=U,
        t=t,
        mode=mode,
        twoSz=spec.twoSz,
        twoS=spec.twoS,
        workflow=workflow,
        root=Path(raw["ROOT"]),
        cache_mode=cache_mode,
        ratio=ratio,
        n_trials=n_trials,
        max_failures=max_failures,
        seed_results=seed_results,
    )


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float") from exc


def _parse_optional_int(raw: dict[str, str], key: str) -> int | None:
    if key not in raw:
        return None
    return _parse_int(raw[key], key)


def _validate_mode_args(N: int, spec) -> None:
    if spec.twoSz is not None:
        if abs(spec.twoSz) > N:
            raise ValueError("twoSz must satisfy |twoSz| <= N")
        if spec.twoSz % 2 != N % 2:
            raise ValueError("twoSz parity must match N")
    if spec.twoS is not None:
        if not (0 <= spec.twoS <= N):
            raise ValueError("twoS must satisfy 0 <= twoS <= N")
        if spec.twoS % 2 != N % 2:
            raise ValueError("twoS parity must match N")
        if spec.twoSz is not None and abs(spec.twoSz) > spec.twoS:
            raise ValueError("twoSz and twoS must satisfy |twoSz| <= twoS")


if __name__ == "__main__":
    raise SystemExit(main())
