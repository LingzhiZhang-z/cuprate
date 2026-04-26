#!/usr/bin/env python3
"""Restartable regression runner per standards/test.md.

Drives `python -m cuprate.main` across the data_test matrix:
    N in {2,3,4,5}
    T in {0.02, 0.04, ..., 0.60}
    MODE in {full, Sz(twoSz=0 even N | twoSz=1 odd N)}
    workflow in {occ, greedy_multi, adiabatic}

Each case is launched as `mpirun -np 4 python -m cuprate.main ROOT=<root> ...`.
Status is tracked in <root>/_status/status.jsonl; a passed record short-circuits
re-runs. Layer A (schema validation) is checked inline as each case finishes.

Layer B/C/D (numerical comparison vs data_test/) are out of scope for this
runner; they belong in a separate validator that consumes <root> outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cuprate.io import SPIN_COUPLINGS_SCHEMA_VERSION  # noqa: E402
from cuprate.paths import (  # noqa: E402
    RESULTS_FILE,
    STAGE_MAIN,
    family_clusters_file,
    family_exchange_file,
    family_projection_file,
    workflow_dir,
)


U_VALUE = 1.0
T_VALUES: list[float] = [round(0.02 * i, 2) for i in range(1, 31)]
N_VALUES: tuple[int, ...] = (2, 3, 4, 5)
MODES: tuple[str, ...] = ("full", "Sz")
WORKFLOWS: tuple[str, ...] = ("occ", "greedy_multi", "adiabatic")


@dataclass(frozen=True)
class Case:
    N: int
    T: float
    mode: str
    twoSz: int | None
    workflow: str

    @property
    def case_id(self) -> str:
        mode_part = self.mode if self.twoSz is None else f"{self.mode}-twoSz{self.twoSz}"
        return f"N{self.N}_T{self.T:.4f}_MODE-{mode_part}_workflow-{self.workflow}"

    def results_path(self, root: Path) -> Path:
        return (
            workflow_dir(
                root,
                STAGE_MAIN,
                self.N,
                self.N,
                U_VALUE,
                self.T,
                self.mode,
                self.workflow,
                twoSz=self.twoSz,
            )
            / RESULTS_FILE
        )


@dataclass
class StatusIndex:
    """In-memory tail of status.jsonl: latest record per case_id."""

    path: Path
    latest: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "StatusIndex":
        index = cls(path=path)
        if not path.exists():
            return index
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = record.get("case_id")
                if isinstance(cid, str):
                    index.latest[cid] = record
        return index

    def append(self, record: dict) -> None:
        cid = record["case_id"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        self.latest[cid] = record

    def passed(self, case_id: str) -> bool:
        record = self.latest.get(case_id)
        return bool(record and record.get("status") == "passed")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enumerate_cases(n_filter: tuple[int, ...] | None = None) -> list[Case]:
    cases: list[Case] = []
    ns = n_filter if n_filter else N_VALUES
    for N in ns:
        for T in T_VALUES:
            for mode in MODES:
                twoSz = None if mode == "full" else (0 if N % 2 == 0 else 1)
                for workflow in WORKFLOWS:
                    cases.append(Case(N=N, T=T, mode=mode, twoSz=twoSz, workflow=workflow))
    return cases


def build_command(case: Case, root: Path, np_workers: int, seed_path: Path | None) -> list[str]:
    cmd: list[str] = [
        "mpirun",
        "-np",
        str(np_workers),
        sys.executable,
        "-m",
        "cuprate.main",
        f"ROOT={root}",
        f"N={case.N}",
        f"U={U_VALUE}",
        f"T={case.T:.4f}",
        f"MODE={case.mode}",
        f"workflow={case.workflow}",
    ]
    if case.twoSz is not None:
        cmd.append(f"twoSz={case.twoSz}")
    cmd.append(f"CACHE_MODE={'save' if case.workflow == 'occ' else 'load'}")
    if case.workflow == "adiabatic":
        if seed_path is None:
            raise RuntimeError(f"missing seed path for adiabatic case {case.case_id}")
        cmd.append(f"SEED_RESULTS={seed_path}")
    return cmd


def adiabatic_seed_path(case: Case, root: Path) -> Path:
    """T=0.02 -> same-(N,mode) occ result; T>=0.04 -> previous-T adiabatic result."""
    if case.workflow != "adiabatic":
        raise ValueError("adiabatic_seed_path only valid for adiabatic workflow")
    seed_workflow = "occ" if abs(case.T - 0.02) < 1e-9 else "adiabatic"
    seed_T = 0.02 if seed_workflow == "occ" else round(case.T - 0.02, 2)
    seed_case = Case(
        N=case.N,
        T=seed_T,
        mode=case.mode,
        twoSz=case.twoSz,
        workflow=seed_workflow,
    )
    return seed_case.results_path(root)


def validate_layer_a(case: Case, root: Path) -> None:
    """Raise ValueError if Layer A schema invariants are violated."""
    results_path = case.results_path(root)
    if not results_path.is_file():
        raise ValueError(f"missing results.json: {results_path}")

    payload = json.loads(results_path.read_text())
    if payload.get("result_kind") != "spin_couplings":
        raise ValueError(f"result_kind != spin_couplings in {results_path}")
    if int(payload.get("schema_version", 0)) != SPIN_COUPLINGS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version != {SPIN_COUPLINGS_SCHEMA_VERSION} in {results_path}"
        )
    if payload.get("complete_family_set") is not True:
        raise ValueError(f"complete_family_set != True in {results_path}")

    rp = payload.get("run_params") or {}
    expected_pairs = [
        ("N", case.N),
        ("U", U_VALUE),
        ("T", case.T),
        ("MODE", case.mode),
        ("workflow", case.workflow),
    ]
    for key, expected in expected_pairs:
        actual = rp.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-9:
                raise ValueError(f"run_params.{key}={actual} != {expected}")
        elif actual != expected:
            raise ValueError(f"run_params.{key}={actual!r} != {expected!r}")
    if case.twoSz is None:
        if rp.get("twoSz") is not None:
            raise ValueError(f"run_params.twoSz expected None, got {rp.get('twoSz')}")
    elif int(rp.get("twoSz", -999)) != case.twoSz:
        raise ValueError(f"run_params.twoSz={rp.get('twoSz')} != {case.twoSz}")

    families = payload.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError(f"missing or empty families in {results_path}")

    family_dir = results_path.parent
    for family in families:
        hole = int(family["hole"])
        class_idx = int(family["class_idx"])
        ex_rel = family.get("exchange_file")
        cl_rel = family.get("clusters_file")
        if not isinstance(ex_rel, str) or not isinstance(cl_rel, str):
            raise ValueError(f"family hole={hole} class={class_idx} missing file refs")
        if Path(ex_rel).name != family_exchange_file(hole, class_idx):
            raise ValueError(f"unexpected exchange filename: {ex_rel}")
        if Path(cl_rel).name != family_clusters_file(hole, class_idx):
            raise ValueError(f"unexpected clusters filename: {cl_rel}")
        ex_path = family_dir / ex_rel
        cl_path = family_dir / cl_rel
        if not ex_path.is_file():
            raise ValueError(f"missing exchange file: {ex_path}")
        if not cl_path.is_file():
            raise ValueError(f"missing clusters file: {cl_path}")
        ex_payload = json.loads(ex_path.read_text())
        for required in ("projection", "operators", "fit"):
            if required not in ex_payload:
                raise ValueError(f"{ex_path} missing {required}")

        artifact_rel = ex_payload["projection"].get("artifact")
        if not isinstance(artifact_rel, str) or not artifact_rel:
            raise ValueError(f"{ex_path} projection missing artifact")
        if Path(artifact_rel).name != family_projection_file(hole, class_idx):
            raise ValueError(f"unexpected projection artifact name: {artifact_rel}")
        artifact_path = family_dir / artifact_rel
        if not artifact_path.is_file():
            raise ValueError(f"missing projection artifact: {artifact_path}")
        _validate_projection_npz(artifact_path)

    if case.workflow == "adiabatic":
        if not rp.get("SEED_RESULTS"):
            raise ValueError(f"adiabatic run missing SEED_RESULTS in {results_path}")
        adiabatic_seed = rp.get("adiabatic_seed") or {}
        seed_workflow = adiabatic_seed.get("run_params", {}).get("workflow")
        expected_seed_workflow = "occ" if abs(case.T - 0.02) < 1e-9 else "adiabatic"
        if seed_workflow != expected_seed_workflow:
            raise ValueError(
                f"adiabatic_seed.run_params.workflow={seed_workflow!r} "
                f"!= expected {expected_seed_workflow!r} for T={case.T}"
            )


def _validate_projection_npz(path: Path) -> None:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        required_top = {"n_blocks", "block_labels", "twoSz", "twoS", "t11_minus_1_norm"}
        missing = sorted(required_top - set(data.files))
        if missing:
            raise ValueError(f"projection npz {path} missing {', '.join(missing)}")
        n_blocks = int(np.asarray(data["n_blocks"]).item())
        for idx in range(n_blocks):
            for stem in ("Heff", "selected_indices", "basis_states", "eigvecs_fock"):
                key = f"block_{idx}_{stem}"
                if key not in data.files:
                    raise ValueError(f"projection npz {path} missing {key}")


def run_case(
    case: Case,
    root: Path,
    np_workers: int,
    timeout_s: float,
) -> tuple[bool, str]:
    if case.workflow == "adiabatic":
        seed_path = adiabatic_seed_path(case, root)
        if not seed_path.is_file():
            return False, f"adiabatic seed missing: {seed_path}"
    else:
        seed_path = None

    cmd = build_command(case, root, np_workers, seed_path)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(SRC_DIR))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s: {' '.join(cmd)}"
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-20:]
        return False, "subprocess failed (rc={}): {}\n{}".format(
            proc.returncode, " ".join(cmd), "\n".join(tail)
        )

    try:
        validate_layer_a(case, root)
    except ValueError as exc:
        return False, f"layer A failed: {exc}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_opus")
    parser.add_argument("--np", dest="np_workers", type=int, default=4)
    parser.add_argument(
        "--n",
        dest="n_filter",
        type=int,
        nargs="*",
        default=None,
        help="restrict to these N values (default: all of 2,3,4,5)",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="abort the whole run on the first failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate cases and exit",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    status_dir = root / "_status"
    status_path = status_dir / "status.jsonl"
    failures_path = status_dir / "failures.jsonl"
    summary_path = status_dir / "summary.json"

    n_filter = tuple(args.n_filter) if args.n_filter else None
    cases = enumerate_cases(n_filter=n_filter)

    if args.dry_run:
        for case in cases:
            print(case.case_id)
        print(f"# total cases: {len(cases)}")
        return 0

    status = StatusIndex.load(status_path)
    root.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[tuple[int, str, int | None], list[Case]] = {}
    for case in cases:
        by_group.setdefault((case.N, case.mode, case.twoSz), []).append(case)

    n_passed = 0
    n_skipped = 0
    n_failed = 0
    failures: list[dict] = []

    for group_key, group_cases in sorted(by_group.items()):
        group_cases.sort(
            key=lambda c: (c.T, WORKFLOWS.index(c.workflow))
        )
        for case in group_cases:
            if status.passed(case.case_id):
                n_skipped += 1
                continue

            results_path = case.results_path(root)
            if results_path.is_file():
                try:
                    validate_layer_a(case, root)
                    record = {
                        "case_id": case.case_id,
                        "stage": "main",
                        "status": "passed",
                        "started_at": now_iso(),
                        "finished_at": now_iso(),
                        "results_json": str(results_path),
                        "note": "validated existing output",
                    }
                    status.append(record)
                    n_passed += 1
                    print(f"[pass-existing] {case.case_id}")
                    continue
                except ValueError as exc:
                    print(
                        f"[stale] {case.case_id} -> existing output failed Layer A: {exc}"
                    )

            started = now_iso()
            print(f"[run] {case.case_id}")
            ok, message = run_case(case, root, args.np_workers, args.timeout)
            finished = now_iso()
            record = {
                "case_id": case.case_id,
                "stage": "main",
                "status": "passed" if ok else "failed",
                "started_at": started,
                "finished_at": finished,
                "results_json": str(results_path),
                "message": message,
            }
            status.append(record)
            if ok:
                n_passed += 1
                print(f"[pass] {case.case_id}")
            else:
                n_failed += 1
                failure = {
                    "case_id": case.case_id,
                    "results_json": str(results_path),
                    "command": build_command(
                        case,
                        root,
                        args.np_workers,
                        adiabatic_seed_path(case, root)
                        if case.workflow == "adiabatic"
                        else None,
                    ),
                    "message": message,
                }
                failures.append(failure)
                failures_path.parent.mkdir(parents=True, exist_ok=True)
                with failures_path.open("a") as fh:
                    fh.write(json.dumps(failure) + "\n")
                print(f"[FAIL] {case.case_id}: {message}")
                if args.stop_on_fail:
                    summary = {
                        "passed": n_passed,
                        "skipped": n_skipped,
                        "failed": n_failed,
                        "stopped_on_fail": True,
                        "total": len(cases),
                    }
                    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                    return 1
                if case.workflow == "adiabatic":
                    print(
                        f"[chain-break] later T values for "
                        f"(N={case.N}, mode={case.mode}, twoSz={case.twoSz}) "
                        "may be skipped due to missing seed"
                    )

    summary = {
        "passed": n_passed,
        "skipped": n_skipped,
        "failed": n_failed,
        "total": len(cases),
        "schema_version_expected": SPIN_COUPLINGS_SCHEMA_VERSION,
        "root": str(root),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
