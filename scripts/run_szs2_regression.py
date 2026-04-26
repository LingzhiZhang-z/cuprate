#!/usr/bin/env python3
"""SzS2eta2 regression runner.

Drives `python -m cuprate.main` for:
    MODE=SzS2eta2 SCOPE=nonnegative
    no fixed twoSz/twoS selectors
    N in {2, 3, 4, 5, 6}
    T = 0.02..0.60 step 0.02 (30 values)
    workflow in {occ, greedy_multi, adiabatic}

Each case launched as `mpirun -np 4 python -m cuprate.main ...`. Restartable
via <root>/_status/szs2eta2.jsonl. Layer A schema validation runs inline.

Adiabatic chain: T=0.02 seeds from the same-(N,SzS2eta2) occ result; T>=0.04
seeds from the previous T's adiabatic result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from cuprate.io import SPIN_COUPLINGS_SCHEMA_VERSION  # noqa: E402
from cuprate.paths import (  # noqa: E402
    RESULTS_FILE,
    STAGE_MAIN,
    family_clusters_file,
    family_exchange_file,
    family_projection_file,
    workflow_dir,
)

from run_data_test_regression import StatusIndex, now_iso  # noqa: E402


U_VALUE = 1.0
T_VALUES: list[float] = [round(0.02 * i, 2) for i in range(1, 31)]
N_VALUES: tuple[int, ...] = (2, 3, 4, 5, 6)
WORKFLOWS: tuple[str, ...] = ("occ", "greedy_multi", "adiabatic")
MODE = "SzS2eta2"
SCOPE = "nonnegative"


def format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def progress_summary(done: int, total: int, started_perf: float) -> str:
    elapsed = time.perf_counter() - started_perf
    eta = 0.0 if done <= 0 else (elapsed / done) * (total - done)
    return (
        f"progress={done}/{total} "
        f"elapsed={format_duration(elapsed)} "
        f"ETA={format_duration(eta)}"
    )


@dataclass(frozen=True)
class Case:
    N: int
    T: float
    workflow: str

    @property
    def case_id(self) -> str:
        return f"SzS2eta2_N{self.N}_T{self.T:.4f}_workflow-{self.workflow}"

    def results_path(self, root: Path) -> Path:
        return (
            workflow_dir(
                root,
                STAGE_MAIN,
                self.N,
                self.N,
                U_VALUE,
                self.T,
                MODE,
                self.workflow,
                scope=SCOPE,
            )
            / RESULTS_FILE
        )


def enumerate_cases(
    n_filter: tuple[int, ...] | None = None,
    t_filter: tuple[float, ...] | None = None,
) -> list[Case]:
    ns = n_filter if n_filter else N_VALUES
    ts = t_filter if t_filter else tuple(T_VALUES)
    cases: list[Case] = []
    for N in ns:
        for T in ts:
            for workflow in WORKFLOWS:
                cases.append(Case(N=N, T=T, workflow=workflow))
    return cases


def adiabatic_seed_path(case: Case, root: Path) -> Path:
    seed_workflow = "occ" if abs(case.T - 0.02) < 1e-9 else "adiabatic"
    seed_T = 0.02 if seed_workflow == "occ" else round(case.T - 0.02, 2)
    return (
        workflow_dir(
            root,
            STAGE_MAIN,
            case.N,
            case.N,
            U_VALUE,
            seed_T,
            MODE,
            seed_workflow,
            scope=SCOPE,
        )
        / RESULTS_FILE
    )


def build_command(case: Case, root: Path, np_workers: int, seed_path: Path | None) -> list[str]:
    cmd = [
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
        f"MODE={MODE}",
        f"SCOPE={SCOPE}",
        f"workflow={case.workflow}",
        f"CACHE_MODE={'save' if case.workflow == 'occ' else 'load'}",
    ]
    if case.workflow == "adiabatic":
        if seed_path is None:
            raise RuntimeError(f"missing seed for {case.case_id}")
        cmd.append(f"SEED_RESULTS={seed_path}")
    return cmd


def validate_layer_a(case: Case, root: Path) -> None:
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
    expected = [
        ("N", case.N),
        ("U", U_VALUE),
        ("T", case.T),
        ("MODE", MODE),
        ("SCOPE", SCOPE),
        ("workflow", case.workflow),
    ]
    for key, want in expected:
        got = rp.get(key)
        if isinstance(want, float):
            if got is None or abs(float(got) - want) > 1e-9:
                raise ValueError(f"run_params.{key}={got!r} != {want!r}")
        elif got != want:
            raise ValueError(f"run_params.{key}={got!r} != {want!r}")
    if rp.get("twoSz") is not None or rp.get("twoS") is not None:
        raise ValueError(f"twoSz/twoS expected None for full SzS2eta2: got {rp}")
    if rp.get("mode_token") != "mode_twoSz_twoS_eta_0":
        raise ValueError(
            f"run_params.mode_token={rp.get('mode_token')!r} "
            "!= 'mode_twoSz_twoS_eta_0'"
        )

    families = payload.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError(f"missing or empty families in {results_path}")
    family_dir = results_path.parent
    for family in families:
        hole = int(family["hole"])
        class_idx = int(family["class_idx"])
        ex_path = family_dir / family["exchange_file"]
        cl_path = family_dir / family["clusters_file"]
        if Path(family["exchange_file"]).name != family_exchange_file(hole, class_idx):
            raise ValueError(f"unexpected exchange filename: {family['exchange_file']}")
        if Path(family["clusters_file"]).name != family_clusters_file(hole, class_idx):
            raise ValueError(f"unexpected clusters filename: {family['clusters_file']}")
        if not ex_path.is_file():
            raise ValueError(f"missing exchange json: {ex_path}")
        if not cl_path.is_file():
            raise ValueError(f"missing clusters json: {cl_path}")
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
        blocks = ex_payload.get("projection", {}).get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"{ex_path} projection missing blocks")
        for block in blocks:
            if block.get("eta") != 0:
                raise ValueError(f"{ex_path} block {block.get('block')!r} eta != 0")
            if "_eta_0" not in str(block.get("block")):
                raise ValueError(f"{ex_path} block label missing eta_0: {block}")
        with np.load(artifact_path) as data:
            if "eta" not in data.files:
                raise ValueError(f"{artifact_path} missing eta array")
            eta_values = data["eta"]
            if eta_values.size == 0 or not np.allclose(eta_values, 0.0):
                raise ValueError(f"{artifact_path} contains nonzero eta labels")

    if case.workflow == "adiabatic":
        if not rp.get("SEED_RESULTS"):
            raise ValueError(f"adiabatic run missing SEED_RESULTS in {results_path}")
        seed = (rp.get("adiabatic_seed") or {}).get("run_params", {})
        expected_seed_wf = "occ" if abs(case.T - 0.02) < 1e-9 else "adiabatic"
        if seed.get("workflow") != expected_seed_wf:
            raise ValueError(
                f"adiabatic_seed.workflow={seed.get('workflow')!r} != {expected_seed_wf!r}"
            )


def run_case(case: Case, root: Path, np_workers: int, timeout_s: float) -> tuple[bool, str]:
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
            stdout=None,
            stderr=subprocess.PIPE,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-15:]
        return False, "rc={}\n{}".format(proc.returncode, "\n".join(tail))
    try:
        validate_layer_a(case, root)
    except ValueError as exc:
        return False, f"layer A failed: {exc}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_opus")
    parser.add_argument("--np", dest="np_workers", type=int, default=4)
    parser.add_argument("--n", dest="n_filter", type=int, nargs="*", default=None)
    parser.add_argument("--t", dest="t_filter", type=float, nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    status_dir = root / "_status"
    status_path = status_dir / "szs2eta2.jsonl"
    failures_path = status_dir / "szs2eta2_failures.jsonl"
    summary_path = status_dir / "szs2eta2_summary.json"

    n_filter = tuple(args.n_filter) if args.n_filter else None
    t_filter = tuple(round(t, 2) for t in args.t_filter) if args.t_filter else None
    cases = enumerate_cases(n_filter=n_filter, t_filter=t_filter)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if args.dry_run:
        for c in cases:
            print(c.case_id)
        print(f"# total cases: {len(cases)}")
        return 0

    status = StatusIndex.load(status_path)
    root.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[int, list[Case]] = {}
    for case in cases:
        by_group.setdefault(case.N, []).append(case)

    n_pass = n_skip = n_fail = 0
    run_started_perf = time.perf_counter()
    for N in sorted(by_group):
        group_cases = sorted(
            by_group[N], key=lambda c: (c.T, WORKFLOWS.index(c.workflow))
        )
        for case in group_cases:
            if status.passed(case.case_id):
                n_skip += 1
                done = n_pass + n_skip + n_fail
                print(
                    f"[skip] {case.case_id} {progress_summary(done, len(cases), run_started_perf)}",
                    flush=True,
                )
                continue
            if case.results_path(root).is_file():
                case_started_perf = time.perf_counter()
                try:
                    validate_layer_a(case, root)
                    duration_s = time.perf_counter() - case_started_perf
                    status.append({
                        "case_id": case.case_id, "stage": "main",
                        "status": "passed", "started_at": now_iso(),
                        "finished_at": now_iso(),
                        "results_json": str(case.results_path(root)),
                        "note": "validated existing output",
                        "duration_s": duration_s,
                    })
                    n_pass += 1
                    done = n_pass + n_skip + n_fail
                    print(
                        f"[pass-existing] {case.case_id} "
                        f"case_time={format_duration(duration_s)} "
                        f"{progress_summary(done, len(cases), run_started_perf)}",
                        flush=True,
                    )
                    continue
                except ValueError as exc:
                    print(f"[stale] {case.case_id} -> {exc}", flush=True)
            started = now_iso()
            case_started_perf = time.perf_counter()
            next_index = n_pass + n_skip + n_fail + 1
            print(f"[run {next_index}/{len(cases)}] {case.case_id}", flush=True)
            ok, msg = run_case(case, root, args.np_workers, args.timeout)
            finished = now_iso()
            duration_s = time.perf_counter() - case_started_perf
            record = {
                "case_id": case.case_id, "stage": "main",
                "status": "passed" if ok else "failed",
                "started_at": started, "finished_at": finished,
                "results_json": str(case.results_path(root)),
                "message": msg,
                "duration_s": duration_s,
            }
            status.append(record)
            if ok:
                n_pass += 1
                done = n_pass + n_skip + n_fail
                print(
                    f"[pass] {case.case_id} case_time={format_duration(duration_s)} "
                    f"{progress_summary(done, len(cases), run_started_perf)}",
                    flush=True,
                )
            else:
                n_fail += 1
                with failures_path.open("a") as fh:
                    fh.write(json.dumps({"case_id": case.case_id, "message": msg}) + "\n")
                done = n_pass + n_skip + n_fail
                print(
                    f"[FAIL] {case.case_id} case_time={format_duration(duration_s)} "
                    f"{progress_summary(done, len(cases), run_started_perf)}: {msg[:200]}",
                    flush=True,
                )
                if args.stop_on_fail:
                    summary_path.write_text(json.dumps({
                        "passed": n_pass, "skipped": n_skip, "failed": n_fail,
                        "stopped_on_fail": True, "total": len(cases),
                        "elapsed_s": time.perf_counter() - run_started_perf,
                    }, indent=2) + "\n")
                    return 1
                if case.workflow == "adiabatic":
                    print(f"[chain-break] later T values for N={case.N} may be skipped", flush=True)

    summary = {
        "passed": n_pass,
        "skipped": n_skip,
        "failed": n_fail,
        "total": len(cases),
        "elapsed_s": time.perf_counter() - run_started_perf,
        "schema_version_expected": SPIN_COUPLINGS_SCHEMA_VERSION,
        "root": str(root),
        "mode": MODE,
        "scope": SCOPE,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
