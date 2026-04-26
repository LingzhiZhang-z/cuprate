#!/usr/bin/env python3
"""Layer D: LCE + embed smoke for the SzS2eta2 custom run.

For each workflow and each T in the regression matrix, write a SEED_SET from
the matching MODE=SzS2eta2 main outputs for N=2..N_max, then run cuprate.lce
and cuprate.embed through the production SEED_SET interface. Validate:
  - LCE reads consecutive N=2..N_max main outputs
  - All LCE reconstruction errors within ATOL_RECON
  - Embed writes embed_results.json, two_site.txt, and cluster sidecars

Restartable via <root>/_status/layer_d_szs2eta2.jsonl. Each (workflow, T) run
records pass/fail with a list of failed checks.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from cuprate.paths import (  # noqa: E402
    EMBED_CLUSTERS_DIR,
    EMBED_RESULTS_FILE,
    EMBED_SUMMARY_FILE,
    EMBED_TWO_SITE_FILE,
    LCE_RESULTS_FILE,
    LCE_SUMMARY_FILE,
    LCE_WEIGHTS_DIR,
    RESULTS_FILE,
    STAGE_MAIN,
    STAGE_EMBED,
    STAGE_LCE,
    seed_stage_dir,
    workflow_dir,
)

from run_data_test_regression import (  # noqa: E402
    StatusIndex,
    T_VALUES,
    U_VALUE,
    now_iso,
)


N_MAX = 6
MODE = "SzS2eta2"
SCOPE = "nonnegative"
ATOL_RECON = 1e-8

WORKFLOWS: tuple[str, ...] = ("occ", "greedy_multi", "adiabatic")


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
class LayerDCase:
    workflow: str
    T: float

    @property
    def case_id(self) -> str:
        return f"LayerD_SzS2eta2_T{self.T:.4f}_workflow-{self.workflow}"

    def seed_set_relpath(self) -> Path:
        return Path("seed_sets") / f"szs2eta2_{self.workflow}_T{self.T:.4f}.txt"


def enumerate_cases(t_filter: tuple[float, ...] | None = None) -> list[LayerDCase]:
    cases = []
    ts = t_filter if t_filter else tuple(T_VALUES)
    for workflow in WORKFLOWS:
        for T in ts:
            cases.append(LayerDCase(workflow=workflow, T=T))
    return cases


def lce_dir(case: LayerDCase, root: Path) -> Path:
    return seed_stage_dir(
        root,
        STAGE_LCE,
        N_MAX,
        N_MAX,
        U_VALUE,
        case.T,
        case.seed_set_relpath(),
    )


def embed_dir(case: LayerDCase, root: Path) -> Path:
    return seed_stage_dir(
        root,
        STAGE_EMBED,
        N_MAX,
        N_MAX,
        U_VALUE,
        case.T,
        case.seed_set_relpath(),
    )


def main_results_path(case: LayerDCase, root: Path, N: int) -> Path:
    return (
        workflow_dir(
            root,
            STAGE_MAIN,
            N,
            N,
            U_VALUE,
            case.T,
            MODE,
            case.workflow,
            scope=SCOPE,
        )
        / RESULTS_FILE
    )


def write_seed_set(case: LayerDCase, root: Path) -> tuple[bool, str]:
    seed_path = root / case.seed_set_relpath()
    rows: list[str] = []
    for N in range(2, N_MAX + 1):
        result = main_results_path(case, root, N)
        if not result.is_file():
            return False, f"missing main result for seed set: {result}"
        rows.append(result.relative_to(root).as_posix())
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text("\n".join(rows) + "\n")
    return True, str(seed_path)


def build_lce_command(case: LayerDCase, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cuprate.lce",
        f"ROOT={root}",
        f"N={N_MAX}",
        f"U={U_VALUE}",
        f"T={case.T:.4f}",
        f"SEED_SET={case.seed_set_relpath().as_posix()}",
    ]


def build_embed_command(case: LayerDCase, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cuprate.embed",
        f"ROOT={root}",
        f"N={N_MAX}",
        f"U={U_VALUE}",
        f"T={case.T:.4f}",
        f"SEED_SET={case.seed_set_relpath().as_posix()}",
    ]


def run_command(cmd: list[str], timeout_s: float) -> tuple[bool, str]:
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
        return False, f"timeout after {timeout_s}s"
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-12:]
        return False, "rc={}\n{}".format(proc.returncode, "\n".join(tail))
    return True, "ok"


def validate_lce(case: LayerDCase, root: Path) -> list[str]:
    """Returns list of failure descriptions; empty list = pass."""
    failures: list[str] = []
    out_dir = lce_dir(case, root)
    manifest_path = out_dir / LCE_RESULTS_FILE
    summary_path = out_dir / LCE_SUMMARY_FILE
    weights_dir = out_dir / LCE_WEIGHTS_DIR

    if not manifest_path.is_file():
        failures.append(f"missing {manifest_path}")
        return failures
    payload = json.loads(manifest_path.read_text())
    if payload.get("result_kind") != "lce_spin_couplings":
        failures.append(f"result_kind != lce_spin_couplings ({payload.get('result_kind')!r})")
    rp = payload.get("run_params", {})
    n_min = rp.get("N_min")
    n_max = rp.get("N_max")
    if n_min != 2 or n_max != N_MAX:
        failures.append(f"N range != [2, {N_MAX}] (got [{n_min}, {n_max}])")
    if not summary_path.is_file():
        failures.append(f"missing {summary_path}")
    if not weights_dir.is_dir():
        failures.append(f"missing weights dir {weights_dir}")
        return failures

    weights = payload.get("weights", [])
    max_recon = 0.0
    max_recon_loc = ""
    for entry in weights:
        weight_rel = entry["weight_file"]
        weight_path = out_dir / weight_rel
        if not weight_path.is_file():
            failures.append(f"missing weight file {weight_path}")
            continue
        weight_payload = json.loads(weight_path.read_text())
        recon = float(weight_payload["diagnostics"]["reconstruction_error"])
        if recon > max_recon:
            max_recon = recon
            max_recon_loc = (
                f"N={weight_payload['N']} hole={weight_payload['hole']} "
                f"class={weight_payload['class_idx']} cluster={weight_payload['cluster_idx']}"
            )
    if max_recon > ATOL_RECON:
        failures.append(
            f"max reconstruction_error={max_recon:.3e} > {ATOL_RECON} at {max_recon_loc}"
        )
    return failures


def validate_embed(case: LayerDCase, root: Path) -> list[str]:
    failures: list[str] = []
    out_dir = embed_dir(case, root)
    manifest_path = out_dir / EMBED_RESULTS_FILE
    if not manifest_path.is_file():
        failures.append(f"missing {manifest_path}")
        return failures
    payload = json.loads(manifest_path.read_text())
    if payload.get("result_kind") != "embedded_spin_couplings":
        failures.append(f"result_kind != embedded_spin_couplings ({payload.get('result_kind')!r})")
    if not (out_dir / EMBED_TWO_SITE_FILE).is_file():
        failures.append(f"missing {EMBED_TWO_SITE_FILE}")
    if not (out_dir / EMBED_SUMMARY_FILE).is_file():
        failures.append(f"missing {EMBED_SUMMARY_FILE}")
    clusters_dir = out_dir / EMBED_CLUSTERS_DIR
    if not clusters_dir.is_dir():
        failures.append(f"missing clusters dir {clusters_dir}")
        return failures
    declared = payload.get("cluster_files", [])
    for entry in declared:
        cluster_rel = entry.get("file")
        if not cluster_rel:
            failures.append(f"cluster entry missing 'file': {entry}")
            continue
        if not (out_dir / cluster_rel).is_file():
            failures.append(f"missing cluster sidecar {cluster_rel}")
    return failures


def run_case(case: LayerDCase, root: Path, timeout_s: float) -> dict:
    started = now_iso()
    started_perf = time.perf_counter()
    seed_ok, seed_msg = write_seed_set(case, root)
    if not seed_ok:
        return {
            "case_id": case.case_id,
            "stage": "layer_d",
            "status": "failed",
            "started_at": started,
            "finished_at": now_iso(),
            "duration_s": time.perf_counter() - started_perf,
            "step": "seed_set",
            "message": seed_msg,
        }
    lce_ok, lce_msg = run_command(build_lce_command(case, root), timeout_s)
    if not lce_ok:
        return {
            "case_id": case.case_id,
            "stage": "layer_d",
            "status": "failed",
            "started_at": started,
            "finished_at": now_iso(),
            "duration_s": time.perf_counter() - started_perf,
            "step": "lce",
            "message": lce_msg,
        }
    lce_failures = validate_lce(case, root)
    if lce_failures:
        return {
            "case_id": case.case_id,
            "stage": "layer_d",
            "status": "failed",
            "started_at": started,
            "finished_at": now_iso(),
            "duration_s": time.perf_counter() - started_perf,
            "step": "lce_validate",
            "failures": lce_failures,
        }
    embed_ok, embed_msg = run_command(build_embed_command(case, root), timeout_s)
    if not embed_ok:
        return {
            "case_id": case.case_id,
            "stage": "layer_d",
            "status": "failed",
            "started_at": started,
            "finished_at": now_iso(),
            "duration_s": time.perf_counter() - started_perf,
            "step": "embed",
            "message": embed_msg,
        }
    embed_failures = validate_embed(case, root)
    if embed_failures:
        return {
            "case_id": case.case_id,
            "stage": "layer_d",
            "status": "failed",
            "started_at": started,
            "finished_at": now_iso(),
            "duration_s": time.perf_counter() - started_perf,
            "step": "embed_validate",
            "failures": embed_failures,
        }
    return {
        "case_id": case.case_id,
        "stage": "layer_d",
        "status": "passed",
        "started_at": started,
        "finished_at": now_iso(),
        "duration_s": time.perf_counter() - started_perf,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_opus")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--t", dest="t_filter", type=float, nargs="*", default=None)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--workflow",
        dest="workflow_filter",
        nargs="*",
        default=None,
        help="restrict to workflows; default all",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    status_dir = root / "_status"
    status_path = status_dir / "layer_d_szs2eta2.jsonl"
    failures_path = status_dir / "layer_d_szs2eta2_failures.jsonl"
    summary_path = status_dir / "layer_d_szs2eta2_summary.json"

    t_filter = tuple(round(t, 2) for t in args.t_filter) if args.t_filter else None
    cases = enumerate_cases(t_filter=t_filter)
    if args.workflow_filter:
        wf = set(args.workflow_filter)
        cases = [c for c in cases if c.workflow in wf]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    status = StatusIndex.load(status_path)
    status_dir.mkdir(parents=True, exist_ok=True)

    n_pass = n_skip = n_fail = 0
    run_started_perf = time.perf_counter()
    for case in cases:
        if status.passed(case.case_id):
            n_skip += 1
            done = n_pass + n_skip + n_fail
            print(
                f"[skip] {case.case_id} {progress_summary(done, len(cases), run_started_perf)}",
                flush=True,
            )
            continue
        next_index = n_pass + n_skip + n_fail + 1
        print(f"[run {next_index}/{len(cases)}] {case.case_id}", flush=True)
        record = run_case(case, root, args.timeout)
        status.append(record)
        duration_s = float(record.get("duration_s", 0.0))
        if record["status"] == "passed":
            n_pass += 1
            done = n_pass + n_skip + n_fail
            print(
                f"[pass] {case.case_id} case_time={format_duration(duration_s)} "
                f"{progress_summary(done, len(cases), run_started_perf)}",
                flush=True,
            )
        else:
            n_fail += 1
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            label = record.get("message") or json.dumps(record.get("failures", []))[:200]
            done = n_pass + n_skip + n_fail
            print(
                f"[FAIL {record.get('step', '?')}] {case.case_id} "
                f"case_time={format_duration(duration_s)} "
                f"{progress_summary(done, len(cases), run_started_perf)}: {label[:200]}",
                flush=True,
            )
            if args.stop_on_fail:
                break

    summary = {
        "passed": n_pass,
        "skipped": n_skip,
        "failed": n_fail,
        "total": len(cases),
        "elapsed_s": time.perf_counter() - run_started_perf,
        "root": str(root),
        "mode": MODE,
        "scope": SCOPE,
        "N_max": N_MAX,
        "workflows": list(WORKFLOWS),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
