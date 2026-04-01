import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_parse_bool_arg_and_iter_cli_assignments():
    from cuprate.io import iter_cli_assignments, parse_bool_arg

    assert parse_bool_arg("true") is True
    assert parse_bool_arg("false") is False
    assert parse_bool_arg("On") is True
    assert parse_bool_arg("0") is False

    assert list(iter_cli_assignments(["cuprate", "N=4", "bad", "TYPE=multi=start"])) == [
        ("N", "4"),
        ("TYPE", "multi=start"),
    ]


def test_build_legacy_result_dirs():
    from cuprate.io import build_legacy_result_dirs

    params = {
        "U": 1.0,
        "t": 0.1,
        "t2": None,
        "sz": 0.0,
        "s2": 0.0,
        "type": "occ",
        "restart": True,
        "match_spin_sectors": True,
    }

    assert build_legacy_result_dirs(params) == (
        "U1.0000_t0.1000",
        "_sz0.0000_s0_match_spin_sectors_occ_restart",
        "_sz0.5000_s0_match_spin_sectors_occ_restart",
    )


def test_find_existing_cluster_report_prefers_primary_and_accepts_json_only(tmp_path):
    from cuprate.io import find_existing_cluster_report

    primary_dir = tmp_path / "U1.0000_t0.1000" / "N2_primary"
    fallback_dir = tmp_path / "U1.0000_t0.1000" / "N2_fallback"
    primary_dir.mkdir(parents=True)
    fallback_dir.mkdir(parents=True)

    (fallback_dir / "hole0_class0_cluster0_results.json").write_text("{}")
    (primary_dir / "hole0_class0_cluster0_results.txt").write_text("legacy")

    report_path, suffix = find_existing_cluster_report(
        str(tmp_path / "U1.0000_t0.1000"),
        2,
        "_primary",
        "_fallback",
        hole=0,
        class_idx=0,
        cluster_idx=0,
    )

    assert suffix == "_primary"
    assert report_path.endswith("N2_primary/hole0_class0_cluster0_results.txt")


def test_find_existing_run_dir_uses_fallback_when_primary_missing(tmp_path):
    from cuprate.io import find_existing_run_dir

    fallback_dir = tmp_path / "U1.0000_t0.1000" / "N2_fallback"
    fallback_dir.mkdir(parents=True)

    resolved_dir, suffix = find_existing_run_dir(
        str(tmp_path / "U1.0000_t0.1000"),
        2,
        "_primary",
        "_fallback",
    )

    assert suffix == "_fallback"
    assert resolved_dir == str(fallback_dir)


def test_find_existing_cluster_report_raises_for_missing_result(tmp_path):
    from cuprate.io import find_existing_cluster_report

    with pytest.raises(FileNotFoundError, match="No result report found"):
        find_existing_cluster_report(
            str(tmp_path / "U1.0000_t0.1000"),
            2,
            "",
            "_fallback",
            hole=0,
            class_idx=0,
            cluster_idx=0,
        )
