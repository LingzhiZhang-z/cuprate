import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.clusters import bond_analysis_spin
from cuprate.embed import parse_arguments as parse_embed_arguments
from cuprate.io import read_coords_file_Block, read_operators
from cuprate.lce import parse_arguments as parse_lce_arguments
from cuprate.lce import write_operators
from cuprate.main import cluster_save_results


def make_spin_coupling_artifact(cluster):
    return {
        "schema_version": 1,
        "mode": "full",
        "result_kind": "spin_couplings",
        "cluster": [[x, y] for x, y in cluster],
        "fit": {
            "relative_error": 0.001,
            "residual": 0.0,
            "r_squared": 0.999,
            "t11_minus_1_norm": 0.02,
            "overlap": 0.98,
        },
        "operators": {
            "constant_term": {"real": 0.5, "imag": 0.0},
            "groups": [
                {
                    "arity": 2,
                    "vector": [1, 0],
                    "label": "J1",
                    "terms": [
                        {
                            "sites": [0, 1],
                            "coefficient": {"real": 0.125, "imag": 0.0},
                        }
                    ],
                },
                {"arity": 4, "terms": []},
                {"arity": 6, "terms": []},
                {"arity": 8, "terms": []},
            ],
        },
    }


def test_embed_cli_defaults_and_boolean_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cuprate.embed", "RESTART=false", "S2_FIX=false"],
    )

    params = parse_embed_arguments(rank=0)

    assert params["restart"] is False
    assert params["s2_fix"] is False
    assert params["Ncell"] == 8
    assert params["Ncut"] == 3


def test_lce_cli_boolean_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cuprate.lce", "RESTART=false", "S2_FIX=false"],
    )

    params = parse_lce_arguments(rank=0)

    assert params["restart"] is False
    assert params["s2_fix"] is False


def test_read_coords_file_block_raises_when_no_upstream_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="No cluster result directories found"):
        read_coords_file_Block(str(tmp_path / "missing"), "", "", 3)


def test_read_operators_rejects_reports_without_couplings(tmp_path):
    report = tmp_path / "analysis_results.txt"
    report.write_text(
        "Hole: 0\n"
        "=== Projection Diagnostics ===\n"
        "Spin couplings are not reported for MODE=fixed_sz_s2.\n"
    )

    with pytest.raises(ValueError, match="No spin-coupling section found"):
        read_operators(str(report))


def test_read_operators_rejects_analysis_only_json(tmp_path):
    report = tmp_path / "analysis_results.txt"
    report.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_kind": "projection_analysis",
                "cluster": [[0, 0], [1, 0]],
                "projection": {"t11_minus_1_norm": 0.2},
            }
        )
    )

    with pytest.raises(ValueError, match="No spin-coupling section found"):
        read_operators(str(report))


def test_cluster_save_results_writes_structured_json_artifact(tmp_path):
    cluster = [(0, 0), (1, 0)]
    bonds = bond_analysis_spin(cluster)

    cluster_save_results(
        str(tmp_path),
        hole=0,
        class_idx=0,
        cluster_idx=0,
        rank=0,
        cluster_time=0.123,
        cluster=cluster,
        bonds=bonds,
        coeffs=[0.5 + 0.0j, np.array([0.125 + 0.0j])],
        error=(0.001, 0.0, 0.999),
        T11m1_norm=0.02,
        overlap=0.98,
    )

    structured = tmp_path / "hole0_class0_cluster0_results.json"
    assert structured.exists()

    payload = json.loads(structured.read_text())
    assert payload["result_kind"] == "spin_couplings"
    assert payload["cluster"] == [[0, 0], [1, 0]]


def test_read_operators_prefers_structured_json_over_text(tmp_path):
    report = tmp_path / "hole0_class0_cluster0_results.txt"
    report.write_text(
        "Hole: 0\n"
        "=== Projection Diagnostics ===\n"
        "Spin couplings are not reported for MODE=fixed_sz_s2.\n"
    )
    report.with_suffix(".json").write_text(json.dumps(make_spin_coupling_artifact([(0, 0), (1, 0)])))

    operators = read_operators(str(report))

    assert operators[0] == pytest.approx(0.5)
    assert operators[1][0][0:2] == [0, 1]
    assert operators[1][0][2] == pytest.approx(0.125 + 0.0j)


def test_read_coords_file_block_prefers_structured_json_and_skips_analysis_only(tmp_path):
    prefix = tmp_path / "Block_U1.0000_t0.1000"
    n2_dir = prefix / "N2"
    n2_dir.mkdir(parents=True)

    (n2_dir / "hole0_class0_cluster0_results.json").write_text(
        json.dumps(make_spin_coupling_artifact([(0, 0), (1, 0)]))
    )
    (n2_dir / "hole0_class1_cluster0_results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "fixed_sz_s2",
                "result_kind": "projection_analysis",
                "cluster": [[0, 0], [0, 1]],
            }
        )
    )

    clusters = read_coords_file_Block(str(prefix), "", "", 2)

    assert clusters[2][0][0][0] == [(0, 0), (1, 0)]
    assert 1 not in clusters[2][0]


def test_read_coords_file_block_keeps_legacy_txt_clusters_without_json(tmp_path):
    prefix = tmp_path / "Block_U1.0000_t0.1000"
    n2_dir = prefix / "N2"
    n2_dir.mkdir(parents=True)

    (n2_dir / "hole0_class0_cluster0_results.json").write_text(
        json.dumps(make_spin_coupling_artifact([(0, 0), (1, 0)]))
    )
    (n2_dir / "hole0_class1_cluster0_results.txt").write_text(
        "Point 0: (0,0)\n"
        "Point 1: (0,1)\n"
    )

    clusters = read_coords_file_Block(str(prefix), "", "", 2)

    assert clusters[2][0][0][0] == [(0, 0), (1, 0)]
    assert clusters[2][0][1][0] == [(0, 0), (0, 1)]


def test_write_operators_writes_structured_json_artifact(tmp_path):
    cluster = [(0, 0), (1, 0)]
    bonds = bond_analysis_spin(cluster)
    path = tmp_path / "hole0_class0_cluster0_results.txt"

    write_operators(
        str(path),
        [0.5, [[0, 1, 0.125 + 0.0j]], [], [], []],
        cluster,
        bonds,
    )

    assert path.with_suffix(".json").exists()
