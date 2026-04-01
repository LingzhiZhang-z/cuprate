import pathlib
import sys
import json

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.clusters import Clusters_Square, bond_analysis_spin, canonical_bond_type
from cuprate.hubbard import Hubbard_SingleBand
from cuprate.io import PathSpec, Params, build_path_spec, load_array_compat, resolve_mode_spec
from cuprate.main import cluster_process_work_item


def make_sz0_model(N=4, U=1.0, t=0.1):
    params = Params(N=N, U=U, t=t, mode="fixed_sz_s2", sz_idx=0, s_idx=0)
    spec = resolve_mode_spec(params)

    clusters = Clusters_Square(params.N)
    clusters.compute_clsuters(t2=params.t2, if_print_time=False)
    clusters.classify_clusters(t2=params.t2, if_print_time=False)
    cluster = clusters.clusters_classified[0][0][0]

    model = Hubbard_SingleBand(params.N, params.U, params.t)
    model.set_mode_spec(spec)
    bonds = bond_analysis_spin(cluster)

    if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 1:
        model.set_bonds_by_class(bonds[0], params.t)
    if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 2:
        model.set_bonds_by_class(bonds[0], params.t2)
    elif len(bonds) >= 2 and canonical_bond_type(cluster, bonds[1][0]) == 2:
        model.set_bonds_by_class(bonds[1], params.t2)

    model.set_states(nsites=params.N, nelec=params.N, sz_set=spec.sz)
    model.load_blocks(params.N)
    model.construct_transform_matrix(params.N)
    model.calc_hamiltonian()
    model.solve()
    model.calc_s2()
    return model, params


def test_sz0_s2_fixed_projection_produces_heff():
    model, params = make_sz0_model()

    model.calc_heff_halffilled(params, {"hole": 0, "class_idx": 0})

    assert model.Heff is not None
    assert model.T11m1 is not None
    assert model.t11_selected_indices is not None
    assert model.Heff.shape == (model.dimspin, model.dimspin)
    assert model.T11m1.shape == (model.dimspin, model.dimspin)
    assert len(model.t11_selected_indices) == model.dimspin


def test_fixed_sz_s2_spin_coeff_is_rejected_for_physical_non_uniqueness():
    model, params = make_sz0_model()
    model.calc_heff_halffilled(params, {"hole": 0, "class_idx": 0})

    clusters = Clusters_Square(params.N)
    clusters.compute_clsuters(t2=params.t2, if_print_time=False)
    clusters.classify_clusters(t2=params.t2, if_print_time=False)
    cluster = clusters.clusters_classified[0][0][0]
    bonds = bond_analysis_spin(cluster)

    with pytest.raises(RuntimeError, match="fixed_sz_s2"):
        model.calc_spin_coeff(bonds, params.s2)


def test_fixed_sz_s2_writes_projection_report_without_couplings(tmp_path):
    params = Params(N=4, U=1.0, t=0.1, mode="fixed_sz_s2", sz_idx=0, s_idx=0)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    params.path_spec = PathSpec(
        work_dir=str(tmp_path),
        base_dir="base",
        data_dir=str(tmp_path / "data"),
        run_dir="run",
        output_dir=str(output_dir),
        restart_dir=str(output_dir),
        tmp_dir=str(output_dir / "tmp"),
    )
    params.result_dir = str(output_dir)

    clusters = Clusters_Square(params.N)
    clusters.compute_clsuters(t2=params.t2, if_print_time=False)
    clusters.classify_clusters(t2=params.t2, if_print_time=False)

    cluster_process_work_item(0, 0, clusters, params, rank=0)

    content = (output_dir / "hole0_class0_cluster0_results.txt").read_text()

    assert "=== Projection Diagnostics ===" in content
    assert "Spin couplings are not reported for MODE=fixed_sz_s2" in content
    assert "=== Individual Bond Coefficients ===" not in content

    payload = json.loads((output_dir / "hole0_class0_cluster0_results.json").read_text())
    assert payload["result_kind"] == "projection_analysis"
    assert sorted(payload["cluster"]) == sorted([list(point) for point in clusters.clusters_classified[0][0][0]])
    assert "operators" not in payload


def test_build_path_spec_matches_mode_sz_s2_format():
    params = Params(N=4, U=1.0, t=0.1, mode="fixed_sz_s2", sz_idx=0, s_idx=0, type="single", restart=True)

    spec = build_path_spec(params)

    assert spec.data_dir.endswith("Block/U1.0000_t0.1000/N4_sz0_s0")
    assert spec.run_dir == "N4_sz0_s0_single_restart"
    assert spec.output_dir.endswith("Block/U1.0000_t0.1000/N4_sz0_s0_single_restart")


def test_save_data_writes_real_npy_files(tmp_path):
    model, params = make_sz0_model()
    model.calc_heff_halffilled(params, {"hole": 0, "class_idx": 0})

    base_filename = tmp_path / "hole0_class0"
    model.save_data(str(base_filename))

    heff = np.load(f"{base_filename}_Heff.npy", allow_pickle=False)
    selected_indices = np.load(f"{base_filename}_t11_selected_indices.npy", allow_pickle=False)
    double_occ = np.load(f"{base_filename}_double_occupation_expectation.npy", allow_pickle=False)

    np.testing.assert_allclose(heff, model.Heff)
    np.testing.assert_array_equal(selected_indices, model.t11_selected_indices)
    np.testing.assert_allclose(double_occ, model.double_occupation_expectation)


def test_load_array_compat_reads_legacy_text_npy(tmp_path):
    legacy_path = tmp_path / "legacy_t11_selected_indices.npy"
    np.savetxt(legacy_path, np.array([3, 1, 4]), fmt="%d")

    loaded = load_array_compat(legacy_path, text_dtype=int)

    np.testing.assert_array_equal(loaded, np.array([3, 1, 4]))
