import json
from pathlib import Path

from cuprate.clusters import Clusters_Square
from cuprate.io import Params, PathSpec, resolve_mode_spec
from cuprate.main import cluster_process_work_item


def _make_projection_run(tmp_path: Path):
    params = Params(N=4, U=1.0, t=0.1, mode="fixed_sz_s2", twoSz=0, twoS=0)
    mode_spec = resolve_mode_spec(params)
    spec = PathSpec(
        work_dir=str(tmp_path),
        base_dir="base",
        data_dir=str(tmp_path / "data"),
        run_dir="run",
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
    )
    Path(spec.data_dir).mkdir(parents=True)
    Path(spec.output_dir).mkdir(parents=True)
    Path(spec.tmp_dir).mkdir(parents=True)

    clusters = Clusters_Square(params.N)
    clusters.compute_clusters(if_print_time=False)
    clusters.classify_clusters(if_print_time=False)

    cluster_process_work_item(0, 0, clusters, params, spec, mode_spec, rank=0)
    return spec, clusters


def test_fixed_sz_s2_writes_projection_report_without_couplings(tmp_path):
    spec, clusters = _make_projection_run(tmp_path)

    content = (Path(spec.output_dir) / "hole0_class0_cluster0_results.txt").read_text()

    assert "=== Projection Diagnostics ===" in content
    assert "Spin couplings are not reported for MODE=fixed_sz_s2" in content
    assert "=== Individual Bond Coefficients ===" not in content
    assert "=== Bond Structure ===" in content

    payload = json.loads(
        (Path(spec.output_dir) / "hole0_class0_cluster0_results.json").read_text()
    )
    assert "projection" in payload
    assert "operators" not in payload
    assert payload["sites"] == [list(point) for point in clusters.clusters_classified[0][0][0]]


def test_fixed_sz_s2_saves_projection_npy_outputs(tmp_path):
    spec, _ = _make_projection_run(tmp_path)

    base = Path(spec.output_dir) / "hole0_class0"
    assert (base.with_name(f"{base.name}_Heff.npy")).exists()
    assert (base.with_name(f"{base.name}_T11m1.npy")).exists()
    assert (base.with_name(f"{base.name}_t11_selected_indices.npy")).exists()
    assert (base.with_name(f"{base.name}_S2_diagonal.npy")).exists()
