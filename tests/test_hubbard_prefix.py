"""Tests for cluster-label threading through HubbardModel cache paths."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate.clusters import Cluster, cluster_label
from cuprate.hubbard import Block, HubbardModel


def _two_site_cluster(class_idx: int = 0) -> Cluster:
    return Cluster(
        sites=((0, 0), (1, 0)),
        bonds=((0, 1),),
        hole=0,
        class_idx=class_idx,
        cluster_idx=0,
    )


def test_cluster_label_format():
    assert cluster_label(0, 5) == "hole0_class5"


def test_block_label():
    block = Block(
        N=2, nelec=2, basis_states=[0], ham=None, eigvals=np.array([0.0]),
        eigvecs=np.eye(1), twoSz=0, twoS=0,
    )
    assert block.label() == "twoSz_0_twoS_0"


def test_cache_path_has_cluster_layer(tmp_path):
    model = HubbardModel(_two_site_cluster(0), U=4.0, t=1.0)
    model.solve(mode="fixed_sz_s2", twoSz=0, twoS=0, cache_dir=tmp_path)
    bucket_dir = tmp_path / model.label() / "hole0_class0_idx0" / "sz_s2_block"
    assert (bucket_dir / "twoSz_0_twoS_0_data.npz").exists()
    assert (bucket_dir / "twoSz_0_twoS_0_label.txt").exists()


def test_cache_hit_round_trip(tmp_path):
    model = HubbardModel(_two_site_cluster(0), U=4.0, t=1.0)
    r1 = model.solve(mode="fixed_sz_s2", twoSz=0, twoS=0, cache_dir=tmp_path)
    r2 = model.solve(mode="fixed_sz_s2", twoSz=0, twoS=0, cache_dir=tmp_path)
    assert np.allclose(r1.blocks[0].eigvals, r2.blocks[0].eigvals)


def test_different_clusters_do_not_collide(tmp_path):
    m1 = HubbardModel(_two_site_cluster(0), U=4.0, t=1.0)
    m2 = HubbardModel(_two_site_cluster(1), U=4.0, t=1.0)
    m1.solve(mode="fixed_sz_s2", twoSz=0, twoS=0, cache_dir=tmp_path)
    m2.solve(mode="fixed_sz_s2", twoSz=0, twoS=0, cache_dir=tmp_path)
    model_dir = tmp_path / m1.label()
    assert (model_dir / "hole0_class0_idx0" / "sz_s2_block" / "twoSz_0_twoS_0_data.npz").exists()
    assert (model_dir / "hole0_class1_idx0" / "sz_s2_block" / "twoSz_0_twoS_0_data.npz").exists()
