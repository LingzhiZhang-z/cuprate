"""Eta-pseudospin diagnostics for half-filled bipartite Hubbard clusters."""

from __future__ import annotations

import pathlib
import sys
from math import comb

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuprate import ATOL
from cuprate.clusters import Cluster, ClusterSets
from cuprate.hubbard import HubbardModel
from cuprate.paths import data_dir_name, mode_token
from cuprate.sectors import (
    S2SectorBlock,
    _sublattice_signs_square,
    build_S2_multiplets,
    build_S2_sectors,
    build_S2_transforms,
    build_S2eta0_sectors,
    build_S2eta0_transforms,
)
from cuprate.states import (
    calc_eta2_matrix_direct,
    calc_eta_plus_matrix,
    generate_states,
    group_states,
    pure_spin_state_indices,
    sort_states,
)


def _square_cluster() -> Cluster:
    return Cluster(
        sites=((0, 0), (1, 0), (0, 1), (1, 1)),
        bonds=((0, 1), (0, 2), (1, 3), (2, 3)),
        hole=0,
        class_idx=0,
        cluster_idx=0,
    )


def _same_sublattice_cluster() -> Cluster:
    return Cluster(
        sites=((0, 0), (1, 0), (2, 0), (3, 0)),
        bonds=((0, 2),),
    )


def _enumerated_clusters(max_N: int) -> list[Cluster]:
    clusters = []
    for N in range(2, max_N + 1):
        clusters.extend(ClusterSets(N).generate().clusters)
    return clusters


def _cluster_id(cluster: Cluster) -> str:
    return f"N{cluster.N}_{cluster.label()}"


def _minimal_nonnegative_twoSz(N: int) -> int:
    return N % 2


def _safe_comb(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0


def _expected_spin_dim(N: int, twoSz: int, twoS: int) -> int:
    if abs(twoSz) > twoS or (twoS - twoSz) % 2:
        return 0
    return _safe_comb(N, (N - twoS) // 2) - _safe_comb(N, (N - twoS) // 2 - 1)


def _eta_operator_pipeline(cluster: Cluster, *, twoSz: int | None = None):
    signs = _sublattice_signs_square(cluster)
    states_N = generate_states(cluster.N, cluster.N, twoSz=twoSz)
    states_Np2 = generate_states(cluster.N, cluster.N + 2, twoSz=twoSz)
    eta_plus = calc_eta_plus_matrix(states_N, states_Np2, cluster.N, signs)
    eta2 = calc_eta2_matrix_direct(states_N, cluster.N, signs)
    ham = HubbardModel(cluster, U=3.0, t=1.0).build_hamiltonian(states_N)
    return signs, states_N, states_Np2, eta_plus, eta2, ham


def _eta_sector_pipeline(cluster: Cluster):
    grouped = group_states(generate_states(cluster.N, cluster.N), cluster.N)
    _hw, multiplets = build_S2_multiplets(grouped, cluster.N)
    sector_blocks = build_S2_sectors(grouped, multiplets)
    transforms = build_S2_transforms(grouped, cluster.N, sector_blocks)
    eta0_sector_blocks = build_S2eta0_sectors(cluster.N, sector_blocks, cluster)
    eta0_transforms = build_S2eta0_transforms(grouped, cluster.N, eta0_sector_blocks)
    return grouped, sector_blocks, transforms, eta0_sector_blocks, eta0_transforms


def _fixed_twoSz_basis(grouped, N: int) -> dict[int, list[int]]:
    basis = {}
    for (twoSz, _D), states in grouped.items():
        basis.setdefault(twoSz, []).extend(states)
    for twoSz in basis:
        basis[twoSz] = sort_states(basis[twoSz], N)
    return basis


def _commutator_norm(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left @ right - right @ left))


def _eta_labels(eigvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eta_raw = (-1.0 + np.sqrt(1.0 + 4.0 * np.maximum(eigvals, 0.0))) / 2.0
    eta = np.rint(eta_raw).astype(int)
    residual = np.abs(eigvals - eta * (eta + 1))
    return eta, residual


@pytest.mark.parametrize("cluster", _enumerated_clusters(6), ids=_cluster_id)
def test_enumerated_cluster_bonds_are_eta_bipartite(cluster: Cluster):
    signs = _sublattice_signs_square(cluster)

    for i, j in cluster.bonds:
        assert signs[i] * signs[j] == -1


@pytest.mark.parametrize("cluster", _enumerated_clusters(6), ids=_cluster_id)
def test_eta_plus_matches_direct_eta2_for_enumerated_clusters(cluster: Cluster):
    twoSz = _minimal_nonnegative_twoSz(cluster.N)
    _signs, _states_N, _states_Np2, eta_plus, eta2, _ham = _eta_operator_pipeline(
        cluster,
        twoSz=twoSz,
    )

    assert np.allclose(eta_plus.conj().T @ eta_plus, eta2, atol=ATOL["tight"])


@pytest.mark.parametrize("cluster", _enumerated_clusters(6), ids=_cluster_id)
def test_eta2_commutes_with_hubbard_for_enumerated_clusters(cluster: Cluster):
    twoSz = _minimal_nonnegative_twoSz(cluster.N)
    _signs, _states_N, _states_Np2, _eta_plus, eta2, ham = _eta_operator_pipeline(
        cluster,
        twoSz=twoSz,
    )

    assert _commutator_norm(ham, eta2) < ATOL["loose"]


def test_eta2_is_hermitian_positive_and_quantized():
    _signs, _states_N, _states_Np2, _eta_plus, eta2, _ham = _eta_operator_pipeline(_square_cluster())

    assert np.allclose(eta2, eta2.conj().T, atol=ATOL["tight"])
    eigvals = np.linalg.eigvalsh(eta2)
    assert np.min(eigvals) >= -ATOL["tight"]

    eta, residual = _eta_labels(eigvals)
    assert set(eta).issubset({0, 1, 2})
    assert np.max(residual) < ATOL["loose"]


def test_hubbard_commutes_with_eta2_in_full_basis():
    cluster = _square_cluster()
    _signs, _states_N, _states_Np2, _eta_plus, eta2, ham = _eta_operator_pipeline(cluster)

    assert _commutator_norm(ham, eta2) < ATOL["loose"]


def test_hubbard_commutes_with_eta2_in_fixed_twoSz_basis():
    cluster = _square_cluster()
    _signs, _states_N, _states_Np2, _eta_plus, eta2, ham = _eta_operator_pipeline(
        cluster,
        twoSz=0,
    )

    assert _commutator_norm(ham, eta2) < ATOL["loose"]


def test_same_sublattice_hopping_breaks_eta2_commutator():
    cluster = _same_sublattice_cluster()
    _signs, _states_N, _states_Np2, _eta_plus, eta2, ham = _eta_operator_pipeline(
        cluster,
        twoSz=0,
    )

    assert _commutator_norm(ham, eta2) > 1.0


def test_eta2_annihilates_pure_spin_subspace():
    _signs, states, _states_Np2, _eta_plus, eta2, _ham = _eta_operator_pipeline(_square_cluster())
    pure = pure_spin_state_indices(states, N=4)

    assert pure
    assert np.allclose(eta2[:, pure], 0.0, atol=ATOL["tight"])
    assert np.allclose(eta2[pure, :], 0.0, atol=ATOL["tight"])


def test_eta2_projects_cleanly_into_existing_S2_blocks():
    cluster = _square_cluster()
    N = cluster.N
    twoSz = 0
    grouped, _sector_blocks, transforms, _eta0_sector_blocks, _eta0_transforms = _eta_sector_pipeline(cluster)
    signs = _sublattice_signs_square(cluster)

    basis_by_twoSz = _fixed_twoSz_basis(grouped, N)

    states = generate_states(N, N, twoSz=twoSz)
    assert states == basis_by_twoSz[twoSz]

    eta2 = calc_eta2_matrix_direct(states, N, signs)
    ham = HubbardModel(cluster, U=3.0, t=1.0).build_hamiltonian(states)
    sector_transforms = {key: U for key, U in transforms.items() if key[0] == twoSz}
    eta2_blocks = {key: U.conj().T @ eta2 @ U for key, U in sector_transforms.items()}

    keys = sorted(eta2_blocks)
    for key in keys:
        U = transforms[key]
        ham_block = U.conj().T @ ham @ U
        eta2_block = eta2_blocks[key]
        assert _commutator_norm(ham_block, eta2_block) < ATOL["loose"]

    for idx, key_a in enumerate(keys):
        U_a = transforms[key_a]
        for key_b in keys[idx + 1:]:
            U_b = transforms[key_b]
            cross = U_a.conj().T @ eta2 @ U_b
            assert np.linalg.norm(cross) < ATOL["loose"]


def test_build_S2eta0_sectors_are_eta2_zero_blocks():
    cluster = _square_cluster()
    N = cluster.N
    _grouped, _sector_blocks, _transforms, eta0_sector_blocks, _eta0_transforms = _eta_sector_pipeline(cluster)

    assert eta0_sector_blocks
    assert all(block.eta == 0 for block in eta0_sector_blocks)

    signs = _sublattice_signs_square(cluster)
    for block in eta0_sector_blocks:
        U = np.asarray(block.transform)
        eta2 = calc_eta2_matrix_direct(block.basis_states, N, signs)
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[1]), atol=ATOL["loose"])
        assert np.allclose(U.conj().T @ eta2 @ U, 0.0, atol=ATOL["loose"])


def test_build_S2eta0_transforms_returns_eta_zero_transforms():
    cluster = _square_cluster()
    N = cluster.N
    _grouped, _sector_blocks, _transforms, _eta0_sector_blocks, transforms = _eta_sector_pipeline(cluster)

    assert transforms
    assert all(eta == 0 for _twoSz, _twoS, eta in transforms)

    signs = _sublattice_signs_square(cluster)
    for (twoSz, _twoS, _eta), U in transforms.items():
        states_src = generate_states(N, N, twoSz=twoSz)
        eta2 = calc_eta2_matrix_direct(states_src, N, signs)
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[1]), atol=ATOL["loose"])
        assert np.allclose(U.conj().T @ eta2 @ U, 0.0, atol=ATOL["loose"])


def test_build_S2eta0_transforms_groups_by_eta_label():
    N = 2
    grouped = group_states(generate_states(N, N), N)
    states = grouped[(0, 0)]
    blocks = [
        S2SectorBlock(0, 0, 0, states, np.eye(len(states), 1), eta=0),
        S2SectorBlock(0, 0, 0, states, np.eye(len(states), 1), eta=1),
    ]

    transforms = build_S2eta0_transforms(grouped, N, blocks)

    assert set(transforms) == {(0, 0, 0), (0, 0, 1)}
    fixed_twoSz_dim = len(generate_states(N, N, twoSz=0))
    assert transforms[(0, 0, 0)].shape == (fixed_twoSz_dim, 1)
    assert transforms[(0, 0, 1)].shape == (fixed_twoSz_dim, 1)


def test_SzS2eta2_mode_uses_eta_zero_blocks_by_default():
    cluster = _square_cluster()
    model = HubbardModel(cluster, U=3.0, t=1.0)
    model.set_symmetry("SzS2eta2", twoSz=0, twoS=0)

    assert mode_token("SzS2eta2", twoSz=0, twoS=0) == "mode_twoSz_0_twoS_0_eta_0"
    assert data_dir_name("SzS2eta2") == "DATA_twoSz_twoS_eta_0"
    assert len(model.blocks) == 1

    block = model.blocks[0]
    assert block.label() == "twoSz_0_twoS_0_eta_0"
    assert block.eta == 0

    signs = _sublattice_signs_square(cluster)
    states = generate_states(cluster.N, cluster.N, twoSz=0)
    eta2 = calc_eta2_matrix_direct(states, cluster.N, signs)
    eta2_block = block.basis_transform.conj().T @ eta2 @ block.basis_transform
    assert np.allclose(eta2_block, 0.0, atol=ATOL["loose"])


def test_SzS2eta2_without_selectors_refines_all_default_SzS2_blocks():
    cluster = _square_cluster()
    model = HubbardModel(cluster, U=3.0, t=1.0)
    model.set_symmetry("SzS2eta2")

    assert model.blocks
    assert all(block.twoSz is not None for block in model.blocks)
    assert all(block.twoS is not None for block in model.blocks)
    assert all(block.eta == 0 for block in model.blocks)
    assert [block.basis_transform.shape[1] for block in model.blocks] == [
        10, 9, 1, 9, 1, 1,
    ]
    assert [block.spin_dim for block in model.blocks] == [2, 3, 1, 3, 1, 1]

    signs = _sublattice_signs_square(cluster)
    for block in model.blocks:
        states = generate_states(cluster.N, cluster.N, twoSz=block.twoSz)
        eta2 = calc_eta2_matrix_direct(states, cluster.N, signs)
        eta2_block = block.basis_transform.conj().T @ eta2 @ block.basis_transform
        assert np.allclose(eta2_block, 0.0, atol=ATOL["loose"])


@pytest.mark.parametrize("cluster", _enumerated_clusters(6), ids=_cluster_id)
def test_SzS2eta2_mode_eta_zero_for_enumerated_clusters(cluster: Cluster):
    model = HubbardModel(cluster, U=3.0, t=1.0)
    model.set_symmetry("SzS2eta2")

    assert model.blocks
    assert all(block.eta == 0 for block in model.blocks)

    signs = _sublattice_signs_square(cluster)
    for block in model.blocks:
        assert block.spin_dim == _expected_spin_dim(
            cluster.N,
            block.twoSz,
            block.twoS,
        )

        states = generate_states(cluster.N, cluster.N, twoSz=block.twoSz)
        eta2 = calc_eta2_matrix_direct(states, cluster.N, signs)
        eta2_block = block.basis_transform.conj().T @ eta2 @ block.basis_transform
        assert np.allclose(eta2_block, 0.0, atol=ATOL["loose"])


def test_SzS2eta2_cache_round_trip_keeps_eta_label(tmp_path):
    cluster = _square_cluster()
    model = HubbardModel(cluster, U=3.0, t=1.0)
    model.set_symmetry("SzS2eta2", twoSz=0, twoS=0)
    model.build_hamiltonians()
    model.solve(cache_mode="save", cache_dir=tmp_path)

    loaded = HubbardModel(cluster, U=3.0, t=1.0)
    loaded.set_symmetry("SzS2eta2", twoSz=0, twoS=0)
    loaded.solve(cache_mode="load", cache_dir=tmp_path)

    assert loaded.blocks[0].eta == 0
    assert loaded.blocks[0].label() == "twoSz_0_twoS_0_eta_0"
    assert np.allclose(model.blocks[0].eigvals, loaded.blocks[0].eigvals)
