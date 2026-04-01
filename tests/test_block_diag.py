import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cuprate.hubbard import Hubbard_SingleBand


def make_model(N, U, t, block_mode="none"):
    """Create a model for a simple chain of N sites."""
    model = Hubbard_SingleBand(N, U, t)
    bonds = [(i, i + 1) for i in range(N - 1)]
    hoppings = [t] * len(bonds)
    model.set_bonds(bonds, hoppings)

    if block_mode == "szs2":
        model.set_block_szs2(True)
    elif block_mode == "sz":
        model.set_block_sz(True)
    model.set_block(block_mode in ("sz", "szs2"))

    model.set_states(nsites=N, nelec=N)

    if model.if_block_szs2:
        model.save_blocks(N)
        model.load_blocks(N)
        model.construct_transform_matrix(N)

    model.calc_hamiltonian()
    model.solve()
    return model


def test_eigenvalues_match():
    """Block-diag eigenvalues must match full-diag eigenvalues."""
    for N in [3, 4, 5]:
        full = make_model(N, 4.0, 1.0, "none")
        sz_only = make_model(N, 4.0, 1.0, "sz")
        szs2 = make_model(N, 4.0, 1.0, "szs2")

        np.testing.assert_allclose(
            full.eigvals,
            sz_only.eigvals,
            atol=1e-10,
            err_msg=f"Sz-only eigenvalues mismatch for N={N}",
        )
        np.testing.assert_allclose(
            full.eigvals,
            szs2.eigvals,
            atol=1e-10,
            err_msg=f"SzS2 eigenvalues mismatch for N={N}",
        )


def test_eigenvectors_satisfy_schrodinger():
    """Reconstructed eigenvectors must satisfy H|v> = E|v>."""
    for N in [3, 4]:
        full = make_model(N, 4.0, 1.0, "none")
        H = full.Ham

        for mode in ["sz", "szs2"]:
            model = make_model(N, 4.0, 1.0, mode)
            for i in range(len(model.eigvals)):
                Hv = H @ model.eigvecs[:, i]
                Ev = model.eigvals[i] * model.eigvecs[:, i]
                np.testing.assert_allclose(
                    Hv,
                    Ev,
                    atol=1e-10,
                    err_msg=f"H|v> != E|v> for mode={mode}, N={N}, state {i}",
                )


if __name__ == "__main__":
    test_eigenvalues_match()
    print("PASS: eigenvalues match")
    test_eigenvectors_satisfy_schrodinger()
    print("PASS: eigenvectors satisfy Schrodinger equation")
