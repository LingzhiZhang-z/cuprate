import numpy as np

from conftest import make_chain_model


def test_eigenvalues_match():
    """Block-diag eigenvalues must match full-diag eigenvalues."""
    for N in [3, 4, 5]:
        full, _ = make_chain_model(N, 4.0, 1.0, "full")
        sz_only, _ = make_chain_model(N, 4.0, 1.0, "block_sz_full")
        szs2, _ = make_chain_model(N, 4.0, 1.0, "block_sz_s2_full")

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
        full, _ = make_chain_model(N, 4.0, 1.0, "full")
        H = full.ham

        for mode in ["block_sz_full", "block_sz_s2_full"]:
            model, _ = make_chain_model(N, 4.0, 1.0, mode)
            for i in range(len(model.eigvals)):
                Hv = H @ model.eigvecs[:, i]
                Ev = model.eigvals[i] * model.eigvecs[:, i]
                np.testing.assert_allclose(
                    Hv,
                    Ev,
                    atol=1e-10,
                    err_msg=f"H|v> != E|v> for mode={mode}, N={N}, state {i}",
                )
