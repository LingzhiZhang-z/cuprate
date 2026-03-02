import numpy as np

def Chop(M, threshold=1e-8):
    M[np.isclose(M.real, 0, atol=threshold) & np.isclose(M.imag, 0, atol=threshold)] = 0
    return M

def Matrix_Out(M):
    try:
        NRow = M.shape[0]
        NCol = M.shape[1]
    except ValueError:
        NRow = len(M)
        NCol = len(M[0])

    for i in range(NRow):
        for j in range(NCol):
            print("{:.5f}".format(M[i][j]), end="    ")
        print()

def objective_function(x, A, b):
    non_zero_indices = np.abs(b) > 1e-6
    residual = (A @ x - b)[non_zero_indices]
    mse = np.mean(residual**2)
    return np.sqrt(mse)

def derivative_objective_function(x, A, b):
    relative_error = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    residual = np.linalg.norm(A @ x - b)
    r2 = 1 - relative_error**2
    return (relative_error, residual, r2)

def norm_matrix(M):
    return np.linalg.norm(M.flatten())
