import numpy as np

def construct_unitary(N):
    # See Mezzadri (2006).
    z = (np.random.randn(N, N) + 1j*np.random.randn(N, N)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)

    d = np.diagonal(r)
    ph = d/np.abs(d)

    return q*ph

def construct_hermitian(lambdas, U):
    """ 
    Constructs a full random Hermitian matrix from the eigendecomposition
        H = U Lambda U^dagger .
    """

    lambda_matrix = np.diag(lambdas)
    U_dagger = U.conj().T

    return U @ lambda_matrix @ U_dagger

def is_hermitian(H, tol = 1e-8):
    # Check up to some tolerance. Hermitian: H = H*.
    H_dagger = H.conj().T
    return np.allclose(H, H_dagger, atol = tol)
