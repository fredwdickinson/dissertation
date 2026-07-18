import numpy as np
import python.densities as densities
import python.integrators as integrators

# ========================================================================================
# Initialisations (random, gue, quartic).

def init_random_eigenvalues(M, N, beta, dt):
    """
    Generates M independent trials of random eigenvalues.
    """

    return np.sort(np.random.normal(0, 2.0*dt/(beta*N), (M, N)))

def init_gue_eigenvalues(M, N):
    """ 
    Generates M independent trials of GUE eigenvalues.
    Returns ndarray, shape (M, N). For large N follows Wigner's semicircle law.
    """

    eigenvalues = np.zeros((M, N))
    for m in range(M):
        A = np.random.normal(0, 1, (N, N)) + 1j*np.random.normal(0, 1, (N, N))
        mat = (A + A.conj().T) / (2.0*np.sqrt(N))
        eigenvalues[m, :] = np.linalg.eigvalsh(mat)

    return eigenvalues

def init_quartic_eigenvalues(M, N, dt = 0.1):
    """
    Run the implicit sampler for T = 1 with small dt.
    """

    # Assume beta=2.
    noise_scale = np.sqrt(2.0*dt/(2.0*N))
    T = 1; num_steps = int(T/dt);
    x = init_gue_eigenvalues(M, N)

    for _ in range(num_steps):
        next_x, _, _ = integrators.implicit_newton_step(x, dt, potential_int = 2, noise_scale = noise_scale)
        x = next_x 
        
    return x

# ========================================================================================
# Reconstructing matrices from the eigenvalues.

def construct_unitary(N):
    """ 
    See Mezzadri (2006) for explanation of the QR decomposition.
    Constructs a random unitary matrix.
    """
    z = (np.random.randn(N, N) + 1j*np.random.randn(N, N))/np.sqrt(2.0)
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

