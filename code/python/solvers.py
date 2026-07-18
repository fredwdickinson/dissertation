import numpy as np
from numba import njit

# ========================================================================================

@njit
def cg(A, b, tol = 1e-9):
    """
    Standard conjugate gradient solve for Ax = b (No preconditioning).
    Otherwise same as above Jacobi-preconditioning.
    """

    N = b.shape[0]; x = np.zeros(N)
    r = b.copy() # Since b - Ax = b - A0 = b.

    total_iterations = 0
    max_iter = N//2 # Can change but seems reasonable.

    p = r.copy()
    w = np.zeros(N)
    rr = np.dot(r, r)

    for _ in range(max_iter):
        total_iterations += 1

        # Compute Ap.
        for i in range(N):
            temp = 0.0
            for j in range(N):
                temp += A[i, j] * p[j]
            
            w[i] = temp

        # NOTE could check pw isn't zero.
        pw = np.dot(p, w)
        if (pw == 0.0):
            break 

        alpha = rr / pw
        # Update x and residual.
        for i in range(N):
            x[i] += alpha*p[i]
            r[i] -= alpha*w[i]

        if (np.max(np.abs(r)) <= tol):
            break 

        # Calculate new residual norm
        next_rr = np.dot(r, r)
        beta = next_rr/rr 

        # Update p.
        for i in range(N):
            p[i] = r[i] + beta*p[i]
        
        rr = next_rr

    return x, total_iterations