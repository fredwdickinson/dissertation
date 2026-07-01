import numpy as np
from numba import njit

@njit
def cg_jacobi(A, b, tol = 1e-6, max_iter = 50):
    """
    Conjugate gradient solve for Ax = b, applies Jacobi preconditioning.
    Numba code: lots of for loops. 
    """

    N = b.shape[0]; x = np.zeros(N)
    r = b.copy() # Since b - Ax = b - A0 = b.

    M_inv = np.zeros(N)
    for i in range(N):
        M_inv[i] = 1.0/A[i, i]
    
    z = M_inv*r # Different to normal CG.
    p = z.copy() # Different to normal CG.

    # Pre-allocate and compute with for loop inside loop.
    w = np.zeros(N)
    rz = np.dot(r, z)

    for k in range(max_iter):
        # Compute Ap
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += A[i, j]*p[j]
            
            w[i] = temp

        # NOTE could check pw isn't zero.
        pw = np.dot(p, w)
        if (pw == 0.0):
            break 

        alpha = rz/pw
        # Update x and residual.
        for i in range(N):
            x[i] += alpha*p[i]
            r[i] -= alpha*w[i]

        if (np.max(np.abs(r)) <= tol):
            break 

        # Update z.
        for i in range(N):
            z[i] = M_inv[i]*r[i]

        next_rz = np.dot(r, z)
        beta = next_rz/rz 

        # Update p.
        for i in range(N):
            p[i] = z[i] + beta*p[i]
        
        rz = next_rz

    return x
        