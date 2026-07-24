import numpy as np
from numba import njit, prange 
from python import forces, solvers
import scipy.linalg

@njit
def construct_hess_coulomb(current_x, dt, v_double_prime, hess, coulomb):
    """
    Fills the Hessian and Coulomb interaction arrays in the same loop.
    """

    N = current_x.shape[0]
    hess.fill(0.0); coulomb.fill(0.0)

    for i in range(N):
        force_i = 0.0; diag_sum_i = 0.0;
        for j in range(N):
            if (i != j):
                diff = current_x[i] - current_x[j]
                inv_diff = 1.0/diff; inv_sq = inv_diff*inv_diff 
                force_i += inv_diff 

                hess[i, j] = -1.0/N*inv_sq
                diag_sum_i += inv_sq    

        coulomb[i] = force_i/N 
        hess[i, i] = diag_sum_i/N + 1.0/dt + 0.5*v_double_prime[i]

    return hess, coulomb

@njit 
def backtracking_step_size(current_x, proposed_x, y, min_alpha = 1/256):
    """
    Finds the largest alpha = 1/2^k so that x - alpha*newton_step remains ordered.
    Returns the step update.

    current_x (ndarray, ): current state.
    proposed_x (ndarray, ): empty temp array for placeholder.
    y (ndarray, ): Newton step direction.
    min_alpha (float, default 1/256): minimum step length.

    """

    N = current_x.shape[0]
    alpha = 1.0;
    while (alpha >= min_alpha):
        # Write the proposal into the temp array.
        for i in range(N):
            proposed_x[i] = current_x[i] - alpha*y[i]

        crossings = False # Check for crossings.
        for i in range(N - 1):
            if (proposed_x[i] >= proposed_x[i+1]):
                crossings = True
                break 
        
        if (not crossings):
            return proposed_x

        alpha = 0.5*alpha # If crossings, update by 0.5.
    
    # Reaching here means minimum alpha reached.
    # Can change to return -999 and handle error somewhere else...
    raise ValueError("Minimum alpha reached in backtracking Newton step.")


@njit
def solve_newton_single(x_m, z_m, dt, potential_int, 
                        max_newton_iters, newton_tol, cg_tol,
                        coulomb, hess, proposed_x):
    """
    Newton solve for one step. Note coulomb, hess, proposed_x are all empty and temp arrays.
    Defaults for max_newton_iters, newton_tol, cg_tol given in parent functions,
    should be ~50, 1e-6, 1e-9. 
    """

    current_x = np.copy(x_m)
    newton_iters = 0; total_cg_iters = 0

    for _ in range(max_newton_iters):
        newton_iters += 1
        v_prime = forces.evaluate_force(current_x, potential_int, 1)
        v_double_prime = forces.evaluate_force(current_x, potential_int, 2)

        hess, coulomb = construct_hess_coulomb(current_x, dt, v_double_prime, hess, coulomb)
        nablaG = (current_x - z_m)/dt - coulomb + 0.5*v_prime

        # Convergence check.
        if (np.max(np.abs(nablaG)) < newton_tol):
            break 

        y, cg_iters = solvers.cg(hess, nablaG, cg_tol)
        total_cg_iters += cg_iters 

        # Backtracking search to avoid crossings.
        next_x = backtracking_step_size(current_x, proposed_x, y)
        current_x = np.copy(next_x)

    if (newton_iters == max_newton_iters):
        print("WARN: reached maximum number of Newton iterations.")

    avg_cg_iters = total_cg_iters/newton_iters
    return current_x, newton_iters, avg_cg_iters
    

# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================

@njit(parallel = True, fastmath = True)
def build_metropolis_hessian(x, beta, dt, v_double_prime):
    """
    Builds the matrix Hessian I_n + dt*beta*N*nabla^2 H_N(x)).
    """

    N = x.shape[0]
    A = np.zeros((N, N), dtype = float)

    if (np.any(v_double_prime < 0.0)):
            raise ValueError(f"V'' is not positive: {v_double_prime}")

    # prange: Numba parallel looping.
    for i in prange(N):
        xi = x[i]; row_sum = 0.0; # Row sum will be -dt*beta/(diff^2).
        diag_constant = 1.0 + 0.5*dt*beta*N*v_double_prime[i]

        for j in range(N):
            if (i == j):
                continue # Ignore diagonals.

            diff = xi - x[j]
            term = (dt*beta)/(diff*diff)
            A[i, j] = -term; row_sum += term 

        # Diagonal elements.
        A[i, i] = diag_constant + row_sum 

    return A

@njit
def cholesky_log_det(x, beta, dt, v_double_prime):
    A = build_metropolis_hessian(x, beta, dt, v_double_prime)

    # For now with NumPy.
    L = np.linalg.cholesky(A)
    return 2.0*np.sum(np.log(np.diag(L)))

@njit
def eval_cholesky_log_det(A):
    L = np.linalg.cholesky(A)
    return 2.0*np.sum(np.log(np.diag(L)))




