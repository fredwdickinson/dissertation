import numpy as np
from numba import njit

from python.forces import grad_quadratic, grad_quartic, grad_quad_quartic, hess_quadratic, hess_quartic, hess_quad_quartic

"""
Chosen to pass all arguments into each integrator even if redundant.
NOTE Will remove by introducing a step pipeline or something similar.
Drift: coulomb interaction and confinement.
"""


@njit
def euler_step(x, coulomb, v_prime, dt, noise_scale, beta):
    # Standard Euler-Maruyama step.
    drift = coulomb - 1/2*v_prime
    noise = np.random.normal(0.0, 1.0, x.shape)
    return x + drift*dt + noise_scale*noise


@njit
def tamed_euler_step(x, coulomb, v_prime, dt, noise_scale, beta):
    # Tamed Euler step as by Li and Menon.
    raw_confinement = 1/2*v_prime 
    tamed_confinement = raw_confinement/(1.0 + dt*np.abs(raw_confinement))

    drift = coulomb - tamed_confinement    
    noise = np.random.normal(0, 1, x.shape)

    return x + drift*dt + noise_scale*noise


@njit
def implicit_newton_step(x, dt, noise_scale, beta, potential_type):
    """
    Solve the implicit (proximal) step using Newton's method
        x_{k+1} = x_k - alpha_k hess(g_k)^{-1} grad(g_k).
    Implement backtracking to ensure particles don't cross (although they shouldn't anyway).
    Drift not used, unlike Euler/tamed version.
    """

    M, N = x.shape
    max_iter, tol = 20, 1e-6 # Hard coded, can change tol to be smaller if needed.

    # Stochastic step.
    next_x = np.zeros_like(x)
    z = np.zeros_like(x)

    # Each trial is independent.
    for m in range(M):
        z[m] = np.sort(x[m] + noise_scale*np.random.normal(0.0, 1.0, N))
        current_x = np.copy(z[m])
    
        for _ in range(max_iter):
            # NOTE Can implement the FMM Coulomb here when made.
            diff = current_x[:, None] - current_x
            for i in range(N): 
                diff[i, i] = np.inf
                
            coulomb = np.sum(1.0/diff, axis = 1)/N
            
            if potential_type == "quadratic":
                v_prime = grad_quadratic(current_x)
                v_double_prime = hess_quadratic(current_x)

            elif potential_type == "quartic":
                v_prime = grad_quartic(current_x)
                v_double_prime = hess_quartic(current_x)

            elif potential_type == "quad-quartic":
                v_prime = grad_quad_quartic(current_x)
                v_double_prime = hess_quad_quartic(current_x)
            else:
                raise ValueError("Invalid potential type specified in implicit Newton step.")
                
            # Generalised grad for any beta (refer to notes).
            nablaG = (current_x - z[m])/dt - coulomb + 1/2*v_prime
            
            if (np.max(np.abs(nablaG)) < tol):
                break
            
            # Again see notes: explicit construction for diags and not.
            # Nondiags are squared diff *-1/N,  diags are sum + 1/Delta t + potential.
            hess = -1/(N*(diff**2))
            for i in range(N): 
                hess[i, i] = 0.0
                
            diags = 1.0/dt + 1/2*v_double_prime - np.sum(hess, axis = 1) # Why inf diags are removed.
            for i in range(N): 
                hess[i, i] = diags[i]
                
            # Cannot use Scipy's CG anymore.
            # NOTE Look at cost of linalg.solve, possibly create own solver that is Numba-ready.
            y = np.linalg.solve(hess, nablaG)
            
            # Backtracking: most sophisticated exists, here just half every time.
            # Just check adjacent diff https://numpy.org/devdocs/reference/generated/numpy.diff.html
            step, rho = 1.0, 0.5
            for _ in range(1000):
                x_new = current_x - step*y
                if np.all(np.diff(x_new) > 1e-12): 
                    break
                
                step *= rho
            current_x = x_new
        
        # End of individual trial.
        next_x[m] = current_x
        
    return next_x