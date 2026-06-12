import numpy as np
import scipy.sparse.linalg as spla
import importlib 
import sampler # includes V, V_prime, V_double_prime, ...
importlib.reload(sampler);


def newton_update(z, dt, N, potential, max_iter = 50, tol = 1e-6, track_steps = False):
    """
    Solve the proximal step using Newton's method
        x_{k+1} = x_k - alpha_{x} hess(g_k)^{-1}nabla(g_k)
    with backtracking.

    Input:
        z (ndarray): stochastic step lambda_n + nosie
        dt (float): timestep
        N (int): dimension
        potential (str): potential type
    """

    x = np.copy(z)

    # NOTE Remove after testing.
    newton_iters = 1

    for _ in range(max_iter):
        # Computes pairwise distance: same as in sampler.py for euler/tamed step.
        diff = np.subtract.outer(x, x)
        np.fill_diagonal(diff, np.inf)
        coulomb_interaction = np.sum(1.0 / diff, axis = 1) / N 
        nablaG = 1/2*sampler.V_prime(x, type = potential) - coulomb_interaction + (x - z)/dt

        # Check convergence.
        # NOTE 2 norm vs inf norm?
        if (np.linalg.norm(nablaG, ord = np.inf)) < tol:
            if (track_steps):
                return x, newton_iters 
            else:
                return x
        
        # Hessian - see notes, split into diag/not.
        # Nondiags are squared diff * -1/N,  diags are sum + 1/Delta t + potential'' .
        hess = -1 / (N * (diff ** 2))
        np.fill_diagonal(hess, 0.0)

        diag_coulomb = -np.sum(hess, axis = 1) # Already removed j=k.
        diags = 1/2*sampler.V_double_prime(x, type = potential) + diag_coulomb + 1/dt
        np.fill_diagonal(hess, diags)

        # Inverting the Hessian: cg?
        # let y = hess(f_k)^{-1}nabla f_k so that hess(f_k)y = nabla f_k
        y, info = spla.cg(hess, nablaG, rtol = tol)

        # Backtracking: more sophisticated exists, here we just half every time.
        # Just check adjacent diff https://numpy.org/devdocs/reference/generated/numpy.diff.html
        step = 1.0
        rho = 0.5
        backtrack_steps = 0
        while (backtrack_steps < 1000):
            x_new = x - step*y 
            if np.all(np.diff(x_new) > 1e-12):
                break 

            # If < then bound step size so they don't cross/get too close.
            step = step*rho
            backtrack_steps += 1

            if (backtrack_steps == 999):
                raise Warning("Exceed maximum number of backtrack steps in Newton update.")
            
        # 
        x = x_new 
        newton_iters += 1
    
    #
    return x