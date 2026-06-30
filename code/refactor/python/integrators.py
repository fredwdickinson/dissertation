import numpy as np
from numba import njit

from python.forces import potential_quadratic, potential_quad_quartic, potential_quartic
from python.forces import grad_quadratic, grad_quad_quartic, grad_quartic
from python.forces import hess_quadratic, hess_quad_quartic, hess_quartic
from python.forces import evaluate_force
from python.forces import coulomb_interaction, log_repulsion

@njit
def euler_step(x, coulomb, v_prime, dt, noise_scale):
    # Standard Euler-Maruyama step.
    drift = coulomb - 1/2*v_prime
    noise = np.random.normal(0.0, 1.0, x.shape)
    return x + drift*dt + noise_scale*noise


@njit
def tamed_euler_step(x, coulomb, v_prime, dt, noise_scale):
    # Tamed Euler step as by Li and Menon.
    raw_confinement = 1/2*v_prime 
    tamed_confinement = raw_confinement/(1.0 + dt*np.abs(raw_confinement))

    drift = coulomb - tamed_confinement    
    noise = np.random.normal(0, 1, x.shape)

    return x + drift*dt + noise_scale*noise


@njit
def implicit_newton_step(x, dt, potential_int, noise_scale):
    """
    Solve the implicit (proximal) step using Newton's method
        x_{k+1} = x_k - alpha_k hess(g_k)^{-1} grad(g_k).
    """

    M, N = x.shape

    # Initialise so only some stored in memory: inner loop fills with zeros.
    coulomb = np.zeros(N); hess = np.zeros((N, N))
    next_x = np.zeros_like(x)
    z = np.zeros_like(x) # NOTE pre-compute all starting? 

    # Each trial is independent.
    max_iter, tol = 20, 1e-6 # Hard coded, can change tol to be smaller if needed.
    for m in range(M):
        z[m] = np.sort(x[m] + noise_scale*np.random.normal(0.0, 1.0, N))
        current_x = np.copy(z[m])

        for _ in range(max_iter):
            # Clear existing arrays, compute Hess/coulomb in the same loop.
            coulomb.fill(0.0); hess.fill(0.0);
            v_prime = evaluate_force(current_x, potential_int, 1)
            v_double_prime = evaluate_force(current_x, potential_int, 2)

            diags = 1.0/dt + 1/2*v_double_prime # to add to diags below

            # Construct Hessian and Coulomb in same loop.
            for i in range(N):
                force_i = 0.0; diag_sum_i = 0.0

                for j in range(N):
                    if (i != j):
                        diff = current_x[i] - current_x[j]
                        inv_diff = 1/diff; inv_sq = inv_diff*inv_diff 
                        force_i += inv_diff 

                        # Hessian: nondiags are squared -1/N*inv_sq,
                        #  diags are 1/N*sum(inv_sq) + 1/dt + 1/2 v_double_prime
                        hess[i, j] = -1/N*inv_sq
                        diag_sum_i += inv_sq 
                
                # Finish Coulomb/Hessian construction.
                coulomb[i] = force_i/N 
                hess[i, i] = diag_sum_i/N + diags[i]

            nablaG = (current_x - z[m])/dt - coulomb + 1/2*v_prime
            if (np.max(np.abs(nablaG)) < tol):
                break

            # CG knowing that the Hessian is SDD.
            # NOTE To implement.
            # y = cg_sdd(hess, nablaG)
            y = np.linalg.solve(hess, nablaG)   
            current_x = current_x - y
        
        # End of individual trial.
        next_x[m] = current_x
        
    return next_x

@njit
def mala_step(x, dt, potential_int, coulomb, noise_scale, beta):
    """
    Propose using Euler-Maruyama, then accept/reject.
    Needs pure potential from forces.py as well as v_prime.
    """

    match potential_int:
        case 0:
            v_prime = grad_quadratic(x)
        case 2:
            v_prime = grad_quartic(x)
        case _:
            raise ValueError(f"MALA integrator does not support potential int {potential_int}.")

    M, N = x.shape
    total_accepts = 0; total_crossing_rejects = 0;
    next_x = np.zeros_like(x)
    next_coulomb = np.zeros_like(coulomb)

    # Proposed step: Euler-Maruyama.
    drift = coulomb - 1/2*v_prime
    noise = np.random.normal(0, 1, x.shape)
    y_proposed = x + drift*dt + noise_scale*noise
  
    for m in range(M):
        current_x = x[m]
        y_prop = y_proposed[m]
        
        # NOTE worth doing? - probably, might be expensive but saves on possible future cost.
        # And lets us look at the number of rejects due to crossings explicitly.
        if np.any(np.diff(y_prop) <= 0):
            next_x[m] = current_x
            next_coulomb[m] = coulomb[m]
            total_crossing_rejects += 1
            continue
        
        match potential_int:
            case 0:
                v_x = potential_quadratic(current_x); v_y = potential_quadratic(y_prop);
                v_prime_y = grad_quadratic(y_prop);
            case 2:
                v_x = potential_quartic(current_x); v_y = potential_quartic(y_prop);
                v_prime_y = grad_quartic(y_prop)
            case _:
                raise ValueError(f"Do not know potential integer {potential_int}.")

        coulomb_y = coulomb_interaction(y_prop)      
        drift_y = coulomb_y - 1/2*v_prime_y # enough for grad H_N
        
        # Now construct H_N.
        log_repulsion_x = log_repulsion(current_x) # Returns ndarray, includes /N.
        log_repulsion_y = log_repulsion(y_prop)

        sum_ham_x = np.sum(v_x)/2 - np.sum(log_repulsion_x)
        sum_ham_y = np.sum(v_y)/2 - np.sum(log_repulsion_y)

        log_q_x_y = np.sum((current_x - y_prop + drift[m]*dt)**2) # backwards
        log_q_y_x = np.sum((y_prop - current_x + drift_y*dt)**2) # forwards

        log_alpha = -beta*N*(sum_ham_y - sum_ham_x) - 1/(2*noise_scale**2)*(log_q_x_y - log_q_y_x)

        # Accept/reject, return the Coulomb!
        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y_prop
            next_coulomb[m] = coulomb_y
            total_accepts += 1
        else:
            next_x[m] = current_x 
            next_coulomb[m] = coulomb[m]  

    return next_x, next_coulomb, total_accepts/M, total_crossing_rejects/M


@njit
def imla_step(x, dt, potential_int, noise_scale, beta, metropolise):
    """
    Metropolise is boolean.
    See NOTE s. Want to reuse Coulomb/Hess code. Some inefficiencies e.g. can reuse Coulomb.
    """ 

    M, N = x.shape
    total_accepts = 0
    total_crossing_rejects = 0 # NOTE currently not implemented.

    coulomb = np.zeros(N); hess = np.zeros((N, N))
    next_x = np.zeros_like(x)
    z = np.zeros_like(x) # NOTE pre-compute all starting noises?
    
    max_iter, tol = 25, 1e-3
    for m in range(M):
        cur_x = x[m]
        z[m] = np.sort(cur_x + noise_scale*np.random.normal(0.0, 1.0, N))

        # NOTE Starting guess at noise (as implicit does).
        y = np.copy(z[m]); # x = lambda^n, y = lambda^(n+1). 

        for _ in range(max_iter):
            """
            if (_ > 20):
                # NOTE testing, should never really reach for 1e-3...
                print(f"iter {_} of Newton.")
            """
          
            coulomb.fill(0.0); hess.fill(0.0);

            u = (y + cur_x) / 2.0 # Midpoint (theta = 1/2).
            v_prime = evaluate_force(u, potential_int, 1)
            v_double_prime = evaluate_force(u, potential_int, 2)
            diags = 1.0/dt + 0.25*v_double_prime # 1/2 for chain rule from midpoint.

            # Construct Hessian and Coulomb in same loop.
            for i in range(N):
                force_i = 0.0; diag_sum_i = 0.0

                for j in range(N):
                    if (i != j):
                        diff = u[i] - u[j] # Same as implicit, just using midpoint.
                        inv_diff = 1/diff; inv_sq = inv_diff*inv_diff 
                        force_i += inv_diff 

                        # Again halved by chain rule.
                        hess[i, j] = -0.5/N*inv_sq
                        diag_sum_i += 0.5*inv_sq 
                
                # Finish Coulomb/Hessian construction.
                coulomb[i] = force_i/N 
                hess[i, i] = diag_sum_i/N + diags[i]

            nablaG = (y - z[m])/dt - coulomb + 0.5 * v_prime
            if (np.max(np.abs(nablaG)) < tol):
                break
            
           # NOTE Use the cg solver.
            step = np.linalg.solve(hess, nablaG) 

            # Backtracking in the midpoint method.
            newton_step_size = 1.0; min_step = 0.125
            y_try = y - step 
    
            while (newton_step_size >= min_step):
                y_try = y - newton_step_size*step
                if np.all(np.diff(y_try) > 1e-12):
                    break 
                newton_step_size *= 0.5
                
            # NOTE Add code here for crossing rejects.
            # Break early.
                
            y = y_try
            # End Newton iteration loop.

        if (not metropolise):
            # IMLA: accept proposal.
            next_x[m] = y
           #  next_coulomb[m] = current_coulomb
            total_accepts += 1
            continue
        
        # MAIMLA: proposal created, now accept/reject.
        # Need to construct H_N in terms of (cur_x, y_prop), for Gibbs measure (log).
        v_x = evaluate_force(cur_x, potential_int, 0); v_y = evaluate_force(y, potential_int, 0)        
        log_repulsion_x = log_repulsion(cur_x); log_repulsion_y = log_repulsion(y)

        sum_ham_x = np.sum(v_x)/2 - np.sum(log_repulsion_x)
        sum_ham_y = np.sum(v_y)/2 - np.sum(log_repulsion_y)
        log_alpha = -beta*N*(sum_ham_y - sum_ham_x)

        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y
            total_accepts += 1
        else:
            next_x[m] = cur_x
            
    return next_x, total_accepts/M