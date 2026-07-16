import numpy as np
from numba import njit

from python.forces import evaluate_force
from python.forces import coulomb_interaction, log_repulsion
from python.solvers import cg_jacobi

@njit
def euler_step(x, coulomb, v_prime, dt, noise_scale):
    # Standard Euler-Maruyama step.
    drift = coulomb - 1/2*v_prime
    noise = np.random.normal(0.0, 1.0, x.shape)
    return x + drift*dt + noise_scale*noise

@njit
def tamed_euler_step(x, coulomb, v_prime, dt, noise_scale):
    # Tamed Euler step as by Li and Menon.
    raw_confinement = v_prime 
    tamed_confinement = raw_confinement/(2.0 + dt*np.abs(raw_confinement))

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

    newton_iterations = np.zeros(M); mean_cg_iterations = np.zeros(M);

    # Newton solver tolerance.
    max_iter, tol = 20, 1e-6 # Hard coded, can change tol to be smaller if needed.
    for m in range(M):
        z[m] = x[m] + noise_scale*np.random.normal(0.0, 1.0, N)
        current_x = np.copy(z[m])
        trial_cg_iters = 0

        for _ in range(max_iter):
            newton_iterations[m] += 1

            # Clear existing arrays, recompute Hessian and Coulomb.
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
            
            y, cg_iters = cg_jacobi(hess, nablaG) # CG solver for inverse Hessian.
            current_x = current_x - y
            trial_cg_iters += cg_iters
        
        # End of Newton iteration,
        next_x[m] = current_x
        mean_cg_iterations[m] = trial_cg_iters/newton_iterations[m]
        
    return next_x, newton_iterations, mean_cg_iterations

@njit
def mala_step(x, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale):
    """
    Propose using Euler-Maruyama, then accept/reject.
    Needs pure potential from forces.py as well as v_prime.
    """

    M, N = x.shape
    total_accepts = 0; total_crossing_rejects = 0;
    next_x = np.copy(x); next_coulomb = np.copy(current_coulomb); next_H_N = np.copy(current_H_N);

    # Proposed step: Euler-Maruyama.
    v_prime = evaluate_force(x, potential_int, 1)
    grad_H_current = current_coulomb - 1/2*v_prime
    noise = np.random.normal(0, 1, x.shape)
    y_proposed = x + grad_H_current*dt + noise_scale*noise
  
    for m in range(M):
        current_x = x[m]
        y_prop = y_proposed[m]

        crossing = False 
        for i in range(N - 1):
            if (y_prop[i] >= y_prop[i + 1]):
                crossing = True 
                break 

        if (crossing):
            total_crossing_rejects += 1
            next_x[m] = current_x 
            next_coulomb[m] = current_coulomb[m]
            continue

        # No crossing, proceed with standard accept/reject logic.
        v_y = evaluate_force(y_prop, potential_int, 0); v_prime_y = evaluate_force(y_prop, potential_int, 1)
        coulomb_y = coulomb_interaction(y_prop)      
        drift_y = coulomb_y - 1/2*v_prime_y # enough for grad H_N
        
        # Now construct proposed H_N and the forward/backward q(x, y).
        log_repulsion_y = log_repulsion(y_prop)
        sum_ham_y = np.sum(v_y)/2 - np.sum(log_repulsion_y)
        sum_ham_x = current_H_N[m] # Avoid recalculating for the log repulsion cost.

        log_q_x_y = -1*np.sum((y_prop - (current_x + grad_H_current[m]*dt))**2) # forwards
        log_q_y_x = -1*np.sum((current_x - (y_prop + drift_y*dt))**2) # backwards
        log_alpha = -beta*N*(sum_ham_y - sum_ham_x) + 1/(2*noise_scale**2)*(log_q_y_x - log_q_x_y)

        # If accept, update, otherwise keep coulomb.
        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y_prop
            next_coulomb[m] = coulomb_y
            next_H_N[m] = sum_ham_y
            total_accepts += 1

    return next_x, next_coulomb, next_H_N, total_accepts/M, total_crossing_rejects/M


@njit
def imla_step(x, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale, metropolise, newton_tol):
    """
    Metropolise is a boolean.
    See NOTE s. Want to reuse Coulomb/Hess code. Some inefficiencies e.g. can reuse Coulomb.
    """ 

    M, N = x.shape
    total_accepts = 0; total_crossing_rejects = 0; line_search_rejects = 0;
    newton_iterations = np.zeros(M); mean_cg_iterations = np.zeros(M);

    # Not to be confused with the midpoint Coulomb/Hessian.
    next_x = np.copy(x); next_coulomb = np.copy(current_coulomb); next_H_N = np.copy(current_H_N);
    coulomb_midpoint = np.zeros(N); hess = np.zeros((N, N)) # For the Newton solver.
    
    noise = np.random.normal(0.0, 1.0, x.shape)
    z = x + noise_scale*noise

    max_newton_iter = 25 # Hard coded: tolerance changed to be parameter.

    for m in range(M):
        cur_x = x[m]
        y = np.copy(cur_x) # NOTE very important!
        trial_cg_iters = 0 # Per trial.

        for _ in range(max_newton_iter):
            line_search_success = True; crossings = False; # Defaults in case solves in one step.
            newton_iterations[m] += 1
            coulomb_midpoint.fill(0.0); hess.fill(0.0);

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
                coulomb_midpoint[i] = force_i/N 
                hess[i, i] = diag_sum_i/N + diags[i]

            nablaG = (y - z[m])/dt - coulomb_midpoint + 0.5*v_prime
            if (np.max(np.abs(nablaG)) < newton_tol):
                break

            step, cg_iters = cg_jacobi(hess, nablaG)
            trial_cg_iters += cg_iters 

            # Backtracking in the midpoint method.
            current_residual_norm = np.sum(nablaG*nablaG)
            alpha = 1.0; min_alpha = 1/16
            line_search_success = False 
            y_try = np.zeros_like(y)

            while (alpha >= min_alpha):
                # Propose y - alpha*step.
                for i in range(N):
                    y_try[i] = y[i] - alpha*step[i]
                
                # Check to see if crossings.
                crossings = False
                for i in range(N - 1):
                    if (y_try[i + 1] - y_try[i] < 0):
                        crossings = True

                if (not crossings):
                    # Need sufficient decrease condition: recalculate residual norm at suggested.
                    u_try = (y_try + cur_x)/2.0
                    v_prime_try = evaluate_force(u_try, potential_int, 1)
                    coulomb_try = coulomb_interaction(u_try)

                    nablaG_try = (y_try - z[m])/dt - coulomb_try + 0.5*v_prime_try 
                    try_residual_norm = np.sum(nablaG_try*nablaG_try)

                    if (try_residual_norm <= current_residual_norm):
                        line_search_success = True 
                        break 
                        
                    # Reaching here means no crossings but no sufficient decrease.
                
                # If crossings, or not sufficient decrease, half line search step.
                alpha = alpha*0.5

            if (line_search_success):
                y = y_try 
            else:
                # No improvement to residual, so break early. Rejected later.
                break 
            
            # End Newton iteration loop. Either succeed line search, or hit minimum alpha.
        
        # Average cg iterations per Newton iteration for this trial.
        mean_cg_iterations[m] = trial_cg_iters/newton_iterations[m]

        # IMLA: calculate coulomb at next step and return.
        coulomb_y = coulomb_interaction(y)

        if (not metropolise):
            # IMLA: always accept proposal. NOTE If using IMLA for burn in for MAIMLA then need to form next_H_N?
            # Probably can do in simulate.py at the end of the burn-in.
            next_x[m] = y; next_coulomb[m] = coulomb_y; next_H_N[m] = 23; # Doesn't use H_N.
            total_accepts += 1
            continue
        
        # MAIMLA: check for line search failure/crossing failure.
        if (not line_search_success):
            # Abort this trial. Don't need to update.
            line_search_rejects += 1
            continue

        # Final check for crossings.
        crossing = False
        for i in range(N - 1):
            if (y[i + 1] - y[i] <= 0):
                crossing = True
                break
        
        if (crossing):
            total_crossing_rejects += 1
            continue

        # MAIMLA: proposal created, now accept/reject.
        # Need to construct H_N AND grad H_N in terms of (cur_x, y_prop).
        v_y = evaluate_force(y, potential_int, 0)        
        log_repulsion_y = log_repulsion(y)
        sum_ham_y = np.sum(v_y)/2 - np.sum(log_repulsion_y)
        sum_ham_x = current_H_N[m] # Stored from last step.

        u = (cur_x + y)/2
        v_prime_u = evaluate_force(u, potential_int, 1)
        coulomb_u = coulomb_interaction(u)

        # (y - x)*grad H_N(u)
        log_q_ratio = beta*N*np.sum((y - cur_x)*(1/2*v_prime_u - coulomb_u))
        log_pi_ratio = -beta*N*(sum_ham_y - sum_ham_x) 
        log_alpha = log_pi_ratio + log_q_ratio

        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y; next_coulomb[m] = coulomb_y; next_H_N[m] = sum_ham_y
            total_accepts += 1

    return next_x, next_coulomb, next_H_N, total_accepts/M, total_crossing_rejects/M, line_search_rejects/M, newton_iterations, mean_cg_iterations


@njit
def hmc_step(x, y, dt, potential_int, current_coulomb, current_H_N, beta, gamma_N =1.0, alpha_N =1.0):
    """ 
    Hybrid Monte Carlo step, scheme as in Chafai and Ferre.
    Update momentum, propose with Leapfrog, accept/reject.
    Note that current_H_N is referring to the *sum* of the current potential.
    """

    M, N = x.shape
    beta_N = beta*N # NOTE Needs changing if generalising to other 
    
    # Update momentum, vectorised.
    eta = np.exp(-gamma_N*alpha_N*dt)
    sdn = np.sqrt((1- eta**2)/beta_N)
    noise = np.random.normal(0, 1, x.shape)
    y_tilde = eta*y + sdn*noise
    
    # Verlet proposal vectorised because current_coulomb already known.
    v_prime = evaluate_force(x, potential_int, 1)
    grad_H_current = 1/2*v_prime - current_coulomb
    y_leapfrog = y_tilde - grad_H_current*(alpha_N*dt/2.0)
    x_prop = x + y_leapfrog*(alpha_N*dt)
    
    # Pre-allocate outputs.
    next_x = np.copy(x); next_y = np.copy(y_tilde) 
    next_coulomb = np.copy(current_coulomb); next_H_N = np.copy(current_H_N)
    total_accepts = 0; total_cross_rejects = 0;
    
    # Trial by trial for accept/reject.
    for m in range(M):
        # Check for potential crossings.
        crossing = False
        for i in range(N - 1):
            if (x_prop[m, i] >= x_prop[m, i + 1]):
                crossing = True
                break
        
        if crossing:
            total_cross_rejects += 1
            next_y[m] = -y_tilde[m] # Momentum is flipped.
            continue # Early exit.
        
        # No crossing, proceed with standard accept/reject logic.
        v_prime_prop = evaluate_force(x_prop[m], potential_int, 1)
        coulomb_prop = coulomb_interaction(x_prop[m])
        grad_H_prop= 1/2*v_prime_prop - coulomb_prop
        y_prop = y_leapfrog[m] - grad_H_prop*(alpha_N*dt/2.0)
        
        # Evaluate Proposal Energy.
        V_prop = evaluate_force(x_prop[m], potential_int, 0) 
        log_repulsion_prop = log_repulsion(x_prop[m])
        sum_ham_prop = np.sum(V_prop)/2 - np.sum(log_repulsion_prop)

        current_total_H = current_H_N[m] + 1/2*np.sum(y_tilde[m]**2)
        prop_total_H = sum_ham_prop + 0.5*np.sum(y_prop**2)
        energy_diff = prop_total_H - current_total_H

        if np.isnan(energy_diff):
            next_y[m] = -y_tilde[m]
            continue
        
        log_prob = -beta_N*energy_diff 
        if (np.log(np.random.random()) < log_prob):
            next_x[m] = x_prop[m]
            next_y[m] = y_prop
            next_coulomb[m] = coulomb_prop 
            next_H_N[m] = sum_ham_prop
            total_accepts += 1
        else:
            next_y[m] = -y_tilde[m]
            
    return next_x, next_y, next_coulomb, next_H_N, total_accepts/M, total_cross_rejects/M