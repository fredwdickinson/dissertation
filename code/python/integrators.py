import numpy as np
from numba import njit

from python.newton import solve_newton_single, cholesky_log_det
from python.forces import evaluate_force
from python.forces import coulomb_interaction, log_repulsion


# ===========================================================================
# Unadjusted methods (Euler, tamed, implicit, implicit midpoint).

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
def implicit_newton_step(x, dt, potential_int, noise_scale,
                         max_newton_iters = 50, newton_tol = 1e-6, cg_tol = 1e-9):
    """ 
    Performs the implicit step with Newton's method,
        x_(k+1) = x_k - alpha_k*hess(g_k)^-1*grad (g_k),
        where g_k is the function being minimised.
    """

    M, N = x.shape
    next_x = np.zeros_like(x)

    newton_iters = np.zeros(M); mean_cg_iters = np.zeros(M, dtype = float); # Counters.
    coulomb = np.zeros(N); hess = np.zeros((N, N)); proposed_x = np.zeros(N) # All act as temp.

    # Noise, included in the solve.
    noise = np.random.normal(0.0, 1.0, x.shape)
    z = x + noise_scale*noise 

    for m in range(M):
        current_x, trial_newton_iters, trial_avg_cg_iters = solve_newton_single(
            x[m], z[m], dt, potential_int, max_newton_iters, newton_tol, cg_tol,
            coulomb, hess, proposed_x
        )

        next_x[m] = current_x 
        newton_iters[m] = trial_newton_iters; mean_cg_iters[m] = trial_avg_cg_iters;

    return next_x, newton_iters, mean_cg_iters

@njit
def maila_newton_step(prev_x, dt, potential_int, beta, noise_scale,
                         max_newton_iters = 50, newton_tol = 1e-6, cg_tol = 1e-9, det_method = "cholesky"):
    """ 
    Performs the implict step with Newton's method with a Metropolis-Hastings accept/reject scheme.
    """

    M, N = prev_x.shape
    next_x = np.copy(prev_x)
    total_accepts = 0 # Divide by M later for probability.

    # Proposal.
    # NOTE Modify implicit_newton_step so that it returns the Coulomb information?
    proposed_x, newton_iters, mean_cg_iters = implicit_newton_step(
                prev_x, dt, potential_int, noise_scale,
                max_newton_iters = max_newton_iters, newton_tol = newton_tol, cg_tol = cg_tol)

    # Now go through and accept/reject.
    for m in range(M):
        x = prev_x[m]; y = proposed_x[m]
        v_x = evaluate_force(x, potential_int, 0); v_y = evaluate_force(y, potential_int, 0)
        v_prime_x = evaluate_force(x, potential_int, 1); v_prime_y = evaluate_force(y, potential_int, 1)
        v_double_prime_x = evaluate_force(x, potential_int, 2); v_double_prime_y = evaluate_force(y, potential_int, 2)
        log_rep_x = log_repulsion(x); log_rep_y = log_repulsion(y)

        sum_H_x = np.sum(v_x)/2.0 - np.sum(log_rep_x) # Log repulsion already divides by N.
        sum_H_y = np.sum(v_y)/2.0 - np.sum(log_rep_y)
        log_pi_ratio = -beta*N*(sum_H_y - sum_H_x)

        grad_H_x = 1/2*v_prime_x - coulomb_interaction(x)
        grad_H_y = 1/2*v_prime_y - coulomb_interaction(y)

        gaussian_part_y_x = -1*np.sum((x - y + dt*grad_H_x)**2)/(2*noise_scale**2)
        gaussian_part_x_y = -1*np.sum((y - x + dt*grad_H_y)**2)/(2*noise_scale**2)

        log_det_x = cholesky_log_det(x, beta, dt, v_double_prime_x)
        log_det_y = cholesky_log_det(y, beta, dt, v_double_prime_y)
        log_q_ratio = (log_det_x - log_det_y) + (gaussian_part_y_x - gaussian_part_x_y)

        # Construct the acceptance ratio.
        log_alpha = log_pi_ratio + log_q_ratio
        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y
            total_accepts += 1 

    return next_x, total_accepts/M, newton_iters, mean_cg_iters

@njit
def imla_newton_step(x, dt, potential_int, noise_scale,
                     max_newton_iters = 50, newton_tol = 1e-6, cg_tol = 1e-9):
    """
    IMLA step, reusing the ILA Newton solver. See the proximal operator equivalence.
    """

    M, N = x.shape
    next_x = np.zeros_like(x)
    z_mid = np.zeros_like(x)

    newton_iters = np.zeros(M); mean_cg_iters = np.zeros(M, dtype = float);
    coulomb = np.zeros(N); hess = np.zeros((N, N)); proposed_x = np.zeros(N)

    # Same noise draw as ILA, target only takes half.
    noise = np.random.normal(0.0, 1.0, x.shape)
    w = x + 0.5*noise_scale*noise # Called z in ILA.
    dt_mid = 0.5*dt 

    for m in range(M):
        current_z, trial_newton_iters, trial_avg_cg_iters = solve_newton_single(
            x[m], w[m], dt_mid, potential_int, max_newton_iters, newton_tol, cg_tol,
            coulomb, hess, proposed_x
        )

        z_mid[m] = current_z
        next_x[m] = 2.0*current_z - x[m] # 2prox - current. Here is where potential cross.
        newton_iters[m] = trial_newton_iters; mean_cg_iters[m] = trial_avg_cg_iters;

    return next_x, z_mid, newton_iters, mean_cg_iters

@njit
def maimla_newton_step(prev_x, dt, potential_int, beta, noise_scale,
                       max_newton_iters = 50, newton_tol = 1e-6, cg_tol = 1e-9):
    """
    MAIMLA step.
    """

    M, N = prev_x.shape
    next_x = np.copy(prev_x)

    total_accepts = 0 # Will be a probability (divide by M).
    total_crossing_rejects = 0 

    # IMLA proposal.
    proposed_x, z_mid, newton_iters, mean_cg_iters = imla_newton_step(
        prev_x, dt, potential_int, noise_scale, max_newton_iters, newton_tol, cg_tol)

    for m in range(M):
        x = prev_x[m]; y = proposed_x[m]
        u = z_mid[m] #

        # Check for crossings, reject if true.
        crossing = False 
        for j in range(N - 1):
            if (y[j] >= y[j + 1]):
                crossing = True 
                break 
        
        if (crossing):
            total_crossing_rejects += 1
            continue 
        
        # Metropolis logic. No crossing, so find V: x, y; log rep: x, y; V': u; Coulomb: u.
        v_x = evaluate_force(x, potential_int, 0); v_y = evaluate_force(y, potential_int, 0)
        v_prime_u = evaluate_force(u, potential_int, 1)
        log_rep_x = log_repulsion(x); log_rep_y = log_repulsion(y)
        coulomb_u = coulomb_interaction(u)
        grad_H_u = 0.5*v_prime_u - coulomb_u 

        sum_H_x = np.sum(v_x)/2.0 - np.sum(log_rep_x) # Log repulsion already divides by N.
        sum_H_y = np.sum(v_y)/2.0 - np.sum(log_rep_y)
        log_pi_ratio = -beta*N*(sum_H_y - sum_H_x)

        # See notes for how it cancels as below.
        log_q_ratio = -beta*N*np.sum((x - y)*grad_H_u)

        log_alpha = log_pi_ratio + log_q_ratio 
        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y 
            total_accepts += 1
    
    return next_x, total_accepts/M, total_crossing_rejects/M, newton_iters, mean_cg_iters

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
def hmc_step(x, p, dt, potential_int, current_coulomb, current_H_N, beta, L = 1, gamma_N =1.0, alpha_N =1.0):
    """ 
    Hybrid Monte Carlo step, scheme as in Chafai and Ferre.
    Update momentum, propose with Leapfrog, accept/reject.
    Note that current_H_N is referring to the *sum* of the current potential.
    ** Updated to include L integrator steps.
        p is momentum, x is current state,
        q is next momentum, y is proposed state.
    """

    M, N = x.shape
    beta_N = beta*N # NOTE Needs changing if generalising to other.
    
    # Update momentum, vectorised.
    eta = np.exp(-gamma_N*alpha_N*dt)
    sdn = np.sqrt((1- eta**2)/beta_N)
    noise = np.random.normal(0, 1, x.shape)
    p_tilde = eta*p + sdn*noise
    
    # Pre-allocate outputs.
    next_x = np.copy(x); next_p = np.copy(p_tilde) 
    next_coulomb = np.copy(current_coulomb); next_H_N = np.copy(current_H_N)
    total_accepts = 0; total_cross_rejects = 0;
    dt_micro = alpha_N*dt/L # For L=1 doesn't change.

    # First step for all M trials.
    v_prime_all = evaluate_force(x, potential_int, 1)
    grad_H_all = 0.5*v_prime_all - current_coulomb

    for m in range(M):
        y = np.copy(x[m]); q = np.copy(p_tilde[m])
        grad_H = grad_H_all[m]; crossing = False
        
        for l in range(L):
            q_tilde = q - grad_H*(dt_micro/2.0) # Leapfrog momentum.
            y = y + q_tilde*dt_micro # Full position step, updates for next l.

            # Check for crossings (if crossed, would be rejected after L steps anyway.)
            for i in range(N - 1):
                if (y[i] >= y[i + 1]):
                    crossing = True 
                    break 

            if (crossing):
                break 

            # For final momentum update.
            v_prime = evaluate_force(y, potential_int, 1)
            coulomb = coulomb_interaction(y)
            grad_H = 0.5*v_prime - coulomb # Updates for next l.
            q = q_tilde - grad_H*(dt_micro/2.0) # Updates for next l.

        # End of leapfrog!
        if (crossing):
            total_cross_rejects += 1
            next_p[m] = -p_tilde[m] 
            continue

        # No crossing, accept/reject logic, need H_N(y) and H_N(x).
        # H_N(x) is in current_H_N
        v_y = evaluate_force(y, potential_int, 0)
        log_repulsion_y = log_repulsion(y)
        sum_ham_y = np.sum(v_y)/2.0 - np.sum(log_repulsion_y)

        prop_total = sum_ham_y + 0.5*np.sum(q**2)
        current_total = current_H_N[m] + 0.5*np.sum(p_tilde[m]**2)

        energy_difference = prop_total - current_total 
        log_alpha = -beta_N*energy_difference # -beta*N*energy_difference

        if (np.log(np.random.random()) < log_alpha):
            next_x[m] = y; next_p[m] = q;
            next_coulomb[m] = coulomb; next_H_N[m] = sum_ham_y
            total_accepts += 1
        else:
            next_p[m] = -p_tilde[m]

    # End of trial loop.
    return next_x, next_p, next_coulomb, next_H_N, total_accepts/M, total_cross_rejects/M


# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================


@njit
def mala_step_wishart(x, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale, c):
    """
    Modified for the Wishart-Laguerre ensembles (takes c, rejects on particles<0).
    """

    M, N = x.shape
    total_accepts = 0; total_crossing_rejects = 0; total_negativity_rejects = 0;
    next_x = np.copy(x); next_coulomb = np.copy(current_coulomb); next_H_N = np.copy(current_H_N);

    # Proposed step: Euler-Maruyama.
    v_prime = evaluate_force(x, potential_int, 1, c)
    grad_H_current = current_coulomb - 1/2*v_prime
    noise = np.random.normal(0, 1, x.shape)
    y_proposed = x + grad_H_current*dt + noise_scale*noise
  
    for m in range(M):
        current_x = x[m]
        y_prop = y_proposed[m]

        any_negative = False
        for i in range(N - 1):
            if (y_prop[i] < 0):
                any_negative = True 
                break 

        if (any_negative):
            total_negativity_rejects += 1
            next_x[m] = current_x
            next_coulomb[m] = current_coulomb[m]
            continue 

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
        v_y = evaluate_force(y_prop, potential_int, 0, c); v_prime_y = evaluate_force(y_prop, potential_int, 1, c)
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

    return next_x, next_coulomb, next_H_N, total_accepts/M, total_crossing_rejects/M, total_negativity_rejects/M