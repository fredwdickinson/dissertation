import numpy as np
from numba import njit

from python.newton import solve_newton_single 
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
def maimla_newton_step(prev_x, dt, potential_int, noise_scale, beta,
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
        # log_q_y_x = np.sum((x - y + dt*beta*N*grad_H_u)**2)
        # log_q_x_y = np.sum((y - x + dt*beta*N*grad_H_u)**2)
        # log_q_ratio = -1/(4*dt)*(log_q_y_x - log_q_x_y)
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