""" 
See Chafai and Ferre, 2018, Algorithm 2.3 and the discussion surrounding it.
"""

import numpy as np 
import importlib 
import sampler, fmm
importlib.reload(sampler); importlib.reload(fmm);

from sampler import V, V_prime


def H(x, N, potential):
    # HN(x): confinement and interaction (i<j, not i!=j).
    potential = np.sum(V(x, name = potential))
    
    diff = fmm.coulomb_difference(x, N, difference_only = True)
    i, j = np.triu_indices(N, k = 1) # k is offset from diag
    interaction = np.sum(np.log(np.abs(diff[i, j]))) / N

    return 1/2*potential - interaction # NOTE 1/2 or not?

def grad_H(x, N, potential):
    # Gradient: use Coulomb 
    coulomb = fmm.coulomb_difference(x, N)
    vp = V_prime(x, name = potential)

    return 1/2*vp - coulomb

def update_momentum(y, dt, beta_N, gamma_N = 1.0, alpha_N = 1.0):
    # Step 1 of the HMC algorithm: update momentum.
    eta = np.exp(-1*gamma_N*alpha_N*dt)
    sdn = np.sqrt((1 - eta**2) / beta_N)

    update = eta*y + sdn*np.random.normal(0, 1, y.shape[0])
    return update

def verlet_proposal(x, y, dt, N, potential, alpha_N = 1.0):
    # Step 2 of the HMC algorithm: Verlet scheme.
    y_leapfrog = y - grad_H(x, N, potential)*alpha_N*dt/2
    x_proposed = x + y_leapfrog*alpha_N*dt
    y_proposed = y_leapfrog - grad_H(x_proposed, N, potential)*alpha_N*dt/2

    return x_proposed, y_proposed

def acceptance_prob(x, y, x_prop, y_prop, N, potential, beta_N):
    # Step 3 of the HMC algorithm: calculate acceptance probability.
    energy_update = H(x_prop, N, potential) + np.dot(y_prop, y_prop)/2 - H(x, N, potential) - np.dot(y, y)/2

    if (np.isnan(energy_update)): # e.g. for eigenvalue crossing.
        return 0.0 
    
    prob = np.exp(-1*beta_N*energy_update)
    return min(1.0, prob)

def hmc_step(x, y, dt, N, potential, beta_N, gamma_N = 1.0, alpha_N = 1.0):
    # Full algorithm (Algorithm 2.3 in Chafai Ferre)
    # Returns x_{k_1}, y_{k+1}, Accepted
    y_tilde = update_momentum(y, dt, beta_N, gamma_N, alpha_N)
    x_prop, y_prop = verlet_proposal(x, y_tilde, dt, N, potential, alpha_N)
    prob = acceptance_prob(x, y_tilde, x_prop, y_prop, N, potential, beta_N)

    # Accept proposal with probability `prob`
    if (np.random.uniform() < prob):
        return x_prop, y_prop, True
    else:
        return x, -1*y_tilde, False
    

def hmc_sampler(N, T, dt, *, num_trials = 500, potential = "quadratic",
                beta_N = None, gamma_N = 1.0, alpha_N = 1.0):
    """ 
    Sampler according to the HMC algorithm (2.3) in Chafai-Ferre.
    beta_n with beta_N = beta*N^2, with beta = 2, is GUE.
    """

    if beta_N is None:
        beta_N = 2*N # for GUE.

    num_steps = int(T / dt)
    total_proposals = num_steps * num_trials 
    accepted_proposals = 0

    # Same initialisiation: Gaussian eigenvalues.
    particles = np.zeros((num_trials, N))
    for trial in range(num_trials):
        A = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))
        M = (A + A.conj().T) / (2.0 * np.sqrt(N))
        particles[trial, :] = np.linalg.eigvalsh(M)

    # Initialise momenta as zero (same as paper).
    momenta = np.zeros((num_trials, N)) # y

    for step in range(num_steps):
        for trial in range(num_trials):
            x_next, momentum_next, accepted = hmc_step(particles[trial, :], momenta[trial, :], dt, 
                                      N, potential, beta_N, gamma_N, alpha_N)
            
            particles[trial, :] = x_next
            momenta[trial, :] = momentum_next
            accepted_proposals += accepted

    return particles.flatten(), accepted_proposals / total_proposals


def target_accept_hmc(N, hmc_steps, target_prob, init = None, potential = "quartic"):
    """ 
    Run the HMC algorithm for hmc_steps, finding the step size that provides the target
    acceptance probability.
        Step sizes updated with dt += dt*(step_accept) / iter   some form of diminishing adaptation? 
    Initialise at equilibrium or zero?

    Input:
        N (int): dimension
        hmc_steps (int) steps to take
        target_prob (float): in [0, 1]
        init (ndarray or None): if True, inits at GUE equilibrium (semicircle).
    """

    if init is None:
        x = np.sort(np.random.normal(0, 1, N)) # particles, can change to be any init.
    else:
        # 
        A = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))
        M = (A + A.conj().T) / (2.0 * np.sqrt(N))
        x = np.linalg.eigvalsh(M)

    y = np.zeros(N) # momentum
    
    # Start step size at 1/N^(1/3). 
    step = np.zeros(hmc_steps + 1)
    step[0] = 1/(N**(3/3)); # NOTE Change from -1/3 to see if converges to correct dt in 10^5 iters.
    accepts = np.zeros(hmc_steps)
    
    for j in range(hmc_steps):
        # Run standard scheme.
        cur_dt = step[j]
        x_next, y_next, accepted = hmc_step(x, y, cur_dt, N, potential = potential, beta_N = 2*N)

        power = 0.75 # somewhere between sqrt and ^1?
        step[j + 1] = cur_dt + cur_dt*(accepted - target_prob)/((j + 1)**power)
        x, y = x_next, y_next
        accepts[j] = int(accepted)
        
    return accepts, step