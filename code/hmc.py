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
    v = np.sum(V(x, type = potential))
    
    # Same naive subtract.outer() but then use triu indices.
    diff = np.subtract.outer(x, x)
    i, j = np.triu_indices(N, k = 1) # k is offset from diag
    interaction = np.sum(np.log(np.abs(diff[i, j]))) / N

    return 1/2*v - interaction # NOTE 1/2 or not?

def grad_H(x, N, potential):
    # Gradient: use Coulomb 
    coulomb = fmm.coulomb_interaction(x, N)
    vp = V_prime(x, type = potential)

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
    
#
#
#

def hmc_sampler(N, T, dt, *, num_trials = 500, potential = "quadratic",
                beta_N = None, gamma_N = 1.0, alpha_N = 1.0):
    """ 
    Sampler according to the HMC algorithm (2.3) in Chafai-Ferre.
    beta_n with beta_N = beta*N^2, with beta = 2, is GUE.
    """

    if beta_N is None:
        beta_N = N**2

    num_steps = int(T / dt)

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

    return particles.flatten()
