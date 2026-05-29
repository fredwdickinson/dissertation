import numpy as np
import matplotlib.pyplot as plt

def hello():
    print("Hello from the sampler file!")

def V(x, type = "quartic"):
    match type:
        case "quartic":
            return x**4 / 4
        case _:
            raise ValueError(f"Potential type '{type}' not found.")
        
def V_prime(x, type = "quartic"):
    match type:
        case "quartic":
            return x**3
        case _:
            raise ValueError(f"Potential type '{type}' not found.")

def update(lambda_n, v, v_prime, dt, N, method = "tamed"):
    """ 
    Potentials v and v_prime have already been evaluated at lambda_n.
    Update according to each method and return next lambda.
    """

    if method not in ["euler", "tamed", "implicit"]:
        raise ValueError(f"Update method {method} not found.")

    if (method == "implicit"):
        raise NotImplementedError("Implicit step.")
    else:
        noise_scale = np.sqrt(2.0*dt / (2*N))

        # Computes pairwise distance matrix, then fill diags with inf: 1/inf = 0.
        diff = np.subtract.outer(lambda_n, lambda_n)
        np.fill_diagonal(diff, np.inf)
        coulomb_interaction = np.sum(1.0 / diff, axis = 1) / N

        if (method == "tamed"):
            tamed_potential = v_prime / (2 + dt*np.abs(v_prime))
            lambda_next = lambda_n + (coulomb_interaction - tamed_potential)*dt + noise_scale*np.random.normal(0, 1, N)
            return lambda_next
        else:
            lambda_next = lambda_n + (coulomb_interaction - 1/2*v_prime)*dt + noise_scale*np.random.normal(0, 1, N)
            return lambda_next

def stochastic_sampler(N, T, num_trials = 500, potential = "quartic", method = "tamed"):
    """ 
    Simulates DBM using the tamed scheme as described by Li and Menon (2013).
    Input:
        N (int): system dimension.
        T (float): final time.
        num_trials (int): number of independent trials (/trajectories).
        potential_type (str): type of potential, e.g. quartic.
        scheme (str): update scheme e.g. tamed, euler, implicit.
    """

    # NOTE More dynamic dt updates with scheme type? tamed is 1/N^2
    # but implicit might by 1/N?
    dt = 1.0 / (N**2)
    num_steps = int(T/dt)

    # Store final positions for each trial.
    final_particles = np.zeros((num_trials, N))

    for trial in range(num_trials):
        # Choose initial vals accoridng to Gaussian eigenvalues.
        # Div by 2 is for half sum, by rootN for spectrum growth O(rootN).
        A = np.random.normal(0, 1, (N, N)) + 1j*np.random.normal(0, 1, (N, N))
        M = (A + A.conj().T) / (2.0*np.sqrt(N))
        lambda_n = np.linalg.eigvalsh(M)

        # NOTE Can modify to save updates if wanted.
        for _ in range(num_steps):
            cur_lambda = np.copy(lambda_n)
            v = V(cur_lambda, type = potential)
            v_prime = V_prime(cur_lambda, type = potential)
            lambda_n = update(cur_lambda, v, v_prime, dt, N, method = method)

        final_particles[trial, :] = lambda_n

    return final_particles.flatten()


def theoretical_density(s, q = 1.0, g = 1.0):
    """
    See (2.1), (2.2), (2.3) in Li and Menon.
    Gives the exact density for different potentials, if mixed assumes x^2/2 + gx^4/4 
    Input:
        s (ndarray): range.
        g (float): quartic coefficient gx^4/4
    """

    density = np.zeros_like(s)
    
    if (q == 1.0):
        # (2.2)
        a = np.sqrt((np.sqrt(1+12*g) - 1) / (6*g))
        condition = np.abs(s) <= 2*a

        # Avoid sqrt runtime errors.
        s_valid = s[condition]
        density[condition] = 1/(2*np.pi)*(1+2*g*(a**2) + g*(s_valid**2))*np.sqrt(4*(a**2) - s_valid**2)

    else:
        a = (3.0*g)**(-1/4)
        condition = np.abs(s) <= 2*a

        s_valid = s[condition]
        density[condition] = 1/(2*np.pi)*(2*g*(a**2) + g*(s_valid**2))*np.sqrt(4*(a**2) - s_valid**2)

    return density