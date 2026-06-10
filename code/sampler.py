import numpy as np
import matplotlib.pyplot as plt
import importlib
import solvers, densities
importlib.reload(solvers); importlib.reload(densities);

def hello():
    print("Hello from the sampler file!")

def V(x, type = "quartic"):
    match type:
        case "quartic":
            return x**4 / 4
        case "quad-quartic":
            return x**2/2 + x**4/4
        case _:
            raise ValueError(f"Potential type '{type}' not found.")
        
def V_prime(x, type = "quartic"):
    match type:
        case "quartic":
            return x**3
        case "quad-quartic":
            return x + x**3
        case _:
            raise ValueError(f"Potential type '{type}' not found.")
        
def V_double_prime(x, type = "quartic"):
    match type:
        case "quartic":
            return 3*(x**2)
        case "quad-quartic":
            return 1 + 3*(x**2)
        case _:
            raise ValueError(f"Potential type '{type}' not found.")

def update(lambda_n, potential, dt, N, method = "tamed"):
    """ 
    Potentials are not evaluated yet.
    Update according to each method and return next lambda.
    """

    if method not in ["euler", "tamed", "implicit"]:
        raise ValueError(f"Update method {method} not found.")
    
    noise_scale = np.sqrt(2*dt / (2*N))
    noise = np.random.normal(0, 1, N)
    v_prime = V_prime(lambda_n, type = potential)


    if (method == "implicit"):
        v_n = lambda_n + noise_scale*noise
        v_n = np.sort(v_n) # In case noise causes a collision.
        
        # Implicit update: Newton's solver for now.
        # NOTE Will add other solvers later.
        lambda_next = solvers.newton_update(v_n, dt, N, potential = potential)
        return lambda_next

    else:
        # Computes pairwise distance matrix, then fill diags with inf: 1/inf = 0.
        diff = np.subtract.outer(lambda_n, lambda_n)
        np.fill_diagonal(diff, np.inf)
        coulomb_interaction = np.sum(1.0 / diff, axis = 1) / N

        if (method == "tamed"):
            tamed_potential = v_prime / (2 + dt*np.abs(v_prime))
            lambda_next = lambda_n + (coulomb_interaction - tamed_potential)*dt + noise_scale*noise

            # if np.any(np.diff(lambda_next) <= 0):
            #    print("Particles have crossed in the tamed scheme!")
            #    print(lambda_next)
            
            return lambda_next
        else:
            lambda_next = lambda_n + (coulomb_interaction - 1/2*v_prime)*dt + noise_scale*noise
            return lambda_next
        
#
#
#

def stochastic_sampler(N, T, dt, *, num_trials = 500, potential = "quartic", method = "tamed",
                       track_distance = False, distance_types = ["ks"],
                       track_crossing = False, T_star = 0, cross_dts = []):
    """ 
    Simulates DBM using the tamed scheme as described by Li and Menon (2013).
    Refactored to track particles (time is outer loop).
    Input:
        N (int): system dimension.
        T (float): final time.
        dt (float): step size; 1/N^2 for euler/tamed and 1/N for implicit.
        num_trials (int): number of independent trials (/trajectories).
        potential_type (str): type of potential, e.g. quartic.
        scheme (str): update scheme e.g. tamed, euler, implicit.
    
    Added track_distance/distance_types and check_crossing/T_star.
    """

    num_steps = int(T/dt)

    # Starting eigenvalues according to Gaussian eigenvalues.
    # Div by 2 is for half sum, by rootN for spectrm growth O(rootN).
    particles = np.zeros((num_trials, N))
    for trial in range(num_trials):
        A = np.random.normal(0, 1, (N, N)) + 1j*np.random.normal(0, 1, (N, N))
        M = (A + A.conj().T) / (2.0*np.sqrt(N))
        particles[trial, :] = np.linalg.eigvalsh(M)

    if (track_distance):
        # Exact target CDF F_N.
        grid = np.linspace(-3, 3, 1000)
        F_exact = densities.compute_exact_cdf(N, potential, grid)
        check_interval = max(1, num_steps // 240)  # NOTE can change to be more dynamic in N e.g. for 1/N vs 1/N^2 dt.
        history_times = [[] for i in range(len(distance_types))]
        history_distances = [[] for i in range(len(distance_types))]

    # Step to check crossing at.
    target_step = int(T_star / dt)
        
    # Loop changed: now update all trials at once for each step.
    for step in range(num_steps):
        # Do crossing checks if running for that.
        if (track_crossing and step == target_step):
            update_params = [potential, N, method]
            if (len(cross_dts) < 1):
                raise ValueError("Provide range of dts to check crossing at.")
            
            crossings = check_crossing(particles.copy(), num_trials, cross_dts, update_params)
            return crossings
        
        cur_time = step*dt
        for trial in range(num_trials):
            particles[trial, :] = update(particles[trial, :], potential, dt, N, method = method)

        if (track_distance) and (step % check_interval == 0):
            # Check every check_interval steps: calculate emp cdf, distance, store.
            for j, distance_type in enumerate(distance_types):
                F_emp = densities.compute_empirical_cdf(particles.flatten(), grid)
                distance = densities.compute_distance(F_emp, F_exact, grid, distance_type = distance_type)
                history_times[j].append(cur_time)
                history_distances[j].append(distance)

    # 
    if (track_distance):
        return particles.flatten(), np.array(history_times), np.array(history_distances)
    
    return particles.flatten()

# 

def check_crossing(particles, trials, dts, update_params):
    # 
    crosses_by_dt = np.zeros_like(dts, dtype = float)
    potential, N, method = update_params
    row_idx, col_idx = np.triu_indices(N, k = 1) # Upper triangular indices.

    for j, dt in enumerate(dts):
        for trial in range(trials):
            # Creates matrix for difference between particle (ij) (for that trial). 
            diff_before = np.subtract.outer(particles[trial, :], particles[trial, :])
            new_particles = update(particles[trial, :], dt = dt, potential = potential, N = N, method = method)
            diff_after = np.subtract.outer(new_particles, new_particles)

            signs_before = np.sign(diff_before[row_idx, col_idx])
            signs_after = np.sign(diff_after[row_idx, col_idx])

            if np.any(signs_before * signs_after <= 0):
                crosses_by_dt[j] += 1
    
    return crosses_by_dt
