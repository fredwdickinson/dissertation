import numpy as np
import python.forces as forces
import python.integrators as integrators
from python.densities import compute_empirical_cdf, compute_distance

potential_ints = {
    "quadratic": 0,
    "quad-quartic": 1,
    "quartic": 2
}

def get_pipeline(method, **kwargs):
    if (method == "euler"):
        return make_euler_pipeline(**kwargs)
    elif (method == "tamed"):
        return make_tamed_pipeline(**kwargs)
    elif (method == "implicit"):
        return make_implicit_pipeline(**kwargs)
    elif (method == "imla"):
        return make_imla_pipeline(**kwargs)
    else:
        raise ValueError(f"Do not know method {method}.")
    
# ==========================================================================================================
    
def make_euler_pipeline(N, dt, potential_type, beta):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        coulomb = forces.coulomb_interaction(state)
        v_prime = forces.evaluate_force(state, potential_int, 1)

        next_x = integrators.euler_step(state, coulomb, v_prime, dt, noise_scale)
        return next_x, {}

    return pipeline

def make_tamed_pipeline(N, dt, potential_type, beta):
    # Step pipeline for tamed Euler.
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        coulomb = forces.coulomb_interaction(state)
        v_prime = forces.evaluate_force(state, potential_int, 1)

        next_x = integrators.tamed_euler_step(state, coulomb, v_prime, dt, noise_scale)
        return next_x, {}

    return pipeline

def make_implicit_pipeline(N, dt, potential_type, beta):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        next_x = integrators.implicit_newton_step(state, dt, potential_int, noise_scale)
        return next_x, {}
    
    return pipeline

def make_imla_pipeline(N, dt, potential_type, beta, metropolise = False, newton_tol = 1e-5):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    current_coulomb = None; current_H_N = None;
    potential_int = potential_ints[potential_type]
    
    def pipeline(state):        
        nonlocal current_coulomb, current_H_N
        # Initialisation (only on first time step).
        if (current_coulomb is None):
            current_coulomb = forces.coulomb_interaction(state)
        if (current_H_N is None):
            current_H_N = forces.sum_hamiltonian(state, potential_int)

        next_x, next_coulomb, next_H_N, accepts, crossing_rejects, line_search_rejects, newton_iters, mean_cg_iters  = integrators.imla_step(
            state, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale, metropolise, newton_tol
        )

        current_coulomb = next_coulomb; current_H_N = next_H_N;
        return next_x, {"accepts": accepts, "cross_rejects": crossing_rejects, "line_search_rejects": line_search_rejects,
                        "newton_iters": newton_iters, "mean_cg_iters": mean_cg_iters}
        
    return pipeline


def make_mala_pipeline(N, dt, potential_type, beta):
    """
    Step pipeline for MALA. 
    Maintains state of forces to avoid recalculation (hence the nonlocal).
    """
    
    current_coulomb = None; current_H_N = None;
    potential_int = potential_ints[potential_type]
    noise_scale = np.sqrt(2.0*dt/(beta*N))

    def pipeline(state):
        nonlocal current_coulomb, current_H_N
        
        # Only calculate on first step.
        if (current_coulomb is None):
            current_coulomb = forces.coulomb_interaction(state)
        if (current_H_N is None):
            current_H_N = forces.sum_hamiltonian(state, potential_int)

        next_x, next_coulomb, next_H_N, accepts, crossing_rejects = integrators.mala_step(
            state, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale
        )

        current_coulomb = next_coulomb; current_H_N = next_H_N;
        return next_x, {"accepts": accepts, "cross_rejects": crossing_rejects}

    return pipeline

def make_hmc_pipeline(N, dt, potential_type, beta):
    """ 
    Step pipeline for the hybrid Monte Carlo algorithm (Chafai and Ferre).
    For now alpha_n and gamma_n are taken as default (both 1).
    As with MALA, keeps Coulomb/Hamiltonian calculations to avoid recalculating on rejects.
    """

    current_y = None; current_coulomb = None; current_H_N = None 
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        nonlocal current_y, current_coulomb, current_H_N
        
        # Initialisation (only on first time step).
        if (current_y is None):
            current_y = np.random.normal(0, 1, state.shape)
        if (current_coulomb is None):
            current_coulomb = forces.coulomb_interaction(state)
        if (current_H_N is None):
            current_H_N = forces.sum_hamiltonian(state, potential_int)

        next_x, next_y, next_coulomb, next_H_N_out, accepts, cross_rejects = integrators.hmc_step(
            state, current_y, dt, potential_int, current_coulomb, current_H_N, beta
        )

        current_y = next_y; current_coulomb = next_coulomb; current_H_N = next_H_N_out
        return next_x, {"accepts": accepts, "cross_rejects": cross_rejects}

    return pipeline

# ==========================================================================================================

def simulate_dbm(init, steps, step_pipeline):
    """
    Generator object for the trajectory: much better performance for memory, and easier experiment running.
    Input:
        init (ndarray): initial particle state, shape MxN.
        dt (float): timestep.
        step_pipeline (func): pipeline for a specific method.
    """
    state = np.copy(init)
    yield state, {}

    for _ in range(steps):
        state, info = step_pipeline(state)
        yield state, info

def analyse_trajectory(trajectory, num_steps, dt = None, track_snapshots = True, burn_in = None, snapshot_interval = 10,
                       track_accepts = False, track_crossings = False, step_star = None,
                       track_distance = False, grid = None, F_exact = None, distance_interval = None, distance_type = "wasserstein",
                       track_newton_info = False):
    """ 
    Master observe function (combined all previous here). By default only track_snapshots is True.
        Snapshots: burn_in, defaults to 1/2 the num_steps.
        Accepts: track_accepts, NOTE right now only MALA returns cross rejects.
        Crossings: track_crossings, step_star.
        Distances: track_distance, grid (that the true cdf is evaluated on), dt, F_exact.
    """

    # Defaults. 
    if burn_in is None:
        burn_in = num_steps//2
        # Default snapshot interval is 10.

    if distance_interval is None:
        distance_interval = max(1, num_steps//240)
        # Don't compute distances at every point.

    # 
    results = {}
    if track_snapshots:
        snapshots = []

    if track_accepts:
        accepts = []
        cross_rejects = []

    if track_distance:
        if (grid is None) or (F_exact is None) or (dt is None):
            raise ValueError("Must provide grid, F_exact, and dt to track distance.")
        
        history_times = []
        history_distances = []

    if track_crossings:
        if step_star is None:
            raise ValueError("Must provide step_star to track crossings.")
        
        prev_ordering = None
        crossings_per_trial = None

    if track_newton_info:
        newton_iters = []
        mean_cg_iters = []
        line_search_rejects = []
        
    # Single pass through the generator.
    # Info is a dictionary: check keys. 
    # NOTE Check the logic with step==0? Should it be step==step_star - 1?
    for step, (state, info) in enumerate(trajectory):
        # Crossings.
        if track_crossings:
            if (step == step_star - 1):
                # Determine ordering at T=0.
                prev_ordering = (state[:, :, None] < state[:, None, :])

            elif (step == step_star):
                current_ordering = (state[:, :, None] < state[:, None, :])
                flips = (current_ordering != prev_ordering)

                # k is teh diagonal offset.
                mask = np.triu(np.ones((state.shape[1], state.shape[1]), dtype = bool), k = 1)
                crossings_per_trial = np.sum(flips & mask, axis = (1, 2))

        # Distance: default is 1-Wasserstein.
        if (track_distance) and (step % distance_interval == 0):
            F_emp = compute_empirical_cdf(state.flatten(), grid)
            distance = compute_distance(F_emp, F_exact, grid, distance_type = distance_type)
            history_times.append(step*dt)
            history_distances.append(distance)

        # Snapshots (for histogram): default every 10 after half steps.
        if (track_snapshots) and (step >= burn_in) and ((step - burn_in) % snapshot_interval == 0):
            snapshots.append(np.copy(state))

        # Acceptance.
        if (track_accepts) and (step > 0):
            if ("accepts" in info):
                accepts.append(info["accepts"])
            if ("cross_rejects" in info):
                cross_rejects.append(info["cross_rejects"])

        # Newton iterations.
        if (track_newton_info):
            if ("newton_iters" in info):
                newton_iters.append(info["newton_iters"])
            if ("mean_cg_iters" in info): 
                mean_cg_iters.append(info["mean_cg_iters"])
            if ("line_search_rejects" in info):
                line_search_rejects.append(info["line_search_rejects"])

    # End of trajectory loop, compile the dictionary and return.
    if track_snapshots:
        results["snapshots"] = np.concatenate(snapshots).flatten()
    
    if track_accepts:
        results["accepts"] = np.array(accepts)
        if cross_rejects:
            results["cross_rejects"] = np.array(cross_rejects)
            
    if track_distance:
        results["distance_times"] = np.array(history_times)
        results["distances"] = np.array(history_distances)
        
    if track_crossings:
        if (crossings_per_trial is None):
            raise ValueError(f"Trajectory finished before reaching step_star ({step_star}).")
        
        results["crossings"] = crossings_per_trial
    
    if track_newton_info:
        results["newton_iters"] = np.array(newton_iters)
        results["mean_cg_iters"] = np.array(mean_cg_iters)
        results["line_search_rejects"] = np.array(line_search_rejects)

    return results


# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================

def imla_target_dt(init, total_steps, potential_int, beta, dt_init, target):
    """
    Generator that yields dt and accept rate (no particles).
    """

    M, N = init.shape
    x = np.copy(init)
    dt = dt_init

    # Will update according to log(dt_next) = log(dt) + gamma_n(accept - target)
    # where gamma_n is the 1/(step+1)^kappa.
    kappa = 0.5
    smooth_avg_accept = target # For init only.
    
    for step_idx in range(total_steps):
        noise_scale = np.sqrt(2*dt/(beta*N))
        x, accept_rate = integrators.imla_step(
            x, dt, potential_int, noise_scale, beta, metropolise = True)
        
        smooth_avg_accept = 0.75*smooth_avg_accept + 0.25*accept_rate
        
        if (step_idx > 30):
            gamma_n = 0.1/((step_idx + 1)**kappa)
            dt = dt*np.exp(gamma_n*(smooth_avg_accept - target))
        
        if (dt < 1e-8) or (dt > 0.75):
            raise ValueError("Step size too big or small in target scheme (dt = {dt}).")

        yield step_idx, dt, accept_rate

