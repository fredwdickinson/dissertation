import numpy as np
import python.forces as forces
import python.integrators as integrators
from python.densities import compute_empirical_cdf, compute_distance

potential_ints = {
    "quadratic": 0,
    "quad-quartic": 1,
    "quartic": 2,
    "wishart-laguerre": 3
}

def get_pipeline(method, **kwargs):
    """
    Give method, kwargs N, dt, potential_type, beta.
    """
    if (method == "euler"):
        return make_euler_pipeline(**kwargs)
    elif (method == "tamed"):
        return make_tamed_pipeline(**kwargs)
    elif (method == "implicit"):
        return make_implicit_pipeline(**kwargs)
    elif (method == "imla"):
        return make_imla_pipeline(**kwargs)
    elif (method == "hmc"):
        return make_hmc_pipeline(**kwargs)
    elif (method == "mala"):
        return make_mala_pipeline(**kwargs)
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

def make_implicit_pipeline(N, dt, potential_type, beta, track_energy = False):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]

    if (track_energy):
        def pipeline(state):
            next_x, newton_iters, mean_cg_iters, log_energy_x, log_energy_y, log_transition_x_y, log_transition_y_x, log_det_x, log_det_y = integrators.implicit_newton_step_with_energy(
                state, dt, potential_int, beta, noise_scale, track_energy = track_energy)
            
            return next_x, {"newton_iters": newton_iters, "mean_cg_iters": mean_cg_iters,
                            "log_energy_x": log_energy_x, "log_energy_y": log_energy_y,
                            "log_transition_x_y": log_transition_x_y, "log_transition_y_x": log_transition_y_x,
                            "log_det_x": log_det_x, "log_det_y": log_det_y}
        
        return pipeline

    # NOTE This is not clean but a fine temp fix for investigating...
    else:
        def pipeline(state):
            next_x, newton_iters, mean_cg_iters = integrators.implicit_newton_step(
                state, dt, potential_int, beta, noise_scale)
            
            return next_x, {"newton_iters": newton_iters, "mean_cg_iters": mean_cg_iters}

        return pipeline


def make_maila_pipeline(N, dt, potential_type, beta):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        next_x, accepts, newton_iters, mean_cg_iters = integrators.maila_newton_step(
            state, dt, potential_int, beta, noise_scale)

        return next_x, {"accepts": accepts, "newton_iters": newton_iters, "mean_cg_iters": mean_cg_iters}

    return pipeline

def make_imla_pipeline(N, dt, potential_type, beta, newton_tol = 1e-6):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]
    
    def pipeline(state):        
        next_x, _, newton_iters, mean_cg_iters  = integrators.imla_newton_step(state, dt, potential_int, noise_scale, newton_tol)
        return next_x, {"newton_iters": newton_iters, "mean_cg_iters": mean_cg_iters}
        
    return pipeline

def make_maimla_pipeline(N, dt, potential_type, beta, newton_tol = 1e-6):
    noise_scale = np.sqrt(2.0*dt/(beta*N))
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        next_x, accepts, crossing_rejects, newton_iters, mean_cg_iters = integrators.maimla_newton_step(
            state, dt, potential_int, beta, noise_scale, newton_tol = newton_tol)   
        return next_x, {"accepts": accepts, "cross_rejects": crossing_rejects, 
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

def make_hmc_pipeline(N, dt, potential_type, beta, L = 1, gamma_N = 1.0, alpha_N = 1.0):
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

        next_x, next_y, next_coulomb, next_H_N, accepts, cross_rejects = integrators.hmc_step(
            state, current_y, dt, potential_int, current_coulomb, current_H_N, beta, L = L, gamma_N = gamma_N, alpha_N = alpha_N
        )

        current_y = next_y; current_coulomb = next_coulomb; current_H_N = next_H_N
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
                       track_distance = False, grid = None, F_exact = None, distance_interval = 10, distance_type = "wasserstein",
                       track_newton_info = False, track_energy = False,
                       track_spacings = False, F_cdf = None):
    """ 
    Master observe function (combined all previous here). By default only track_snapshots is True.
        Snapshots: burn_in, defaults to 1/2 the num_steps.
        Accepts: track_accepts, NOTE right now only MALA returns cross rejects.
        Crossings: track_crossings, step_star.
        Distances: track_distance, grid (that the true cdf is evaluated on), dt, F_exact.
        Spacings: track_spacings, F_cdf (function)
    """

    # Defaults. 
    if burn_in is None:
        burn_in = num_steps//10 # Assumed initialised near equilibrium.

    results = {}
    if track_snapshots:
        snapshots = []
    
    if track_spacings:
        spacings = []

    if track_accepts:
        accepts = []
        cross_rejects = []
        negativity_rejects = []

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

    if track_energy:
        log_energy_x = []; log_energy_y = []
        log_transition_x_y = []; log_transition_y_x = []
        log_det_x = []; log_det_y = []
        
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
            if ("negativity_rejects" in info):
                negativity_rejects.append(info["negativity_rejects"])

        # Newton iterations.
        if (track_newton_info):
            if ("newton_iters" in info):
                newton_iters.append(info["newton_iters"])
            if ("mean_cg_iters" in info): 
                mean_cg_iters.append(info["mean_cg_iters"])
            if ("line_search_rejects" in info):
                line_search_rejects.append(info["line_search_rejects"])

        # Mean spacings.
        if (track_spacings):
            M, N = state.shape # Just reads metadata so no bother about speed.
            low = int(N/5); high = int(4*N/5);
            middle = state[:, low:high]

            xis = N*F_cdf(middle)
            gaps = np.diff(xis, axis = 1)
            if (np.abs(1 - gaps.mean()) > 0.05):
                raise ValueError("Mean spacings after transform is not 1.")
            
            spacings.append(gaps)

        if (track_energy):
            if ("log_energy_x" in info):
                log_energy_x.append(info["log_energy_x"])
            if ("log_energy_y" in info):
                log_energy_y.append(info["log_energy_y"])
            if ("log_transition_x_y" in info):
                log_transition_x_y.append(info["log_transition_x_y"])
            if ("log_transition_y_x" in info):
                log_transition_y_x.append(info["log_transition_y_x"])
            if ("log_det_x" in info):
                log_det_x.append(info["log_det_x"])
            if ("log_det_y" in info):
                log_det_y.append(info["log_det_y"])

    # End of trajectory loop, compile the dictionary and return.
    if track_snapshots:
        results["snapshots"] = np.array(snapshots)

    if (track_spacings):
        results["spacings"] = np.concatenate(spacings)
    
    if track_accepts:
        results["accepts"] = np.array(accepts)
        if cross_rejects:
            results["cross_rejects"] = np.array(cross_rejects)
        if negativity_rejects:
            results["negativity_rejects"] = np.array(negativity_rejects)
            
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

    if track_energy:
        results["log_energy_x"] = np.array(log_energy_x)
        results["log_energy_y"] = np.array(log_energy_y)
        results["log_transition_x_y"] = np.array(log_transition_x_y)
        results["log_transition_y_x"] = np.array(log_transition_y_x)
        results["log_det_x"] = np.array(log_det_x)
        results["log_det_y"] = np.array(log_det_y)

    return results

# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================

def target_dt(method, init, burn_steps, total_steps, potential_name, beta, dt_init, target, 
              L = 1, kappa = 0.7, newton_tol = 1e-6):
    """ 
    Accept MALA, HMC, MAIMLA.
    """

    M, N = init.shape
    x = np.copy(init); dt = dt_init 
    potential_int = potential_ints[potential_name]

    noise_scale = np.sqrt(2*dt/(beta*N))
    current_coulomb = forces.coulomb_interaction(x)
    current_H_N = forces.sum_hamiltonian(x, potential_int)
    if (method == "hmc"):
        current_y = np.random.normal(0, 1, x.shape)

    # Run the burn in.
    # NOTE now obselete since running from equilibrium.
    for _ in range(burn_steps):
        if (method == "hmc"):
            next_x, next_y, next_coulomb, next_H_N, accepts, _ = integrators.hmc_step(
                x, current_y, dt, potential_int, current_coulomb, current_H_N, beta, L = L)
            current_y = next_y 
        
        elif (method == "mala"):
            next_x, next_coulomb, next_H_N, accepts, _ = integrators.mala_step(
                x, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale)       

        elif (method == "maimla"):
            next_x, accepts, crossing_rejects, newton_iters, mean_cg_iters = integrators.maimla_newton_step(
                x, dt, potential_int, beta, noise_scale, newton_tol = newton_tol)

        x = next_x;

    # Post burn in: update according to log(dt_next) = log(dt) + log(1 - (accept - target)/(iter+1)^kappa),
    # translates to standard dt_next = dt + dt(accept-target)/(iter+1)^kappa.
    accept_history = np.zeros(total_steps); dt_history = np.zeros(total_steps);

    for step_idx in range(total_steps):
        noise_scale = np.sqrt(2*dt/(beta*N)) # Update every step because dt changes.

        # Run the integrator for this dt.
        if (method == "hmc"):
            next_x, next_y, next_coulomb, next_H_N, accepts, _ = integrators.hmc_step(
                x, current_y, dt, potential_int, current_coulomb, current_H_N, beta, L = L)
            current_y = next_y 
        
        elif (method == "mala"):
            next_x, next_coulomb, next_H_N, accepts, _ = integrators.mala_step(
                x, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale)       

        elif (method == "maimla"):
            next_x, accepts, _, _, _  = integrators.maimla_newton_step(
                x, dt, potential_int, beta, noise_scale, newton_tol = newton_tol)

        elif (method == "maila"):
            next_x, accepts, _, _ = integrators.maila_newton_step(
                x, dt, potential_int, beta, noise_scale, newton_tol = newton_tol)

        # Update dt and then all parameters.
        dt_next = dt + dt*(accepts - target)/((step_idx + 1)**kappa)
        dt = dt_next

        x = next_x
        if (method not in["maimla", "maila"]):
            current_coulomb = next_coulomb; current_H_N = next_H_N; 

        # Save history.
        accept_history[step_idx] = accepts
        dt_history[step_idx] = dt

    # End of range.
    return accept_history, dt_history

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


def make_mala_pipeline_wishart(N, dt, potential_type, beta, c = 1/2):
    """
    Temp modification to MALA pipe for the Wishart-Laguerre ensembles,
    needs c >0  passing in and needs an explicit positivty constraint on the
    particles because of the logarithm in the potential.
    """
    
    current_coulomb = None; current_H_N = None;
    potential_int = potential_ints[potential_type]
    if (potential_int != 3):
        raise ValueError("Don't call Wishart modification with wrong potential (use <wishart-laguerre>).")

    noise_scale = np.sqrt(2.0*dt/(beta*N))

    def pipeline(state):
        nonlocal current_coulomb, current_H_N
        
        # Only calculate on first step.
        if (current_coulomb is None):
            current_coulomb = forces.coulomb_interaction(state)
        if (current_H_N is None):
            current_H_N = forces.sum_hamiltonian(state, potential_int, c)

        next_x, next_coulomb, next_H_N, accepts, crossing_rejects, negativity_rejects = integrators.mala_step_wishart(
            state, dt, potential_int, current_coulomb, current_H_N, beta, noise_scale, c
        )

        current_coulomb = next_coulomb; current_H_N = next_H_N;
        return next_x, {"accepts": accepts, "cross_rejects": crossing_rejects, "negativity_rejects": negativity_rejects}

    return pipeline