import numpy as np
import python.forces as forces
import python.integrators as integrators

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
    
def make_euler_pipeline(dt, noise_scale, potential_type):
    def pipeline(state):
        coulomb = forces.coulomb_interaction(state)
        v_prime_func = forces.get_force(potential_type, "grad")
        v_prime = v_prime_func(state)

        next_x = integrators.euler_step(state, coulomb, v_prime, dt, noise_scale)
        return next_x, {}

    return pipeline

def make_tamed_pipeline(dt, noise_scale, potential_type):
    # Step pipeline for tamed Euler.
    def pipeline(state):
        coulomb = forces.coulomb_interaction(state)
        v_prime_func = forces.get_force(potential_type, "grad")
        v_prime = v_prime_func(state)

        next_x = integrators.tamed_euler_step(state, coulomb, v_prime, dt, noise_scale)
        return next_x, {}

    return pipeline

def make_implicit_pipeline(dt, noise_scale, potential_type):
    # Step pipeline for implicit methods: skips the Coulomb pre-computation.
    def pipeline(state):
        potential_int = potential_ints[potential_type]
        next_x = integrators.implicit_newton_step(state, dt, potential_int, noise_scale)
        return next_x, {}
    
    return pipeline

def make_imla_pipeline(dt, noise_scale, potential_type, beta, metropolise):
    # 
    def pipeline(state):
        potential_int = potential_ints[potential_type]
        next_x, accepts = integrators.imla_step(state, dt, potential_int, noise_scale, beta, metropolise)
        return next_x, {"accepts": accepts}
    
    return pipeline


def make_mala_pipeline(dt, noise_scale, potential_type, beta):
    """
    Step pipeline for MALA. 
    Maintains state of forces to avoid recalculation (hence the nonlocal).
    """
    
    current_coulomb = None
    potential_int = potential_ints[potential_type]

    def pipeline(state):
        nonlocal current_coulomb
        
        # Only calculate on first step.
        if (current_coulomb is None):
            current_coulomb = forces.coulomb_interaction(state)

        next_x, next_coulomb, accepts, crossing_rejects = integrators.mala_step(state, dt, potential_int, current_coulomb, noise_scale, beta)

        current_coulomb = next_coulomb
        return next_x, {"accepts": accepts, "cross_rejects": crossing_rejects}

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

# ====================================================================================================================
# "Observers" that use the trajectory information.
# collect_snapshots produces the hist every X after burn in, count_crossings looks for unique eigenvalue crossing, etc.

def metropolis_experiment(trajectory, num_steps, burn_in = None, interval = None):
    """ 
    Histogram plotter and acceptance rate.
    """
    if (burn_in is None):
        burn_in = int(3/4*num_steps)
        interval = int(num_steps / 20)

    accepts = []; cross_rejects = [];
    snapshots = []
    for step, (state, info) in enumerate(trajectory):
        # For histogram.
        if (step >= burn_in) and ((step - burn_in) % interval == 0):
            snapshots.append(np.copy(state))

        if (step > 0):
            accepts.append(info["accepts"])

    return np.concatenate(snapshots).flatten(), np.array(accepts)

def collect_snapshots(trajectory, num_steps, burn_in = None, interval = None):
    """ 
    Looks at the trajectory's positions every 20th step after a long burn-in period.
    Room to change the burn-in or interval as parameters.
    """

    if (burn_in is None):
        burn_in = int(3/4*num_steps)
        interval = int(num_steps / 20)
     
    snapshots = []
    for step, (state, info) in enumerate(trajectory):
        if (step >= burn_in) and ((step - burn_in) % interval == 0):
            snapshots.append(np.copy(state))

    return np.concatenate(snapshots).flatten()

def acceptance_rate(trajectory):
    """
    
    """

    accepts = []; cross_rejects = [];
    for step, (state, info) in enumerate(trajectory):
        if (step > 0):
            accepts.append(info["accepts"])

    return np.array(accepts)


def count_crossings(trajectory, step_star):
    """
    Determines how many particles cross across all trials in a simulation at T_star (give step_star).
    Returns the total number of crossings.
    """
    total_crossings = 0
    prev_ordering = None

    for step, (state, info) in enumerate(trajectory):
        if (step == 0):
            prev_ordering = (state[:, :, None] < state[:, None, :])
        elif (step == step_star):
            # Similar to before, but handles all trials at once.
            current_ordering = (state[:, :, None] < state[:, None, :])
            flips = (current_ordering != prev_ordering)

            # k is the diagonal offset.
            return np.sum(np.triu(flips, k = 1))
        
    raise ValueError(r"Never reached $T^*$ in count_crossings().")