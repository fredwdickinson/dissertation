import numpy as np
import python.forces as forces
import python.integrators as integrators

def get_pipeline(method, **kwargs):
    if (method == "euler"):
        return make_euler_pipeline(**kwargs)
    elif (method == "tamed"):
        return make_tamed_pipeline(**kwargs)
    elif (method == "implicit"):
        return make_implicit_pipeline(**kwargs)
    else:
        raise ValueError(f"Do not know method {method}.")
    
# ==========================================================================================================
    
def make_euler_pipeline(dt, noise_scale, beta, potential_type):
    def pipeline(state):
        coulomb, v_prime = forces.compute_forces(state, beta, potential_type)
        return integrators.euler_step(state, coulomb, v_prime, dt, noise_scale, beta)

    return pipeline

def make_tamed_pipeline(dt, noise_scale, beta, potential_type):
    # Step pipeline for tamed Euler.
    def pipeline(state):
        coulomb, v_prime = forces.compute_forces(state, beta, potential_type)
        return integrators.tamed_euler_step(state, coulomb, v_prime, dt, noise_scale, beta)

    return pipeline

def make_implicit_pipeline(dt, noise_scale, beta, potential_type):
    # Step pipeline for implicit methods: skips the Coulomb pre-computation.
    def pipeline(state):
        return integrators.implicit_newton_step(state, dt, noise_scale, beta, potential_type)
    
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
    yield state 

    for _ in range(steps):
        state = step_pipeline(state)
        yield state

# ====================================================================================================================
# "Observers" that use the trajectory information.
# collect_snapshots produces the hist every X after burn in, count_crossings looks for unique eigenvalue crossing, etc.

def collect_snapshots(trajectory, num_steps, burn_in = None, interval = None):
    """ 
    Looks at the trajectory's positions every 20th step after a long burn-in period.
    Room to change the burn-in or interval as parameters.
    """

    if (burn_in is None):
        burn_in = int(3/4*num_steps)
        interval = int(num_steps / 20)
     
    snapshots = []
    for step, state in enumerate(trajectory):
        if (step >= burn_in) and ((step - burn_in) % interval == 0):
            snapshots.append(np.copy(state))

    return np.concatenate(snapshots).flatten()


def count_crossings(trajectory, step_star):
    """
    Determines how many particles cross across all trials in a simulation at T_star (give step_star).
    Returns the total number of crossings.
    """
    total_crossings = 0
    prev_ordering = None

    for step, state in enumerate(trajectory):
        if (step == 0):
            prev_ordering = (state[:, :, None] < state[:, None, :])
        elif (step == step_star):
            # Similar to before, but handles all trials at once.
            current_ordering = (state[:, :, None] < state[:, None, :])
            flips = (current_ordering != prev_ordering)

            # k is the diagonal offset.
            return np.sum(np.triu(flips, k = 1))
        
    raise ValueError(r"Never reached $T^*$ in count_crossings().")