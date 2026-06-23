import numpy as np
import python.forces as forces
import python.integrators as integrators

def make_tamed_pipeline(dt, noise_scale, beta, potential_type):
    # Step pipeline for explicit methods (e.g. euler-maruyama, tamed version).
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
        init (ndarray): initial particle state.
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
    
    """

    if (burn_in is None):
        burn_in = int(3/4*num_steps)
        interval = int(num_steps / 20)
     
    snapshots = []
    for step, state in enumerate(trajectory):
        if (step >= burn_in) and ((step - burn_in) % interval == 0):
            snapshots.append(np.copy(state))

    return np.concatenate(snapshots).flatten()


def count_crossings(trajectory):
    """

    """
    total_crossings = 0
    prev_ordering = None
    for step, state in enumerate(trajectory):
        # An NxN boolean matrix with True := state[i] < state[j].
        current_ordering = (state[:, None] < state)

        if step == 0:
            prev_ordering = current_ordering
            continue
        
        flips = (current_ordering != prev_ordering)
        crossings_in_step = np.sum(np.triu(flips, k = 1)) # upper triangular only (k is diag offset).
        total_crossings += crossings_in_step
        prev_ordering = current_ordering
        
    return total_crossings