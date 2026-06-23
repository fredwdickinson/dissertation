import numpy as np
from numba import njit 

@njit
def potential_quadratic(x):
    return np.sum((x**2)/2.0)

@njit
def potential_quartic(x):
    return np.sum((x**4)/4.0)

@njit
def potential_quad_quartic(x):
    return np.sum((x**2)/2.0 + (x**4)/4.0)

#
# First derivatives for e.g. drift term.
#


@njit
def grad_quadratic(x):
    return x

@njit
def grad_quartic(x):
    return x**3

@njit
def grad_quad_quartic(x):
    return x + x**3

#
# Second derivatives for Newton solver and ... NOTE.
#

@njit
def hess_quadratic(x):
    return np.ones_like(x)

@njit
def hess_quartic(x):
    return 3.0*(x**2)

@njit
def hess_quad_quartic(x):
    return 1.0 + 3.0*(x**2)

#
# Coulomb interaction: for now do naive, later re-implement 
# the FMM that is already Numba prepped.
#

@njit
def coulomb_interaction_1d(x):
    """ 
    Returns 1/N*sum(1/x_i - x_j).
    NOTE may change to compute distances only.
    NOTE will be replaced by FMM when N is large.
    """
    N = len(x)
    interaction = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if (i != j):
                interaction[i] += 1.0/(x[i] - x[j])
    
    return interaction/N

#
# Drift for general beta-ensembles...
# NOTE Check this is right against notes.
#


@njit
def compute_forces(x, beta, potential_type):
    """
    Calculates the Coulomb interaction and the confinement potential, returning them separately.
    Input x is of shape (M, N), returns two lots of shape M, N.
    """
    M, N = x.shape 
    coulomb_interaction = np.zeros((M, N))

    # NOTE Should make Coulomb more efficient (but Numba does like loops for m, at least).
    # USE global Coulomb.
    for m in range(M):
        for i in range(N):
            for j in range(N):
                if (i != j):
                    coulomb_interaction[m, i] += 1.0/(x[m, i] - x[m, j])

    coulomb_interaction = coulomb_interaction/N

    # Find potential and add on to coulomb interaction term.
    if potential_type == "quadratic":
        confinement = grad_quadratic(x)

    elif potential_type == "quartic":
        confinement = grad_quartic(x)

    elif potential_type == "quad-quartic":
        confinement = grad_quad_quartic(x)
    else:
        raise ValueError("Invalid potential type specified in compute_drift().")
    
    return coulomb_interaction, confinement