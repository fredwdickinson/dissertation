import numpy as np
from numba import njit 

def get_force(name, deriv):
    """
    Returns the function for potential 'name' and deriv e.g. None, "grad", "hess".
    """

    match name:
        case "quartic":
            match deriv:
                case None:
                    return potential_quartic
                case "grad":
                    return grad_quartic
                case "hess":
                    return hess_quartic
        case "quadratic":
            match deriv:
                case None:
                    return potential_quadratic
                case "grad":
                    return grad_quadratic
                case "hess":
                    return hess_quadratic
        case _:
            raise ValueError(f"Could not find force: {name}, {deriv}.")
        
# =================================================================================================

@njit
def potential_quadratic(x):
    return (x**2)/2.0

@njit
def potential_quartic(x):
    return (x**4)/4.0

@njit
def potential_quad_quartic(x):
    return (x**2)/2.0 + (x**4)/4.0

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
def coulomb_interaction(x):
    """
    Calculates Coulomb interaction in 1d or 2d (1/)
    NOTE Use FMM for N >= 5000?
    """

    if (x.ndim == 1):
        N = x.shape[0]
        coulomb = np.zeros(N)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    # += inherently handles the summation.
                    coulomb[i] += 1.0 / (x[i] - x[j])
                    
        return coulomb/N
    elif (x.ndim == 2):
        M, N = x.shape
        coulomb = np.zeros((M, N))
        
        for m in range(M):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        # += inherently handles the summation.
                        coulomb[m, i] += 1.0 / (x[m, i] - x[m, j])
                        
        return coulomb/N
    else:
        raise ValueError(f"Input array to coulomb interaction wrong shape ({x.shape}).")

@njit
def log_repulsion(x):
    """ 
    Calculates log repulsion (sum i<j log(x_j) - log(x_i)),
    as in the standard Hamiltonian.
    """

    if (x.ndim == 1):
        N = x.shape[0]
        repulsion = np.zeros(N)

        for i in range(N):
            for j in range(i + 1, N):
                repulsion[i] += np.log(np.abs(x[j] - x[i]))
        
        return repulsion/N
    
    elif (x.ndim == 2):
        raise NotImplementedError("Dimension 2 (with trials) in log repulsion.")

    else:
        raise ValueError(f"Input array to log repulsion wrong shape ({x.shape}).") 