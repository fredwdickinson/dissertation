import numpy as np
import matplotlib.pyplot as plt

def numerical_int(vals, grid):
    # Standard numerical integration.
    dx = grid[1] - grid[0]
    return np.sum(vals) * dx

def orthogonal_polys(N, potential, grid):
    """ 
    Builds N monic, orthogonal polynomials w.r.t. exp(-NV) over a
    grid through the Gran-Schmidt process.
    """

    weight = np.exp(-N*potential(grid))
    pis = np.zeros((N, len(grid)))
    c_sqrs = np.zeros(N)

    # Base case: pi_0 = 1.
    pis[0] = np.ones_like(grid)
    c_sqrs[0] = numerical_int(pis[0]**2 * weight, grid)

    # Build higher order terms with the Gran-Schmidt process.
    # e.g. pi1(x) = x - proj_pi0(x) is linear term.
    for k in range(1, N):
        # Define highest order, subtract projections.
        pk = grid**k

        for j in range(k):
            # Subtract projection of all terms up to.
            # proj_pij(p) = <p, pij> / <pij, pij> * pij
            integrand = pk * pis[j] * weight
            proj_coeff = numerical_int(integrand, grid) / c_sqrs[j]

            pk = pk - proj_coeff*pis[j]

        # Finished building kth polynomial, store csqrs.
        pis[k] = pk 
        c_sqrs[k] = numerical_int(pis[k]**2 * weight, grid)

    return pis, c_sqrs

def construct_kernel(N, potential, grid, pis, c_sqrs):
    """ 
    Construct K_N by (a) calculating the wavefunctions and then (b)
    taking the outer product --- see Li and Menon.
    """

    # NOTE Here there's a /2 in the Gibbs factor.
    weight = np.exp(-N*potential(grid) / 2.0)
    phis = np.zeros((N, len(grid)))

    for k in range(N):
        phis[k] = (1.0 / np.sqrt(c_sqrs[k])) * weight * pis[k]

    # Evaluate sum as matrix multiplication:
    # (grid, N) * (N, grid): rows (r) x cols (s) --> (grid, grid).
    return phis.T @ phis

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