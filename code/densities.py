import numpy as np
import matplotlib.pyplot as plt

def numerical_int(vals, grid):
    # Standard numerical integration.
    dx = grid[1] - grid[0]
    return np.sum(vals) * dx

def get_potential(type):
    if (type == "quad-quartic"):
        return lambda x: x**2/2 + x**4/4
    elif (type == "quartic"):
        return lambda x: x**4/4
    elif (type == "quadratic"):
        return lambda x: x**2/2
    else:
        raise ValueError(f"Potential {type} not found.")

def orthogonal_polys(N, potential_func, grid):
    """ 
    Builds N monic, orthogonal polynomials w.r.t. exp(-NV) over a
    grid through the Gran-Schmidt process.
    """

    weight = np.exp(-N*potential_func(grid))
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

def construct_kernel(N, potential_func, grid, pis, c_sqrs):
    """ 
    Construct K_N by (a) calculating the wavefunctions and then (b)
    taking the outer product --- see Li and Menon.
    """

    # NOTE Here there's a /2 in the Gibbs factor.
    weight = np.exp(-N*potential_func(grid) / 2.0)
    phis = np.zeros((N, len(grid)))

    for k in range(N):
        phis[k] = (1.0 / np.sqrt(c_sqrs[k])) * weight * pis[k]

    # Evaluate sum as matrix multiplication:
    # (grid, N) * (N, grid): rows (r) x cols (s) --> (grid, grid).
    return phis.T @ phis

def compute_exact_cdf(N, potential, grid, cdf_tol = 0.05):
    """ 
    Construct true finite_N cdf F_N(x) from K_N (construct_kernel above).
    See (2.5) in Li and Menon.
    """

    potential_func = get_potential(potential)
    pis, c_sqrs = orthogonal_polys(N, potential_func, grid)
    K_N = construct_kernel(N, potential_func, grid, pis, c_sqrs)
    rho_N = np.diagonal(K_N) / N 

    dx = grid[1] - grid[0]
    F_exact = np.cumsum(rho_N)*dx

    # For floating point/discretisation errors.
    diff = np.abs(F_exact[-1] - 1)
    if (diff > cdf_tol):
        raise RuntimeWarning(f"True CDF more than {cdf_tol} from 1.")
    
    F_exact = F_exact / F_exact[-1]
    return F_exact


def compute_empirical_cdf(particles, grid):
    """ 
    Compute empirircal distribution from particle positions,
        F_{n, M} = 1/NM sum_M sum_N ind(lambda_emp <= lambda).

    Particles is a flatten array of ALL particles across all trials (N*M length).
    """

    sorted_particles = np.sort(particles)
    # https://numpy.org/devdocs/reference/generated/numpy.searchsorted.html
    counts = np.searchsorted(sorted_particles, grid, side = "right")

    F_emp = counts / len(sorted_particles)
    return F_emp

def theoretical_density(s, q = 1.0, g = 1.0):
    """
    See (2.1), (2.2), (2.3) in Li and Menon.
    Gives the exact density for different potentials, if mixed assumes x^2/2 + gx^4/4 
    Input:
        s (ndarray): range.
        g (float): quartic coefficient gx^4/4
    """

    density = np.zeros_like(s)
    q = float(q); g = float(g);
    
    if (q == 1.0 and g > 0):
        # (2.2), mixed case.
        a = np.sqrt((np.sqrt(1+12*g) - 1) / (6*g))
        condition = np.abs(s) <= 2*a

        # Avoid sqrt runtime errors.
        s_valid = s[condition]
        density[condition] = 1/(2*np.pi)*(1+2*g*(a**2) + g*(s_valid**2))*np.sqrt(4*(a**2) - s_valid**2)

    elif (g == 0) and (q == 1):
        # Pure quadratic case.
        R = 2.0
        condition = np.abs(s) <= R
        s_valid = s[condition]
        density[condition] = 2/(np.pi*(R**2))*np.sqrt((R**2) - (s_valid**2))
        
    else:   
        # Pure quartic case.
        a = (3.0*g)**(-1/4)
        condition = np.abs(s) <= 2*a

        s_valid = s[condition]
        density[condition] = 1/(2*np.pi)*(2*g*(a**2) + g*(s_valid**2))*np.sqrt(4*(a**2) - s_valid**2)

    return density

def compute_distance(P, Q, grid, distance_type = "ks", p = 1):
    """
    Computes the distance between P and Q (e.g. empirircal vs exact CDFs).
    NOTE might implement Wasserstein, KL, ... or can use SciPy's versions.
    """

    abs_difference = np.abs(P - Q)
    if (distance_type == "ks"):
        distance = np.max(abs_difference)
        return distance

    elif (distance_type == "wasserstein"):
        if (p == 1):
            # Special case: just integration.
            dx = grid[1] - grid[0]
            distance = np.sum(abs_difference)*dx
            return distance
        else:
            # Estimate inverse CDF by interpolating for range of p's.
            p_grid = np.linspace(0, 1, 2*len(grid))
            inv_P = np.interp(p_grid, P, grid)
            inv_Q = np.interp(p_grid, Q, grid)

            integral = np.sum(np.abs(inv_P - inv_Q)**p)*(p_grid[1] - p_grid[0])
            distance = integral**(1.0/p)
            return distance

    elif (distance_type == "kl"):
        distance = 23
        raise NotImplementedError("Implement KL (densities.py).")