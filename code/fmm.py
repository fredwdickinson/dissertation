import numpy as np
import warnings
from math import comb # binomial coefficients
# In comments C(n, k) = nCk

def M2M(child_moments, child_centre, parent_centre, p):
    """
    Re-expand child multipole around parent centre.
    """
    delta = child_centre - parent_centre
    out = np.zeros(p + 1)

    for k in range(p + 1):
        for n in range(k + 1):
            out[k] += comb(k, n)*(delta**(k - n))*child_moments[n]

    return out

def M2L(source_moments, source_centre, target_centre, p):
    """
    Convert multipole at source box into local expansion at target box.
    """
    R = target_centre - source_centre
    out = np.zeros(p + 1)

    for n in range(p + 1):
        s = sum(comb(n + k, k)*source_moments[k]/R**(n + k + 1) for k in range(p + 1))
        out[n] = ((-1) ** n) * s

    return out

def L2L(parent_local, parent_centre, child_centre, p):
    """
    Shift local expansion from parent box to child box.
    """
    d = child_centre - parent_centre
    out = np.zeros(p + 1)

    for m in range(p + 1):
        for n in range(m, p + 1):
            out[m] += comb(n, m)*parent_local[n]*(d**(n - m))

    return out

def fmm_coulomb_1d(lambdas, p = 10, s = None):
    """
    Approximate the coulomb interaction as described.
    Particles assumed to be sorted on input.
    Input:
        lambdas (ndarry): sorted 1D array of N particles.
        p (int): expansion order.
        s (int): max particles per leaf box.
    Output:
        f (ndarray): array of shape (N), the Coulomb sum for each particle.
    """


    N = len(lambdas)
    if (N < 100):
        warnings.warn(f"Dimension in FMM coulomb interaction is small (N = {N}).")

    if s is None:
        s = max(p, min(20, N // 4))

    # Determine tree structure.
    num_levels = int(np.ceil(np.log2(N / s))) + 1
    num_levels = max(num_levels, 2)
    L = num_levels - 1 # leaf index

    def box_slice(l, b):
        # At level l there are 2^l boxes.
        # Returns the start/end indices of a box b.
        size = N / 2**l
        start = int(np.round(b * size))
        end = int(np.round((b + 1) * size))

        return start, end

    def box_centre(l, b):
        # Centre of a box.
        start, end = box_slice(l, b)
        return (lambdas[start] + lambdas[end - 1]) / 2.0

    slices  = {(l, b): box_slice(l, b)  for l in range(num_levels) for b in range(2**l)}
    centres = {(l, b): box_centre(l, b) for l in range(num_levels) for b in range(2**l)}

    # Storage: moments[l][b] and locals_[l][b] are arrays of shape (p+1,).
    moments = [[None] * (2**l) for l in range(num_levels)]
    locals_ = [[np.zeros(p+1) for _ in range(2**l)] for l in range(num_levels)]

    # Leaf boxes: compute moments directly from particles.
    # M_k = sum_{j in box} (lambda_j - c_box)^k
    for b in range(2**L):
        start, end = slices[(L, b)]
        if start >= end:
            moments[L][b] = np.zeros(p + 1)
            continue

        dx = lambdas[start:end] - centres[(L, b)]
        moments[L][b] = (dx[:, None] ** np.arange(p + 1)).sum(axis=0)

    # Internal nodes: propagate up using M2M (multipole to multipole, upward pass).
    for l in range(num_levels - 2, -1, -1):
        for b in range(2**l):
            b_left, b_right = 2*b, 2*b + 1
            moments[l][b] = (M2M(moments[l+1][b_left],  centres[(l+1, b_left)],  centres[(l, b)], p) +
                M2M(moments[l+1][b_right], centres[(l+1, b_right)], centres[(l, b)], p))

    #
    # Downward pass.
    for l in range(1, num_levels):
        num_boxes = 2**l

        for b in range(num_boxes):
            parent = b//2
            b_centre = centres[(l, b)]

            # Accumulate far field for all boxes in interaction list.
            for dp in [-1, 0, 1]: # parent's neighbours
                bp = parent + dp
                if bp < 0 or bp >= 2**(l-1):
                    continue
                
                # Children of parent's neighbour
                for dc in [0, 1]: 
                    b_src = 2*bp + dc
                    if b_src < 0 or b_src >= num_boxes:
                        continue
                    
                    # Adjacent to b: skip
                    if abs(b_src - b) <= 1:          
                        continue

                    locals_[l][b] += M2L(moments[l][b_src], centres[(l, b_src)], b_centre, p)

            # Children inheret local expansion.
            locals_[l][b] += L2L(locals_[l-1][parent], centres[(l-1, parent)], b_centre, p)

    # Evaluate leaves as sum of near and far.
    f = np.zeros(N)
    for b in range(2**L):
        start, end = slices[(L, b)]
        if start >= end:
            continue

        dx = lambdas[start:end] - centres[(L, b)]
        powers = dx[:, None] ** np.arange(p + 1) # shape (block, p+1).
        f[start:end] += powers @ locals_[L][b] # explicit MV product.

        # Near field: direct sum over this box and its adjacent boxes.
        b_left = max(b - 1, 0)
        b_right = min(b + 1, 2**L - 1)
        s_near, _ = slices[(L, b_left)]; _, e_near = slices[(L, b_right)]
        near_idx = np.arange(s_near, e_near) 

        for i in range(start, end):
            mask = near_idx != i
            diffs = lambdas[i] - lambdas[near_idx[mask]]
            f[i] += np.sum(1.0 / diffs)

    coulomb_approx = f/N
    return coulomb_approx

# 

def coulomb_difference(x, N, difference_only = False):
    """ 
    Computes the Coulomb difference for some particles x \in R^N.
    """

    if (N <= 5000):
        diff = np.subtract.outer(x, x)           
        np.fill_diagonal(diff, np.inf)

        if (difference_only):
            return diff
        else:
            coulomb = np.sum(1.0 / diff, axis = 1) / N
            return coulomb
    else:
        raise NotImplementedError("(fmm.py) Coulomb interaction for large $N$.")


