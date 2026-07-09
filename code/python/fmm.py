import numpy as np
import warnings
from numba import njit

"""
Numba ready FMM. See notes for a detailed explanation of the algorithms.
"""

@njit
def comb(n, k):
    """ 
    Binomial coefficients nCk.
    """
    if (k < 0) or (k > n):
        return 0
    if (k == 0) or (k == n):
        return 1
    
    k = min(k, n - k); result = 1
    for i in range(k):
        result = result*(n - i)//(i + 1)

    return result

# ==========================================================================================================

@njit
def M2M(child_moments, child_centre, parent_centre, p):
    # Multipole to multipole.
    delta = child_centre - parent_centre
    out = np.zeros(p + 1)

    for k in range(p + 1):
        for n in range(k + 1):
            out[k] += comb(k, n)*(delta**(k - n))*child_moments[n]

    return out

@njit
def M2L(source_moments, source_centre, target_centre, p):
    # Multipole to local.
    R = target_centre - source_centre
    out = np.zeros(p + 1)

    for n in range(p + 1):
        s = 0
        for k in range(p + 1):
            s += comb(n + k, k)*source_moments[k]/R**(n + k + 1)
        out[n] = ((-1)**n)*s

    return out

@njit
def L2L(parent_local, parent_centre, child_centre, p):
    # Local to local.
    d = child_centre - parent_centre
    out = np.zeros(p + 1)

    for m in range(p + 1):
        for n in range(m, p + 1):
            out[m] += comb(n, m)*parent_local[n]*(d**(n - m))
    return out

# ==========================================================================================================

@njit
def bh_evaluate(lambdas, moments_arr, centres_arr, slices_start, slices_end, num_levels, L_leaf, p, N, theta):
    """
    Barnes-Hut Evaluator. Traverses the tree for every particle.
    Uses a stack (different to the FMM implementation below).
    """

    f = np.zeros(N)
    
    # Pre-allocate the Depth-First Search (DFS) stack. 
    # Because we push 2 children per level, max depth is 2*num_levels.
    stack_l = np.zeros(num_levels*2, dtype = np.int32)
    stack_b = np.zeros(num_levels*2, dtype = np.int32)
    
    for i in range(N):
        x = lambdas[i]; force = 0.0
        
        # Start the search at the root of the tree (level 0, box 0).
        stack_ptr = 0; stack_l[stack_ptr] = 0; stack_b[stack_ptr] = 0
        stack_ptr += 1
        
        while (stack_ptr > 0):
            # Pop the current box off the stack.
            stack_ptr -= 1
            l = stack_l[stack_ptr]; b = stack_b[stack_ptr]
            start = int(slices_start[l, b]); end = int(slices_end[l, b])
            
            if (start >= end):
                continue # Skip empty boxes.
                
            if (l == L_leaf):
                # Calculate direct interactions.
                for j in range(start, end):
                    if (j != i):
                        force += 1/(x - lambdas[j])
            else:
                # Recursive case: check MAC.
                c = centres_arr[l, b]
                
                # Spatial width defined by the extreme particles in the box.
                box_width = lambdas[end - 1] - lambdas[start]
                r = x - c
                
                # MAC: (Width of Box) / (Distance to Box) < Theta
                # We include abs(r) > 1e-14 to prevent division by zero.
                if (abs(r) > 1e-14) and ((box_width/abs(r)) < theta):
                    # MAC PASSES: The box is "far enough" to be a multipole.
                    for k in range(p + 1):
                        force += moments_arr[l, b, k] / (r**(k + 1))
                else:
                    # The box is too close. "Open" it and push children to stack.
                    stack_l[stack_ptr] = l + 1
                    stack_b[stack_ptr] = 2 * b
                    stack_ptr += 1 # left child.
                    
                    stack_l[stack_ptr] = l + 1
                    stack_b[stack_ptr] = 2 * b + 1
                    stack_ptr += 1 # right child.
            
        f[i] = force
        
    return f / N

def barnes_hut_coulomb_1d(lambdas, p = 10, s = None, theta = 0.5):
    """
    Approximate the coulomb interaction using Barnes-Hut.
    Input:
        lambdas (ndarray): sorted 1D array of N particles.
        p (int): expansion order.
        s (int): max particles per leaf box.
        theta (float): Multipole Acceptance Criterion. Lower = more accurate/slower.
    """

    N = len(lambdas)
    if (s is None):
        s = max(p, min(20, N//4))

    # Exactly as in FMM implementation.
    num_levels = int(np.ceil(np.log2(N/s))) + 1
    num_levels = max(num_levels, 2)
    L_leaf = num_levels - 1

    def box_slice(l, b):
        size = N/2**l
        start = int(np.round(b*size))
        end = int(np.round((b + 1)*size))
        return start, end

    def box_centre(l, b):
        start, end = box_slice(l, b)
        return (lambdas[start] + lambdas[end - 1])/2.0

    slices = {(l, b): box_slice(l, b)  for l in range(num_levels) for b in range(2**l)}
    centres = {(l, b): box_centre(l, b) for l in range(num_levels) for b in range(2**l)}

    max_boxes = 2**num_levels
    slices_start = np.zeros((num_levels, max_boxes))
    slices_end = np.zeros((num_levels, max_boxes))
    centres_arr = np.zeros((num_levels, max_boxes))

    for l in range(num_levels):
        for b in range(2**l):
            slices_start[l, b] = slices[(l, b)][0]
            slices_end[l, b] = slices[(l, b)][1]
            centres_arr[l, b] = centres[(l, b)]

    moments = [[None]*(2**l) for l in range(num_levels)]

    #  Upward pass.
    for b in range(2**L_leaf):
        start, end = slices[(L_leaf, b)]
        if (start >= end):
            moments[L_leaf][b] = np.zeros(p + 1)
            continue

        dx = lambdas[start:end] - centres[(L_leaf, b)]
        moments[L_leaf][b] = (dx[:, None]**np.arange(p + 1)).sum(axis = 0)

    for l in range(num_levels - 2, -1, -1):
        for b in range(2**l):
            b_left, b_right = 2*b, 2*b + 1
            moments[l][b] = (
                M2M(moments[l + 1][b_left], centres[(l + 1, b_left)], centres[(l, b)], p) +
                M2M(moments[l + 1][b_right], centres[(l + 1, b_right)], centres[(l, b)], p)
            )

    moments_arr = np.zeros((num_levels, max_boxes, p + 1), dtype=np.float64)
    for l in range(num_levels):
        for b in range(2**l):
            if moments[l][b] is not None:
                moments_arr[l, b, :] = moments[l][b]

    # Barnes-Hut: NO LOCAL ARRAYS.
    return bh_evaluate(lambdas, moments_arr, centres_arr, slices_start, slices_end, 
                       num_levels, L_leaf, p, N, theta)

# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================


@njit
def fmm_downward_and_eval(lambdas, moments_arr, locals_arr, centres_arr,
                            slices_start, slices_end, num_levels, L, p, N):
    # Downward pass.
    for l in range(1, num_levels):
        num_boxes = 2**l
        for b in range(num_boxes):
            parent = b//2
            b_centre = centres_arr[l, b]

            # Accumulate far field for all boxes in interaction list.
            for dp in range(-1, 2): # Parent's neighbours.
                bp = parent + dp
                if (bp < 0) or (bp >= 2**(l - 1)):
                    continue

                for dc in range(0, 2): # Children of parent's neighbour.
                    b_src = 2*bp + dc
                    if b_src < 0 or b_src >= num_boxes:
                        continue

                    if abs(b_src - b) <= 1: # Adjacent to b: skip.
                        continue

                    locals_arr[l, b] += M2L(moments_arr[l, b_src], centres_arr[l, b_src], b_centre, p)

            # Children inherit local expansion.
            locals_arr[l, b] += L2L(locals_arr[l - 1, parent], centres_arr[l - 1, parent], b_centre, p)

    # Evaluate leaves as sum of near and far.
    f = np.zeros(N)
    for b in range(2**L):
        start = slices_start[L, b]
        end = slices_end[L, b]
        if start >= end:
            continue

        # Far field: local expansion.
        for i in range(start, end):
            dx = lambdas[i] - centres_arr[L, b]
            for k in range(p + 1):
                f[i] += locals_arr[L, b, k]*dx**k

        # Near field: direct sum over this box and its adjacent boxes.
        b_left = max(b - 1, 0)
        b_right = min(b + 1, 2**L - 1)
        s_near = slices_start[L, b_left]
        e_near = slices_end[L, b_right]

        for i in range(start, end):
            for j in range(s_near, e_near):
                if (j != i):
                    f[i] += 1.0 / (lambdas[i] - lambdas[j])

    return f/N

def fmm_coulomb_1d(lambdas, p = 10, s = None):
    """
    Approximate the coulomb interaction as described.
    Particles assumed to be sorted on input.
    Input:
        lambdas (ndarry): sorted 1D array of N particles.
        p (int): expansion order.
        s (int): max particles per leaf box.
    Output:
        f (ndarray): the Coulomb sum for each particle.
    """

    N = len(lambdas)
    if s is None:
        s = max(p, min(20, N//4))

    # Determine tree structure.
    num_levels = int(np.ceil(np.log2(N / s))) + 1
    num_levels = max(num_levels, 2)
    L = num_levels - 1

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

    slices = {(l, b): box_slice(l, b)  for l in range(num_levels) for b in range(2**l)}
    centres = {(l, b): box_centre(l, b) for l in range(num_levels) for b in range(2**l)}

    max_boxes = 2**num_levels
    slices_start = np.zeros((num_levels, max_boxes))
    slices_end = np.zeros((num_levels, max_boxes))
    centres_arr = np.zeros((num_levels, max_boxes))

    for l in range(num_levels):
        for b in range(2**l):
            slices_start[l, b] = slices[(l, b)][0]
            slices_end[l, b] = slices[(l, b)][1]
            centres_arr[l, b] = centres[(l, b)]

    # Storage: moments[l][b] shape (p + 1,).
    moments = [[None]*(2**l) for l in range(num_levels)]

    # Leaf boxes: compute moments directly from particles.
    # M_k = sum_{j in box} (lambda_j - c_box)^k
    for b in range(2**L):
        start, end = slices[(L, b)]
        if (start >= end):
            moments[L][b] = np.zeros(p + 1)
            continue

        dx = lambdas[start:end] - centres[(L, b)]
        moments[L][b] = (dx[:, None]**np.arange(p + 1)).sum(axis = 0)

    # Internal nodes: propagate up using M2M.
    for l in range(num_levels - 2, -1, -1):
        for b in range(2**l):
            b_left, b_right = 2*b, 2*b + 1
            moments[l][b] = (
                M2M(moments[l + 1][b_left], centres[(l + 1, b_left)], centres[(l, b)], p) +
                M2M(moments[l + 1][b_right], centres[(l + 1, b_right)], centres[(l, b)], p)
            )

    # Convert moments and locals to 3D numpy arrays for Numba.
    moments_arr = np.zeros((num_levels, max_boxes, p + 1), dtype = np.float64)
    locals_arr = np.zeros((num_levels, max_boxes, p + 1), dtype = np.float64)

    for l in range(num_levels):
        for b in range(2**l):
            if moments[l][b] is not None:
                moments_arr[l, b, :] = moments[l][b]

    # Run the downward pass and evaluate.
    return fmm_downward_and_eval(lambdas, moments_arr, locals_arr, centres_arr,
                                  slices_start, slices_end, num_levels, L, p, N)
