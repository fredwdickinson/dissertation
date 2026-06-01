import numpy as np
import matplotlib.pyplot as plt
import importlib
import time 

import sampler
importlib.reload(sampler);


def get_results(N, T, num_trials, potential, *, methods = None):
    results = []
    infos = []

    if (methods == None):
        methods = ["euler", "tamed", "implicit"]

    for method in methods:
        dt = 1/N if (method == "implicit") else 1/(N**2)
        start = time.perf_counter()
        res = sampler.stochastic_sampler(N, T, dt, num_trials = num_trials, potential = potential, method = method)
        end = time.perf_counter()
        time_taken = round(end - start, 2)
        mini = round(np.min(res), 2); maxi = round(np.max(res), 2)

        info = f"{method.capitalize()} in {time_taken}s over ({mini}, {maxi})."
        infos.append(info)
        results.append(res)

    return methods, results, infos

# ===========================================================
# NOTE Should change these from being two functions/refactor?

def plot_results(potential, methods, results):
    q = 1 if potential == "quad-quartic" else 0
    g = 1

    s_vals = np.linspace(-2.5, 2.5, 500)
    density = sampler.theoretical_density(s_vals, q, g)

    fig, axes = plt.subplots(1, len(methods), figsize = (21, 5))
    axes = np.atleast_1d(axes)
    for method, res, ax in zip(methods, results, axes):
        plot(res, s_vals, density, title = method.capitalize(), ax = ax)

    plt.show()

def plot(results, density_range, density, *, 
                 bins = 50, title = None, ax = None, plot = False, save = False, crop = True):
    """ 
    
    """

    if (ax == None):
        fig, ax = plt.subplots()

    ax.hist(results, bins = bins, density = True, alpha = 0.75, color = "steelblue", edgecolor = "none")
    ax.plot(density_range, density, 'r--', lw = 2)

    ax.grid(True, alpha = 0.3)
    ax.set_title(title)
    ax.set_xlim(-2.35, 2.35)

    if (not crop):
        ax.autoscale()
        ax.set_ylim(0, 0.005)

    if (save):
        randint = np.random.randint(1, 999)
        plt.savefig(f"images/sampler{randint}.png", format = "png", dpi = 150)
        plt.close()
    
    if (plot):
        plt.show()
        return
    
    # print(f"With N = {N} over {num_trials} trials with {potential} potential run for T = {T}.")
    return ax
    
