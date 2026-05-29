import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 5)
import importlib
import sampler
importlib.reload(sampler);

def run_experiment(N, T, num_trials, potential, method = "tamed", bins = 100, crop = True, save = False):
    """ 
    
    """
    sim_evals = sampler.stochastic_sampler(N = N, T = T, num_trials = num_trials, potential = potential, method = method)
    
    # NOTE will have to change if using different potentials.
    q = 1.0 if potential == "mixed" else 0.0
    s_vals = np.linspace(-2.15, 2.15, 500)
    theoretical_density = sampler.theoretical_density(s_vals, q = q, g = 1.0)
    
    fig, ax = plt.subplots()
    ax.hist(sim_evals, bins = bins, density = True, alpha = 0.75, color = "steelblue",
            edgecolor = "black", label = f"N = {N} over {num_trials} trials")

    ax.plot(s_vals, theoretical_density, 'r--', lw = 2, label = "Theoretical Density")
    ax.grid(True, alpha = 0.3)
    ax.set_title(f"DBM ({method}) for N = {N}, T = {T}, over {num_trials} trials.", fontsize = 14)
    if (crop):
        ax.set_xlim(-2.35, 2.35)

    # plt.legend(fontsize = 12)
    print(f"With N = {N} over {num_trials} trials with {potential} potential run for T = {T}.")

    if (save):
        randint = np.random.randint(1, 999)
        plt.savefig(f"images/sampler{randint}.png", format = "png", dpi = 150)
        plt.close()

    plt.show()

    
