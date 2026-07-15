
from python.densities import get_density
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dissertation.mplstyle')

def plot_hist(ax, particles, num_bins = 100, with_limit = True, potential_name = None):
    """
    Histogram plotter.
    """
    ax.hist(particles, bins = num_bins, density = True, color = "steelblue", edgecolor = "black", alpha = 0.75)

    if (with_limit):
        density_range, limiting_density = get_density(potential_name)
        ax.plot(density_range, limiting_density, color = "red", lw = 1.5)

    ax.grid(False)
    ax.set_yticks([0, 0.3])

    return ax

def plot_accepts(ax, accepts):
    """
    Acceptance rate plotter.
    """
    x = list(range(accepts.shape[0]))

    ax.plot(x, accepts, label = "acceptance rate")
    ax.axhline(np.mean(accepts), color = "orange", linestyle = "--", lw = 1.5, label = f"average {np.mean(accepts):.2f}")
    ax.legend(loc = "upper right")

    return ax

def plot_distances(ax, info):
    """
    Distance plotter.
    """

    if ("distance_times" not in info) or ("distances" not in info):
        raise ValueError("Info passed does not contain distance information.")

    ax.plot(info["distance_times"], np.log10(info["distances"]), lw = 1.5)
    return ax