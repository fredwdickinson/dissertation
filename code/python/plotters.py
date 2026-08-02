
from python.densities import get_density, wigner_surmise
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dissertation.mplstyle')

def plot_hist(ax, particles, num_bins = 100, with_limit = True, potential_name = None, c = None):
    """
    Histogram plotter.
    """

    edgecolour = "black" if num_bins <= 100 else None
    ax.hist(particles, bins = num_bins, density = True, color = "steelblue", edgecolor = edgecolour, alpha = 0.75)

    if (with_limit):
        density_range, limiting_density = get_density(potential_name, c = c)
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

def plot_distances(ax, info, label = None):
    """
    Distance plotter.
    """

    if ("distance_times" not in info) or ("distances" not in info):
        raise ValueError("Info passed does not contain distance information.")
    
    if (label):
        ax.plot(info["distance_times"], np.log10(info["distances"]), lw = 1.5, label = label)
    else:
        ax.plot(info["distance_times"], np.log10(info["distances"]), lw = 1.5)
    return ax


def plot_spacings(ax, spacings, beta, num_bins = 100, with_limit = True, hist_label = None):
    """
    Spacings plotter.
    """
    edgecolour = "black" if num_bins <= 100 else None
    ax.hist(spacings, bins = num_bins, density = True, color = "steelblue", edgecolor = edgecolour, alpha = 0.75, label = hist_label)
    
    if (with_limit):
        space = np.linspace(0, 3, 500)
        surmise = wigner_surmise(space, beta)
        ax.plot(space, surmise, color = "red", lw = 1.5, label = rf"$\beta =$ {beta}")
        # ax.legend(fontsize = 14)

    ax.grid(False)
    ax.set_yticks([0, 0.5, 1])

    return ax

