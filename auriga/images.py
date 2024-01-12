import matplotlib.pyplot as plt
import numpy as np
from auriga.cosmology import Cosmology


def figure_setup():
    params = {'axes.labelsize': 10.0,
              'axes.titlesize': 8.0,
              'text.usetex': True,
              'figure.dpi': 500,
              'figure.facecolor': 'white',
              'font.size': 8.0,
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'font.family': 'serif',
              'xtick.top': 'on',
              "xtick.labelsize": 8.0,
              'xtick.major.width': 0.5,
              'xtick.major.size': 1.5,
              'xtick.minor.width': 0.25,
              'xtick.minor.size': 1.5,
              'ytick.right': 'on',
              'ytick.major.width': 0.5,
              'ytick.major.size': 1.5,
              'ytick.minor.width': 0.25,
              'ytick.minor.size': 1.5,
              "ytick.labelsize": 8.0,
              'savefig.dpi': 500,
              'savefig.bbox': 'tight',
              'savefig.pad_inches': 0.02}
    plt.rcParams.update(params)


def add_redshift(ax: plt.Axes) -> None:
    """
    This method add the redshift to a given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to add the redshift.
    """

    cosmology = Cosmology()
    ax2 = ax.twiny()
    ax2.tick_params(which='both', direction="in")
    ax2_label_values = [0.1, 0.5, 1, 2, 3, 10]
    ax2_ticklabels = ['0.1', '0.5', '1', '2', '3', '10']
    ax2_ticks = [cosmology.redshift_to_time(float(item)) for item in
                 ax2_label_values]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax2_ticks)
    if ax.get_subplotspec().is_first_row():
        ax2.set_xticklabels(ax2_ticklabels)
        ax2.set_xlabel('$z$')
    else:
        ax2.set_xticklabels([])


def set_axs_configuration(xlim: tuple, ylim: tuple,
                          xticks: list, yticks: list,
                          xlabel: str, ylabel: str,
                          axs: np.ndarray, n_used: int,
                          xscale: str = "linear", yscale: str = "linear",
                          xticklabels: list = None, yticklabels: list = None):
    """
    This method configures the axes properly for large figures.

    Parameters
    ----------
    xlim : tuple
        The limits of the x-axis.
    ylim : tuple
        The limits of the y-axis.
    xticks : list
        The ticks of the x-axis.
    yticks : list
        The ticks of the y-axis.
    xlabel : str
        The label of the x-axis.
    ylabel : str
        The label of the y-axis.
    axs : np.ndarray
        An array with all the axes.
    n_used : int
        The number of panels used.
    xscale : str
        The scale of the x-axis. `linear` by default.
    yscale : str
        The scale of the y-axis. `linear` by default.
    """
    for ax in axs.flat:
        ax.tick_params(which='both', direction="in")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    row_idx = (n_used - 1) // axs.shape[1]
    col_idx = n_used - row_idx * axs.shape[1] - 1

    for i in range(axs.shape[0]):
        axs[i, 0].set_ylabel(ylabel)
        axs[i, 0].set_yticks(yticks)
        for j in range(axs.shape[1]):
            if (i == row_idx and j <= col_idx) \
                    or (i == row_idx - 1 and j > col_idx):
                axs[i, j].xaxis.set_tick_params(labelbottom=True)
                axs[i, j].set_xlabel(xlabel)
                axs[i, j].set_xticks(xticks)
    
    for ax in axs.flat:  # After using set_xticks() and set_yticks()
        if xticklabels is not None: ax.set_xticklabels(xticklabels)
        if yticklabels is not None: ax.set_yticklabels(yticklabels)
