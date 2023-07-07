import matplotlib.pyplot as plt
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
              'ytick.right': 'on',
              'xtick.major.width': 0.5,
              'xtick.major.size': 1.5,
              'xtick.minor.width': 0.25,
              'xtick.minor.size': 1.5,
              'ytick.major.width': 0.5,
              'ytick.major.size': 1.5,
              'ytick.minor.width': 0.25,
              'ytick.minor.size': 1.5,
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
