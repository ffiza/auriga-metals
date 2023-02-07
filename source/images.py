import matplotlib.pyplot as plt
from cosmology import Cosmology


TICK_MAJOR_WIDTH: float = 0.5
TICK_MAJOR_SIZE: float = 1.5
TICK_MINOR_WIDTH: float = 0.25
TICK_MINOR_SIZE: float = 1.5

DPI: int = 800
SAVEFIG_PAD_INCHES: float = 0.02

LABEL_SIZE: float = 10.0
FONTSIZE: float = 8.0
TITLE_FONTSIZE: float = 8.0
GALAXY_LABEL_FONTSIZE: float = 7.0
LEGEND_FONTSIZE: float = 4.0

HALF_WIDTH: float = 3.0
FULL_WIDTH: float = 2.0 * HALF_WIDTH
STANDARD_HEIGHT: float = 2.0
FIGSIZE: tuple = (HALF_WIDTH, HALF_WIDTH)
SIZE_FACT: float = 1.2

MARKER_SIZE: float = 4.0
MARKER_EDGE_WIDTH: float = 1.5
LINE_WIDTH: float = 1.0


def figure_setup():
    params = {'axes.labelsize': LABEL_SIZE,
              'axes.titlesize': TITLE_FONTSIZE,
              'text.usetex': True,
              'figure.dpi': DPI,
              'figure.figsize': FIGSIZE,
              'figure.facecolor': 'white',
              'font.size': FONTSIZE,
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'font.family': 'serif',
              'xtick.top': 'on',
              'ytick.right': 'on',
              'xtick.major.width': TICK_MAJOR_WIDTH,
              'xtick.major.size': TICK_MAJOR_SIZE,
              'xtick.minor.width': TICK_MINOR_WIDTH,
              'xtick.minor.size': TICK_MINOR_SIZE,
              'ytick.major.width': TICK_MAJOR_WIDTH,
              'ytick.major.size': TICK_MAJOR_SIZE,
              'ytick.minor.width': TICK_MINOR_WIDTH,
              'ytick.minor.size': TICK_MINOR_SIZE,
              'savefig.dpi': DPI,
              'savefig.bbox': 'tight',
              'savefig.pad_inches': SAVEFIG_PAD_INCHES}
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
