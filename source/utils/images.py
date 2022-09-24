import matplotlib.pyplot as plt
from source.auriga.cosmology import Cosmology

TICK_MAJOR_WIDTH = .5
TICK_MAJOR_SIZE = 3
TICK_MINOR_WIDTH = .25
TICK_MINOR_SIZE = 1.5
LABEL_SIZE = 9
FONTSIZE = 7
GALAXY_LABEL_FONTSIZE = 6
DPI = 800
FIGSIZE = [2.3, 2]
HALF_WIDTH = 3
FULL_WIDTH = 3 * HALF_WIDTH


def figure_setup() -> None:
    """
    This method configures the rcParams of MatPlotLib.
    """

    params = {'axes.labelsize': LABEL_SIZE,
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
              'savefig.pad_inches': .02}
    plt.rcParams.update(params)


def add_label(x: float, y: float, text: str, ax: plt.Axes,
              ha: str = 'left', va: str = 'bottom') -> None:
    # TODO: Document this function.
    ax.text(x, y, text, ha=ha, va=va, size=GALAXY_LABEL_FONTSIZE,
            bbox={"facecolor": "white",
                  "edgecolor": "k",
                  "pad": .2,
                  'boxstyle': 'round',
                  'lw': .5})


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
    ax2_ticks = [cosmology.redhift_to_time(float(item)) for item in
                 ax2_label_values]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax2_ticks)
    if ax.get_subplotspec().is_first_row():
        ax2.set_xticklabels(ax2_ticklabels)
        ax2.set_xlabel('$z$')
    else:
        ax2.set_xticklabels([])
