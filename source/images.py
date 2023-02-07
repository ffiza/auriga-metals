import matplotlib.pyplot as plt
from cosmology import Cosmology
from settings import Settings


def figure_setup():
    settings = Settings()

    params = {'axes.labelsize': settings.label_size,
              'text.usetex': True,
              'figure.dpi': settings.dpi,
              'figure.facecolor': 'white',
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'font.family': 'serif',
              'xtick.top': 'on',
              'ytick.right': 'on',
              'xtick.major.width': settings.tick_major_width,
              'xtick.major.size': settings.tick_major_size,
              'xtick.minor.width': settings.tick_minor_width,
              'xtick.minor.size': settings.tick_minor_size,
              'ytick.major.width': settings.tick_major_width,
              'ytick.major.size': settings.tick_major_size,
              'ytick.minor.width': settings.tick_minor_width,
              'ytick.minor.size': settings.tick_minor_size,
              'savefig.dpi': self.dpi,
              'savefig.bbox': 'tight',
              'savefig.pad_inches': settings.savefig_pad_inches}
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
