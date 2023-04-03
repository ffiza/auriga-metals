from sys import stdout
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

from cosmology import Cosmology
from snapshot import Snapshot
from settings import Settings
from paths import Paths
from support import timer, create_or_load_dataframe
from images import add_redshift, figure_setup


def plot_energy_circularity(galaxy: int, rerun: bool,
                            resolution: int, snapnum: int,
                            savefig: bool = True) -> None:
    """This method creates a plot of the normalized specific orbital energy
    of stars versus their circularity as a color map.

    Parameters
    ----------
    galaxy : int
        The galaxy number.
    rerun : bool
        True for the reruns and False for the original simulations.
    resolution : int
        The resolution of the simulation.
    snapnum : int
        The snapshot number to plot.
    savefig : bool, optional
        If True, save the figure. If not, only show the plot.
    """

    settings = Settings()
    paths = Paths(galaxy, rerun, resolution)

    s = Snapshot(galaxy, rerun, resolution, snapnum)
    s.calc_circularity()
    s.drop_types([0, 1, 2, 3, 5])
    s.drop_winds()
    s.keep_only_halo()
    # s.calc_referenced_potential()
    # s.calc_normalized_orbital_energy()
    s.calc_normalized_potential()

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = gs.subplots(sharex=True, sharey=True)

    ax.tick_params(which='both', direction="in")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1, 0)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, -.8, -.6, -.4, -.2, 0])
    ax.set_xlabel(r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$')
    ax.set_ylabel(r'$\tilde{e} = e \, \left| e \right|_\mathrm{max}^{-1}$')

    _, _, _, im = ax.hist2d(s.df["Circularity"],
                            s.df["NormalizedPotential"],
                            cmap='nipy_spectral',
                            bins=100,
                            range=[ax.get_xlim(), ax.get_ylim()],
                            norm=mcolors.LogNorm(vmin=1E0, vmax=1E4))

    ax.text(-.9, -.1, 'Halo', fontsize=5)
    ax.text(-.9, -.9, 'Bulge', fontsize=5)
    ax.text(.55, -.35, 'Warm Disc', fontsize=5, rotation=90)
    ax.text(.85, -.35, 'Cold Disc', fontsize=5, rotation=90)

    ax.plot([settings.disc_min_circ, settings.disc_min_circ],
             ax.get_ylim(), color='k', ls='--', lw=.5)
    ax.plot([1 - settings.cold_disc_delta_circ,
             1 - settings.cold_disc_delta_circ],
            ax.get_ylim(), color='k', ls='--', lw=.5)
    ax.plot([ax.get_xlim()[0], settings.disc_min_circ],
            [settings.bulge_max_specific_energy,
             settings.bulge_max_specific_energy], color='k', ls='--', lw=.5)

    ax.plot([1, 1], ax.get_ylim(), color='k', ls='-.', lw=.25)
    ax.plot([-1, -1], ax.get_ylim(), color='k', ls='-.', lw=.25)

    # add_label(ax.get_xlim()[1] - np.diff(ax.get_xlim())/10,
    #           ax.get_ylim()[0] + np.diff(ax.get_ylim())/10,
    #           f'Au{galaxy}', ha='right', ax=ax)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                        label=r'$N_\mathrm{stars}$',
                        pad=0)
    cbar.ax.set_yticks([1E0, 1E1, 1E2, 1E3, 1E4])

    if savefig:
        for extension in settings.figure_extensions:
            fig.savefig(f"{paths.images}/energy_vs_circularity.{extension}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # figure_setup()
    plot_energy_circularity(6, False, 4, 127, True)
