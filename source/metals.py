import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from snapshot import Snapshot
from settings import Settings
from paths import Paths
from images import figure_setup


def create_age_metallicity_plot(galaxy: int, rerun: bool,
                                resolution: int, snapnum: int,
                                savefig: bool = True) -> None:
    """
    This method creates the age-metallicity plot.

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
    s.tag_particles_by_region()
    s.keep_only_feats(["RegionTag", "FeFraction",
                       "Circularity", "PartTypes",
                       "NormalizedPotential",
                       "HFraction", "StellarFormationTime"])
    s.calc_metal_abundance(of="Fe", to="H")
    s.drop_feats(["HFraction", "FeFraction", "PartTypes"])
    s.calculate_stellar_age()
    s.drop_feats(["StellarFormationTime"])

    fig = plt.figure(figsize=settings.long_horizontal_fig_size)
    gs = fig.add_gridspec(nrows=1, ncols=4,
                          hspace=0, wspace=0,
                          width_ratios=[1, 1, 1, 1])
    axs = gs.subplots(sharex=False, sharey=True)

    for ax in axs.flatten():
        ax.tick_params(which='both', direction="in")
        ax.set_xlim(0, 14)
        ax.set_ylim(-4, 3)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([-3, -2, -1, 0, 1, 2])
        ax.set_xlabel('Age [Gyr]')
        ax.set_ylabel('[Fe/H]')
        ax.label_outer()

    for i in range(0, 4):
        region = settings.components[i]
        x = s.df.loc[s.df["RegionTag"] == region, "StellarAge"]
        y = s.df.loc[s.df["RegionTag"] == region, "FeH_Abundance"]
        _, _, _, im = axs[i].hist2d(
            x=x, y=y, cmap='nipy_spectral', bins=100,
            range=[axs[i].get_xlim(), axs[i].get_ylim()],
            norm=mcolors.LogNorm(vmin=1E0, vmax=5E3))
        
        axs[i].plot(axs[i].get_xlim(), [np.nanmedian(y), np.nanmedian(y)],
                    lw=.25, color='k')
        axs[i].plot([np.nanmedian(x), np.nanmedian(x)], axs[i].get_ylim(),
                    lw=.25, color='k')
        axs[i].plot(np.nanmedian(x), np.nanmedian(y),
                    marker='o', mfc='k', ms=2, mew=0)
        
        axs[i].text(axs[i].get_xlim()[1] - np.diff(axs[i].get_xlim())/10,
                    axs[i].get_ylim()[1] - np.diff(axs[i].get_ylim())/10,
                    settings.component_labels[i], ha='right', va='center',
                    fontsize=6)
        
        # add_label(axs[i].get_xlim()[0] + np.diff(axs[i].get_xlim())/10,
        #         axs[i].get_ylim()[0] + np.diff(axs[i].get_ylim())/10,
        #         f'Au{galaxy}', ax=axs[i])


    cbar = fig.colorbar(im, ax=axs.ravel().tolist(),
                        label=r'$N_\mathrm{stars}$', pad=0)
    cbar.ax.set_yticks([1E0, 1E1, 1E2, 1E3])

    if savefig:
        for extension in settings.figure_extensions:
            fig.savefig(f"{paths.images}/age_metallicity.{extension}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    figure_setup()
    create_age_metallicity_plot(6, False, 4, 127)