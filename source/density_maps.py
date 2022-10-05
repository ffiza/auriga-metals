from multiprocessing import Pool
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy
# import warnings
# from matplotlib import rc
from auriga.settings import Settings
from auriga.simulation import Simulation
from auriga.snapshot import Snapshot
from utils.paths import Paths
from utils.timer import timer
from utils.images import figure_setup, FULL_WIDTH
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


class DensityMaps:
    # TODO: Add class documentation.

    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        # TODO: Add constructor documentation.

        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution

        simulation = Simulation(self._rerun, self._resolution)
        self._n_snapshots = simulation.n_snapshots

        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

        settings = Settings()
        self._color_maps = settings.color_maps
        self._box_size = settings.box_size
        self._n_bins = settings.n_bins
        self._bin_area = (self._box_size / self._n_bins)**2
        self._hist_range = [[-self._box_size / 2, self._box_size / 2],
                            [-self._box_size / 2, self._box_size / 2]]

    @timer
    def make_plots(self) -> None:
        # TODO: Add documentation.

        figure_setup()

        snapnums = list(range(self._n_snapshots))
        Pool().map(self._make_snapshot_plot, snapnums)

    def _make_snapshot_plot(self, snapnum: int) -> None:
        # TODO: Add documentation.

        # Read snapshot.
        s = Snapshot(self._galaxy, self._rerun, self._resolution, snapnum)
        s.keep_only_halo()

        # Keep only particles inside the box.
        s.df = s.df[(np.abs(s.df.xCoordinates) <= self._box_size / 2)
                    & (np.abs(s.df.yCoordinates) <= self._box_size / 2)
                    & (np.abs(s.df.zCoordinates) <= self._box_size / 2)]

        fig, axes = plt.subplots(ncols=3, nrows=3,
                                 figsize=(FULL_WIDTH, FULL_WIDTH),
                                 gridspec_kw={'width_ratios': [1, 1, 1]})

        # Set aspect ratio
        for ax in axes.flatten():
            ax.set_aspect('equal', adjustable='box')

        # Add panels.
        # TODO: Choose more appropriate color maps.
        # TODO: Put labels inside of axes.
        # TODO: Remove tick labels (but keep ticks) and remove white space.
        for idx, ptype in enumerate([0, 1, 4]):
            is_ptype = s.df.PartTypes == ptype
            self._plot_panel(x=s.df.xCoordinates[is_ptype],
                             y=s.df.yCoordinates[is_ptype],
                             vx=s.df.xVelocities[is_ptype],
                             vy=s.df.yVelocities[is_ptype],
                             mass=s.df.Masses[is_ptype], ptype=ptype,
                             xlabel=r'$x$ [$\mathrm{ckpc}$]',
                             ylabel=r'$y$ [$\mathrm{ckpc}$]',
                             ax=axes[idx, 0])
            self._plot_panel(x=s.df.yCoordinates[is_ptype],
                             y=s.df.zCoordinates[is_ptype],
                             vx=s.df.yVelocities[is_ptype],
                             vy=s.df.zVelocities[is_ptype],
                             mass=s.df.Masses[is_ptype], ptype=ptype,
                             xlabel=r'$x$ [$\mathrm{ckpc}$]',
                             ylabel=r'$y$ [$\mathrm{ckpc}$]',
                             ax=axes[idx, 1])
            self._plot_panel(x=s.df.xCoordinates[is_ptype],
                             y=s.df.zCoordinates[is_ptype],
                             vx=s.df.xVelocities[is_ptype],
                             vy=s.df.zVelocities[is_ptype],
                             mass=s.df.Masses[is_ptype], ptype=ptype,
                             xlabel=r'$x$ [$\mathrm{ckpc}$]',
                             ylabel=r'$y$ [$\mathrm{ckpc}$]',
                             ax=axes[idx, 2])

        # Add anotations.
        bbox_props = {"facecolor": "white", "edgecolor": "None",
                      "pad": 1, "alpha": 0.8}
        axes[0, 0].text(-38, -38,
                        f'$z = $ {round(s.redshift, 3)}',
                        size=5, bbox=bbox_props, va='bottom', ha='left')
        axes[0, 1].text(-38, -38,
                        f'{round(s.time, 3)} Gyr',
                        size=5, bbox=bbox_props, va='bottom', ha='left')
        axes[0, 2].text(-38, -38,
                        snapnum, size=5, bbox=bbox_props,
                        va='bottom', ha='left')

        plt.tight_layout()
        # TODO: Maybe use another file format (eps, pdf, svg).
        plt.savefig(f'{self._paths.images}density_maps/snapshot_{snapnum}.png',
                    bbox_inches='tight', pad_inches=0.02, dpi=800)
        plt.close(fig)

    def _plot_panel(self, x, y, vx, vy, mass, ptype,
                    xlabel, ylabel, ax) -> None:
        # TODO: Add documentation.

        # Define color map for density plot.
        color_map = copy.copy(mpl.cm.get_cmap(self._color_maps[ptype]))
        color_map.set_bad((0, 0, 0))

        vx_quiver, xedges, yedges = np.histogram2d(x, y, bins=32,
                                                   range=self._hist_range,
                                                   weights=vx)
        vy_quiver = np.histogram2d(x, y, bins=32, range=self._hist_range,
                                   weights=vy)[0]
        n_quiver = np.histogram2d(x, y, bins=32, range=self._hist_range)[0]

        xcenter = xedges[:-1] + (xedges[1] - xedges[0]) / 2
        ycenter = yedges[:-1] + (yedges[1] - yedges[0]) / 2

        xcenter = np.asarray([xcenter for _ in range(32)])
        ycenter = np.asarray([ycenter for _ in range(32)])

        vx_quiver /= n_quiver
        vy_quiver /= n_quiver

        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.hist2d(x, y, bins=(self._n_bins, self._n_bins),
                  weights=mass / self._bin_area,
                  norm=mpl.colors.LogNorm(vmin=1E4, vmax=1E9),
                  rasterized=True, cmap=color_map,
                  range=self._hist_range)
        # TODO: Use black arrows if the color maps require it.
        ax.quiver(np.transpose(xcenter).flatten(), ycenter.flatten(),
                  vx_quiver.flatten(), vy_quiver.flatten(), color='white')


if __name__ == '__main__':
    # TODO: Run plots for all galaxies.
    density_maps = DensityMaps(6, False, 4)
    density_maps._make_snapshot_plot(127)
