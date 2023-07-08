from multiprocessing import Pool
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy
import warnings
import os
from sys import stdout
import argparse
import shutil
from PyPDF2 import PdfFileMerger
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from auriga.settings import Settings
from auriga.support import make_snapshot_number, timer
from auriga.snapshot import Snapshot
from auriga.paths import Paths
from auriga.parser import parse
from auriga.images import figure_setup


class DensityMaps:
    """
    A class to manage the creation of density maps.
    """

    def __init__(self, simulation: str) -> None:
        """
        Parameters
        ----------
        simulation : str
            The simulation to load.
        """

        self._simulation = simulation
        galaxy, rerun, resolution = parse(simulation=simulation)
        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

        settings = Settings()
        self._first_snap = settings.first_snap
        self._color_maps = settings.color_maps
        self._box_size = settings.box_size
        self._n_bins = settings.n_bins
        self._bin_area = (self._box_size / self._n_bins)**2
        self._hist_range = [[-self._box_size / 2, self._box_size / 2],
                            [-self._box_size / 2, self._box_size / 2]]

        # Density map style
        self._bbox_props = {"facecolor": "white",
                            "edgecolor": "None",
                            "pad": 0.2,
                            "alpha": 0.8,
                            "boxstyle": "round"}
        self._fontsize = 12
        self._spine_width = 3.0
        self._quiver_plot_bins = 32

    @timer
    def make_plots(self) -> None:
        """
        This method creates the density maps for all snapshots of the
        galaxy.
        """

        # Make temporary folder to store plots
        if not os.path.isdir(f"images/density_maps/{self._simulation}/"):
            os.makedirs(f"images/density_maps/{self._simulation}/",
                        exist_ok=False)

        snapnums = list(range(self._first_snap, self._n_snapshots))
        Pool(2).map(self._make_snapshot_plot, snapnums)
        self._merge_pdfs()

    def _make_snapshot_plot(self, snapnum: int) -> None:
        """
        This method creates the density maps for the given snapshot number.

        Parameters
        ----------
        snapnum : int
            The snapshot number.
        """

        # Read snapshot
        s = Snapshot(simulation=f"{self._simulation}_s{snapnum}",
                     loadonlytype=[0, 1, 4])

        # Keep only particles inside the box
        is_box = (np.abs(s.pos[:, 0]) <= self._box_size / 2) \
            & (np.abs(s.pos[:, 1]) <= self._box_size / 2) \
            & (np.abs(s.pos[:, 2]) <= self._box_size / 2)

        is_real_part = (s.type == 0) | (s.type == 1) \
            | ((s.type == 4) & (s.stellar_formation_time > 0))

        x = s.pos[is_box & is_real_part, 0]
        y = s.pos[is_box & is_real_part, 1]
        z = s.pos[is_box & is_real_part, 2]
        vx = s.vel[is_box & is_real_part, 0]
        vy = s.vel[is_box & is_real_part, 1]
        vz = s.vel[is_box & is_real_part, 2]
        mass = s.mass[is_box & is_real_part]

        # Create figure
        fig, axes = plt.subplots(ncols=3, nrows=3,
                                 figsize=(7.2, 7.2),
                                 gridspec_kw={"width_ratios": [1, 1, 1],
                                              "height_ratios": [1, 1, 1],
                                              "hspace": 0.1,
                                              "wspace": 0.17})

        # Set style of axes
        for ax in axes.flatten():
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(length=6, width=1)

        # Add panels
        for idx, ptype in enumerate([0, 1, 4]):
            is_type = s.type[is_box & is_real_part] == ptype
            self._plot_panel(x=x[is_type],
                             y=y[is_type],
                             vx=vx[is_type],
                             vy=vy[is_type],
                             mass=mass[is_type],
                             ptype=ptype,
                             xlabel=r"$x$", ylabel=r"$y$",
                             ax=axes[idx, 0])
            self._plot_panel(x=y[is_type],
                             y=z[is_type],
                             vx=vy[is_type],
                             vy=vz[is_type],
                             mass=mass[is_type],
                             ptype=ptype,
                             xlabel=r"$y$", ylabel=r"$z$",
                             ax=axes[idx, 1])
            self._plot_panel(x=x[is_type],
                             y=z[is_type],
                             vx=vx[is_type],
                             vy=vz[is_type],
                             mass=mass[is_type],
                             ptype=ptype,
                             xlabel=r"$x$", ylabel=r"$z$",
                             ax=axes[idx, 2])

        # Add anotations
        axes[0, 0].text(x=-0.95 * self._box_size / 2,
                        y=0.95 * self._box_size / 2,
                        ha="left",
                        va="top",
                        s=r'\textbf{Gas}',
                        color='white',
                        size=10)
        axes[1, 0].text(x=-0.95 * self._box_size / 2,
                        y=0.95 * self._box_size / 2,
                        ha="left",
                        va="top",
                        s=r'\textbf{Dark Matter}',
                        color='white',
                        size=10)
        axes[2, 0].text(x=-0.95 * self._box_size / 2,
                        y=0.95 * self._box_size / 2,
                        ha="left",
                        va="top",
                        s=r'\textbf{Stars}',
                        color='white',
                        size=10)
        axes[0, 0].text(x=-self._box_size / 2,
                        y=self._box_size / 2 + 0.025 * self._box_size,
                        s=r"$\textbf{" + str(self._simulation.upper()) + "}$",
                        color='black',
                        size=10)
        axes[0, 2].text(x=-self._box_size / 2,
                        y=self._box_size / 2 + 0.025 * self._box_size,
                        s=f"Redshift: {round(s.redshift, 3)}",
                        size=10)
        axes[0, 1].text(x=-self._box_size / 2,
                        y=self._box_size / 2 + 0.025 * self._box_size,
                        s=f"Time: {round(s.time, 3)} Gyr",
                        size=10)
        axes[0, 0].text(x=self._box_size / 2,
                        y=self._box_size / 2 + 0.025 * self._box_size,
                        s=f"({snapnum})",
                        size=10,
                        ha="right")

        # plt.tight_layout()
        plt.savefig(
            f"images/density_maps/{self._simulation}/snapshot_{snapnum}.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=500)
        plt.close(fig)

    def _plot_panel(self, x: np.ndarray,
                    y: np.ndarray,
                    vx: np.ndarray,
                    vy: np.ndarray,
                    mass: np.ndarray,
                    ptype: int,
                    xlabel: str,
                    ylabel: str,
                    ax: plt.Axes
                    ) -> None:
        """
        The method plots a density map with the given properties to the
        specified axis.

        Parameters
        ----------
        x : np.ndarray
            The coordinates for the horizontal axis.
        y : np.ndarray
            The coordinates for the vertical axis.
        vx : np.ndarray
            The velocities for the horizontal axis.
        vy : np.ndarray
            The velocities for the vertical axis.
        mass : np.ndarray
            The masses.
        ptype : int
            The particle type (this is only used to select the color maps).
        xlabel : str
            The label for the horizontal axis.
        ylabel : str
            The label for the vertical axis.
        ax : plt.Axes
            The axis in which to plot the density map.
        """

        # Define color map for density plot
        color_map = copy.copy(mpl.cm.get_cmap(self._color_maps[ptype]))
        color_map.set_bad(color_map(0))

        vx_quiver, xedges, yedges = np.histogram2d(x, y,
                                                   bins=self._quiver_plot_bins,
                                                   range=self._hist_range,
                                                   weights=vx)
        vy_quiver = np.histogram2d(x, y, bins=self._quiver_plot_bins,
                                   range=self._hist_range,
                                   weights=vy)[0]
        n_quiver = np.histogram2d(x, y, bins=self._quiver_plot_bins,
                                  range=self._hist_range)[0]

        xcenter = xedges[:-1] + (xedges[1] - xedges[0]) / 2
        ycenter = yedges[:-1] + (yedges[1] - yedges[0]) / 2

        xcenter = np.asarray([xcenter for _ in range(self._quiver_plot_bins)])
        ycenter = np.asarray([ycenter for _ in range(self._quiver_plot_bins)])

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            vx_quiver /= n_quiver
            vy_quiver /= n_quiver

        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if ax.get_subplotspec().is_last_row():
            ax.set_xticklabels(["$-40$", "$-20$", 0, 20, 40])
        if ax.get_subplotspec().is_first_col():
            ax.set_yticklabels(["$-40$", "$-20$", 0, 20, 40])

        ax.set_xlabel(f"{xlabel} [ckpc]", labelpad=1)
        ax.set_ylabel(f"{ylabel} [ckpc]", labelpad=1)

        ax.hist2d(x=x,
                  y=y,
                  bins=(self._n_bins, self._n_bins),
                  weights=mass / self._bin_area,
                  norm=mpl.colors.LogNorm(vmin=1E4, vmax=1E9),
                  rasterized=True,
                  cmap=color_map,
                  range=self._hist_range)
        ax.quiver(np.transpose(xcenter).flatten(),
                  ycenter.flatten(),
                  vx_quiver.flatten(),
                  vy_quiver.flatten(),
                  color="white")

    def _merge_pdfs(self):
        path = f"images/density_maps/{self._simulation}/"
        pdfs = [a for a in os.listdir(path=path) if a.endswith(".pdf")]
        pdfs = sorted(pdfs, key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Merge files
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(f"images/density_maps/{self._simulation}/{pdf}")
        merger.write(f'images/density_maps/{self._simulation}.pdf')
        merger.close()

        # Delete files
        shutil.rmtree(f"images/density_maps/{self._simulation}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulations to consider.")
    args = parser.parse_args()
    for simulation in args.simulations:
        stdout.write(f"Analyzing {simulation.upper()}... ")
        density_maps = DensityMaps(simulation=simulation)
        density_maps.make_plots()
        stdout.write(" Done.\n")


if __name__ == "__main__":
    figure_setup()
    main()
