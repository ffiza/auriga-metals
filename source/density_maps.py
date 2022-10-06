from multiprocessing import Pool
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy
import warnings
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

warnings.filterwarnings('ignore', category=UserWarning)


class DensityMaps:
    """A class to manage the creation of density maps.

    Attributes
    ----------
    _galaxy : int
        The galaxy number.
    _rerun : bool
        A bool to indicate if this is a original run or a rerun.
    _resolution : int
        The resolution level of the simulation.
    _paths : Paths
        An instance of the Paths class.
    _n_snapshots : int
        The total amount of snapshots in this simulation.
    _first_snap : int
        The first snapshot to analyze.
    _color_maps : dict
        A dictionary with the color maps to use for each particle type.
    _box_size : int
        The size of the box to plot.
    _n_bins : int
        The number of bins to consider for the density plots.
    _bin_area : float
        The area of each two-dimensional bin in ckpc^2.
    _hist_range : list
        The range of the two-dimensional histogram.
    _box_props : dict
        A dictionary with the style of the text boxes.
    _fontsize : int
        The size of the fonts.
    _spine_width : float
        The width of the spines.
    _quiver_plot_bins : int
        The number of bins to use for the quiver (velocity arrows) plot.

    Methods
    -------
    make_plots()
        This method creates the density maps for all snapshots of the
        galaxy.
    _make_snapshot_plot(snapnum)
        This method creates the density maps for the given snapshot number.
    _plot_panel(self, x, y, vx, vy, mass, ptype, xlabel, ylabel, ax)
        The method plots a density map with the given properties to the
        specified axis.
    """

    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        """
        Parameters
        ----------
        galaxy : int
            The galaxy number.
        rerun : bool
            A bool to indicate if this is a original run or a rerun.
        resolution : int
            The resolution level of the simulation.
        """

        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution

        simulation = Simulation(self._rerun, self._resolution)
        self._n_snapshots = simulation.n_snapshots

        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

        settings = Settings()
        self._first_snap = settings.first_snap
        self._color_maps = settings.color_maps
        self._box_size = settings.box_size
        self._n_bins = settings.n_bins
        self._bin_area = (self._box_size / self._n_bins)**2
        self._hist_range = [[-self._box_size / 2, self._box_size / 2],
                            [-self._box_size / 2, self._box_size / 2]]

        # Density map style.
        self._bbox_props = {"facecolor": "white", "edgecolor": "None",
                            "pad": 0.2, "alpha": 0.8, "boxstyle": "round"}
        self._fontsize = 12
        self._spine_width = 3.0
        self._quiver_plot_bins = 32

    @timer
    def make_plots(self) -> None:
        """This method creates the density maps for all snapshots of the
        galaxy.
        """

        snapnums = list(range(self._first_snap, self._n_snapshots))
        Pool().map(self._make_snapshot_plot, snapnums)

    def _make_snapshot_plot(self, snapnum: int) -> None:
        """This method creates the density maps for the given snapshot number.

        Parameters
        ----------
        snapnum : int
            The snapshot number.
        """

        # Read snapshot.
        s = Snapshot(self._galaxy, self._rerun, self._resolution, snapnum)
        s.keep_only_halo()
        s.drop_winds()

        # Keep only particles inside the box.
        s.df = s.df[(np.abs(s.df.xCoordinates) <= self._box_size / 2)
                    & (np.abs(s.df.yCoordinates) <= self._box_size / 2)
                    & (np.abs(s.df.zCoordinates) <= self._box_size / 2)]

        fig, axes = plt.subplots(ncols=3, nrows=3,
                                 figsize=(FULL_WIDTH, FULL_WIDTH),
                                 gridspec_kw={"width_ratios": [1, 1, 1],
                                              "height_ratios": [1, 1, 1],
                                              "hspace": 0, "wspace": 0})

        # Set style of axes.
        for ax in axes.flatten():
            ax.set_aspect("equal", adjustable="box")
            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_linewidth(self._spine_width)

        # Add panels.
        for idx, ptype in enumerate([0, 1, 4]):
            is_ptype = s.df.PartTypes == ptype
            self._plot_panel(x=s.df.xCoordinates[is_ptype],
                             y=s.df.yCoordinates[is_ptype],
                             vx=s.df.xVelocities[is_ptype],
                             vy=s.df.yVelocities[is_ptype],
                             mass=s.df.Masses[is_ptype], ptype=ptype,
                             xlabel=r"$x$", ylabel=r"$y$",
                             ax=axes[idx, 0])
            self._plot_panel(x=s.df.yCoordinates[is_ptype],
                             y=s.df.zCoordinates[is_ptype],
                             vx=s.df.yVelocities[is_ptype],
                             vy=s.df.zVelocities[is_ptype],
                             mass=s.df.Masses[is_ptype], ptype=ptype,
                             xlabel=r"$y$", ylabel=r"$z$",
                             ax=axes[idx, 1])
            self._plot_panel(x=s.df.xCoordinates[is_ptype],
                             y=s.df.zCoordinates[is_ptype],
                             vx=s.df.xVelocities[is_ptype],
                             vy=s.df.zVelocities[is_ptype],
                             mass=s.df.Masses[is_ptype], ptype=ptype,
                             xlabel=r"$x$", ylabel=r"$z$",
                             ax=axes[idx, 2])

        # Add anotations.
        axes[0, 0].text(0, 45,
                        f"$z = $ {round(s.redshift, 3)}",
                        size=self._fontsize, bbox=self._bbox_props,
                        va="top", ha="center")
        axes[0, 1].text(0, 45,
                        f"{round(s.time, 3)} Gyr",
                        size=self._fontsize, bbox=self._bbox_props,
                        va="top", ha="center")
        axes[0, 2].text(0, 45,
                        snapnum, size=self._fontsize, bbox=self._bbox_props,
                        va="top", ha="center")

        plt.tight_layout()
        plt.savefig(f"{self._paths.images}density_maps/snapshot_{snapnum}.png",
                    bbox_inches="tight", pad_inches=0.02, dpi=100)
        plt.close(fig)

    def _plot_panel(self, x: np.ndarray, y: np.ndarray,
                    vx: np.ndarray, vy: np.ndarray, mass: np.ndarray,
                    ptype: int, xlabel: str, ylabel: str,
                    ax: plt.Axes) -> None:
        """The method plots a density map with the given properties to the
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

        # Define color map for density plot.
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
            warnings.simplefilter("ignore")
            vx_quiver /= n_quiver
            vy_quiver /= n_quiver

        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.text(0, -45, xlabel, size=self._fontsize, bbox=self._bbox_props,
                va="bottom", ha="center")
        ax.text(-45, 0, ylabel, size=self._fontsize, bbox=self._bbox_props,
                va="center", ha="left")

        ax.hist2d(x, y, bins=(self._n_bins, self._n_bins),
                  weights=mass / self._bin_area,
                  norm=mpl.colors.LogNorm(vmin=1E4, vmax=1E9),
                  rasterized=True, cmap=color_map,
                  range=self._hist_range)
        ax.quiver(np.transpose(xcenter).flatten(), ycenter.flatten(),
                  vx_quiver.flatten(), vy_quiver.flatten(), color="black")


if __name__ == "__main__":
    figure_setup()

    settings = Settings()
    for galaxy in settings.galaxies:
        print(f"Analyzing Au{galaxy}... ", end='')
        density_maps = DensityMaps(galaxy, False, 4)
        density_maps.make_plots()
        print(" Done.")

    # TODO: Run plots for all galaxies.
