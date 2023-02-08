from loadmodules import gadget_readsnap, load_subfind
from settings import Settings
from paths import Paths
from support import timer, make_snapshot_number
from images import add_redshift, figure_setup
from multiprocessing import Pool
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


class SubhaloVelocityAnalysis:
    """
    A class to manage the calculations regarding the velocity of the main
    galaxy.

    Attributes
    ----------
    _galaxy : int
        The galaxy number.
    _rerun : bool
        A bool to indicate if this is a original run or a rerun.
    _resolution : int
        The resolution level of the simulation.
    _distance : float
        The distance to consider stars for velocity calculation.
    _paths : Paths
        An instance of the Paths class.
    _n_snapshots : int
        The total amount of snapshots in this simulation.
    _subhalo_velocities : np.ndarray
        An array with the subhalo velocity for each snapshot of this
        simulation.
    _halo_idxs : np.ndarray
        An array with the indices of the main halo.
    _subhalo_idxs : np.ndarray
        An array with the indices of the main subhalo.

    Methods
    -------
    calculate_subhalo_velocities()
        This method calculates the velocity of the main subhalo for all
        snapshots in this simulation.
    _calculate_subhalo_velocity(snapnum)
        This method calculates the velocity of the main subhalo in this
        snapshot.
    _save_data()
        This method saves the data.
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
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

        # Set halo/subhalo indices.
        df = pd.read_csv(f"{self._paths.data}temporal_data.csv",
                         usecols=["MainHaloIdx", "MainSubhaloIdx"])
        self._halo_idxs = df["MainHaloIdx"].to_numpy()
        self._subhalo_idxs = df["MainSubhaloIdx"].to_numpy()

    @timer
    def calculate_subhalo_velocities(self) -> None:
        """
        This method calculates the velocity of the main subhalo for all
        snapshots in this simulation.
        """

        snapnums = list(range(self._n_snapshots))
        self._subhalo_velocities = np.array(
            Pool().map(self._calculate_subhalo_velocity, snapnums))

        self._save_data()

    def _calculate_subhalo_velocity(self, snapnum: int) -> None:
        """
        This method calculates the velocity of the main subhalo in this
        snapshot.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.
        """

        settings = Settings()

        if snapnum < settings.first_snap:
            return np.array([np.nan, np.nan, np.nan])
        else:
            halo_idx = self._halo_idxs[snapnum]
            subhalo_idx = self._subhalo_idxs[snapnum]

            sf = gadget_readsnap(snapshot=snapnum,
                                 snappath=self._paths.snapshots,
                                 loadonlytype=[4], lazy_load=True,
                                 cosmological=False,
                                 applytransformationfacs=False)
            sb = load_subfind(id=snapnum, dir=self._paths.snapshots,
                              cosmological=False)
            sf.calc_sf_indizes(sf=sb)

            # Find the index of the subhalo in the subfind table.
            subhalo_grouptab_idx = sb.data["ffsh"][halo_idx] + subhalo_idx

            pos = (sf.pos - sb.data["spos"][subhalo_grouptab_idx]
                   / sf.hubbleparam) * 1E3  # ckpc
            del sb
            vel = sf.vel * np.sqrt(sf.time)  # km/s
            mass = sf.mass * 1E10  # Msun
            age = sf.age

            r = np.linalg.norm(pos, axis=1)  # ckpc
            del pos

            is_main_inner_star = (age > 0) \
                & (r < settings.subh_vel_distance) \
                & (sf.halo == halo_idx) \
                & (sf.subhalo == subhalo_idx)
            del sf, age, r

            if is_main_inner_star.sum() == 0:
                # No stars were found with the condition (early snapshots).
                return np.array([np.nan, np.nan, np.nan])
            else:
                vel_cm = \
                    mass[is_main_inner_star].T @ vel[is_main_inner_star] / \
                    mass[is_main_inner_star].sum()  # km/s
                return vel_cm

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        np.savetxt(f"{self._paths.data}subhalo_vels.csv",
                   self._subhalo_velocities)

    # def make_plot(self) -> None:
    #     """This method makes a plot of the absolute value of the velocity
    #     as a function of time. Bear in mind that it ignores the parameters
    #     of the constructor of the class - it plots all galaxies.
    #     """

    #     figure_setup()

    #     fig, axs = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH),
    #                             nrows=6, ncols=5,
    #                             sharex=True, sharey=True)
    #     fig.subplots_adjust(wspace=0, hspace=0)

    #     for ax_idx, ax in enumerate(axs.flat):
    #         ax.label_outer()
    #         ax.grid(True, ls='-', lw=0.5, c="silver")
    #         ax.tick_params(which="both", direction="in")
    #         ax.set_xlim(0, 14)
    #         # ax.set_ylim(-0.5, 5.5)
    #         ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
    #         # ax.set_yticks([0, 1, 2, 3, 4, 5])
    #         for spine in ["top", "bottom", "left", "right"]:
    #             ax.spines[spine].set_linewidth(1.5)

    #         galaxy = ax_idx + 1

    #         paths = Paths(galaxy, False, 4)
    #         simulation = Simulation(False, 4)

    #         data = np.loadtxt(f"{paths.data}subhalo_vels.csv")
    #         vel_norm = np.linalg.norm(data, axis=1)

    #         ax.plot(simulation.times, vel_norm, c='k', lw=2, zorder=10)

    #         add_redshift(ax)
    #         ax.text(0.95, 0.95, f"Au{galaxy}", size=6,
    #                 ha="right", va="top",
    #                 transform=ax.transAxes,
    #                 bbox={"facecolor": "silver", "edgecolor": "white",
    #                       "pad": .2, "boxstyle": "round", "lw": 1})

    #         if ax.get_subplotspec().is_first_col():
    #             ax.set_ylabel(
    #                 r"$v_\mathrm{sh}$ [$\mathrm{km} \, \mathrm{s}^{-1}$]")
    #         if ax.get_subplotspec().is_last_row():
    #             ax.set_xlabel("Time [Gyr]")

    #     fig.savefig(f"images/level{self._resolution}/subhalo_velocity.pdf")
    #     plt.close(fig)
