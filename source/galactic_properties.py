from multiprocessing import Pool
from support import make_snapshot_number
from sys import stdout
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from cosmology import Cosmology
from loadmodules import gadget_readsnap, load_subfind
from settings import Settings
from paths import Paths
from support import timer, create_or_load_dataframe
from images import add_redshift, figure_setup


class GalacticPropertiesAnalysis:
    """
    A class to manage the calculations regarding the reference potential.

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
    _df : pd.DataFrame
        A DataFrame with all the temporal data for a given simulation. It is
        loaded if it exists or created if not.

    Methods
    -------
    calculate_subhalo_velocities()
        This method calculates the velocity of the main subhalo for all
        snapshots in this simulation.
    _calculate_subhalo_velocity(snapnum)
        This method calculates the velocity of the main subhalo in this
        snapshot.
    _create_or_load_dataframe()
        This method loads the temporal data frame if it exists or creates it
        if it doesn't.
    save_data()
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
        self._df = create_or_load_dataframe(
            f"{self._paths.data}temporal_data.csv")

    def _calc_properties_in_snapshot(self, snapshot_number: int) -> tuple:
        """
        This method calculates the snapshot number, cosmic time, lookback
        time, expansion factor and redshift for a given snapshot.

        Parameters
        ----------
        snapnum : int
            The snapshot number.

        Returns
        -------
        tuple
            A tuple with the calculated values.
        """

        settings = Settings()
        cosmology = Cosmology()

        sf = gadget_readsnap(snapshot=snapshot_number,
                             snappath=self._paths.snapshots,
                             onlyHeader=True,
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False)
        time = cosmology.redshift_to_time(sf.redshift)
        lookback_time = cosmology.redshift_to_lookback_time(sf.redshift)
        expansion_factor = sf.time
        redshift = sf.redshift
        del sf

        if snapshot_number >= settings.first_snap:
            sb = load_subfind(id=snapshot_number,
                              dir=self._paths.snapshots,
                              cosmological=False)

            halo_idx = self._df["MainHaloIdx"].loc[snapshot_number]
            subhalo_idx = self._df["MainSubhaloIdx"].loc[snapshot_number]
            subhalo_grouptab_idx = sb.data["ffsh"][halo_idx] + subhalo_idx

            virial_radius = sb.data['frc2'][subhalo_grouptab_idx] * 1E3 \
                / sb.hubbleparam  # ckpc
            virial_mass = sb.data['fmc2'][subhalo_grouptab_idx] \
                / sb.hubbleparam  # 1E10 Msun

            # For reference, also consider the virial properties of the (0, 0)
            # object (same units)
            virial_radius_00 = sb.data['frc2'][0] * 1E3 / sb.hubbleparam
            virial_mass_00 = sb.data['fmc2'][0] / sb.hubbleparam

            del sb
        else:
            virial_radius = np.nan
            virial_mass = np.nan
            virial_radius_00 = np.nan
            virial_mass_00 = np.nan

        return (snapshot_number, time, lookback_time, redshift,
                expansion_factor, virial_radius, virial_mass,
                virial_radius_00, virial_mass_00)

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the properties for all snapshots in
        this simulation.
        """

        settings = Settings()

        snapnums = list(range(self._n_snapshots))
        data = np.array(Pool(settings.processes).map(
            self._calc_properties_in_snapshot, snapnums))

        self._df["SnapshotNumber"] = data[:, 0].astype(np.int)
        self._df["Time_Gyr"] = np.round(data[:, 1], 7)
        self._df["LookbackTime_Gyr"] = np.round(data[:, 2], 7)
        self._df["Redshift"] = np.round(data[:, 3], 7)
        self._df["ExpansionFactor"] = np.round(data[:, 4], 7)
        self._df["VirialRadius_ckpc"] = np.round(data[:, 5], 7)
        self._df["VirialMass_1E10Msun"] = np.round(data[:, 6], 7)
        self._df["VirialRadius00_ckpc"] = np.round(data[:, 7], 7)
        self._df["VirialMass00_1E10Msun"] = np.round(data[:, 8], 7)

        self._save_data()

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        self._df.set_index(keys="SnapshotNumber")
        self._df.to_csv(f"{self._paths.data}temporal_data.csv",
                        index=False)


def run_analysis(galaxy: int, rerun: bool, resolution: int) -> None:
    stdout.write(f"Analyzing Au{galaxy}... ")
    analysis = GalacticPropertiesAnalysis(galaxy, rerun, resolution)
    analysis.analyze_galaxy()
    stdout.write(" Done.\n")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        run_analysis(galaxy=galaxy, rerun=False, resolution=4)
        if galaxy in settings.reruns:
            run_analysis(galaxy=galaxy, rerun=True, resolution=4)


def plot_virial_radius(savefig: bool) -> None:
    """
    This method creates a plot to visualize the virial radius of each
    galaxy.
    """

    settings = Settings()

    fig = plt.figure(figsize=settings.big_fig_size)
    gs = fig.add_gridspec(nrows=settings.big_fig_nrows,
                          ncols=settings.big_fig_ncols,
                          hspace=settings.big_fig_hspace,
                          wspace=settings.big_fig_wspace)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax_idx, ax in enumerate(axs.flat):
        ax.label_outer()
        ax.grid(True, ls='-', lw=settings.big_fig_grid_lw, c='silver')
        ax.tick_params(which='both', direction="in")
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 400)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_yticks([100, 200, 300, 400])

        galaxy = ax_idx + 1

        paths = Paths(galaxy, False, 4)
        df = pd.read_csv(f"{paths.data}temporal_data.csv")
        ax.plot(df["Time_Gyr"], df["VirialRadius_ckpc"], c='k',
                lw=settings.big_fig_line_lw, zorder=10)

        if galaxy in settings.reruns:
            paths = Paths(galaxy, True, 4)
            df = pd.read_csv(f"{paths.data}temporal_data.csv")
            ax.plot(df["Time_Gyr"], df["VirialRadius_ckpc"], c='tab:red',
                    lw=settings.big_fig_line_lw, ls="--", zorder=11)

        add_redshift(ax)
        ax.text(0.95, 0.95, f'Au{galaxy}', size=6,
                ha='right', va='top',
                transform=ax.transAxes,
                bbox={"facecolor": "silver", "edgecolor": "white",
                      "pad": .2, 'boxstyle': 'round', 'lw': 1})

        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r"$R_{200}$ [ckpc]")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Time [Gyr]')

    if savefig:
        for extension in settings.figure_extensions:
            fig.savefig(f"images/level4/virial_radius.{extension}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    figure_setup()
    plot_virial_radius(savefig=True)
