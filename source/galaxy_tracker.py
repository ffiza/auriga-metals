import numpy as np
from scipy.stats import mode
import pandas as pd
from loadmodules import gadget_readsnap, load_subfind
from auriga.settings import Settings
from auriga.simulation import Simulation
from utils.support import find_indices
from utils.paths import Paths
from utils.timer import timer
from utils.images import add_redshift, figure_setup, FULL_WIDTH
from typing import Tuple
from matplotlib import pyplot as plt


def load_dm_snapshot(galaxy: int, rerun: bool,
                     resolution: int, snapnum: int) -> pd.DataFrame:
    """This method loads a snapshot given the parameters, keeping only
    dark matter particles.

    Parameters
    ----------
    galaxy : int
        The galaxy number.
    rerun : bool
        If True, load the rerun version of the snapshots.
    resolution : int
        The resolution to consider.
    snapnum : int
        The snapshot number to load.

    Returns
    -------
    pd.DataFrame
        A data frame with the particle IDs, halo, subhalo and potential.
    """

    paths = Paths(galaxy, rerun, resolution)
    sf = gadget_readsnap(snapshot=snapnum,
                         snappath=paths.snapshots,
                         loadonlytype=[1], lazy_load=True,
                         cosmological=False,
                         applytransformationfacs=False)
    sb = load_subfind(id=snapnum, dir=paths.snapshots,
                      cosmological=False)
    sf.calc_sf_indizes(sf=sb)

    return pd.DataFrame({'ParticleIDs': sf.id,
                         'Halo': sf.halo,
                         'Subhalo': sf.subhalo,
                         'Potential': sf.pot / sf.time  # (km/s)^2
                         })


class GalaxyTracker:
    """A class to manage the tracking of the main object in each simulation.

    Attributes
    ----------
    _galaxy : int
        The galaxy number to track.
    _rerun : bool
        If True, consider the rerun version of the simulations.
    _resolution : int
        The resolution of the simulation.
    _n_snapshots : int
        The total number of snapshots in the simulation.
    _paths : Paths
        An instance of the Paths class with this parameters.
    _settings : Settings
        An instance of the Settings class.
    _target_ids : np.ndarray
        An array of dark matter IDs to track, selected from the present given
        the tracking number set in Settings.
    _df : pd.DataFrame
        A data frame that contains the information of the track process.

    Methods
    -------
    _find_present_day_most_bound_dm_ids()
        This method finds an array of IDs that belong to the present-day
        most bound dark matter particles. The amount of particles to consider
        is defined in Settings.
    _find_location_of_target_ids(snapnum)
        This method finds the most common halo/subhalo index in the current
        snapshot for the particles being tracked.
    track_galaxy()
        This method tracks the galaxy through all snapshots.
    save_data()
        This method saves the data to a csv file.
    """

    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        """
        Parameters
        ----------
        galaxy : int
            The galaxy number to track.
        rerun : bool
            If True, consider the rerun version of the simulations.
        resolution : int
            The resolution of the simulation.
        """

        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._settings = Settings()
        self._n_snapshots = Simulation(self._rerun,
                                       self._resolution).n_snapshots

        self._df = pd.DataFrame()
        self._df['MainHaloIDX'] = 0 * np.ones(self._n_snapshots,
                                              dtype=np.int32)
        self._df['MainSubhaloIDX'] = 0 * np.ones(self._n_snapshots,
                                                 dtype=np.int32)
        # Set present day halo and subhalo index.
        self._df.MainHaloIDX.iloc[-1] = 0
        self._df.MainSubhaloIDX.iloc[-1] = 0

    def _find_most_bound_dm_ids(self, snapnum: int, halo: int = 0,
                                subhalo: int = 0) -> np.ndarray:
        """This method finds the IDs of the most bound dark matter particles
        in a given snapshot and a given halo and subhalo index. The amount of
        particles to track is defined in Settings.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.
        halo : int, optional
            The index of the halo to search.
        subhalo : int, optional
            The index of the subhalo to search.

        Returns
        -------
        np.ndarray
            An array that contains the IDs of the most bound dark matter
            particles.
        """

        df = load_dm_snapshot(self._galaxy, self._rerun,
                              self._resolution, snapnum)

        # Keep only particles from the selected halo and subhalo.
        df = df[(df.Halo == halo) & (df.Subhalo == subhalo)]

        df.sort_values(by=['Potential'], ascending=True, inplace=True)

        most_bound_ids = \
            df.ParticleIDs.iloc[:self._settings.n_track_dm_parts]

        return most_bound_ids.to_numpy()

    def _find_location_of_dm_ids(self, snapnum: int,
                                 target_ids: np.ndarray) -> Tuple[int, int]:
        """This method finds the most common halo/subhalo index in the current
        snapshot for the particle IDs indicated.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.
        target_ids : np.ndarray
            The array of particle IDs to search for.

        Returns
        -------
        Tuple[int, int]
            A tuple that contains the halo and subhalo indices.

        Raises
        ------
        Exception
            If not all target IDs were found in the particle IDs data for the
            current snapshot.
        """

        if snapnum < self._settings.first_snap:
            # Return NaNs for snapshots outside of the analysis scope.
            return np.nan, np.nan
        else:
            df = load_dm_snapshot(self._galaxy, self._rerun,
                                  self._resolution, snapnum)

            target_idxs = find_indices(df.ParticleIDs.to_numpy(),
                                       target_ids, invalid_specifier=-1)

            if target_idxs.min() == -1:
                raise Exception('-1 detected in target indices.')

            target_halo_idxs = df.Halo.iloc[target_idxs].to_numpy()
            target_subhalo_idxs = df.Subhalo.iloc[target_idxs].to_numpy()

            # Remove particles in the inner or outer fuzz.
            is_not_fuzz = \
                (target_halo_idxs != -1) & (target_subhalo_idxs != -1)

            target_halo = mode(target_halo_idxs[is_not_fuzz])[0][0]
            target_subhalo = mode(target_subhalo_idxs[is_not_fuzz])[0][0]

            return target_halo, target_subhalo

    @timer
    def track_galaxy(self) -> None:
        """This method tracks the galaxy through all snapshots.
        """

        for snapnum in range(self._n_snapshots - 1, -1, -1):
            if (snapnum != self._n_snapshots - 1)\
               & (snapnum >= self._settings.first_snap):
                target_halo = self._df.MainHaloIDX.iloc[snapnum + 1]
                target_subhalo = self._df.MainSubhaloIDX.iloc[snapnum + 1]
                target_ids = self._find_most_bound_dm_ids(snapnum + 1,
                                                          target_halo,
                                                          target_subhalo)
                new_target_halo, new_target_subhalo = \
                    self._find_location_of_dm_ids(snapnum, target_ids)
                self._df.MainHaloIDX.iloc[snapnum] = new_target_halo
                self._df.MainSubhaloIDX.iloc[snapnum] = new_target_subhalo

    def save_data(self) -> None:
        """This method saves the data to a csv file.
        """

        self._df.to_csv(f'{self._paths.data}main_object_idxs.csv')

    @staticmethod
    def make_plot(self) -> None:
        """This method creates a plot to visualize the main object halo/subhalo
        index. Note that this plots all galaxies and hence the galaxy and
        rerun parameter of the class constructor are not used.
        """

        figure_setup()

        fig, axs = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH),
                                nrows=6, ncols=5,
                                sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)

        for ax_idx, ax in enumerate(axs.flat):
            ax.label_outer()
            ax.grid(True, ls='-', lw=0.5, c='silver')
            ax.tick_params(which='both', direction="in")
            ax.set_xlim(0, 14)
            ax.set_ylim(-0.5, 5.5)
            ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_linewidth(1.5)

            galaxy = ax_idx + 1

            paths = Paths(galaxy, False, 4)
            simulation = Simulation(False, 4)

            df = pd.read_csv(f'{paths.data}main_object_idxs.csv')

            ax.plot(simulation.times, df.MainHaloIDX, c='tab:red', lw=2,
                    label='Halo', zorder=10)
            ax.plot(simulation.times, df.MainSubhaloIDX, c='tab:green', lw=1,
                    label='Subhalo', zorder=11)

            if galaxy == 1:
                ax.legend(loc='upper left', ncol=1, fontsize=5, framealpha=0,
                          bbox_to_anchor=(0.05, 0.95))

            add_redshift(ax)
            ax.text(0.95, 0.95, f'Au{galaxy}', size=6,
                    ha='right', va='top',
                    transform=ax.transAxes,
                    bbox={"facecolor": "silver", "edgecolor": "white",
                          "pad": .2, 'boxstyle': 'round', 'lw': 1})

            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel('Index')
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel('Time [Gyr]')

        fig.savefig(f'images/level{self._resolution}/galaxy_tracker.png')
        plt.close(fig)


if __name__ == '__main__':
    # Analysis.
    # settings = Settings()
    # for galaxy in settings.galaxies:
    #     print(f'Analyzing Au{galaxy}... ', end='')
    #     galaxy_tracker = GalaxyTracker(galaxy, False, 4)
    #     galaxy_tracker.track_galaxy()
    #     galaxy_tracker.save_data()
    #     print(' Done.')

    # Plotting.
    galaxy_tracker = GalaxyTracker(6, False, 4)
    galaxy_tracker.make_plot()
