import numpy as np
from scipy.stats import mode
import pandas as pd
from loadmodules import gadget_readsnap, load_subfind
from utils.support import find_indices
from multiprocessing import Pool
from auriga.snapshot import Snapshot
from auriga.settings import Settings
from utils.paths import Paths
from typing import Tuple


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
    df = pd.DataFrame({'ParticleIDs': sf.id,
                       'Halo': sf.halo,
                       'Subhalo': sf.subhalo,
                       'Potential': sf.pot/sf.time  # (km/s)^2
                       })
    
    return df


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
        self._n_snapshots = 252 if self._rerun else 128
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._settings = Settings()

        self._target_ids = self._find_present_day_most_bound_dm_ids()

        self._df = pd.DataFrame()

    def _find_present_day_most_bound_dm_ids(self) -> np.ndarray:
        """This method finds an array of IDs that belong to the present-day
        most bound dark matter particles. The amount of particles to consider
        is defined in Settings.

        Returns
        -------
        np.ndarray
            An array that contains the IDs of the most bound dark matter
            particles.
        """
 
        present_day_snapnum = 251 if self._rerun else 127

        df = load_dm_snapshot(self._galaxy, self._rerun,
                              self._resolution, present_day_snapnum)

        # Keep only particles from the halo 0, subhalo 0.
        df = df[(df.Halo == 0) & (df.Subhalo == 0)]

        df.sort_values(by=['Potential'], ascending=True, inplace=True)

        most_bound_ids = \
            df.ParticleIDs.iloc[0:self._settings.n_track_dm_parts]

        return most_bound_ids.to_numpy()

    def _find_location_of_target_ids(self, snapnum: int) -> Tuple[int, int]:
        """This method finds the most common halo/subhalo index in the current
        snapshot for the particles being tracked.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.

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
            return 0, 0
        else:
            df = load_dm_snapshot(self._galaxy, self._rerun,
                                  self._resolution, snapnum)

            target_idxs = find_indices(df.ParticleIDs.to_numpy(),
                                       self._target_ids, invalid_specifier=-1)

            if target_idxs.min() == -1:
                raise Exception('-1 detected in target indices.')

            target_halo_idxs = df.Halo.iloc[target_idxs].to_numpy()
            target_subhalo_idxs = df.Subhalo.iloc[target_idxs].to_numpy()

            target_halo = mode(target_halo_idxs)[0][0]
            target_subhalo = mode(target_subhalo_idxs)[0][0]

            return target_halo, target_subhalo

    def track_galaxy(self) -> None:
        """This method tracks the galaxy through all snapshots.
        """      
  
        snapnums = [i for i in range(self._n_snapshots)]
        data = np.array(
            Pool().map(self._find_location_of_target_ids, snapnums))

        self._df['MainHaloIDX'] = data[:, 0]
        self._df['MainSubhaloIDX'] = data[:, 1]

    def save_data(self) -> None:
        """This method saves the data to a csv file.
        """

        self._df.to_csv(f'{self._paths.data}main_object_idxs.csv')


if __name__ == '__main__':
    settings = Settings()

    for galaxy in settings.galaxies:
        galaxy_tracker = GalaxyTracker(galaxy, False, 4)
        galaxy_tracker.track_galaxy()
        galaxy_tracker.save_data()
