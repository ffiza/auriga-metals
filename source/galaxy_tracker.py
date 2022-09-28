import numpy as np
from scipy.stats import mode
import pandas as pd
from utils.support import find_indices
from multiprocessing import Pool
from auriga.snapshot import Snapshot
from auriga.settings import Settings
from utils.paths import Paths
from typing import Tuple


class GalaxyTracker:
    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = 252 if self._rerun else 128
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._settings = Settings()

        self._target_ids = self._find_present_day_most_bound_dm_ids()

        self.df = pd.DataFrame()

    def _find_present_day_most_bound_dm_ids(self) -> np.ndarray:
        present_day_snapnum = 251 if self._rerun else 127
        s = Snapshot(self._galaxy, self._rerun, self._resolution,
                     snapnum=present_day_snapnum)

        # We consider the present-day halo 0 and subhalo 0 to be the main
        # object.
        s.keep_only_halo(0, 0)

        # Keep only dark matter particles.
        s.drop_types([0, 2, 3, 4, 5])

        # Remove all unused features.
        s.keep_only_feats(['Halo', 'Subhalo', 'ParticleIDs', 'Potential'])

        s.df.sort_values(by=['Potential'], ascending=True, inplace=True)

        most_bound_ids = \
            s.df.ParticleIDs.iloc[0:self._settings.n_track_dm_parts]

        return most_bound_ids.to_numpy()

    def _find_location_of_target_ids(self, snapnum: int) -> Tuple[int, int]:
        if snapnum < self._settings.first_snap:
            return 0, 0
        else:
            s = Snapshot(self._galaxy, self._rerun, self._resolution, snapnum)
            s.drop_types([0, 2, 3, 4, 5])
            s.keep_only_feats(['Halo', 'Subhalo', 'ParticleIDs'])

            target_idxs = find_indices(s.df.ParticleIDs.to_numpy(),
                                       self._target_ids, invalid_specifier=-1)

            if target_idxs.min() == -1:
                raise Exception('-1 detected in target indices.')

            target_halo_idxs = s.df.Halo.iloc[target_idxs].to_numpy()
            target_subhalo_idxs = s.df.Subhalo.iloc[target_idxs].to_numpy()

            target_halo = mode(target_halo_idxs, keepdims=False)[0]
            target_subhalo = mode(target_subhalo_idxs, keepdims=False)[0]

            return target_halo, target_subhalo

    def track_galaxy(self) -> None:
        snapnums = [i for i in range(self._n_snapshots)]
        data = np.array(
            Pool().map(self._find_location_of_target_ids, snapnums))

        self.df['MainHaloIDX'] = data[:, 0]
        self.df['MainSubhaloIDX'] = data[:, 1]

    def save_data(self) -> None:
        self.df.to_csv(f'{self._paths.data}main_object_idxs.csv')


if __name__ == '__main__':
    settings = Settings()

    for galaxy in settings.galaxies:
        galaxy_tracker = GalaxyTracker(galaxy, False, 4)
        galaxy_tracker.track_galaxy()
        galaxy_tracker.save_data()
