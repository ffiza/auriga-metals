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

        df = load_dm_snapshot(present_day_snapnum)

        df.sort_values(by=['Potential'], ascending=True, inplace=True)

        most_bound_ids = \
            df.ParticleIDs.iloc[0:self._settings.n_track_dm_parts]

        return most_bound_ids.to_numpy()

    def _find_location_of_target_ids(self, snapnum: int) -> Tuple[int, int]:
        if snapnum < self._settings.first_snap:
            return 0, 0
        else:
            df = load_dm_snapshot(snapnum)

            target_idxs = find_indices(df.ParticleIDs.to_numpy(),
                                       self._target_ids, invalid_specifier=-1)

            if target_idxs.min() == -1:
                raise Exception('-1 detected in target indices.')

            target_halo_idxs = df.Halo.iloc[target_idxs].to_numpy()
            target_subhalo_idxs = df.Subhalo.iloc[target_idxs].to_numpy()

            target_halo = mode(target_halo_idxs, keepdims=False)[0]
            target_subhalo = mode(target_subhalo_idxs, keepdims=False)[0]

            return target_halo, target_subhalo

    def track_galaxy(self) -> None:
        snapnums = [i for i in range(self._n_snapshots)]
        data = np.array(
            Pool(2).map(self._find_location_of_target_ids, snapnums))

        self.df['MainHaloIDX'] = data[:, 0]
        self.df['MainSubhaloIDX'] = data[:, 1]

    def save_data(self) -> None:
        self.df.to_csv(f'{self._paths.data}main_object_idxs.csv')


if __name__ == '__main__':
    galaxy_tracker = GalaxyTracker(6, False, 4)
    galaxy_tracker.track_galaxy()
    galaxy_tracker.save_data()
