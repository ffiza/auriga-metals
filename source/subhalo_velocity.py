from multiprocessing import Pool
from loadmodules import gadget_readsnap, load_subfind
from auriga.settings import Settings
from utils.paths import Paths
from typing import Optional
import pandas as pd
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


class SubhaloVelocities:
    """
    A class to manage the calculations regarding the velocity of the main
    galaxy.

    Attributes
    ----------
    _galaxy : int
        The snapshot in which to start the analysis.
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
    save_data()
        This method saves the data.
    """

    def __init__(self, galaxy: int, rerun: bool, resolution: int,
                 distance: Optional[float] = 10) -> None:
        """
        Parameters
        ----------
        galaxy : int
            The snapshot in which to start the analysis.
        rerun : bool
            A bool to indicate if this is a original run or a rerun.
        resolution : int
            The resolution level of the simulation.
        distance : float, optional
            The distance to consider stars for velocity calculation.
        """

        self._galaxy = galaxy
        self._rerun = rerun
        self._n_snapshots = 252 if self._rerun else 128
        self._resolution = resolution
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._distance = distance

        # Set halo/subhalo indices.
        main_obj_df = pd.read_csv(f'{self._paths.data}main_object_idxs.csv')
        self._halo_idxs = main_obj_df.MainHaloIDX.to_numpy()
        self._subhalo_idxs = main_obj_df.MainSubhaloIDX.to_numpy()

    def calculate_subhalo_velocities(self) -> None:
        """
        This method calculates the velocity of the main subhalo for all
        snapshots in this simulation.
        """

        snapnums = [i for i in range(self._n_snapshots)]
        self._subhalo_velocities = np.array(
            Pool(2).map(self._calculate_subhalo_velocity, snapnums))

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
            sf = gadget_readsnap(snapshot=snapnum,
                                 snappath=self._paths.snapshots,
                                 loadonlytype=[4], lazy_load=True,
                                 cosmological=False,
                                 applytransformationfacs=False)
            sb = load_subfind(id=snapnum, dir=self._paths.snapshots,
                              cosmological=False)
            sf.calc_sf_indizes(sf=sb)

            pos = (sf.pos - sb.data['spos'][0] / sf.hubbleparam) * 1E3  # ckpc
            del sb
            vel = sf.vel * np.sqrt(sf.time)  # km/s
            mass = sf.mass * 1E10  # Msun
            age = sf.age

            r = np.linalg.norm(pos, axis=1)  # ckpc
            del pos

            is_main_inner_star = (age > 0) & (r < self._distance) & \
                                 (sf.halo == self._halo_idxs[snapnum]) & \
                                 (sf.subhalo == self._subhalo_idxs[snapnum])
            del sf, age, r

            if is_main_inner_star.sum() == 0:
                # No stars were found with the condition (early snapshots).
                return np.array([np.nan, np.nan, np.nan])
            else:
                vel_cm = \
                    mass[is_main_inner_star].T @ vel[is_main_inner_star] / \
                    mass[is_main_inner_star].sum()  # km/s
                return vel_cm

    def save_data(self) -> None:
        """
        This method saves the data.
        """

        np.savetxt(f'{self._paths.data}subhalo_vels.csv',
                   self._subhalo_velocities)


if __name__ == '__main__':
    subhalo_vels = SubhaloVelocities(6, False, 4)
    subhalo_vels.calculate_subhalo_velocities()
    subhalo_vels.save_data()
