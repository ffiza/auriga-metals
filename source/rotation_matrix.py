from loadmodules import gadget_readsnap, load_subfind
from auriga.settings import Settings
from utils.paths import Paths
from multiprocessing import Pool
from scipy import linalg as linalg
import pandas as pd
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


class RotationMatrices:
    """
    A class to manage the calculations regarding the rotation matrices of the
    main galaxy.

    Attributes
    ----------
    _galaxy : int
        The snapshot in which to start the analysis.
    _rerun : bool
        A bool to indicate if this is a original run or a rerun.
    _resolution : int
        The resolution level of the simulation.
    _distance : float
        The distance to consider stars for inertia matrix calculation.
    _paths : Paths
        An instance of the Paths class.
    _n_snapshots : int
        The total amount of snapshots in this simulation.
    _rotation_matrices : np.ndarray
        An array with the rotation matrix of each snapshot of this simulation.
    _halo_idxs : np.ndarray
        An array with the indices of the main halo.
    _subhalo_idxs : np.ndarray
        An array with the indices of the main subhalo.

    Methods
    -------
    calculate_rotation_matrices()
        This method calculates the rotation matrix of the main subhalo for all
        snapshots in this simulation.
    _calculate_rotation_matrix(snapnum)
        This method calculates the rotation matrix of the main subhalo in this
        snapshot.
    save_data()
        This method saves the data.
    """

    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        """
        Parameters
        ----------
        galaxy : int
            The snapshot in which to start the analysis.
        rerun : bool
            A bool to indicate if this is a original run or a rerun.
        resolution : int
            The resolution level of the simulation.
        """

        self._galaxy = galaxy
        self._rerun = rerun
        self._n_snapshots = 252 if self._rerun else 128
        self._resolution = resolution
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

        settings = Settings()
        self._distance = settings.rot_mat_distance

        # Set halo/subhalo indices.
        main_obj_df = pd.read_csv(f'{self._paths.data}main_object_idxs.csv')
        self._halo_idxs = main_obj_df.MainHaloIDX.to_numpy()
        self._subhalo_idxs = main_obj_df.MainSubhaloIDX.to_numpy()

    def calculate_rotation_matrices(self) -> None:
        """
        This method calculates the rotation matrices of the main subhalo for
        all snapshots in this simulation.
        """

        snapnums = [i for i in range(self._n_snapshots)]
        self._rotation_matrices = np.array(
            Pool(2).map(self._calculate_rotation_matrix, snapnums))

    def _calculate_rotation_matrix(self, snapnum: int) -> np.ndarray:
        """
        This method calculates the rotation matrix of the main subhalo in this
        snapshot.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.
        """

        settings = Settings()
        if snapnum < settings.first_snap:
            return np.nan * np.ones((9,))
        else:
            halo_idx = self._halo_idxs[snapnum]
            subhalo_idx = self._subhalo_idxs[snapnum]

            subhalo_vel = np.loadtxt(f'{self._paths.data}/subhalo_vels.csv')

            sf = gadget_readsnap(snapshot=snapnum,
                                 snappath=self._paths.snapshots,
                                 loadonlytype=[4], lazy_load=True,
                                 cosmological=False,
                                 applytransformationfacs=False)
            sb = load_subfind(id=snapnum, dir=self._paths.snapshots,
                              cosmological=False)
            sf.calc_sf_indizes(sf=sb)

            # Find the index of the subhalo in the subfind table.
            subhalo_grouptab_idx = sb.data['ffsh'][halo_idx] + subhalo_idx

            pos = (sf.pos - sb.data['spos'][subhalo_grouptab_idx]
                   / sf.hubbleparam) * 1000  # ckpc
            vel = sf.vel * np.sqrt(sf.time) - subhalo_vel[snapnum]  # km/s
            mass = sf.mass * 1E10  # Msun

            r = np.linalg.norm(pos, axis=1)  # ckpc
            is_inner = r < self._distance
            is_main_obj = (sf.halo == halo_idx) & (sf.subhalo == subhalo_idx)

            pos = pos[is_inner & is_main_obj]
            vel = vel[is_inner & is_main_obj]
            mass = mass[is_inner & is_main_obj]

            # Inertia matrix.
            inertia_tensor = np.nan * np.ones((3, 3))
            inertia_tensor[0, 0] = np.sum([mass * (pos[:, 1]**2
                                                   + pos[:, 2]**2)])
            inertia_tensor[1, 1] = np.sum([mass * (pos[:, 0]**2
                                                   + pos[:, 2]**2)])
            inertia_tensor[2, 2] = np.sum([mass * (pos[:, 0]**2
                                                   + pos[:, 1]**2)])
            inertia_tensor[0, 1] = -np.sum([mass * pos[:, 0] * pos[:, 1]])
            inertia_tensor[1, 0] = inertia_tensor[0][1]
            inertia_tensor[0, 2] = -np.sum([mass * pos[:, 0] * pos[:, 2]])
            inertia_tensor[2, 0] = inertia_tensor[0][2]
            inertia_tensor[1, 2] = -np.sum([mass * pos[:, 1] * pos[:, 2]])
            inertia_tensor[2, 1] = inertia_tensor[1][2]

            # Diagonalization.
            e_vals, e_vecs = linalg.eigh(inertia_tensor)
            x = np.array([1, 0, 0])
            X = e_vecs[:, 0]
            y = np.array([0, 1, 0])
            Y = e_vecs[:, 1]
            z = np.array([0, 0, 1])
            Z = e_vecs[:, 2]

            # Angular momentum.
            j = np.nan * np.ones(3)
            j[0] = np.sum(mass * (pos[:, 1] * vel[:, 2]
                                  - pos[:, 2] * vel[:, 1]))
            j[1] = np.sum(mass * (pos[:, 2] * vel[:, 0]
                                  - pos[:, 0] * vel[:, 2]))
            j[2] = np.sum(mass * (pos[:, 0] * vel[:, 1]
                                  - pos[:, 1] * vel[:, 0]))
            j /= np.linalg.norm(j)  # Normalize.

            # Verify direction of principal axes with angular momentum.
            direction = np.dot(j, Z)
            if direction < 0:
                X *= 1
                Y *= -1
                Z *= -1
            if np.sign(np.cross(X, Y)[2]) != np.sign(Z[2]):
                X *= -1

            # Define rotation matrix.
            rotation_matrix = np.nan * np.ones((3, 3))
            rotation_matrix[0, 0] = np.dot(X, x)
            rotation_matrix[0, 1] = np.dot(X, y)
            rotation_matrix[0, 2] = np.dot(X, z)
            rotation_matrix[1, 0] = np.dot(Y, x)
            rotation_matrix[1, 1] = np.dot(Y, y)
            rotation_matrix[1, 2] = np.dot(Y, z)
            rotation_matrix[2, 0] = np.dot(Z, x)
            rotation_matrix[2, 1] = np.dot(Z, y)
            rotation_matrix[2, 2] = np.dot(Z, z)

            return rotation_matrix.reshape((9,))

    def save_data(self) -> None:
        """
        This method saves the data.
        """

        np.savetxt(f'{self._paths.data}rotation_matrices.csv',
                   self._rotation_matrices)


if __name__ == '__main__':
    rotation_mats = RotationMatrices(6, False, 4)
    rotation_mats.calculate_rotation_matrices()
    rotation_mats.save_data()
