from typing import Optional
from loadmodules import gadget_readsnap, load_subfind
from utils.paths import Paths
from multiprocessing import Pool
from auriga.settings import Settings
from scipy import linalg as linalg
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

    def calculate_rotation_matrices(self) -> None:
        """
        This method calculates the rotation matrices of the main subhalo for
        all snapshots in this simulation.
        """

        snapnums = [i for i in range(self._n_snapshots)]
        self._rotation_matrices = np.array(
            Pool().map(self._calculate_rotation_matrix, snapnums))

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
        if snapnum <= settings.first_snap:
            return np.nan * np.ones((9,))
        else:
            subhalo_vel = np.loadtxt(f'{self._paths.data}/subhalo_vels.csv')

            sf = gadget_readsnap(snapshot=snapnum,
                                 snappath=self._paths.snapshots,
                                 loadonlytype=[4], lazy_load=True,
                                 cosmological=False,
                                 applytransformationfacs=False)
            sb = load_subfind(id=snapnum, dir=self._paths.snapshots,
                              cosmological=False)
            sf.calc_sf_indizes(sf=sb)

            pos = (sf.pos - sb.data['spos'][0] / sf.hubbleparam) * 1000  # ckpc
            vel = sf.vel * np.sqrt(sf.time) - subhalo_vel[snapnum]  # km/s
            mass = sf.mass * 1E10  # Msun

            r = np.linalg.norm(pos, axis=1)  # ckpc
            isInner = r < self._distance

            pos = pos[isInner]
            vel = vel[isInner]
            mass = mass[isInner]

            # Inertia matrix.
            inertia_tensor = np.nan * np.ones((3, 3), dtype='float64')
            inertia_tensor[0, 0] = np.sum([mass*(pos[:, 1]**2 + pos[:, 2]**2)])
            inertia_tensor[1, 1] = np.sum([mass*(pos[:, 0]**2 + pos[:, 2]**2)])
            inertia_tensor[2, 2] = np.sum([mass*(pos[:, 0]**2 + pos[:, 1]**2)])
            inertia_tensor[0, 1] = -np.sum([mass*pos[:, 0]*pos[:, 1]])
            inertia_tensor[1, 0] = inertia_tensor[0][1]
            inertia_tensor[0, 2] = -np.sum([mass*pos[:, 0]*pos[:, 2]])
            inertia_tensor[2, 0] = inertia_tensor[0][2]
            inertia_tensor[1, 2] = -np.sum([mass*pos[:, 1]*pos[:, 2]])
            inertia_tensor[2, 1] = inertia_tensor[1][2]

            # Diagonalization.
            e_vals, e_vecs = linalg.eigh(inertia_tensor)
            x = [1, 0, 0]
            X = e_vecs[:, 0]
            y = [0, 1, 0]
            Y = e_vecs[:, 1]
            z = [0, 0, 1]
            Z = e_vecs[:, 2]

            # Angular momentum.
            j = np.nan * np.ones(3, dtype='float64')
            j[0] = np.sum(mass*(pos[:, 1]*vel[:, 2] - pos[:, 2]*vel[:, 1]))
            j[1] = np.sum(mass*(pos[:, 2]*vel[:, 0] - pos[:, 0]*vel[:, 2]))
            j[2] = np.sum(mass*(pos[:, 0]*vel[:, 1] - pos[:, 1]*vel[:, 0]))
            j /= np.linalg.norm(j)  # Normalize.

            # Verify direction of principal axes.
            direction = np.dot(j, Z)
            if direction < 0:
                X = X
                Y = -Y
                Z = -Z
            if np.sign(np.cross(X, Y)[2]) != np.sign(Z[2]):
                X = -X

            # Define rotation matrix.
            rotation_matrix = np.nan * np.ones((3, 3), dtype='float64')
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
