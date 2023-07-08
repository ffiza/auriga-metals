from multiprocessing import Pool
from scipy import linalg as linalg
import pandas as pd
import os
from sys import stdout
import argparse
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from loadmodules import gadget_readsnap, load_subfind
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.support import timer, make_snapshot_number
from auriga.parser import parse


class RotationMatrixAnalysis:
    """
    A class to manage the calculations regarding the rotation matrices of the
    main galaxy.
    """

    def __init__(self, simulation: str) -> None:
        """
        Parameters
        ----------
        simulation : str
            The simulation to load.
        """

        self._simulation = simulation
        galaxy, rerun, resolution = parse(simulation=simulation)
        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

        # Set halo/subhalo indices.
        df = pd.read_csv(f"{self._paths.results}temporal_data.csv",
                         usecols=["MainHaloIdx", "MainSubhaloIdx"])
        self._halo_idxs = df["MainHaloIdx"].to_numpy()
        self._subhalo_idxs = df["MainSubhaloIdx"].to_numpy()

    @timer
    def calculate_rotation_matrices(self) -> None:
        """
        This method calculates the rotation matrices of the main subhalo for
        all snapshots in this simulation.
        """

        snapnums = list(range(self._n_snapshots))
        self._rotation_matrices = np.array(
            Pool().map(self._calculate_rotation_matrix, snapnums))

        self._save_data()

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

        halo_idx = self._halo_idxs[snapnum]
        subhalo_idx = self._subhalo_idxs[snapnum]

        subhalo_vel = np.loadtxt(f'{self._paths.results}/subhalo_vels.csv')

        sf = gadget_readsnap(snapshot=snapnum,
                             snappath=self._paths.snapshots,
                             loadonlytype=[4],
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False)
        sb = load_subfind(id=snapnum, dir=self._paths.snapshots,
                          cosmological=False)
        sf.calc_sf_indizes(sf=sb)

        # Find the index of the subhalo in the subfind table.
        subhalo_grouptab_idx = sb.data['ffsh'][halo_idx] + subhalo_idx

        pos = (sf.pos - sb.data['spos'][
            subhalo_grouptab_idx] / sf.hubbleparam) * 1000  # ckpc
        vel = sf.vel * np.sqrt(sf.time) - subhalo_vel[snapnum]  # km/s
        mass = sf.mass * 1E10  # Msun

        r = np.linalg.norm(pos, axis=1)  # ckpc
        is_inner = r < settings.rot_mat_distance
        is_main_obj = (sf.halo == halo_idx) & (sf.subhalo == subhalo_idx)

        pos = pos[is_inner & is_main_obj]
        vel = vel[is_inner & is_main_obj]
        mass = mass[is_inner & is_main_obj]

        # Inertia matrix.
        inertia_tensor = np.nan * np.ones((3, 3))
        inertia_tensor[0, 0] = np.sum([mass * (pos[:, 1]**2 + pos[:, 2]**2)])
        inertia_tensor[1, 1] = np.sum([mass * (pos[:, 0]**2 + pos[:, 2]**2)])
        inertia_tensor[2, 2] = np.sum([mass * (pos[:, 0]**2 + pos[:, 1]**2)])
        inertia_tensor[0, 1] = -np.sum([mass * pos[:, 0] * pos[:, 1]])
        inertia_tensor[1, 0] = inertia_tensor[0][1]
        inertia_tensor[0, 2] = -np.sum([mass * pos[:, 0] * pos[:, 2]])
        inertia_tensor[2, 0] = inertia_tensor[0][2]
        inertia_tensor[1, 2] = -np.sum([mass * pos[:, 1] * pos[:, 2]])
        inertia_tensor[2, 1] = inertia_tensor[1][2]

        # Diagonalize the inertia tensor
        _, e_vecs = linalg.eigh(inertia_tensor)
        x = np.array([1, 0, 0])
        X = e_vecs[:, 0]
        y = np.array([0, 1, 0])
        Y = e_vecs[:, 1]
        z = np.array([0, 0, 1])
        Z = e_vecs[:, 2]

        # Calculate and normalize the angular momentum
        j = np.nan * np.ones(3)
        j[0] = np.sum(mass * (pos[:, 1] * vel[:, 2] - pos[:, 2] * vel[:, 1]))
        j[1] = np.sum(mass * (pos[:, 2] * vel[:, 0] - pos[:, 0] * vel[:, 2]))
        j[2] = np.sum(mass * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
        j /= np.linalg.norm(j)

        # Verify direction of principal axes with angular momentum
        direction = np.dot(j, Z)
        if direction < 0:
            X *= 1
            Y *= -1
            Z *= -1
        if np.sign(np.cross(X, Y)[2]) != np.sign(Z[2]):
            X *= -1

        # Define rotation matrix
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

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        np.savetxt(f'{self._paths.results}rotation_matrices.csv',
                   self._rotation_matrices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulations to consider.")
    args = parser.parse_args()
    for simulation in args.simulations:
        stdout.write(f"Analyzing {simulation.upper()}... ")
        analysis = RotationMatrixAnalysis(simulation=simulation)
        analysis.calculate_rotation_matrices()
        stdout.write(" Done.\n")


if __name__ == "__main__":
    main()
