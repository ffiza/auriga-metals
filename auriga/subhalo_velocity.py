from multiprocessing import Pool
import pandas as pd
import os
import argparse
from sys import stdout
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from loadmodules import gadget_readsnap, load_subfind

from auriga.settings import Settings
from auriga.paths import Paths
from auriga.support import timer, make_snapshot_number
from auriga.parser import parse


class SubhaloVelocityAnalysis:
    """
    A class to manage the calculations regarding the velocity of the main
    galaxy.
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

        halo_idx = self._halo_idxs[snapnum]
        subhalo_idx = self._subhalo_idxs[snapnum]

        sf = gadget_readsnap(snapshot=snapnum,
                             snappath=self._paths.snapshots,
                             loadonlytype=[4],
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False)
        sb = load_subfind(id=snapnum,
                          dir=self._paths.snapshots,
                          cosmological=False)
        sf.calc_sf_indizes(sf=sb)

        # Find the index of the subhalo in the subfind table.
        subhalo_grouptab_idx = sb.data["ffsh"][halo_idx] + subhalo_idx

        pos = (sf.pos - sb.data["spos"][
            subhalo_grouptab_idx] / sf.hubbleparam) * 1E3  # ckpc
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

        np.savetxt(f"{self._paths.results}subhalo_vels.csv",
                   self._subhalo_velocities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulation to consider.")
    args = parser.parse_args()
    for simulation in args.simulations:
        stdout.write(f"Analyzing {simulation.upper()}... ")
        analysis = SubhaloVelocityAnalysis(simulation=simulation)
        analysis.calculate_subhalo_velocities()
        stdout.write(" Done.\n")


if __name__ == "__main__":
    main()
