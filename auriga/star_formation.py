from multiprocessing import Pool
from sys import stdout
import os
import argparse
import pandas as pd
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.support import timer, make_snapshot_number
from auriga.support import create_or_load_dataframe
from auriga.parser import parse


class StarFormationAnalysis:
    def __init__(self, simulation: str) -> None:
        """
        Parameters
        ----------
        simulation : str
            The simulation to analyse.
        """

        galaxy, rerun, resolution = parse(simulation=simulation)

        # Start analysis at snapshot 36 because of some issues in Au11
        self._first_snap: int = 36

        self._simulation = simulation
        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._df = create_or_load_dataframe(
            f"{self._paths.results}temporal_data.csv")

    def _calc_sfr(self, snapnum: int) -> float:
        """
        This method calculates the star formation rate by galactic component.

        Parameters
        ----------
        snapnum : int
            The snapshot number.

        Returns
        -------
        list
            The SFR for the halo, bulge, cold disc, and warm disc (in that
            order).
        """

        settings = Settings()

        sfr_by_region = np.nan * np.ones(4)

        if snapnum >= self._first_snap:
            s = Snapshot(simulation=f"{self._simulation}_s{snapnum}",
                         loadonlytype=[0, 1, 2, 3, 4, 5])
            s.tag_particles_by_region(
                disc_std_circ=settings.disc_std_circ,
                disc_min_circ=settings.disc_min_circ,
                cold_disc_delta_circ=settings.cold_disc_delta_circ,
                bulge_max_specific_energy=settings.bulge_max_specific_energy)
            sfr_by_region = s.calculate_sfr_by_region()

        return sfr_by_region

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the SFR for all snapshots.
        """

        snapnums = list(range(self._n_snapshots))
        sfr_by_region = np.array(Pool().map(self._calc_sfr, snapnums))

        self._df["SFR_H_Msun/yr"] = sfr_by_region[:, 0]
        self._df["SFR_B_Msun/yr"] = sfr_by_region[:, 1]
        self._df["SFR_CD_Msun/yr"] = sfr_by_region[:, 2]
        self._df["SFR_WD_Msun/yr"] = sfr_by_region[:, 3]

        self._save_data()

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        self._df.set_index(keys="SnapshotNumber")
        self._df.to_csv(f"{self._paths.results}temporal_data.csv", index=False)


def main():
    settings = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulations to consider.")
    args = parser.parse_args()
    for simulation in args.simulations:
        stdout.write(f"Analyzing {simulation.upper()}... ")
        analysis = StarFormationAnalysis(simulation=simulation)
        analysis.analyze_galaxy()
        stdout.write(" Done.\n")


if __name__ == "__main__":
    main()
