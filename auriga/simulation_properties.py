from multiprocessing import Pool
from support import make_snapshot_number
from sys import stdout
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import argparse

from loadmodules import gadget_readsnap, load_subfind
from auriga.cosmology import Cosmology
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.support import timer, create_or_load_dataframe
from auriga.parser import parse


class SimulationPropertiesAnalysis:
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
        self._df = create_or_load_dataframe(
            f"{self._paths.results}temporal_data.csv")

    def _calc_properties_in_snapshot(self, snapshot_number: int) -> tuple:
        """
        This method calculates the snapshot number, cosmic time, lookback
        time, expansion factor and redshift for a given snapshot.

        Parameters
        ----------
        snapnum : int
            The snapshot number.

        Returns
        -------
        tuple
            A tuple with the calculated values.
        """

        cosmology = Cosmology()

        sf = gadget_readsnap(snapshot=snapshot_number,
                             snappath=self._paths.snapshots,
                             onlyHeader=True,
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False)
        time = cosmology.redshift_to_time(sf.redshift)
        lookback_time = cosmology.redshift_to_lookback_time(sf.redshift)
        expansion_factor = sf.time
        redshift = sf.redshift

        return (snapshot_number, time, lookback_time, redshift,
                expansion_factor)

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the properties for all snapshots in
        this simulation.
        """

        settings = Settings()

        snapnums = list(range(self._n_snapshots))
        data = np.array(Pool(settings.processes).map(
            self._calc_properties_in_snapshot, snapnums))

        self._df["SnapshotNumber"] = data[:, 0].astype(np.uint16)
        self._df["Time_Gyr"] = np.round(data[:, 1], 7)
        self._df["LookbackTime_Gyr"] = np.round(data[:, 2], 7)
        self._df["Redshift"] = np.round(data[:, 3], 7)
        self._df["ExpansionFactor"] = np.round(data[:, 4], 7)

        self._save_data()

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        self._df.set_index(keys="SnapshotNumber")
        self._df.to_csv(f"{self._paths.results}temporal_data.csv",
                        index=False)


def run_analysis(simulation: str) -> None:
    stdout.write(f"Analyzing {simulation.upper()}... ")
    analysis = SimulationPropertiesAnalysis(simulation=simulation)
    analysis.analyze_galaxy()
    stdout.write(" Done.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulation to consider.")
    args = parser.parse_args()
    for simulation in args.simulations:
        run_analysis(simulation=simulation)


if __name__ == "__main__":
    main()
