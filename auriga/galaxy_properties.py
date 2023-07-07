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


class GalaxyPropertiesAnalysis:
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

        settings = Settings()

        if snapshot_number >= settings.first_snap:
            sb = load_subfind(id=snapshot_number,
                              dir=self._paths.snapshots,
                              cosmological=False)

            halo_idx = self._df["MainHaloIdx"].loc[snapshot_number]
            subhalo_idx = self._df["MainSubhaloIdx"].loc[snapshot_number]
            subhalo_grouptab_idx = sb.data["ffsh"][halo_idx] + subhalo_idx

            virial_radius = sb.data['frc2'][subhalo_grouptab_idx] * 1E3 \
                / sb.hubbleparam  # ckpc
            virial_mass = sb.data['fmc2'][subhalo_grouptab_idx] \
                / sb.hubbleparam  # 1E10 Msun

            virial_radius_00 = sb.data['frc2'][0] * 1E3 / sb.hubbleparam
            virial_mass_00 = sb.data['fmc2'][0] / sb.hubbleparam

            del sb
        else:
            virial_radius = np.nan
            virial_mass = np.nan
            virial_radius_00 = np.nan
            virial_mass_00 = np.nan

        return virial_radius, virial_mass, virial_radius_00, virial_mass_00

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the properties for all snapshots in
        this simulation.
        """

        snapnums = list(range(self._n_snapshots))
        data = np.array(Pool().map(
            self._calc_properties_in_snapshot, snapnums))

        self._df["VirialRadius_ckpc"] = np.round(data[:, 0], 3)
        self._df["VirialMass_1E10Msun"] = np.round(data[:, 1], 7)
        self._df["VirialRadius00_ckpc"] = np.round(data[:, 2], 3)
        self._df["VirialMass00_1E10Msun"] = np.round(data[:, 3], 7)

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
    analysis = GalaxyPropertiesAnalysis(simulation=simulation)
    analysis.analyze_galaxy()
    stdout.write(" Done.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulations to consider.")
    args = parser.parse_args()
    for simulation in args.simulations:
        run_analysis(simulation=simulation)


if __name__ == "__main__":
    main()
