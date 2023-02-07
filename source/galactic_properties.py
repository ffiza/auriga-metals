from multiprocessing import Pool
from os.path import exists
import pandas as pd
from support import make_snapshot_number
import os
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from cosmology import Cosmology
from pylib import gadget, gadget_subfind
from settings import Settings
from paths import Paths
from support import timer


class GalacticPropertiesAnalysis:
    """
    A class to manage the calculations regarding the reference potential.

    Attributes
    ----------
    _galaxy : int
        The galaxy number.
    _rerun : bool
        A bool to indicate if this is a original run or a rerun.
    _resolution : int
        The resolution level of the simulation.
    _paths : Paths
        An instance of the Paths class.
    _n_snapshots : int
        The total amount of snapshots in this simulation.
    _df : pd.DataFrame
        A DataFrame with all the temporal data for a given simulation. It is
        loaded if it exists or created if not.

    Methods
    -------
    calculate_subhalo_velocities()
        This method calculates the velocity of the main subhalo for all
        snapshots in this simulation.
    _calculate_subhalo_velocity(snapnum)
        This method calculates the velocity of the main subhalo in this
        snapshot.
    _create_or_load_dataframe()
        This method loads the temporal data frame if it exists or creates it
        if it doesn't.
    save_data()
        This method saves the data.
    """

    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        """
        Parameters
        ----------
        galaxy : int
            The galaxy number.
        rerun : bool
            A bool to indicate if this is a original run or a rerun.
        resolution : int
            The resolution level of the simulation.
        """

        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._df = self._create_or_load_dataframe()

    def _calc_properties_in_snapshot(self, snapshot_number: int) -> tuple:
        """
        This method calculates the snapshot number, cosmic time, expansion
        factor and redshift for a given snapshot.

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

        sb = gadget_subfind.load_subfind(id=snapshot_number,
                                         dir=self.snapshot_path,
                                         cosmological=False)

        time = cosmology.redshift_to_time(sb.redshift)
        expansion_factor = sb.time
        redshift = sb.redshift
        virial_radius = sb.data['frc2'][0] * 1E3 / sb.hubbleparam  # ckpc
        virial_mass = sb.data['fmc2'][0] / sb.hubbleparam  # 1E10 Msun

        return (snapshot_number, time, redshift,
                expansion_factor, virial_radius, virial_mass)

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

        self._df["SnapshotNumber"] = data[:, 0]
        self._df["Time_Gyr"] = data[:, 1]
        self._df["Redshift"] = data[:, 2]
        self._df["ExpansionFactor"] = data[:, 3]
        self._df["VirialRadius_ckpc"] = data[:, 4]
        self._df["VirialMass_1E10Msun"] = data[:, 5]

        self._save_data()

    def _create_or_load_dataframe(self) -> pd.DataFrame:
        """
        This method loads the temporal data frame if it exists or creates it
        if it doesn't.

        Returns
        -------
        pd.DataFrame
            The data frame.
        """

        if exists(f"{self._paths.data}temporal_data.csv"):
            df = pd.read_csv(f"{self._paths.data}temporal_data.csv")
        else:
            df = pd.DataFrame()
        return df

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        self._df.to_csv(f"{self._paths.data}temporal_data.csv")


def run_analysis(galaxy: int, rerun: bool, resolution: int) -> None:
    print(f"Analyzing Au{galaxy}... ", end='')
    analysis = GalacticPropertiesAnalysis(galaxy, rerun, resolution)
    analysis.analyze_galaxy()
    print(" Done.")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        run_analysis(galaxy=galaxy, rerun=False, resolution=4)
        if galaxy in settings.reruns:
            run_analysis(galaxy=galaxy, rerun=True, resolution=4)


if __name__ == "__main__":
    main()
