from multiprocessing import Pool
from support import make_snapshot_number
from sys import stdout
import os
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from cosmology import Cosmology
from pylib import gadget, gadget_subfind
from settings import Settings
from paths import Paths
from support import timer, create_or_load_dataframe


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
        self._df = create_or_load_dataframe(
            f"{self._paths.data}temporal_data.csv")

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
        cosmology = Cosmology()

        sf = gadget.gadget_readsnap(snapshot=snapshot_number,
                                    snappath=self._paths.snapshots,
                                    onlyHeader=True,
                                    lazy_load=True,
                                    cosmological=False,
                                    applytransformationfacs=False)
        time = cosmology.redshift_to_time(sf.redshift)
        lookback_time = cosmology.redshift_to_lookback_time(sf.redshift)
        expansion_factor = sf.time
        redshift = sf.redshift
        del sf

        if snapshot_number >= settings.first_snap:
            sb = gadget_subfind.load_subfind(id=snapshot_number,
                                             dir=self._paths.snapshots,
                                             cosmological=False)
            virial_radius = sb.data['frc2'][0] * 1E3 / sb.hubbleparam  # ckpc
            virial_mass = sb.data['fmc2'][0] / sb.hubbleparam  # 1E10 Msun
            del sb
        else:
            virial_radius = np.nan
            virial_mass = np.nan

        return (snapshot_number, time, lookback_time, redshift,
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

        self._df["SnapshotNumber"] = data[:, 0].astype(np.int)
        self._df["Time_Gyr"] = data[:, 1]
        self._df["LookbackTime_Gyr"] = data[:, 2]
        self._df["Redshift"] = data[:, 3]
        self._df["ExpansionFactor"] = data[:, 4]
        self._df["VirialRadius_ckpc"] = data[:, 5]
        self._df["VirialMass_1E10Msun"] = data[:, 6]

        self._save_data()

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        self._df.set_index(keys="SnapshotNumber")
        self._df.to_csv(f"{self._paths.data}temporal_data.csv",
                        index=False)


def run_analysis(galaxy: int, rerun: bool, resolution: int) -> None:
    stdout.write(f"Analyzing Au{galaxy}... ")
    analysis = GalacticPropertiesAnalysis(galaxy, rerun, resolution)
    analysis.analyze_galaxy()
    stdout.write(" Done.\n")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        run_analysis(galaxy=galaxy, rerun=False, resolution=4)
        if galaxy in settings.reruns:
            run_analysis(galaxy=galaxy, rerun=True, resolution=4)


if __name__ == "__main__":
    main()
