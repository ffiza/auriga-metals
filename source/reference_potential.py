from multiprocessing import Pool
from sys import stdout
import os
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from snapshot import Snapshot
from settings import Settings
from paths import Paths
from support import find_idx_ksmallest, timer, make_snapshot_number
from support import create_or_load_dataframe


class ReferencePotentialAnalysis:
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
    _save_data()
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

    def _calc_reference_potential(self, snapnum: int) -> float:
        """
        This method calculates the reference potential for a given snapshot.

        Parameters
        ----------
        snapnum : int
            The snapshot number.

        Returns
        -------
        float
            The reference potential for this snapshot.

        Raises
        ------
        ValueError
            Raise ValueError if not enough dark matter particles are detected.
        """

        settings = Settings()

        if snapnum >= settings.first_snap:
            self._virial_radius = self._df["VirialRadius_ckpc"].iloc[snapnum]

            s = Snapshot(self._galaxy, self._rerun, self._resolution, snapnum)
            s.drop_types([0, 2, 3, 4, 5])
            s.calc_extra_coordinates()

            # Find the indices of the smallest k values in array
            if len(s.df.index) < settings.neighbor_number:
                raise ValueError('Too few DM particles detected.')
            else:
                idx = find_idx_ksmallest(
                    np.abs(s.df["rCoordinates"] - settings.infinity_factor
                           * self._virial_radius).to_numpy(),
                    settings.neighbor_number)

                # Calculate the mean potential for selected DM particles
                reference_potential = s.df.Potential.iloc[idx].mean()
        else:
            reference_potential = np.nan

        return reference_potential

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the reference potential for all snapshots.
        """

        settings = Settings()

        snapnums = list(range(self._n_snapshots))
        reference_potential = np.array(
            Pool(settings.processes).map(
                self._calc_reference_potential, snapnums))

        self._df["ReferencePotential_(km/s)^2"] = reference_potential

        self._save_data()

    def _save_data(self) -> None:
        """
        This method saves the data.
        """

        self._df.set_index(keys="SnapshotNumber")
        self._df.to_csv(f"{self._paths.data}temporal_data.csv",
                        index=False)


if __name__ == "__main__":
    analysis = ReferencePotentialAnalysis(1, False, 4)
    analysis._calc_reference_potential(30)
