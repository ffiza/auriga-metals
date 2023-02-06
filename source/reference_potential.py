import numpy as np
from multiprocessing import Pool
from os.path import exists
import pandas as pd

from snapshot import Snapshot
from settings import Settings
from utils.paths import Paths
from utils.support import find_idx_ksmallest, timer


class ReferencePotential:
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
    _reference_potentials : np.ndarray
        An array with the reference potential for each snapshot of this
        simulation.
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
        self._n_snapshots = 252 if self._rerun else 128
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._df = self._create_or_load_dataframe()

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

        self._virial_radius = np.loadtxt(
            f'{self._paths.data}virial_radius.csv')[self._snapnum]

        s = Snapshot(self.galaxy, self.rerun, self.resolution, snapnum)
        s.drop_types([0, 2, 3, 4, 5])

        # Find the indices of the smallest k values in array
        if len(s.df.index) < self.settings.neighbor_number:
            raise ValueError('Too few DM particles detected.')
        else:
            idx = find_idx_ksmallest(
                np.abs(s.df.SphericalRadius
                       - self.settings.infinity_factor * self._virial_radius),
                settings.neighbor_number)

            # Calculate the mean potential for selected DM particles
            reference_potential = s.df.Potential[idx].mean()

        return reference_potential

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the reference potential for all snapshots.
        """

        snapnums = list(range(self._n_snapshots))
        self._reference_potentials = np.array(
            Pool().map(self._calc_reference_potential, snapnums))

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
            df = pd.loadcsv(f"{self._paths.data}temporal_data.csv")
        else:
            df = pd.DataFrame()
        return df

    def save_data(self) -> None:
        """
        This method saves the data.
        """

        self.df.to_csv(f"{self._paths.data}temporal_data.csv")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        print(f"Analyzing Au{galaxy}... ", end='')
        reference_potentials = ReferencePotential(galaxy, False, 4)
        reference_potentials.analyze_galaxy()
        reference_potentials.save_data()
        print(" Done.")


if __name__ == "__main__":
    main()
