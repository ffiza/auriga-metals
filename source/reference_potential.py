import numpy as np
from multiprocessing import Pool
from snapshot import Snapshot
from settings import Settings
from utils.paths import Paths
from utils.support import find_idx_ksmallest, timer


class ReferencePotential:
    # TODO: Add documentation.

    def __init__(self, galaxy: int, rerun: bool, resolution: int) -> None:
        # TODO: Add documentation.

        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = 252 if self._rerun else 128
        self._settings = Settings()
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)

    def _calc_reference_potential(self, snapnum: int) -> None:
        # TODO: Add documentation.

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
                self._settings.neighbor_number)

            # Calculate the mean potential for selected DM particles
            reference_potential = s.df.Potential[idx].mean()

        return reference_potential

    @timer
    def analyze_galaxy(self) -> None:
        # TODO: Add documentation.

        snapnums = list(range(self._n_snapshots))
        self._reference_potentials = np.array(
            Pool().map(self._calc_reference_potential, snapnums))

    def save_data(self) -> None:
        """This method saves the data.
        """

        np.savetxt(f"{self._paths.data}subhalo_vels.csv",
                   self._reference_potentials)


if __name__ == "__main__":
    settings = Settings()
    for galaxy in settings.galaxies:
        print(f"Analyzing Au{galaxy}... ", end='')
        reference_potentials = ReferencePotential(galaxy, False, 4)
        reference_potentials.analyze_galaxy()
        reference_potentials.save_data()
        print(" Done.")
