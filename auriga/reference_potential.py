from multiprocessing import Pool
from sys import stdout
import os
import argparse
os.environ["MKL_NUM_THREADS"] = "1"  # Limits threads in Numpy
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.support import find_idx_ksmallest, timer, make_snapshot_number
from auriga.support import create_or_load_dataframe
from auriga.parser import parse


class ReferencePotentialAnalysis:
    """
    A class to manage the calculations regarding the reference potential.
    """

    def __init__(self,
                 simulation: str,
                 infinity_factor: float,
                 neighbor_number: int,
                 feat_name: str) -> None:
        """
        Parameters
        ----------
        snapnum : int
            The snapshot number.
        infinity_factor : float
            The distance at which to calculate the reference potential in
            units of the virial radius.
        neightbor_number : int
            The number of dark matter particles to use in the calculation.
        feat_name : str
            The name of the reference potential feature in the target
            data frame.
        """

        self._simulation = simulation
        galaxy, rerun, resolution = parse(simulation=self._simulation)
        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._df = create_or_load_dataframe(
            f"{self._paths.results}temporal_data.csv")

        self._infinity_factor = infinity_factor
        self._neighbor_number = neighbor_number
        self._feat_name = feat_name

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

        if snapnum < settings.first_snap:
            return np.nan

        self._virial_radius = self._df["VirialRadius_ckpc"].iloc[snapnum]

        s = Snapshot(simulation=f"{self._simulation}_s{snapnum}",
                     loadonlytype=[1])
        s.add_extra_coordinates()

        idx = find_idx_ksmallest(
            np.abs(s.r - self._infinity_factor * self._virial_radius),
            self._neighbor_number)

        if np.isnan(s.potential[idx]).sum() != 0:
            raise RuntimeError("NaNs detected in result.")

        return s.potential[idx].mean()

    @timer
    def analyze_galaxy(self) -> None:
        """
        This method calculates the reference potential for all snapshots.
        """

        snapnums = list(range(self._n_snapshots))
        reference_potential = np.array(
            Pool().map(self._calc_reference_potential, snapnums))

        self._df[self._feat_name] = np.round(reference_potential, 3)
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
    parser.add_argument("--infinity_factor",
                        type=float,
                        required=False,
                        default=settings.infinity_factor,
                        help="The distance at which to calculate the "
                             "reference potential in units of the virial "
                             "radius. Default value is "
                             f"{settings.infinity_factor}.")
    parser.add_argument("--neighbor_number",
                        type=int,
                        required=False,
                        default=settings.neighbor_number,
                        help="The number of dark matter particles to use in "
                             "the calculation. Default value is "
                             f"{settings.neighbor_number}.")
    parser.add_argument("--feat_name",
                        type=str,
                        required=False,
                        default="ReferencePotential_(km/s)^2",
                        help="The number of dark matter particles to use in "
                             "the calculation. Default value is "
                             "ReferencePotential_(km/s)^2.")
    args = parser.parse_args()
    for simulation in args.simulations:
        stdout.write(f"Analyzing {simulation.upper()}... ")
        analysis = ReferencePotentialAnalysis(
            simulation=simulation,
            infinity_factor=args.infinity_factor,
            neighbor_number=args.neighbor_number,
            feat_name=args.feat_name)
        analysis.analyze_galaxy()
        stdout.write(" Done.\n")


if __name__ == "__main__":
    main()
