import numpy as np
from scipy.stats import mode
import pandas as pd
from typing import Tuple
import argparse
from sys import stdout
from loadmodules import gadget_readsnap, load_subfind
from auriga.settings import Settings
from auriga.support import find_indices, create_or_load_dataframe, timer
from auriga.support import make_snapshot_number
from auriga.paths import Paths
from auriga.parser import parse


def _load_dm_snapshot(galaxy: int,
                      rerun: bool,
                      resolution: int,
                      snapnum: int,
                      ) -> pd.DataFrame:
    """
    This method loads a snapshot given the parameters, keeping only
    dark matter particles.

    Parameters
    ----------
    simulation : str
        The simulation to load.

    Returns
    -------
    pd.DataFrame
        A data frame with the particle IDs, halo, subhalo and potential.
    """

    paths = Paths(galaxy, rerun, resolution)
    sf = gadget_readsnap(snapshot=snapnum,
                         snappath=paths.snapshots,
                         loadonlytype=[1], lazy_load=True,
                         cosmological=False,
                         applytransformationfacs=False)
    sb = load_subfind(id=snapnum, dir=paths.snapshots,
                      cosmological=False)
    sf.calc_sf_indizes(sf=sb)

    return pd.DataFrame({'ParticleIDs': sf.id,
                         'Halo': sf.halo,
                         'Subhalo': sf.subhalo,
                         'Potential': sf.pot / sf.time  # (km/s)^2
                         })


class GalaxyTracker:
    """
    A class to manage the tracking of the main object in each simulation.
    """

    def __init__(self, simulation: str) -> None:
        """
        Parameters
        ----------
        simulation : str
            The simulation to load.
        """

        galaxy, rerun, resolution = parse(simulation=simulation)

        self._galaxy = galaxy
        self._rerun = rerun
        self._resolution = resolution
        self._n_snapshots = make_snapshot_number(self._rerun, self._resolution)
        self._paths = Paths(self._galaxy, self._rerun, self._resolution)
        self._df = create_or_load_dataframe(
            f"{self._paths.results}temporal_data.csv")

    def _find_most_bound_dm_ids(self,
                                snapnum: int,
                                n_part: int,
                                halo: int = 0,
                                subhalo: int = 0,
                                ) -> np.ndarray:
        """
        This method finds the IDs of the most bound dark matter particles
        in a given snapshot and a given halo and subhalo index. The amount of
        particles to track is defined in Settings.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.
        n_part : int
            The number of dark matter particles to find.
        halo : int, optional
            The index of the halo to search.
        subhalo : int, optional
            The index of the subhalo to search.

        Returns
        -------
        np.ndarray
            An array that contains the IDs of the most bound dark matter
            particles.
        """

        df = _load_dm_snapshot(galaxy=self._galaxy,
                               rerun=self._rerun,
                               resolution=self._resolution,
                               snapnum=snapnum)

        # Keep only particles from the selected halo and subhalo.
        df = df[(df.Halo == halo) & (df.Subhalo == subhalo)]

        # Order the data frame by potential
        df.sort_values(by=['Potential'], ascending=True, inplace=True)

        # The first (most bound) particles
        most_bound_ids = df.ParticleIDs.iloc[:n_part]

        return most_bound_ids.to_numpy()

    def _find_location_of_dm_ids(self,
                                 snapnum: int,
                                 target_ids: np.ndarray,
                                 ) -> Tuple[int, int]:
        """
        This method finds the most common halo/subhalo index in the current
        snapshot for the particle IDs indicated.

        Parameters
        ----------
        snapnum : int
            The snapshot number to analyze.
        target_ids : np.ndarray
            The array of particle IDs to search for.

        Returns
        -------
        Tuple[int, int]
            A tuple that contains the halo and subhalo indices.

        Raises
        ------
        Exception
            If not all target IDs were found in the particle IDs data for the
            current snapshot.
        """

        settings = Settings()

        if snapnum < settings.first_snap:
            # Return NaNs for snapshots outside of the analysis scope.
            return np.nan, np.nan
        df = _load_dm_snapshot(galaxy=self._galaxy,
                               rerun=self._rerun,
                               resolution=self._resolution,
                               snapnum=snapnum)

        target_idxs = find_indices(a=df.ParticleIDs.to_numpy(),
                                   b=target_ids,
                                   invalid_specifier=-1)

        if target_idxs.min() == -1:
            raise RuntimeError('-1 detected in target indices.')

        target_halo_idxs = df.Halo.iloc[target_idxs].to_numpy()
        target_subhalo_idxs = df.Subhalo.iloc[target_idxs].to_numpy()

        # Remove particles in the inner or outer fuzz.
        is_not_fuzz = (target_halo_idxs != -1) & (target_subhalo_idxs != -1)

        target_halo = mode(target_halo_idxs[is_not_fuzz])[0][0]
        target_subhalo = mode(target_subhalo_idxs[is_not_fuzz])[0][0]

        return target_halo, target_subhalo

    @timer
    def track_galaxy(self,
                     track: bool,
                     n_part: int,
                     halo_feat_name: str = "MainHaloIdx",
                     subhalo_feat_name: str = "MainSubhaloIdx") -> None:
        """
        This method tracks the galaxy through all snapshots.

        Parameters
        ----------
        track : bool
            If True, track galaxy. If False, return the indices 0 and 0 for
            the main halo and subhalo.
        n_part : int
            The number of dark matter particles to find.
        halo_feat_name : str, optional
            The name of the feature of the main halo index in the data frame.
            Defaults to "MainHaloIdx".
        subhalo_feat_name : str, optional
            The name of the feature of the main subhalo index in the data
            frame. Defaults to "MainSubhaloIdx".
        """

        settings = Settings()

        main_halo_idx = 0 * np.ones(self._n_snapshots, dtype=np.int32)
        main_subhalo_idx = 0 * np.ones(self._n_snapshots, dtype=np.int32)

        if track:
            for snapnum in range(self._n_snapshots - 1, -1, -1):
                if (snapnum != self._n_snapshots - 1)\
                   & (snapnum >= settings.first_snap):
                    target_halo = main_halo_idx[snapnum + 1]
                    target_subhalo = main_subhalo_idx[snapnum + 1]
                    target_ids = self._find_most_bound_dm_ids(snapnum + 1,
                                                              n_part,
                                                              target_halo,
                                                              target_subhalo)
                    new_target_halo, new_target_subhalo = \
                        self._find_location_of_dm_ids(snapnum, target_ids)
                    main_halo_idx[snapnum] = new_target_halo
                    main_subhalo_idx[snapnum] = new_target_subhalo

        self._df[halo_feat_name] = main_halo_idx
        self._df[subhalo_feat_name] = main_subhalo_idx

        self._save_data()

    def _save_data(self) -> None:
        """
        This method saves the data to a csv file.
        """

        self._df.set_index(keys="SnapshotNumber")
        self._df.to_csv(f"{self._paths.results}temporal_data.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations",
                        type=str,
                        nargs="+",
                        required=True,
                        help="The simulation to consider.")
    parser.add_argument("--n_part",
                        type=int,
                        required=True,
                        help="The number of DM particles to consider in the "
                             "tracking.")
    parser.add_argument("--track",
                        type=str,
                        choices=["yes", "no"],
                        required=True,
                        help="If True, track galaxy. If False, return the "
                             "indices 0 and 0 for the halo and subhalo.")
    parser.add_argument("--halo_feat_name",
                        type=str,
                        required=False,
                        default="MainHaloIdx",
                        help="The name of the simulation file.")
    parser.add_argument("--subhalo_feat_name",
                        type=str,
                        required=False,
                        default="MainSubhaloIdx",
                        help="The name of the simulation file.")
    args = parser.parse_args()

    for simulation in args.simulations:
        stdout.write(f"Analyzing {simulation.upper()}... ")
        tracker = GalaxyTracker(simulation=simulation)
        tracker.track_galaxy(track=args.track == "yes",
                             n_part=args.n_part,
                             halo_feat_name=args.halo_feat_name,
                             subhalo_feat_name=args.subhalo_feat_name)
        stdout.write(" Done.\n")


if __name__ == "__main__":
    main()
