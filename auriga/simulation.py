import numpy as np
import json


class Simulation:
    """
    A class used to manage global simulation properties.
    """

    def __init__(self, rerun: bool, resolution: int):
        """
        Parameters
        ----------
        rerun : bool
            A bool indicating if the simulation is the original run or
            the rerun.
        resolution : int
            The resolution of the simulation.
        """

        self._rerun = rerun
        self._resolution = resolution

        self._type = 'Rerun' if self._rerun else 'Original'

        with open(f'data/level{self._resolution}/simulation.json') as f:
            data = json.load(f)

            self.snapshot_numbers = np.array(
                data[self._type]['SnapshotNumber'])
            self.times = np.array(
                data[self._type]['Time_Gyr'])
            self.redshifts = np.array(
                data[self._type]['Redshift'])
            self.expansion_factors = np.array(
                data[self._type]['ExpansionFactor'])

        self.n_snapshots = 252 if self._rerun else 128
