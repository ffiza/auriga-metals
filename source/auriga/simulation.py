import numpy as np
import json


class Simulation:
    """
    A class used to manage global simulation properties.

    Attributes
    ----------
    snapshot_numbers : np.array
        An array with the snapshot numbers of each snapshot.
    times : np.array
        An array with the times (age of the universe in Gyr) of each
        snapshot.
    redshifts : np.array
        An array with the redshift of each snapshot.
    expansion_factors : np.array
        An array with the expansion factor of each snapshot.
    """

    def __init__(self, rerun=False):
        """
        Parameters
        ----------
        rerun : bool
            A bool indicating if the simulation is the original run or
            the rerun.
        """

        self.type = 'Rerun' if rerun is True else 'Original'
        with open('data/simulation.json') as f:
            data = json.load(f)

            self.snapshot_numbers = np.array(data[self.type]['SnapshotNumber'])
            self.times = np.array(data[self.type]['Time_Gyr'])
            self.redshifts = np.array(data[self.type]['Redshift'])
            self.expansion_factors = np.array(
                data[self.type]['ExpansionFactor'])
