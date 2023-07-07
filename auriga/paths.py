import os
import pathlib


class Paths:
    """
    A class to manage paths.

    Attributes
    ----------
    snapshot : str
        The path to the snapshot files.
    data : str
        The path to the data files.
    images : str
        The path to the image files.
    """
    def __init__(self, galaxy: int, rerun: bool, resolution: int):
        """
        Parameters
        ----------
        galaxy : int
            The number of the halo.
        rerun : bool
            A bool to indicate if this is a original run or a rerun.
        resolution : int
            The resolution level of the simulation.
        """
        rerun_text = "re" if rerun else "or"

        # Path to snapshots
        dir_name = 'RerunsHighFreqStellarSnaps' if rerun else 'Original'
        self.snapshots = f'/virgotng/mpa/Auriga/level{resolution}/' + \
            f'{dir_name}/halo_{galaxy}/output/'

        # Path to results directory
        self.results = f"results/au{galaxy}_{rerun_text}_l{resolution}/"

        # Path to image directory
        self.images = f"images/au{galaxy}_{rerun_text}_l{resolution}/"

        # Absolute path to the results directory
        self.results_abs = pathlib.Path(
            os.path.dirname(
                os.path.abspath(
                    __file__))).parent.absolute().joinpath(self.results)
