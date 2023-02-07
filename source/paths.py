import os


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
        rerun_text = '_rerun' if rerun else ''

        # Path to snapshot files.
        if os.uname()[1] == 'virgo':
            dir_name = 'RerunsHighFreqStellarSnaps' if rerun else 'Original'
            self.snapshots = f'/virgotng/mpa/Auriga/level{resolution}/' + \
                f'{dir_name}/halo_{galaxy}/output/'
        elif os.uname()[1] == 'neuromancer':
            self.snapshots = '/media/federico/Elements1/Simulations/' + \
                f'au{galaxy}{rerun_text}/'

        # Path to data files.
        self.data = f"data/level{resolution}/au{galaxy}{rerun_text}/"

        # Path to image files.
        self.images = f"images/level{resolution}/au{galaxy}{rerun_text}/"
