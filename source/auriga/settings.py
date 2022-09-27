class Settings:
    """
    A class to manage several configuration variables.

    Attributes
    ----------
    repo_name : str
        The name of the respository for this project.
    first_snap : int
        The first snapshot to consider in all analysis.
    galaxies : list
        A list of all galaxies.
    reruns : list
        A list of the rerun galaxies.
    window_length : int
        The window length used in the Savitzky-Golay filter for smoothing.
    polyorder : ing
        The order of the polynomial used in the Savitzky-Golay filter.
    """

    def __init__(self) -> None:
        self.repo_name = 'auriga-metals'
        self.first_snap = 15
        self.galaxies = [i for i in range(1, 31)]
        self.reruns = [5, 6, 9, 13, 17, 23, 24, 26, 28]

        # Smoothing configuration.
        self.window_length = 9
        self.polyorder = 1
