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
    n_track_dm_parts : int
        The number of dark matter particles to use in the galaxy tracker.
    """

    def __init__(self) -> None:
        self.repo_name = 'auriga-metals'
        self.first_snap = 30
        self.galaxies = list(range(1, 31))
        self.reruns = [5, 6, 9, 13, 17, 23, 24, 26, 28]

        # Smoothing configuration.
        self.window_length = 9
        self.polyorder = 1

        # Galaxy tracker configuration.
        self.n_track_dm_parts = 128

        # Subhalo velocity calculation configuration.
        self.subh_vel_distance = 10  # ckpc

        # Rotation matrix calculation configuration.
        self.rot_mat_distance = 10  # ckpc

        # Configuration for the density maps.
        self.box_size = 100  # ckpc
        self.n_bins = 200
        self.color_maps = {0: 'Blues',
                           1: 'Greens',
                           4: 'Reds'}
