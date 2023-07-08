class Settings:
    """
    A class to manage several configuration variables.
    """

    def __init__(self) -> None:
        # General configuration variables
        self.repo_name: str = "auriga-metals"
        self.first_snap: int = 30
        self.galaxies: list = list(range(1, 31))
        self.reruns: list = [5, 6, 9, 13, 17, 23, 24, 26, 28]

        # Galaxy tracker configuration
        self.n_track_dm_parts: int = 32
        self.galaxies_to_track: list = [1, 11]

        # Subhalo velocity calculation configuration
        self.subh_vel_distance: float = 10.0  # ckpc

        # Rotation matrix calculation configuration
        self.rot_mat_distance: float = 10.0  # ckpc

        # Configuration for the density maps
        self.box_size: float = 100.0  # ckpc
        self.n_bins: int = 200
        self.color_maps: dict = {0: 'plasma',
                                 1: 'inferno',
                                 4: 'viridis'}

        # Milky Way-like groups
        self.groups: dict = {"MilkyWayLike": [2, 3, 6, 8, 9, 10, 11, 13, 14,
                                              16, 17, 18, 21, 22, 23, 24, 25,
                                              26, 27],
                             "NotMilkyWayLike": [4, 7, 12, 15, 20],
                             "Excluded": [1, 5, 19, 28, 29, 30]}

        # Parameters to calculate the reference potential
        self.infinity_factor: float = 3.0
        self.neighbor_number: int = 16

        # Smoothing configuration
        self.window_length: int = 9
        self.polyorder: int = 1

        # Galaxy indicators
        self.markers: list = ["o", "^", "d", "s", "p", "v"] * 5
        self.colors: list = ["tab:blue", "tab:red", "tab:orange",
                             "tab:green", "tab:purple"] * 6

        # Parameters for the galactic decomposition
        self.disc_min_circ: float = 0.4
        self.disc_std_circ: float = 1.0
        self.cold_disc_delta_circ: float = 0.25
        self.bulge_max_specific_energy: float = -0.75

        # Galactic decomposition
        self.components: list = ["H", "B", "CD", "WD"]
        self.component_labels: dict = {"H": "Halo",
                                       "B": "Bulge",
                                       "CD": "Cold Disc",
                                       "WD": "Warm Disc"}
        self.component_colors: dict = {"H": "tab:blue",
                                       "B": "tab:green",
                                       "CD": "tab:red",
                                       "WD": "tab:orange"}
        self.component_colormaps: dict = {"H": "Blues",
                                          "B": "Greens",
                                          "CD": "Reds",
                                          "WD": "Oranges"}
