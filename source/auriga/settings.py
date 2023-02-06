class Settings:
    """
    A class to manage several configuration variables.
    """

    def __init__(self) -> None:
        self.repo_name: str = "auriga-metals"
        self.first_snap: int = 30
        self.galaxies: list = list(range(1, 31))
        self.reruns: list = [5, 6, 9, 13, 17, 23, 24, 26, 28]
        self.processes: int = 2

        # Smoothing configuration
        self.window_length: int = 9
        self.polyorder: int = 1

        # Galaxy tracker configuration
        self.n_track_dm_parts: int = 32

        # Subhalo velocity calculation configuration
        self.subh_vel_distance: int = 10  # ckpc

        # Rotation matrix calculation configuration.
        self.rot_mat_distance = 10  # ckpc

        # Configuration for the density maps
        self.box_size: float = 100.0  # ckpc
        self.n_bins: int = 200
        self.color_maps: dict = {0: 'Blues',
                                 1: 'Greens',
                                 4: 'Reds'}

        # Parameters for the galactic decomposition.
        self.disc_min_circ: float = 0.4
        self.cold_disc_delta_circ: float = 0.25
        self.bulge_max_specific_energy: float = -0.75

        # Parameters to calculate the reference potential.
        self.infinity_factor: int = 3
        self.neighbor_number: int = 64
