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

        # Galaxy markers
        self.markers: list = ["o", "^", "d", "s", "p", "v"] * 5

        # Galaxy colors
        self.colors: list = ["tab:blue", "tab:red", "tab:orange",
                             "tab:green", "tab:purple"] * 6

        # Rotation matrix calculation configuration
        self.rot_mat_distance = 10  # ckpc

        # Galaxies to track using the DM particles
        self.galaxies_to_track: list = [1, 11]

        # Configuration for the density maps
        self.box_size: float = 100.0  # ckpc
        self.n_bins: int = 200
        self.color_maps: dict = {0: 'Blues',
                                 1: 'Greens',
                                 4: 'Reds'}

        # Parameters for the galactic decomposition
        self.disc_min_circ: float = 0.4
        self.cold_disc_delta_circ: float = 0.25
        self.bulge_max_specific_energy: float = -0.75

        # Parameters to calculate the reference potential
        self.infinity_factor: int = 3
        self.neighbor_number: int = 64

        # Milky Way-like groups
        self.groups: dict = {"MilkyWayLike": [2, 3, 6, 8, 9, 10, 11, 13, 14,
                                              16, 17, 18, 21, 22, 23, 24, 25,
                                              26, 27],
                             "NotMilkyWayLike": [4, 7, 12, 15, 20],
                             "Excluded": [1, 5, 19, 28, 29, 30]}

        # Figure settings
        self.big_fig_size: tuple = (7.2, 7.2)
        self.big_fig_ncols: int = 5
        self.big_fig_nrows: int = 6
        self.big_fig_hspace: float = 0.0
        self.big_fig_wspace: float = 0.0
        self.big_fig_grid_lw: float = 0.25
        self.big_fig_tick_labelsize: float = 5.0
        self.big_fig_axis_fontsize: float = 6.0
        self.big_fig_legend_fontsize: float = 4.0
        self.big_fig_standard_lw: float = 1.0
        self.big_fig_label_fontsize: float = 5.0
        self.big_fig_line_lw: float = 0.5
        self.big_fig_marker_size: float = 2.5
        self.big_fig_marker_ew: float = 0.5
        self.big_fig_annotations_fontsize: float = 4.8

        self.medium_fig_size: tuple = (6.0, 6.0)
        self.medium_fig_ncols: int = 3
        self.medium_fig_nrows: int = 3
        self.medium_fig_hspace: float = 0.0
        self.medium_fig_wspace: float = 0.0
        self.medium_fig_legend_fontsize: float = 5.0
        self.medium_fig_grid_lw: float = 0.25
        self.medium_fig_standard_lw: float = 1.0
        self.medium_fig_marker_size: float = 3.0
        self.medium_fig_marker_ew: float = 0.5
        self.medium_fig_annotations_fontsize: float = 6.0
        self.medium_fig_label_fontsize: float = 6.0
        self.medium_fig_axis_fontsize: float = 8.0
        self.medium_fig_capsize: float = 2.0
        self.medium_fig_capthick: float = 1.0
        self.medium_fig_elinewidth: float = 0.5
        self.medium_fig_reference_lw: float = 0.5

        self.small_fig_size: tuple = (7.2, 1.5)
        self.small_fig_ncols: int = 5
        self.small_fig_nrows: int = 1
        self.small_fig_grid_lw: float = 0.25
        self.small_fig_line_lw: float = 1.0
        self.small_fig_marker_size: float = 3.0
        self.small_fig_marker_ew: float = 0.5
        self.small_fig_label_fontsize: float = 6.0
        self.small_fig_hspace: float = 0.0
        self.small_fig_wspace: float = 0.0

        self.individual_fig_size: tuple = (3.0, 3.0)
        self.individual_fig_grid_lw: float = 0.25
        self.individual_fig_ms: float = 3.0
        self.individual_fig_elinewidth: float = 0.5
        self.individual_fig_capsize: float = 2.0
        self.individual_fig_capthick: float = 1.0
        self.individual_fig_marker_ew: float = 0.5
        self.individual_fig_marker_size: float = 3.0
        self.individual_fig_standard_lw: float = 1.0
        self.individual_fig_annotation_lw: float = 0.5
        self.individual_fig_legend_fontsize: float = 5.0
        self.individual_fig_scatter_maker_size: float = 35.0

        self.long_horizontal_fig_size: tuple = (6.0, 1.5)
        self.long_horizontal_fig_bar_width: float = 0.5
        self.long_horizontal_fig_marker_size: float = 25.0
        self.long_horizontal_fig_marker_ew: float = 0.5

        self.figure_extensions: list = ["pdf", "png"]
