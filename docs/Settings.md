# Description of Settings

The following table provides a description of each variable in the `settings.py` configuration file.

| Name | Unit | Type | Description |
|--------------|:-----:|:-----------:|----|
| `repo_name` | - | `str` | The name of the repository. |
| `first_snap` |  - | `int` | The number of the first snapshot to analyze. This number ensures that structures are detected in the simulations. |
| `galaxies` | - | `list` | The number of all the L4 resolution galaxies. |
| `reruns` | - | `list` | The number of all the L4 resolution galaxies that contain tracer particles. |
| `n_track_dm_parts` | - | `int` | The number of dark matter particles to use as default for the tracking of the main halo/subhalo. |
| `galaxies_to_track` | - | `list` | A list of galaxies to track. |
| `subh_vel_distance` | $\mathrm{ckpc}$ | `float` | The distance inside which stars are considered to calculate the subhalo velocity. |
| `rot_mat_distance` | $\mathrm{ckpc}$ | `float` | The distance inside which stars are considered to calculate the rotation matrix. |
| `box_size` | $\mathrm{ckpc}$ | `float` | The length of the side of the box in the density maps. |
| `n_bins` | - | `int` | The number of bix in each axis of the box in the density maps. |
| `color_maps` | - | `dict` | A dictionary that contains a color map to use for each particle type (gas, dark matter and stars). |
| `groups` | - | `dict` | The group found by Iza et al. (2022) that group galaxies by formation history. |
| `infinity_factor` | - | `float` | The distance (in units of the virial radius) at which to calculate the reference potential. |
| `neighbor_number` | - | `int` | The number of neighbors to use in the calculation of the reference potential. |
| `window_length` | - | `int` | The window length to use when smoothing data with the Savitzky-Golay filter. This should be mostly unused. | 
| `polyord` | - | `int` | The order of the polynomial  to use when smoothing data with the Savitzky-Golay filter. This should be mostly unused. |
| `markers` | - | `list` | A list of markers to designate each galaxy (from 1 to 30). |
| `colors` | - | `list` | A list of colors to designate each galaxy (from 1 to 30). |
| `disc_min_circ` | - | `float` | The minimum circulariy of the (warm) disc. | 
| `disc_std_circ` | - | `float` | The standard circularity of the cold disc. | 
| `cold_disc_delta_circ` | - | `float` | The circularity dispersion of the cold disc. | 
| `bulge_max_specific_energy` | - | `float` | The maximum specific energy (orbital or potential) of the bulge. | 
| `components` | - | `list` | A symbol for each galactic component. | 
| `component_labels` | - | `dict` | The labels of each galactic component. | 
| `component_colors` | - | `dict` | A color for each galactic component. | 
| `component_colormaps` | - | `dict` | A color map for each galactic component. |
| `photometric_bands` | - | `list` | A list of all the photometric bands available in the Auriga simulations. Further details can be found [here](https://www.tng-project.org/data/docs/specifications/#parttype4:~:text=Stellar%20magnitudes%20in,%2C%20section%203.2.1.). |