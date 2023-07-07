# Main Pipeline Description

## General Remarks

According to module `auriga/settings.py`:

We are currently starting or analysis at snapshot
```[python]
    first_snap: int = 30
```
Note that snapshot 30 can be located at different times depending on the simulation (specifically, depending on the amount of snapshots saved). This, however, is irrelevant for our analysis since snapshots below that number represent very early times, which are out of our scope.

## Calculate Basic Simulation Data

Calculte basic properties stored in the snapshots and save them to the `temporal_data.csv` files. The properties are snapshot number, cosmic time, lookback time, redshift and expansion factor.

For example, running
```[python]
    python auriga/simulation_properties.py --simulations au1_or_l4
```
will analyze the original version of Au1 of resolution level 4 and will save the results to `temporal_data.csv`.

## Track Galaxy

The first step in our analysis is tracking the main object through cosmic time.
We asume the main object to be halo 0 and subhalo 0 in the last snapshot of each simulation ($z=0$).

This can be done using the module `auriga/galaxy_tracker.py` and choosing a given number of dark matter particles.

The fiducial configuration, defined in `auriga/settings.py` is as follows:
- The number of dark matter particles to consider when tracking is
```[python]
    n_track_dm_parts: int = 32
```
- The galaxies currently tracked (for the rest, consider halo 0 and subhalo 0 as the main objects) are
```[python]
    galaxies_to_track: list = [1, 11]
```

For example, running
```[python]
    python auriga/galaxy_tracker.py --simulations au1_or_l4 --n_part 32 --track yes --halo_feat_name MainHaloIdx_32part --subhalo_feat_name MainSubhaloIdx_32part
```
will analyze the original version of Au1 of resolution level 4 and will track the (0, 0) object at $z=0$ using 32 dark matter particles and will store the results in `results/au1_or_l4/temporal_data.csv` under the headers `MainHaloIdx_32part` and `MainSubhaloIdx_32part`.

## Calculate Galaxy Properties

Once the indices of the main halo and subhalo have been calculated, we calculate the properties of the galaxy (virial radius and mass).

For example, running
```[python]
    python auriga/galaxy_properties.py --simulations au1_or_l4
```
will calculate the virial mass and virial radius of the original version of Au1 of resolution level 4 and will save the results to `temporal_data.csv`.


## Calculate the Velocity of the Main Subhalo

With the information of the indices of the main halo and subhalo, we calculate the velocity of the main object using stars inside
```[python]
    subh_vel_distance: float = 10.0  # ckpc
```

Running
```[python]
    python auriga/subhalo_velocity.py --simulations au1_or_l4
```
will calculate the velocity of the main subhalo and save the results to `subhalo_vels.csv`.

## Calculate Rotation Matrices

In order to orient the galaxy in space and to keep the galactic disc in the $xy$-plane, we calculate a rotation matrix based on the eigenvectors of the inertia tensor of the stars inside
```[python]
    rot_mat_distance: float = 10.0  # ckpc
```

Running
```[python]
    python auriga/rotation_matrix.py --simulations au1_or_l4
```
will calculate the rotation matrix of Au1 in its original version and resolutio level 4 and save the results to `rotation_matrices.csv`.

## Density Maps

Projected mass density maps can be created using the script `auriga/density_maps.py`. Configuration variables (box size, bin number, and color maps) can be found in `auriga/setting.py`. To create density maps for *all* snapshots and merge them into a PDF, run, for example:
```[python]
    python auriga/rotation_matrix.py --simulations au1_or_l4
```

## Reference Potential

Since the simulations consist of a periodical box, the gravitational potential is not well defined. Therefore, we calculate a reference potential to use as a standard value in order to be able to compare different simulations.

For this calculation, we use the dark matter particles at a given distance from the center of the main halo. The default values are set in `auriga/settings.py`: they are the number of dark matter particles to consider (`neighbor_number`) and the distance from the center of the main halo in units of the virial radius (`infinity_factor`).

To calculate the reference potential with the standard values (for the original run of Au1 at resolution L4, for example), run
```[python]
    python auriga/reference_potential.py --simulation au1_or_l4
```
To print a help docstring, use `python auriga/reference_potential.py -h`.

## Galactic Decomposition

## 
