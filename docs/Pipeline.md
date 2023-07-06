# Main Pipeline Description

## General Remarks

According to module `auriga/settings.py`:

We are currently starting or analysis at snapshot
```[python]
    first_snap: int = 30
```
Note that snapshot 30 can be located at different times depending on the simulation (specifically, depending on the amount of snapshots saved). This, however, is irrelevant for our analysis since snapshots below that number represent very early times, which are out of our scope.

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

## Calculate Basic Simulation Data

