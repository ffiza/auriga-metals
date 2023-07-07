<div align="center">
    <h1>The Distribution and Origin of Metals in the Auriga Simulations</h1>
</div>

<p align="center">
    <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a>
</p>

Code written for an ongoing analysis of the distribution of metals in the galaxies
from [the Auriga project](https://wwwmpa.mpa-garching.mpg.de/auriga/index.html).

## Introduction

## Simulations

### The Auriga Simulations

### Tracer Particles

## Analysis

### Tracking the Main Object

We take the main object to be the most massive subhalo (index 0) in the most massive halo (index 0) at $z=0$. In order to keep track of this object, we adopt the following procedure.

At $z=0$ (snapshot 127 of the original simulations), we find the 32 (set in the attribute ```n_track_dm_parts``` of the ```Settings``` class defined in ```source/auriga/settings.py```) most bound dark matter particles in the main object. Then, we load the previous snapshot (126 in the original simulations) and find the index of the halo and subhalo that contains most (using the mathematical mode) of the targeted dark matter particles and consider this to be the main object.

This procedure is then applied to all the previous snapshots. The first snapshot
to analyze (defined in the attribute ```first_snap``` of the ```Settings``` class defined in ```source/auriga/settings.py```) is number 30; for earlier snapshots (not considered in any analysis) the indices 0 and 0 are stored.

The halo/subhalo index pair can be found in the file ```main_object_idxs.csv``` for each galaxy (these can be found in the ```data/``` directory).

### Centering and Rotating the Galaxies

*Te be completed.*

### The Circularity Parameter

*Te be completed.*

### The Gravitational Potential

*Te be completed.*

### Galactic Decomposition

## The Distribution of Metals

## The Origin of Metals

## Discussion and Conclusions

<!-- ## Apenddices -->

<!-- ### Appendix A: Some Title -->