<div align="center">
    <h1>The Distribution and Origin of Metals in the Auriga Simulations</h1>
</div>

<p align="center">
    <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a>
</p>

Code written for an ongoing analysis of the distribution of metals in the galaxies
from [the Auriga project](https://wwwmpa.mpa-garching.mpg.de/auriga/index.html).

## Description

This repository is organized as follows:

- The `auriga/` directory contains all the code related to reading and analyzing the results of the simulations. The main script is `auriga/snapshot.py`, that has the principal methods used in the calculation of properties of interest.
- The `docs/` directory contains all the documentation of the project.
    - The `Data.md` file has a description of the files in the `results/` directory.
    - The `Pipeline.md` file describes the order in which we perform the analysis of the simulations.
    - The `Settings.md` file describes the variable of the class `Settings` defined in `auriga/settings.py`. This file acts as the configuration file and contains all the standard values for quantities of interest in out analysis.
    - The `Analysis.md` contains a paper-like description of the analysis and results.
- The `images/` directory contains important plots. It contains one folder for each simulation, named as explained below. Plots with more than one simulation are store in the root directory.
- The `results/` directory contains files (mostly `csv` files) with results. It contains one folder for each simulation, named as explained below.
- The `scripts/` directory contains notebooks and other scripts.
- The `tests/` directory contains test written for different sections of the code.

## Naming Convention

Throughout this repostiry, we name complete simulations as `AUXX_YY_LZ`, where `XX` is the number of simulation, `YY` indicates if the simulation is the original (`OR`) or the rerun version (`RE`) that includes tracer particles, and `Z` is the resolution number (typically 4).

So, for example, the original version of galaxy Au17 in the resolution 4 level is labelled as `AU17_OR_L4`.

For individual snapshots, we extend the previous convention to `AUXX_YY_LZ_SNNN`, where `NNN` is the snapshot number. For example, snapshot 112 of galaxy Au5 in the rerun version and resolution level 4 is labelled as `AU5_RE_L4_S112`.