# Description of Data

## `rotation_matrices.csv`

The rotation matrix of the main subhalo (with indices stored in `temporal_data.csv` as `MainHaloIdx` and `MainSubhaloIdx`). Each row contains the nine components of the matrix for a given snapshot (the row number). To recover the matrix once the data is read, use `reshape((3, 3))`.

## `subhalo_vels.csv`

The velocity of the main subhalo (with indices stored in `temporal_data.csv` as `MainHaloIdx` and `MainSubhaloIdx`). Each row contains the velocity in the $x$-, $y$- and $z$-directions for a given snapshot (the row number).

## `temporal_data.csv`

The following table provides a description of each variable stored in the `temporal_data.csv` files, including header name, unit, mathematical symbol and description.

| Name | Unit | Symbol | Description |
|--------------|:-----:|:-----------:|----|
| `SnapshotNumber` | - | - | The number of the snapshot according to the simulation outputs. |
| `Time_Gyr` |  $\mathrm{Gyr}$ | $t$ | The cosmic time or age of the universe. |
| `LookbackTime_Gyr` | $\mathrm{Gyr}$ | $t_\mathrm{lookback}$ | The lookback time or time before the present.|
| `Redshift` | - | $z$ | The cosmological redshift. |
| `ExpansionFactor` | - | $a$ | The scale factor of the universe. |
| `VirialRadius_ckpc` | $\mathrm{ckpc}$ | $R_{200}$ | The virial radius of the main object (calculated as the radius in which the density is 200 times the critical density of the universe). The index of the main object is stored in variable `MainHaloIdx`. |
| `VirialMass_1E10Msun` | $10^{10} ~ \mathrm{M}_\odot$ | $M_{200}$ | The virial mass of the main object (all the mass inside the virial radius). The index of the main object is stored in variable `MainHaloIdx`. |
| `MainHaloIdx` | - | - | The index of the main halo. |
| `MainSubhaloIdx` | - | - | The index of the main subhalo. |
| `VirialRadius00_ckpc` | $\mathrm{ckpc}$ | $R_{200}$ | The virial radius of halo 0 and subhalo 0 (calculated as the radius in which the density is 200 times the critical density of the universe). |
| `VirialMass00_1E10Msun` | $10^{10} ~ \mathrm{M}_\odot$ | $M_{200}$ | The virial mass  of halo 0 and subhalo 0 (all the mass inside the virial radius). |
| `ReferencePotential_(km/s)^2` | $\left( \mathrm{km}/\mathrm{s} \right)^2$ | $V_\mathrm{ref}$ | The reference gravitational potential calculated using dark matter particles at a given distance of the centre of the main object. |