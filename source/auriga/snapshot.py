import pandas as pd
import numpy as np
import warnings
from loadmodules import gadget_readsnap, load_subfind
from auriga.cosmology import Cosmology
from auriga.simulation import Simulation
from auriga.physics import Physics
from utils.paths import Paths


class Snapshot:
    """
    A class to manage both the global properties of snapshots and particle
    data.

    Attributes
    ----------
    galaxy : int
        The galaxy to analyze.
    df : pd.DataFrame
        A pandas data frame that contains the data of the particles
        in this snapshot.
    rerun : bool
        A bool to indicate if this is a original run or a rerun.
    resolution : int
        The resolution level of the simulation.
    snapnum : int
        The snapshot number.
    rotation_matrix : np.array
        The rotation matrix of this snapshot.
    subhalo_vel : np.array
        The velocity of the main subhalo of this snapshot.
    redshift : float
        The redshift of this snapshot.
    expansion_factor : float
        The expansion factor of this snapshot.
    time : float
        The age of the universe of this snapshot.
    _paths : Paths
        An instance of the Paths class.

    Methods
    -------
    keep_only_halo(int, int)
        Remove all particles that do not belong to the indicated halo/subhalo
        pair.
    drop_types(list[int])
        Remove all particles of the indicated types.
    drop_winds()
        Remove all stellar particles with negative formation time.
    calc_extra_coordinates()
        Calculate the cylindrical and spherical radius for each particle.
    calc_birth_snapnum()
        Calculate the snapshot number in which every star appears for the
        first time.
    calc_circularity()
        This method calculates the circularity parameter for the stellar
        particles in the main halo/subhalo.
    keep_only_feats(feats)
        This method removes all feats from the data frame, keeping only those
        specified in the argument.
    """

    def __init__(self, galaxy: int, rerun: bool, resolution: int,
                 snapnum: int) -> None:
        """
        Parameters
        ----------

        galaxy : int
            The number of the halo.
        rerun : bool
            A bool to indicate if this is a original run or a rerun.
        resolution : int
            The resolution level of the simulation.
        snapnum : int
            The snapshot number.
        """

        self.df = pd.DataFrame()
        self.galaxy = galaxy
        self.rerun = rerun
        self.resolution = resolution
        self._paths = Paths(self.galaxy, self.rerun, self.resolution)
        self.snapnum = snapnum

        sf = gadget_readsnap(snapshot=self.snapnum,
                             snappath=self._paths.snapshots,
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False)
        sb = load_subfind(id=self.snapnum,
                          dir=self._paths.snapshots,
                          cosmological=False)
        sf.calc_sf_indizes(sf=sb)

        self.rotation_matrix = np.loadtxt(
            f'{self._paths.data}'
            + 'rotation_matrices.csv')[self.snapnum].reshape((3, 3))

        self.subhalo_vel = np.loadtxt(
            f'{self._paths.data}subhalo_vels.csv')[self.snapnum]

        pos = (self.rotation_matrix @ (sf.pos - sb.data['spos'][0]
                                       / sf.hubbleparam).T).T * 1E3
        vel = (self.rotation_matrix @ (sf.vel * np.sqrt(sf.time)
                                       - self.subhalo_vel).T).T

        # Snapshot properties.
        self.redshift = sf.redshift
        self.expansion_factor = sf.time
        cosmology = Cosmology()
        self.time = cosmology.redshift_to_time(self.redshift)

        # General particle properties.
        self.df['xCoordinates'] = pos[:, 0]  # ckpc
        self.df['yCoordinates'] = pos[:, 1]  # ckpc
        self.df['zCoordinates'] = pos[:, 2]  # ckpc
        self.df['xVelocities'] = vel[:, 0]  # km/s
        self.df['yVelocities'] = vel[:, 1]  # km/s
        self.df['zVelocities'] = vel[:, 2]  # km/s
        self.df['PartTypes'] = sf.type
        self.df['ParticleIDs'] = sf.id
        self.df['Potential'] = sf.pot / self.expansion_factor  # (km/s)^2
        self.df['Masses'] = sf.mass * 1E10  # Msun
        self.df['Halo'] = sf.halo
        self.df['Subhalo'] = sf.subhalo

        # Stellar formation time.
        formation_time = np.nan * np.ones(sf.mass.shape[0])
        formation_time[sf.type == 4] = sf.age
        self.df['StellarFormationTime'] = formation_time

    def keep_only_feats(self, feats: list) -> None:
        """
        This method removes all feats from the data frame, keeping only those
        specified in the argument.

        Parameters
        ----------
        feats : list
            A list of features to keep in the data frame.
        """

        drop_feats = self.df.columns.to_list()
        for feat in feats:
            drop_feats.remove(feat)
        self.df.drop(columns=drop_feats, inplace=True)

    def calc_circularity(self) -> None:
        """
        This method calculates the circularity parameter for the stellar
        particles in the main halo/subhalo.
        """

        physics = Physics()

        # Check if all particles are loaded in the data frame.
        for include_type in [0, 1, 2, 3, 4, 5]:
            if include_type not in self.df.PartTypes:
                raise ValueError(f'Type {include_type} particles/cells not'
                                 f'loaded.')

        # Check if spherical coordinates were calculated.
        if 'rCoordinates' not in self.df.columns.values:
            self.calc_extra_coordinates()

        pos = self.df[['xCoordinates', 'yCoordinates',
                       'zCoordinates']].to_numpy()
        vel = self.df[['xVelocity', 'yVelocity', 'zVelocity']].to_numpy()
        jz = np.cross(pos, vel)[:, 2] * self.expansion_factor  # kpc km/s
        del pos, vel

        is_galaxy = (self.df.Halo == 0) & (self.df.Subhalo == 0)
        is_star = (self.df.PartTypes == 4) & (self.df.StellarFormationTime > 0)

        bins = np.asarray(
            [0] + list(np.sort(self.df.rCoordinates[is_galaxy & is_star])))
        ii = self.df.rCoordinates[is_galaxy & is_star].argsort().argsort()
        enclosed_mass = np.add.accumulate(
            np.histogram(
                self.df.rCoordinates,
                weights=self.df.Masses, bins=bins)[0])[ii]
        del bins, ii

        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero.
            warnings.simplefilter('ignore', RuntimeWarning)
            jc = self.df.rCoordinates[is_galaxy & is_star] \
                * self.expansion_factor \
                * np.sqrt(physics.gravitational_constant * enclosed_mass
                          / (1E3 * self.df.rCoordinates[is_galaxy & is_star]
                             * self.expansion_factor))  # kpc km/s
        del enclosed_mass

        # Calculate circularity parameter.
        eps = np.nan * np.ones(self.df.Masses.shape[0])
        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero.
            warnings.simplefilter('ignore', RuntimeWarning)
            eps[is_galaxy & is_star] = jz[is_galaxy & is_star] / jc
        del jz, jc, is_galaxy, is_star
        self.df['Circularity'] = eps

    def keep_only_halo(self, halo: int, subhalo: int) -> None:
        """
        This method removes all particles in the data frame that do not belong
        to the chose halo/subhalo index.

        Args:
            halo: The halo to keep.
            subhalo: The subhalo to keep.
        """
        self.df = self.df[(self.df.Halo == halo)
                          & (self.df.Subhalo == subhalo)]

    def drop_types(self, particle_types: list) -> None:
        """
        This method removes all particles that match the selected types.

        Args:
            particle_types: A list of particle types to remove from the
             data frame.
        """
        for particle_type in particle_types:
            self.df = self.df[self.df.PartTypes != particle_type]

    def drop_winds(self) -> None:
        """
        This method removes all wind particles (with StellarFromationTime
        below or equal to zero) from the data frame.
        """
        self.df = self.df[~(self.df.StellarFormationTime <= 0)]

    def calc_extra_coordinates(self) -> None:
        """
        This method calculates the radii of particles in cylindrical and
        spherical coordinates.
        """
        pos = self.df[['xCoordinates', 'yCoordinates',
                       'zCoordinates']].to_numpy()
        self.df['rxyCoordinates'] = np.linalg.norm(pos[:, 0:2], axis=1)
        self.df['rCoordinates'] = np.linalg.norm(pos, axis=1)

    def calc_birth_snapnum(self) -> None:
        """
        This method calculates the snapshot number in which the stars appear
        for the first time (StellarFormationSnapshotNumber). All other particle
        types (including winds) get a NaN.
        """
        simulation = Simulation(self.rerun)
        stellar_birth_snapnum = np.nan * np.ones(self.df.Masses.shape[0])
        exp_fact_diff = self.df.StellarFormationTime
        for i, exp_fact in enumerate(simulation.expansion_factors):
            new_exp_fact_diff = self.df.StellarFormationTime - exp_fact
            stellar_birth_snapnum[(new_exp_fact_diff < 0)
                                  & (exp_fact_diff > 0)] = i
            exp_fact_diff = new_exp_fact_diff
        self.df['StellarFormationSnapshotNumber'] = stellar_birth_snapnum
