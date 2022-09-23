from utils import make_paths, find_indices
import pandas as pd
import numpy as np
from loadmodules import gadget, load_subfind
from source.auriga.cosmology import Cosmology
from source.auriga.simulation import Simulation
from source.auriga.settings import Settings


class SnapshotData:
    """
    A class to manage both the global properties of snapshots and particle
    data.

    Attributes
    ----------
    galaxy : int
        The snapshot in which to start the analysis.
    df : pd.DataFrame
        A pandas data frame that contains the data of the particles
        in this snapshot.
    rerun : bool
        A bool to indicate if this is a original run or a rerun.
    resolution : int
        The resolution level of the simulation.
    snapnum : int
        The snapshot number.
    snapshot_path : str
        The snapshot path.
    data_path : str
        The path to the data directory.
    rotation_matrix : np.array
        The rotation matrix of this snapshot.
    subhalo_vel : np.array
        The velocity of the main subhalo.
    redshift : float
        The redshift of this snapshot.
    expansion_factor : float
        The expansion factor of this snapshot.
    time : float
        The age of the universe of this snapshot.
    rd : float
        The radius of the stellar disc at this time.
    hd : float
        The height of the stellar disc at this time.

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
    tag_disc_particles()
        Add a new feature in the data frame with a boolean that indicates if
        the particle is in the stellar disc or not.
    calc_birth_snapnum()
        Calculate the snapshot number in which every star appears for the
        first time.
    tag_ex_situ_stars()
        Add a new feature in the data frame with a boolean that indicates if
        the star particle was outside (True) or inside (False) the disc when
        it first appears in the simulation. Keep in mind that this analysis
        can take some time to run.
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
        self.snapnum = snapnum

        self.snapshot_path, self.data_path, _ = make_paths(self.galaxy,
                                                           self.rerun,
                                                           self.resolution)

        sf = gadget.gadget_readsnap(snapshot=self.snapnum,
                                    snappath=self.snapshot_path,
                                    lazy_load=True,
                                    cosmological=False,
                                    applytransformationfacs=False)
        sb = load_subfind(id=self.snapnum,
                          dir=self.snapshot_path,
                          cosmological=False)
        sf.calc_sf_indizes(sf=sb)

        self.rotation_matrix = np.load(f'{self.data_path}/rmatrix/snap_'
                                       f'{str(int(self.snapnum)).zfill(3)}'
                                       f'.npy')

        self.subhalo_vel = np.loadtxt(
            f'{self.data_path}subhalo_vel.csv')[self.snapnum]

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
        self.df['Potential'] = sf.pot  # TODO: Check units.
        self.df['Masses'] = sf.mass * 1E10  # Msun
        self.df['Halo'] = sf.halo
        self.df['Subhalo'] = sf.subhalo

        # Stellar formation time.
        formation_time = np.nan * np.ones(sf.mass.shape[0])
        formation_time[sf.type == 4] = sf.age
        self.df['StellarFormationTime'] = formation_time

        # Size of the disc.
        self.rd = np.loadtxt(f'{self.data_path}disc_radius.csv')[self.snapnum]
        self.hd = np.loadtxt(f'{self.data_path}disc_height.csv')[self.snapnum]

    def keep_only_halo(self, halo: int, subhalo: int) -> None:
        """
        This method removes all particles in the data frame that do not belong
        to the chose halo/subhalo index.

        Args:
            halo: The halo to keep.
            subhalo: The subhalo to keep.
        """
        self.df = self.df[(self.df.Halo == halo) &
                          (self.df.Subhalo == subhalo)]

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

    def tag_disc_particles(self) -> None:
        """
        This method adds a feature in the data frame that indicates if
        particles belong to the disc or not.
        """
        # Check if cylindrical coordinates are in data frame.
        if 'rxyCoordinates' not in self.df.columns.values:
            self.calc_extra_coordinates()

        self.df['isDisc'] = (self.df.rxyCoordinates <= self.rd) \
            & (np.abs(self.df.zCoordinates) <= self.hd)

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

    def tag_ex_situ_stars(self) -> None:
        """
        This method adds a tag for each star particle in the disc of the
        current snapshot. The tag indicates if the star was found inside the
        disc when it is first detected (False) or not (True).
        """

        # TODO: Add warning because this function takes some time to run.

        settings = Settings()
        if 'isDisc' not in self.df.columns.values:
            self.tag_disc_particles()
        if 'StellarFormationSnapshotNumber' not in self.df.columns.values:
            self.calc_birth_snapnum()
        self.df['isExSituStar'] = np.nan * np.ones(self.df.Masses.shape[0])
        for snapnum in range(settings.first_snapshot, self.snapnum):
            target_s = SnapshotData(self.galaxy, self.rerun, self.resolution,
                                    snapnum)
            target_s.drop_types([0, 1, 2, 3, 5])
            target_s.df.drop(
                columns=['xVelocities', 'yVelocities', 'zVelocities',
                         'Potential'], inplace=True)
            target_s.tag_disc_particles()

            # Select the ParticleIDs in the present of the stars born in this
            # snapshot.
            target_ids = self.df.ParticleIDs[
                self.df.isDisc & (self.df.StellarFormationSnapshotNumber
                                  == snapnum)].to_numpy()

            idx = find_indices(target_s.df.ParticleIDs.to_numpy(),
                               target_ids,
                               invalid_specifier=-1)
            self.df.isExSituStar = target_s.df.isDisc.iloc[idx]
