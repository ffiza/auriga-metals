import pandas as pd
import numpy as np
import warnings
import time
from loadmodules import gadget_readsnap, load_subfind

from cosmology import Cosmology
from simulation import Simulation
from physics import Physics
from paths import Paths
from settings import Settings


class Snapshot:
    """
    A class to manage both the global properties of snapshots and particle
    data.
    """

    def __init__(self, galaxy: int, rerun: bool,
                 resolution: int, snapnum: int,
                 loadonlytype: list = [0, 1, 2, 3, 4, 5]) -> None:
        """
        The class constructor.

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
        loadonlytype : list, optional
            A list with the particle types to load.
        """

        self.df = pd.DataFrame()
        self.galaxy = galaxy
        self.rerun = rerun
        self.resolution = resolution
        self._paths = Paths(self.galaxy, self.rerun, self.resolution)
        self.snapnum = snapnum
        self.units = {}

        sf = gadget_readsnap(snapshot=self.snapnum,
                             snappath=self._paths.snapshots,
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False,
                             loadonlytype=loadonlytype)
        sb = load_subfind(id=self.snapnum,
                          dir=self._paths.snapshots,
                          cosmological=False)
        sf.calc_sf_indizes(sf=sb)

        self.rotation_matrix = np.loadtxt(
            f"{self._paths.data}" + "rotation_matrices.csv")[
                self.snapnum].reshape((3, 3))

        self.subhalo_vel = np.loadtxt(
            f"{self._paths.data}subhalo_vels.csv")[self.snapnum]

        # Set halo/subhalo indices.
        df = pd.read_csv(f"{self._paths.data}temporal_data.csv",
                         usecols=["MainHaloIdx", "MainSubhaloIdx"])
        self._halo_idx = df["MainHaloIdx"].to_numpy()[self.snapnum]
        self._subhalo_idx = df["MainSubhaloIdx"].to_numpy()[self.snapnum]

        subhalo_grouptab_idx = sb.data["ffsh"][self._halo_idx] \
            + self._subhalo_idx

        pos = (
            self.rotation_matrix @ (sf.pos
                                    - sb.data["spos"][subhalo_grouptab_idx]
                                    / sf.hubbleparam).T).T * 1e3
        vel = (self.rotation_matrix @ (sf.vel * np.sqrt(sf.time)
               - self.subhalo_vel).T).T

        # Snapshot properties
        self.redshift = sf.redshift
        self.expansion_factor = sf.time
        cosmology = Cosmology()
        self.time = cosmology.redshift_to_time(self.redshift)

        # General particle properties
        self.df["xCoordinates"] = pos[:, 0]
        self.df["yCoordinates"] = pos[:, 1]
        self.df["zCoordinates"] = pos[:, 2]
        self.df["xVelocities"] = vel[:, 0]
        self.df["yVelocities"] = vel[:, 1]
        self.df["zVelocities"] = vel[:, 2]
        self.df["ParticleIDs"] = pd.Series(sf.id, dtype="uint64")
        self.df["Potential"] = sf.pot / self.expansion_factor
        self._add_reference_to_potential()
        self.df["Halo"] = sf.halo
        self.df["Subhalo"] = sf.subhalo

        self.units["xCoordinates"] = "ckpc"
        self.units["yCoordinates"] = "ckpc"
        self.units["zCoordinates"] = "ckpc"
        self.units["xVelocities"] = "km/s"
        self.units["yVelocities"] = "km/s"
        self.units["zVelocities"] = "km/s"
        self.units["Potential"] = "(km/s)^2"

        del pos

        # Specific orbital energy
        self.df["SpecificOrbitalEnergy"] = 0.5 \
            * np.linalg.norm(vel, axis=1)**2 + self.df["Potential"].to_numpy()
        self.units["SpecificOrbitalEnergy"] = "(km/s)^2"
        
        del vel

        if len(loadonlytype) > 1:
            self.df["PartTypes"] = pd.Series(sf.type, dtype='uint8')
        
        # Calculate stellar formation time if stars were loaded
        if 4 in loadonlytype:
            formation_time = np.nan * np.ones(sf.type.shape[0])
            formation_time[sf.type == 4] = sf.age
            self.df["StellarFormationTime"] = formation_time
            self.units["StellarFormationTime"] = "a"

        # Mass of particles in Msun
        if len(loadonlytype) == 1 and loadonlytype[0] == 1:
            self.df["Masses"] = sf.masses[1] * np.ones(sf.type.shape[0]) * 1e10
        else:
            self.df["Masses"] = sf.mass * 1e10
        self.units["Masses"] = "Msun"
    
        # Add metals to data frame.
        self._add_metals_to_dataframe(sf)

    def calculate_stellar_age(self) -> None:
        """
        This method calculates the stellar age based on the stellar formation
        time.
        """
      
        cosmology = Cosmology()

        stellar_formation_times = cosmology.expansion_factor_to_time(
            self.df["StellarFormationTime"])
        self.df["StellarAge"] = cosmology.present_time \
            - stellar_formation_times


    def _add_metals_to_dataframe(self, sf) -> None:
        """
        This method adds a feat for the metal fraction of each species to the
        data frame.

        Parameters
        ----------
        sf : Snapshot
            The loaded snapshot file.
        """

        is_baryon = (sf.type == 0) | (sf.type == 4)

        metals = np.nan * np.ones((len(sf.type), sf.gmet.shape[1]))
        metals[is_baryon] = sf.gmet

        physics = Physics()

        for idx, metal in enumerate(physics.metals):
            self.df[f"{metal}Fraction"] = metals[:, idx]

    def _add_reference_to_potential(self) -> None:
        """
        Change the raw potential to a potential with a reference, calculated
        somewhere else.
        """

        df = pd.read_csv(f"{self._paths.data}temporal_data.csv",
                         index_col="SnapshotNumber")
        ref_pot = df["ReferencePotential_(km/s)^2"].loc[self.snapnum]
        del df
        self.df["Potential"] = self.df["Potential"] - ref_pot  # (km/s)^2

    def calc_normalized_potential(self) -> None:
        """
        Normalize the potential to the range [-1, 0).
        """

        if 4 not in self.df["PartTypes"].values:
            raise ValueError("Stars not found in the snapshot.")

        is_star = self.df["PartTypes"] == 4 
        is_wind = self.df["StellarFormationTime"] <= 0

        max_potential = np.abs(
            self.df.loc[is_star & ~is_wind, "Potential"]).max()

        self.df["NormalizedPotential"] = self.df["Potential"] / max_potential
        self.units["NormalizedPotential"] = "1"

    def calc_normalized_orbital_energy(self) -> None:
        """
        Normalize the specific orbital energy to the range [-1, 0).
        """

        if 4 not in self.df["PartTypes"].values:
            raise ValueError("Stars not found in the snapshot.")

        is_star = self.df["PartTypes"] == 4
        is_wind = self.df["StellarFormationTime"] <= 0
            
        max_energy = np.abs(
            self.df.loc[is_star & ~is_wind, "SpecificOrbitalEnergy"]).max()

        col_name = "NormalizedSpecificOrbitalEnergy"
        self.df[col_name] = np.nan
        self.df.loc[is_star, col_name] = \
            self.df.loc[is_star, "SpecificOrbitalEnergy"] / max_energy
        self.units[col_name] = "1"

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

    def drop_feats(self, feats: list) -> None:
        """
        This method removes the selected feats from the data frame.

        Parameters
        ----------
        feats : list
            A list of feats to remove.
        """

        self.df.drop(columns=feats, inplace=True)

    def calc_circularity(self) -> None:
        """
        This method calculates the circularity parameter for the stellar
        particles in the main halo/subhalo.
        """

        physics = Physics()

        # Check if all particles are loaded in the data frame.
        for include_type in [0, 1, 2, 3, 4, 5]:
            if include_type not in self.df.PartTypes:
                raise ValueError(
                    f"Type {include_type} particles/cells not" f"loaded.")

        # Check if spherical coordinates were calculated.
        if "rCoordinates" not in self.df.columns.values:
            self.calc_extra_coordinates()

        pos = self.df[["xCoordinates", "yCoordinates",
                       "zCoordinates"]].to_numpy()
        vel = self.df[["xVelocities", "yVelocities", "zVelocities"]].to_numpy()
        jz = np.cross(pos, vel)[:, 2] * self.expansion_factor  # kpc km/s
        del pos, vel

        is_galaxy = (self.df.Halo == self._halo_idx) & (
            self.df.Subhalo == self._subhalo_idx
        )
        is_star = (self.df.PartTypes == 4) & (self.df.StellarFormationTime > 0)

        bins = np.asarray(
            [0] + list(np.sort(self.df.rCoordinates[is_galaxy & is_star]))
        )
        ii = self.df.rCoordinates[is_galaxy & is_star].argsort().argsort()
        enclosed_mass = np.add.accumulate(
            np.histogram(self.df.rCoordinates, weights=self.df.Masses,
                         bins=bins)[0]
        )[ii]
        del bins, ii

        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero.
            warnings.simplefilter("ignore", RuntimeWarning)
            jc = (
                self.df.rCoordinates[is_galaxy & is_star]
                * self.expansion_factor
                * np.sqrt(
                    physics.gravitational_constant
                    * enclosed_mass
                    / (
                        1e3
                        * self.df.rCoordinates[is_galaxy & is_star]
                        * self.expansion_factor
                    )
                )
            )  # kpc km/s
        del enclosed_mass

        # Calculate circularity parameter.
        eps = np.nan * np.ones(self.df.Masses.shape[0])
        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero.
            warnings.simplefilter("ignore", RuntimeWarning)
            eps[is_galaxy & is_star] = jz[is_galaxy & is_star] / jc
        del jz, jc, is_galaxy, is_star
        
        self.df["Circularity"] = eps
        self.units["Circularity"] = 1

    def keep_only_halo(self, halo: int = None, subhalo: int = None) -> None:
        """
        This method removes all particles in the data frame that do not belong
        to the chose halo/subhalo index.

        Parameters
        ----------
        halo : int, optional
            The index of the halo to keep. If not supplied, it defaults to
            the main object halo index.
        subhalo : int, optional
            The index of the subhalo subhalo to keep. If not supplied, it
            defaults to the main object subhalo index.
        """

        if halo is None:
            halo = self._halo_idx
        if subhalo is None:
            subhalo = self._subhalo_idx

        self.df = self.df[(self.df.Halo == halo)
                          & (self.df.Subhalo == subhalo)]

        self.df.reset_index(inplace=True, drop=True)

    def drop_types(self, particle_types: list) -> None:
        """
        This method removes all particles that match the selected types.

        Parameters
        ----------
        particle_types : list
            A list of particle types to remove from the data frame.
        """

        for particle_type in particle_types:
            self.df = self.df[self.df.PartTypes != particle_type]

        self.df.reset_index(inplace=True, drop=True)

    def drop_winds(self) -> None:
        """
        This method removes all wind particles (with StellarFromationTime
        below or equal to zero) from the data frame.
        """

        self.df = self.df[~(self.df.StellarFormationTime <= 0)]

        self.df.reset_index(inplace=True, drop=True)

    def calc_extra_coordinates(self) -> None:
        """
        This method calculates the radii of particles in cylindrical and
        spherical coordinates.
        """

        pos = self.df[["xCoordinates", "yCoordinates",
                       "zCoordinates"]].to_numpy()
        self.df["rxyCoordinates"] = np.linalg.norm(pos[:, 0:2], axis=1)
        self.df["rCoordinates"] = np.linalg.norm(pos, axis=1)

        self.units["rxyCoordinates"] = "ckpc"
        self.units["yCoordinates"] = "ckpc"

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
        self.df["StellarFormationSnapshotNumber"] = stellar_birth_snapnum
    
    def tag_particles_by_region(self) -> None:
        """
        Calculate the region to which each particle belongs based on their
        circularity and normalized potential. The regions can be Cold Disc,
        Warm Disc, Halo and Bulge. All particles belong to a region. If the
        properties used are not found in the snapshot, it calls the methods
        to calculate them.
        """

        if "Circularity" not in self.df.columns.values:
            self.calc_circularity()

        if "NormalizedPotential" not in self.df.columns.values:
            self.calc_normalized_potential()

        settings = Settings()

        region_tag = ["Halo"] * self.df["Masses"].shape[0]
        region_tag = np.array(region_tag, dtype="<U8")

        circ = self.df["Circularity"].to_numpy()
        pot = self.df["NormalizedPotential"].to_numpy()

        region_tag[
            (np.abs(circ - settings.disc_std_circ) \
                <= settings.cold_disc_delta_circ)] = "ColdDisc"

        region_tag[
            (np.abs(circ - settings.disc_std_circ) \
                > settings.cold_disc_delta_circ) \
            & (circ >= settings.disc_min_circ)] = "WarmDisc"
    
        region_tag[(pot <= settings.bulge_max_specific_energy) \
                   & (circ < settings.disc_min_circ)] = "Bulge"
    
        region_tag[
            (pot > settings.bulge_max_specific_energy) \
            & (circ < settings.disc_min_circ)] = "Halo"
        
        self.df["RegionTag"] = pd.Categorical(region_tag)

    def calc_metal_abundance(self, of: str, to: str) -> None:
        """
        Calculates the abundance [X/Y] of metal x relative to metal Y.

        Parameters
        ----------
        of : str
            The metal to calculate the abundance of.
        to : str
            The metal to use as relative to.
        """

        if (4 not in self.df["PartTypes"].values) \
            and (0 not in self.df["PartTypes"].values):
            raise ValueError("No gas nor star cells found.")
        
        if f"{of}Fraction" not in self.df.columns.values:
            raise ValueError(f"{of}Fraction not found in data frame.")
        if f"{to}Fraction" not in self.df.columns.values:
            raise ValueError(f"{to}Fraction not found in data frame.")

        physics = Physics()

        col_name = f"{of}{to}_Abundance"

        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero
            warnings.simplefilter("ignore", RuntimeWarning)

            self.df[col_name] = np.log10(
                self.df[f"{of}Fraction"] / self.df[f"{to}Fraction"])
            self.df[col_name] -= np.log10(
                physics.atomic_numbers[of] / physics.atomic_numbers[to])
            self.df[col_name] -= physics.solar_abundances[of] \
                                - physics.solar_abundances[to]
        
        if of == "Mg" and to == "H":  # Apply correction for [Mg/H]
            self.df[col_name] += 0.4


if __name__ == "__main__":
    s = Snapshot(1, False, 4, 127)
    s.tag_particles_by_region()
    print(s.df.info())
    # print(s.df)
    # print(s.df.describe(include="all"))
    # df = pd.read_csv("data/level4/au1/temporal_data.csv",
    #                  index_col="SnapshotNumber")
    # print(df)
    # print(df["ReferencePotential_(km/s)^2"].to_numpy()[-1])

