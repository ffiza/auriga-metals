import pandas as pd
import numpy as np
import warnings
import os
from loadmodules import gadget_readsnap, load_subfind
from auriga.cosmology import Cosmology
from auriga.physics import Physics
from auriga.paths import Paths
from auriga.parser import parse
from auriga.support import find_indices
from auriga.settings import Settings


def read_raw_snapshot(simulation: str,
                      loadonlytype: list,
                      ) -> tuple:
    """
    This method reads a raw version of the snapshot (with no post
    processing) for a given simulation and a given list of particle
    types.

    Parameters
    ----------
    simulation : str
        The simulation to read.
    loadonlytype : list
        A list with the particle types to load.

    Returns
    -------
    sf
        The snapshot.
    sb
        The SUBFIND catalogue.
    """

    galaxy, rerun, resolution, snapnum = parse(simulation=simulation)
    paths = Paths(galaxy=galaxy, rerun=rerun, resolution=resolution)

    sf = gadget_readsnap(snapshot=snapnum,
                         snappath=paths.snapshots,
                         lazy_load=True,
                         cosmological=False,
                         applytransformationfacs=False,
                         loadonlytype=loadonlytype)
    sb = load_subfind(id=snapnum,
                      dir=paths.snapshots,
                      cosmological=False)
    sf.calc_sf_indizes(sf=sb)

    return sf, sb


def read_header(simulation: str,
                ) -> tuple:
    """
    This method reads only the header of a given snapshot.

    Parameters
    ----------
    simulation : str
        The simulation to read.

    Returns
    -------
    header
        The snapshot header.
    """

    galaxy, rerun, resolution, snapnum = parse(simulation=simulation)
    paths = Paths(galaxy=galaxy, rerun=rerun, resolution=resolution)

    header = gadget_readsnap(snapshot=snapnum,
                             snappath=paths.snapshots,
                             lazy_load=True,
                             cosmological=False,
                             applytransformationfacs=False,
                             onlyHeader=True)

    return header


class Snapshot:
    def __init__(self, simulation: str, loadonlytype: list):
        """
        The class constructor.

        Parameters
        ----------
        simulation : str
            The simulation to read.
        loadonlytype : list
            A list with the particle types to load.
        """

        galaxy, rerun, resolution, snapnum = parse(simulation=simulation)
        self.simulation = simulation
        self.loadonlytype = loadonlytype
        self.galaxy = galaxy
        self.rerun = rerun
        self.resolution = resolution
        self.snapnum = snapnum

        self._has_referenced_pot = False
        self._has_circularity = False
        self._has_normalized_potential = False
        self._has_metals = False
        self._has_stellar_formation_snapshot = False

        paths = Paths(galaxy=galaxy, rerun=rerun, resolution=resolution)

        sf, sb = read_raw_snapshot(simulation=simulation,
                                   loadonlytype=loadonlytype)

        # Load rotation matrix and subhalo velocity
        rotation_matrix = np.loadtxt(
            f"{paths.results_abs}/rotation_matrices.csv")[
                self.snapnum].reshape((3, 3))
        subhalo_vel = np.loadtxt(
            f"{paths.results_abs}/subhalo_vels.csv")[
                self.snapnum]

        # Set the indices of the main object
        df = pd.read_csv(f"{paths.results_abs}/temporal_data.csv",
                         usecols=["MainHaloIdx", "MainSubhaloIdx"])
        self.halo_idx = df["MainHaloIdx"].to_numpy()[self.snapnum]
        self.subhalo_idx = df["MainSubhaloIdx"].to_numpy()[self.snapnum]

        subhalo_grouptab_idx = sb.data["ffsh"][self.halo_idx] \
            + self.subhalo_idx

        self.pos = (
            rotation_matrix @ (sf.pos - sb.data["spos"][subhalo_grouptab_idx]
                               / sf.hubbleparam).T).T * 1e3
        self.vel = (
            rotation_matrix @ (sf.vel * np.sqrt(sf.time)
                               - subhalo_vel).T).T
        self.ids = sf.id

        self.expansion_factor = sf.time
        self.time = Cosmology().redshift_to_time(sf.redshift)
        self.redshift = sf.redshift
        self.lookback_time = Cosmology().present_time - self.time
        self.type = sf.type
        self.potential = sf.pot / self.expansion_factor
        self.halo = sf.halo
        self.subhalo = sf.subhalo

        # Stellar formation time as scale factor
        if 4 in loadonlytype:
            self.stellar_formation_time = np.nan * np.ones(sf.type.shape[0])
            self.stellar_formation_time[sf.type == 4] = sf.age

        # Mass of particles in Msun
        if len(loadonlytype) == 1 and loadonlytype[0] == 1:
            self.mass = sf.masses[1] * np.ones(sf.type.shape[0]) * 1e10
        else:
            self.mass = sf.mass * 1e10

        self.metal_abundance = {}

    def add_stellar_age(self):
        """
        This method calculates the age of each star (not the birth time) in
        Gyr.
        """

        cosmology = Cosmology()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            stellar_formation_times = cosmology.expansion_factor_to_time(
                self.stellar_formation_time)
        # Age should be relative to the time of the current snapshot!
        self.stellar_age = self.time - stellar_formation_times

    def add_metals(self):
        """
        This method adds an array with the mass fraction of each metal to the
        class.
        """

        sf, _ = read_raw_snapshot(simulation=self.simulation,
                                  loadonlytype=self.loadonlytype)

        if 0 in self.loadonlytype or 4 in self.loadonlytype:
            self.metals = np.nan * np.ones((len(sf.type), sf.gmet.shape[1]))
            self.metals[(self.type == 0) | (self.type == 4)] = sf.gmet
        else:
            warnings.warn(message="No stars nor gas found in the simulation. "
                                  "Metals not loaded.")

        self._has_metals = True

    def add_metallicity(self):
        """
        This method calculates the metallicity Z = 1 - X - Y for the
        particles in mass fraction.
        """

        if not self._has_metals:
            self.add_metals()

        self.metallicity = 1 - np.nansum(self.metals[:, :2], axis=1)

    def add_metal_abundance(self, of: str, to: str):
        """
        Calculates the abundance [X/Y] of metal x relative to metal Y.

        Parameters
        ----------
        of : str
            The metal to calculate the abundance of.
        to : str
            The metal to use as relative to.
        """

        physics = Physics()

        if of not in physics.metals or to not in physics.metals:
            raise NotImplementedError("Metal not implemented in "
                                      "the simulation.")

        if 0 in self.loadonlytype or 4 in self.loadonlytype:
            if not self._has_metals:
                self.add_metals()

            of_idx = physics.metals.index(of)
            to_idx = physics.metals.index(to)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                ab = np.log10(self.metals[:, of_idx] / self.metals[:, to_idx])
                ab -= np.log10(
                    physics.atomic_numbers[of] / physics.atomic_numbers[to])
                ab -= physics.solar_abundances[of] \
                    - physics.solar_abundances[to]

            # # Apply correction for [Mg/H]
            # if of == "Mg" and to == "H":
            #     ab += 0.4

            self.metal_abundance[f"{of}/{to}"] = ab

        else:
            warnings.warn(message="No stars nor gas found in the simulation. "
                                  "Abundance not calculated.")

    def add_reference_to_potential(self):
        """
        Change the raw potential to a potential with a reference.
        """

        paths = Paths(galaxy=self.galaxy,
                      rerun=self.rerun,
                      resolution=self.resolution)

        df = pd.read_csv(
            os.path.abspath(f"{paths.results_abs}/temporal_data.csv"),
            index_col="SnapshotNumber")
        ref_pot = df["ReferencePotential_(km/s)^2"].loc[self.snapnum]
        self.potential -= ref_pot  # (km/s)^2

        self._has_referenced_pot = True

    def add_extra_coordinates(self):
        """
        This method calculates the radii of particles in cylindrical and
        spherical coordinates.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            self.rho = np.linalg.norm(self.pos[:, 0:2], axis=1)
            self.r = np.linalg.norm(self.pos, axis=1)

            self.v_rho = (self.pos[:, 0] * self.vel[:, 0]
                          + self.pos[:, 1] * self.vel[:, 1]) / self.rho
            self.v_phi = (self.pos[:, 0] * self.vel[:, 1]
                          - self.pos[:, 1] * self.vel[:, 0]) / self.rho

    def add_circularity(self) -> None:
        """
        This method calculates the circularity parameter for the stellar
        particles in the main halo/subhalo. Stars not in the main object
        or other particle types are assigned NaNs.
        """

        physics = Physics()

        # Check if all particles are loaded in the data frame.
        for include_type in [0, 1, 2, 3, 4, 5]:
            if include_type not in self.loadonlytype:
                raise ValueError(
                    f"Type {include_type} particles/cells not loaded. "
                    "All particle types must be loaded to calculate the "
                    "circularity parameter.")

        self.add_extra_coordinates()

        # Specific angular momentum in kpc km/s
        jz = np.cross(self.pos, self.vel)[:, 2] * self.expansion_factor

        is_galaxy = (self.halo == self.halo_idx) \
            & (self.subhalo == self.subhalo_idx)
        is_star = (self.type == 4) & (self.stellar_formation_time > 0)

        bins = np.asarray([0] + list(np.sort(self.r[is_galaxy & is_star])))
        ii = self.r[is_galaxy & is_star].argsort().argsort()
        enclosed_mass = np.add.accumulate(
            np.histogram(self.r, weights=self.mass, bins=bins)[0])[ii]
        del bins, ii

        # Calculate circular momentum
        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero.
            warnings.simplefilter("ignore", RuntimeWarning)
            jc = (
                self.r[is_galaxy & is_star]
                * self.expansion_factor
                * np.sqrt(
                    physics.gravitational_constant
                    * enclosed_mass
                    / (1e3 * self.r[is_galaxy & is_star]
                        * self.expansion_factor)))  # kpc km/s
        del enclosed_mass

        # Calculate circularity parameter
        circularity = np.nan * np.ones(self.mass.shape[0])
        with warnings.catch_warnings():
            # Ignore RuntimeWarnings due to division by zero
            warnings.simplefilter("ignore", RuntimeWarning)
            circularity[is_galaxy & is_star] = jz[is_galaxy & is_star] / jc
        del jz, jc, is_galaxy, is_star

        self.circularity = circularity
        self._has_circularity = True

    def add_normalized_potential(self):
        """
        This method calculates the normalized potential using the maximum
        absolute value of the stellar particles in the main halo/subhalo.
        """

        is_main_obj = (self.halo == self.halo_idx) \
            & (self.subhalo == self.subhalo_idx)
        is_real_star = (self.type == 4) & (self.stellar_formation_time > 0)
        norm_pot = self.potential / np.abs(
            self.potential[is_real_star & is_main_obj]).max()
        self.normalized_potential = norm_pot
        self._has_normalized_potential = True

    def tag_particles_by_region(self,
                                disc_std_circ: float,
                                disc_min_circ: float,
                                cold_disc_delta_circ: float,
                                bulge_max_specific_energy: float):
        """
        This method adds a tag for each particle that indicates to which
        galactic component they belong (-1: does not belong to the main
        subhalo, 0: belongs to the halo, 1: belongs to the bulge,
        2: belongs to the cold disc, 3: belongs to the warm disc), base
        on the input parameters.

        Parameters
        ----------
        disc_std_circ : float
            The standard circularity of the disc (usually 1.0).
        disc_min_circ : float
            The minimum circularity of the disc. This value separaes the
            rotating components from the spheroidal components.
        cold_disc_delta_circ : float
            The deviation from the standard circularity for the cold disc.
        bulge_max_specific_energy : float
            The maximum specific energy of the bulge.
        """

        if not self._has_circularity:
            self.add_circularity()
        if not self._has_referenced_pot:
            self.add_reference_to_potential()
        if not self._has_normalized_potential:
            self.add_normalized_potential()

        region_tag = -1 * np.ones(self.mass.shape[0], dtype=np.int8)

        # Tag halo particles
        region_tag[
            (self.halo == self.halo_idx)
            & (self.subhalo == self.subhalo_idx)] = 0

        # Tag bulge particles
        region_tag[
            (self.normalized_potential <= bulge_max_specific_energy)
            & (self.circularity < disc_min_circ)
            & (self.halo == self.halo_idx)
            & (self.subhalo == self.subhalo_idx)] = 1

        # Tag cold disc particles
        region_tag[
            (np.abs(self.circularity - disc_std_circ) <= cold_disc_delta_circ)
            & (self.halo == self.halo_idx)
            & (self.subhalo == self.subhalo_idx)] = 2

        # Tag warm disc particles
        region_tag[
            (np.abs(self.circularity - disc_std_circ) > cold_disc_delta_circ)
            & (self.circularity > disc_min_circ)
            & (self.halo == self.halo_idx)
            & (self.subhalo == self.subhalo_idx)] = 3

        self.region_tag = region_tag

    def _load_snapshot_exp_facts(self):
        """
        This method adds the cosmic time in Gyr (age of the universe) for
        each snapshot.
        """

        expansion_factors = np.nan * np.ones(self.snapnum + 1)
        for i in range(self.snapnum + 1):
            header = read_header(
                simulation=self.simulation.split("_s")[0] + f"_s{i}")
            expansion_factors[i] = header.time

        return expansion_factors

    def calculate_stellar_formation_snapshot(self):
        """
        This method calculates the snapshot at which each star is found for
        the first time (the `formation snapshot`). If the particle is not a
        star, it gets a -1.
        """

        if 4 not in self.loadonlytype:
            raise ValueError("No stars found in snapshot.")

        exp_facts = self._load_snapshot_exp_facts()
        self.stellar_formation_snapshot = -1 * np.ones(
            self.mass.shape[0], dtype=np.int8)

        for i in range(self.snapnum + 1):
            self.stellar_formation_snapshot[
                self.stellar_formation_time > exp_facts[i - 1]] = i

        self._has_stellar_formation_snapshot = True

    def add_stellar_origin(self):
        """
        This method calculates the halo and subhalo in which each star was
        born. If the particle is not a star or the star was born at very early
        times, it gets a -1. The first column indicates the halo index and the
        second the subhalo index.
        """

        settings = Settings()

        self.stellar_origin_idx = -1 * np.ones((self.mass.shape[0], 2),
                                               dtype=np.int32)
        if not self._has_stellar_formation_snapshot:
            self.calculate_stellar_formation_snapshot()
        for i in range(settings.first_snap, self.snapnum + 1):
            is_star_born_here = (self.type == 4) \
                & (self.stellar_formation_time > 0) \
                & (self.stellar_formation_snapshot == i)
            if is_star_born_here.sum() != 0:
                ids_born_here = self.ids[is_star_born_here]
                sf, _ = read_raw_snapshot(
                    simulation=self.simulation.split("_s")[0] + f"_s{i}",
                    loadonlytype=[4])
                idxs = find_indices(a=sf.id, b=ids_born_here)
                if idxs.min() == -1:
                    raise ValueError("-1 found in idxs.")
                self.stellar_origin_idx[is_star_born_here, 0] = sf.halo[idxs]
                self.stellar_origin_idx[
                    is_star_born_here, 1] = sf.subhalo[idxs]
