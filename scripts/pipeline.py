from sys import stdout

from auriga.parser import parse
from auriga.settings import Settings
from auriga.galaxy_tracker import GalaxyTracker
from auriga.galactic_properties import GalacticPropertiesAnalysis
from auriga.subhalo_velocity import SubhaloVelocityAnalysis
from auriga.rotation_matrix import RotationMatrixAnalysis
from auriga.reference_potential import ReferencePotentialAnalysis
from auriga.density_maps import DensityMaps


class MainPipeline():

    def __init__(self, simulation: str):
        galaxy, rerun, resolution = parse(simulation=simulation)
        self._simulation: str = simulation
        self._galaxy: int = galaxy
        self._rerun: bool = rerun
        self._resolution: int = resolution

    def run_pipeline(self,
                     track_galaxy: bool,
                     calculate_basic_properties: bool,
                     calculate_subhalo_velocity: bool,
                     calculate_rotation_matrices: bool,
                     calculate_reference_potential: bool,
                     create_density_maps: bool) -> None:

        settings = Settings()

        # Track galaxy to find the main halo and subhalo idx
        stdout.write("Tracking main object... ")
        tracker = GalaxyTracker(self._simulation)
        tracker.track_galaxy(track=self._galaxy in settings.galaxies_to_track,
                             n_part=settings.n_track_dm_parts)

        # Calculate basic simulation data
        if calculate_basic_properties:
            stdout.write("Calculating basic properties... ")
            analysis = GalacticPropertiesAnalysis(self._galaxy,
                                                  self._rerun,
                                                  self._resolution)
            analysis.analyze_galaxy()

        # Calculate the velocity of the main subhalo
        if calculate_subhalo_velocity:
            stdout.write("Calculating subhalo velocity... ")
            analysis = SubhaloVelocityAnalysis(self._galaxy,
                                               self._rerun,
                                               self._resolution)
            analysis.calculate_subhalo_velocities()

        # Calculate the rotation matrix
        if calculate_rotation_matrices:
            stdout.write("Calculating rotation matrices... ")
            analysis = RotationMatrixAnalysis(self._galaxy,
                                              self._rerun,
                                              self._resolution)
            analysis.calculate_rotation_matrices()

        # # Calculate the reference potential
        if calculate_reference_potential:
            stdout.write("Calculating reference potential... ")
            analysis = ReferencePotentialAnalysis(self._galaxy,
                                                  self._rerun,
                                                  self._resolution)
            analysis.analyze_galaxy()

        # Plot the density maps
        if create_density_maps:
            stdout.write("Plotting density maps... ")
            analysis = DensityMaps(self._galaxy,
                                   self._rerun,
                                   self._resolution)
            analysis.make_plots()


def run_analysis(galaxy: int, rerun: bool, resolution: int) -> None:
    stdout.write(f"Analyzing Au{galaxy}... ")
    pipeline = MainPipeline(galaxy, rerun, resolution)
    pipeline.run_pipeline()
    stdout.write("Done.\n")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        run_analysis(galaxy=galaxy, rerun=False, resolution=4)
        if galaxy in settings.reruns:
            run_analysis(galaxy=galaxy, rerun=True, resolution=4)


if __name__ == "__main__":
    main()
