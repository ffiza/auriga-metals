from sys import stdout

from settings import Settings
from galaxy_tracker import GalaxyTracker
from galactic_properties import GalacticPropertiesAnalysis
from subhalo_velocity import SubhaloVelocityAnalysis
from rotation_matrix import RotationMatrixAnalysis

track_galaxy: bool = False
calculate_basic_properties: bool = False
calculate_subhalo_velocity: bool = True
calculate_rotation_matrices: bool = False


class MainPipeline():

    def __init__(self, galaxy: int, rerun: bool, resolution: int):
        self._galaxy: int = galaxy
        self._rerun: bool = rerun
        self._resolution: int = resolution

    def run_pipeline(self) -> None:
        # Track galaxy to find the main halo and subhalo idx
        if track_galaxy:
            tracker = GalaxyTracker(self._galaxy,
                                    self._rerun,
                                    self._resolution)
            tracker.track_galaxy()

        # Calculate basic simulation data
        if calculate_basic_properties:
            analysis = GalacticPropertiesAnalysis(self._galaxy,
                                                  self._rerun,
                                                  self._resolution)
            analysis.analyze_galaxy()

        # Calculate the velocity of the main subhalo
        if calculate_subhalo_velocity:
            analysis = SubhaloVelocityAnalysis(self._galaxy,
                                               self._rerun,
                                               self._resolution)
            analysis.calculate_subhalo_velocities()

        # Calculate the rotation matrix
        if calculate_rotation_matrices:
            analysis = RotationMatrixAnalysis(self._galaxy,
                                              self._rerun,
                                              self._resolution)
            analysis.calculate_rotation_matrices()

        # # Calculate the reference potential
        # analysis = ReferencePotentialAnalysis(self._galaxy,
        #                                       self._rerun,
        #                                       self._resolution)
        # analysis.analyze_galaxy()


        # Plot the density maps


def run_analysis(galaxy: int, rerun: bool, resolution: int) -> None:
    stdout.write(f"Analyzing Au{galaxy}... ")
    pipeline = MainPipeline(galaxy, rerun, resolution)
    pipeline.run_pipeline()
    stdout.write(" Done.\n")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        run_analysis(galaxy=galaxy, rerun=False, resolution=4)
        if galaxy in settings.reruns:
            run_analysis(galaxy=galaxy, rerun=True, resolution=4)


if __name__ == "__main__":
    main()
