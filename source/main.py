from sys import stdout

from settings import Settings
from galactic_properties import GalacticPropertiesAnalysis
from galaxy_tracker import GalaxyTracker
from reference_potential import ReferencePotentialAnalysis


class MainPipeline():

    def __init__(self, galaxy: int, rerun: bool, resolution: int):
        self._galaxy: int = galaxy
        self._rerun: bool = rerun
        self._resolution: int = resolution

    def run_pipeline(self) -> None:
        # # Calculate basic simulation data
        # analysis = GalacticPropertiesAnalysis(self._galaxy,
        #                                       self._rerun,
        #                                       self._resolution)
        # analysis.analyze_galaxy()

        # Track galaxy to find the main halo and subhalo idx
        tracker = GalaxyTracker(self._galaxy,
                                self._rerun,
                                self._resolution)
        tracker.track_galaxy()

        # # Calculate the reference potential
        # analysis = ReferencePotentialAnalysis(self._galaxy,
        #                                       self._rerun,
        #                                       self._resolution)
        # analysis.analyze_galaxy()

        # Calculate the velocity of the main subhalo

        # Calculate the rotation matrix

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
