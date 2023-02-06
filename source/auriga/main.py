from galactic_properties import GalacticPropertiesAnalysis
from auriga.settings import Settings


class MainPipeline():

    def __init__(self, galaxy: int, rerun: bool, resolution: int):
        self._galaxy: int = galaxy
        self._rerun: bool = rerun
        self._resolution: int = resolution

    def run_pipeline(self) -> None:
        # Calculate basic simulation data
        analysis = GalacticPropertiesAnalysis(self._galaxy,
                                              self._rerun,
                                              self._resolution)
        analysis.analyze_galaxy()

        # Calculate the reference potential
        # Calculate the velocity of the main subhalo
        # Calculate the rotation matrix
        # Plot the density maps


def run_analysis(galaxy: int, rerun: bool, resolution: int) -> None:
    print(f"Analyzing Au{galaxy}... ", end='')
    pipeline = MainPipeline(galaxy, rerun, resolution)
    pipeline.run_pipeline()
    print(" Done.")


def main() -> None:
    settings = Settings()
    for galaxy in settings.galaxies:
        run_analysis(galaxy=galaxy, rerun=False, resolution=4)
        if galaxy in settings.reruns:
            run_analysis(galaxy=galaxy, rerun=True, resolution=4)


if __name__ == "__main__":
    pipeline = MainPipeline(galaxy=6, rerun=False, resolution=4)
    pipeline.run_pipeline()
