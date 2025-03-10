import numpy as np
from scipy.stats import binned_statistic
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.parser import parse
from auriga.images import figure_setup


def read_data(simulation: str) -> pd.DataFrame:
    """
    This method returns a Pandas dataframe with the gravitational potential
    and the radius of the dark matter particles.

    Parameters
    ----------
    simulation : str
        The simulation to consider.

    Returns
    -------
    pd.DataFrame
        The properties in a Pandas DataFrame.
    """
    s = Snapshot(simulation=simulation, loadonlytype=[1])
    s.add_extra_coordinates()

    props = {
        "GravPotential_(km/s)^2": s.potential,
        "CylindricalRadius_ckpc": s.rho,
        "SphericalRadius_ckpc": s.r
        }

    df = pd.DataFrame(props)
    df.simulation = simulation
    return df


def bin_data(df: pd.DataFrame) -> None:
    STATISTIC = "mean"
    N_BINS = 1000
    RANGE = (0.0, 1000.0)

    stat, bin_edges, _ = binned_statistic(
        x=df["SphericalRadius_ckpc"],
        values=df["GravPotential_(km/s)^2"],
        statistic=STATISTIC, bins=N_BINS, range=RANGE)
    bin_centers = bin_edges[1:] - np.diff(bin_edges)[0]
    
    data = pd.DataFrame({
        "GravPotential_(km/s)^2": stat,
        "SphericalRadius_ckpc": bin_centers,
    })

    galaxy, rerun, resolution, _ = parse(df.simulation)
    paths = Paths(galaxy, rerun, resolution)

    data.to_csv(f"{paths.results}reference_potential_vs_radius.csv")


def run_analysis_for_simulation(simulation: str) -> None:
    df = read_data(simulation)
    bin_data(df)


def plot_ref_pot_for_sample(sample: list) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlabel(r"$r$ [ckpc]")
    ax.set_ylabel(
        "Grav. Potential "
        r"[$10^5 \, \left( \mathrm{km} / \mathrm{s} \right)^2$]")
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    for simulation in sample:
        galaxy, rerun, resolution, _ = parse(simulation)
        paths = Paths(galaxy, rerun, resolution)
        df = pd.read_csv(f"{paths.results}reference_potential_vs_radius.csv")

        ax.plot(df["SphericalRadius_ckpc"],
                df["GravPotential_(km/s)^2"].to_numpy() / 1E5,
                ls="-", lw=0.75, color='k', zorder=10)

    fig.savefig("images/reference_potential_vs_radius.pdf")
    plt.close(fig)


def main():
    settings = Settings()

    sample = [f"au{i}_or_l4_s127" for i in settings.groups["Included"]]
    Pool(8).map(run_analysis_for_simulation, sample)

    figure_setup()
    plot_ref_pot_for_sample(sample)


if __name__ == "__main__":
    main()