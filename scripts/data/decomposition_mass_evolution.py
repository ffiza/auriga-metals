import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import argparse

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.parser import parse
from auriga.support import make_snapshot_number


def read_data(simulation: str, config: dict) -> pd.DataFrame:
    """
    This method returns data related to this analysis.

    Parameters
    ----------
    simulation : str
        The simulation to consider.
    config : dict
        A dictionary with the configuration values.

    Returns
    -------
    pd.DataFrame
        The properties in a Pandas DataFrame.
    """
    s = Snapshot(simulation=simulation, loadonlytype=[0, 1, 2, 3, 4, 5])
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)

    df = pd.DataFrame({
        "Mass_Msun": s.mass[is_real_star & is_main_obj],
        "ComponentTag": s.region_tag[is_real_star & is_main_obj],
        })
    
    df.time = s.time
    df.redshift = s.redshift
    df.expansion_factor = s.expansion_factor
    
    return df


def calculate_evolution(simulation: str, config: dict) -> pd.DataFrame:
    """
    This method calculates and saves the data.

    Parameters
    ----------
    simulation : str
        The simulation to analyze.
    config : dict
        A dictionary with the configuration values.

    Returns
    -------
    pd.DataFrame
        The statistical properties of `df`.
    """
    settings = Settings()
    galaxy, rerun, resolution = parse(simulation)
    paths = Paths(galaxy, rerun, resolution)

    n_snapshots = make_snapshot_number(rerun, resolution)
    features = ["Snapshot", "Time_Gyr", "Redshift", "ExpansionFactor",
                "Mass_Msun", "Mass_H_Msun", "Mass_B_Msun", "Mass_CD_Msun",
                "Mass_WD_Msun"]
    data = np.nan * np.ones((n_snapshots, len(features)))
    for i in tqdm(range(settings.first_snap, n_snapshots),
                  desc=f"Au{galaxy}".rjust(4), ncols=100,
                  unit="snapshot", colour="#9467bd"):
        df = read_data(f"{simulation}_s{i}", config)
        data[i, 0] = i
        data[i, 1] = df.time
        data[i, 2] = df.redshift
        data[i, 3] = df.expansion_factor
        data[i, 4] = df["Mass_Msun"].sum()
        for c in range(len(settings.components)):
            data[i, 5 + c] = df["Mass_Msun"][df["ComponentTag"] == c].sum()

    data = pd.DataFrame(data=data, columns=features)
    data.to_csv(f"{paths.results}/"
                f"decomposition_mass_evolution{config['FILE_SUFFIX']}.csv",
                index=False)
    return data


def main():
    settings = Settings()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Run the analysis
    # for simulation in [f"au{i}_or_l4" for i in settings.groups["Included"]]:
            # calculate_evolution(simulation, config)
    for i in settings.groups["Included"]:
        if i >= 21:
            calculate_evolution(f"au{i}_or_l4", config)


if __name__ == "__main__":
    main()