import numpy as np
import pandas as pd
import yaml
import argparse
from multiprocessing import Pool

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.parser import parse


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
    s.add_stellar_age()
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    mask = is_main_obj & is_real_star

    props = {
        "StellarAge_Gyr": s.stellar_age[mask],
        "ComponentTag": s.region_tag[mask]}

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def calculate_values_for_galaxy(simulation: str, config: dict):
    galaxy, _, _, _ = parse(simulation)
    label = f"Au{galaxy}"

    df = read_data(simulation, config)

    data = [np.nan] * 10

    data[0] = df["StellarAge_Gyr"].median()
    data[1] = df["StellarAge_Gyr"][df["ComponentTag"] == 0].median()
    data[2] = df["StellarAge_Gyr"][df["ComponentTag"] == 1].median()
    data[3] = df["StellarAge_Gyr"][df["ComponentTag"] == 2].median()
    data[4] = df["StellarAge_Gyr"][df["ComponentTag"] == 3].median()
    data[5] = df["StellarAge_Gyr"].mean()
    data[6] = df["StellarAge_Gyr"][df["ComponentTag"] == 0].mean()
    data[7] = df["StellarAge_Gyr"][df["ComponentTag"] == 1].mean()
    data[8] = df["StellarAge_Gyr"][df["ComponentTag"] == 2].mean()
    data[9] = df["StellarAge_Gyr"][df["ComponentTag"] == 3].mean()

    print(f"{label.rjust(4)}... Ok.")
    return data


def calculate_late_sf_fraction(simulation: str, config: dict,) -> None:
    df = read_data(simulation, config)
    return [(df["StellarAge_Gyr"] <= 5.0).sum() / len(df),
            (df["StellarAge_Gyr"] <= 4.0).sum() / len(df),
            (df["StellarAge_Gyr"] <= 3.0).sum() / len(df),
            (df["StellarAge_Gyr"] <= 2.0).sum() / len(df),
            (df["StellarAge_Gyr"] <= 1.0).sum() / len(df)]


def main():
    settings = Settings()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    sample = [(
        f"au{i}_or_l4_s127", config) for i in settings.groups["Included"]]

    # Run the analysis
    data = Pool(8).starmap(calculate_values_for_galaxy, sample)
    columns = ["MedianStellarAge_Gyr"]
    for c in settings.components:
        columns.append(f"MedianStellarAge_{c}_Gyr")
    columns.append("MeanStellarAge_Gyr")
    for c in settings.components:
        columns.append(f"MeanStellarAge_{c}_Gyr")
    data = pd.DataFrame(data=data,
                        index=[f"Au{i}" for i in settings.groups["Included"]],
                        columns=columns)
    data.to_csv(f"results/stellar_age{config['FILE_SUFFIX']}.csv")
    
    # data = Pool(8).starmap(calculate_late_sf_fraction, sample)
    # columns = [f"FractionOfStarsYoungerThan{i}Gyr" for i in [5, 4, 3, 2, 1]]
    # data = pd.DataFrame(data=data,
    #                     index=[f"Au{i}" for i in settings.groups["Included"]],
    #                     columns=columns)
    # data.to_csv(f"results/"
    #             f"fraction_of_young_stars{config['FILE_SUFFIX']}.csv")


if __name__ == "__main__":
    main()