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
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    mask = is_main_obj & is_real_star

    props = {
        "Mass_Msun": s.mass[mask],
        "ComponentTag": s.region_tag[mask]}

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def calculate_values_for_galaxy(simulation: str, config: dict):
    settings = Settings()
    galaxy, _, _, _ = parse(simulation)
    label = f"Au{galaxy}"

    df = read_data(simulation, config)

    data = [np.nan] * 4

    mass = df["Mass_Msun"].sum()
    for i, c in enumerate(settings.components):
        data[i] = df["Mass_Msun"][
            df["ComponentTag"] == settings.component_tags[c]].sum() / mass

    print(f"{label.rjust(4)}... Ok.")
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
    args = [(f"au{i}_or_l4_s127", config) for i in settings.groups["Included"]]
    data = Pool(8).starmap(calculate_values_for_galaxy, args)

    columns = [f"{c}ToTotal" for c in settings.components]

    data = pd.DataFrame(data=data,
                        index=[f"Au{i}" for i in settings.groups["Included"]],
                        columns=columns)
    data.to_csv(f"results/"
                f"stellar_mass_fractions{config['FILE_SUFFIX']}.csv")


if __name__ == "__main__":
    main()