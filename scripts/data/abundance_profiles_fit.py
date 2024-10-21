import numpy as np
import pandas as pd
import yaml
from multiprocessing import Pool
import argparse
from scipy.stats import linregress
import json

from auriga.snapshot import Snapshot
from auriga.settings import Settings


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
    s.add_metal_abundance(of="Fe", to="H")
    s.add_metal_abundance(of="O", to="H")
    s.add_metal_abundance(of="O", to="Fe")
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    mask = is_main_obj & is_real_star

    props = {
        "CylindricalRadius_ckpc": s.rho[mask],
        "[Fe/H]": s.metal_abundance["Fe/H"][mask],
        "[O/H]": s.metal_abundance["O/H"][mask],
        "[O/Fe]": s.metal_abundance["O/Fe"][mask],
        "ComponentTag": s.region_tag[mask]}

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def fit_profiles(simulation: str, config: dict):
    settings = Settings()
    short_sim = "_".join(simulation.split("_")[:-1])

    df = read_data(simulation, config)

    min_radius = config["ABUNDANCE_PROFILE_FIT"]["MIN_RADIUS_CKPC"]
    max_radius = config["ABUNDANCE_PROFILE_FIT"]["MAX_RADIUS_CKPC"]
    mask = (df["CylindricalRadius_ckpc"] <= max_radius) & \
        (df["CylindricalRadius_ckpc"] >= min_radius) & \
            (df["ComponentTag"] == settings.component_tags["CD"])
    x = df["CylindricalRadius_ckpc"][mask]
    y = df["[Fe/H]"][mask]
    lreg = linregress(x=x, y=y)
    reg_dict = {"slope": lreg.slope,
                "intercept": lreg.intercept,
                "rvalue": lreg.rvalue,
                "pvalue": lreg.pvalue,
                "stderr": lreg.stderr,
                "intercept_stderr": lreg.intercept_stderr}
    with open(f"results/{short_sim}/"
              f"FeH_abundance_profile_stars_fit{config['FILE_SUFFIX']}"
              f".json",
              'w') as f:
        json.dump(reg_dict, f)


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
    Pool(8).starmap(fit_profiles, args)


if __name__ == "__main__":
    main()
