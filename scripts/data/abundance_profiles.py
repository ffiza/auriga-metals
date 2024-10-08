import numpy as np
from scipy.stats import binned_statistic
import pandas as pd
import yaml
import argparse
from scipy.stats import linregress
import json

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
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
    s.add_metal_abundance(of="Fe", to="H")
    s.add_metal_abundance(of="O", to="H")
    s.add_metal_abundance(of="O", to="Fe")
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_gas = (s.type == 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    mask = is_main_obj & (is_gas | is_real_star)

    props = {
        "ParticleType": s.type[mask],
        "CylindricalRadius_ckpc": s.rho[mask],
        "[Fe/H]": s.metal_abundance["Fe/H"][mask],
        "[O/H]": s.metal_abundance["O/H"][mask],
        "[O/Fe]": s.metal_abundance["O/Fe"][mask],
        "ComponentTag": s.region_tag[mask]}

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def calculate_profile(simulation: str, config: dict):
    settings = Settings()
    galaxy, rerun, resolution, _ = parse(simulation)
    label = f"Au{galaxy}"
    print(f"{label.rjust(4)}... ", end="")
    paths = Paths(galaxy, rerun, resolution)
    abundances = [("Fe", "H"), ("O", "H"), ("O", "Fe")]

    df = read_data(simulation, config)

    data = {}

    # region Stars
    is_star = df["ParticleType"] == 4
    for of, to in abundances:
        binned_stat, bin_edges, _ = binned_statistic(
            x=df["CylindricalRadius_ckpc"][is_star],
            values=df[f"[{of}/{to}]"][is_star],
            statistic=np.nanmean,
            bins=config["ABUNDANCE_PROFILES"]["N_BINS"],
            range=(config["ABUNDANCE_PROFILES"]["MIN_RADIUS_CKPC"],
                config["ABUNDANCE_PROFILES"]["MAX_RADIUS_CKPC"]))
        bin_centers = bin_edges[1:] - np.diff(bin_edges)[0] / 2
        data["CylindricalRadius_ckpc"] = bin_centers
        data[f"[{of}/{to}]_Stars"] = binned_stat
        for i, c in enumerate(settings.components):
            is_component = df["ComponentTag"] == i
            data[f"[{of}/{to}]_{c}_Stars"] = binned_statistic(
                x=df["CylindricalRadius_ckpc"][is_star & is_component],
                values=df["[Fe/H]"][is_star & is_component],
                statistic=np.nanmean,
                bins=config["ABUNDANCE_PROFILES"]["N_BINS"],
                range=(config["ABUNDANCE_PROFILES"]["MIN_RADIUS_CKPC"],
                    config["ABUNDANCE_PROFILES"]["MAX_RADIUS_CKPC"]))[0]
    # endregion

    # region Gas
    is_gas = df["ParticleType"] == 0
    for of, to in abundances:
        binned_stat = binned_statistic(
            x=df["CylindricalRadius_ckpc"][is_gas],
            values=df[f"[{of}/{to}]"][is_gas],
            statistic=np.nanmean,
            bins=config["ABUNDANCE_PROFILES"]["N_BINS"],
            range=(config["ABUNDANCE_PROFILES"]["MIN_RADIUS_CKPC"],
                config["ABUNDANCE_PROFILES"]["MAX_RADIUS_CKPC"]))[0]
        bin_centers = bin_edges[1:] - np.diff(bin_edges)[0] / 2
        data["CylindricalRadius_ckpc"] = bin_centers
        data[f"[{of}/{to}]_Gas"] = binned_stat
        for i, c in enumerate(settings.components):
            is_component = df["ComponentTag"] == i
            data[f"[{of}/{to}]_{c}_Gas"] = binned_statistic(
                x=df["CylindricalRadius_ckpc"][is_gas & is_component],
                values=df["[Fe/H]"][is_gas & is_component],
                statistic=np.nanmean,
                bins=config["ABUNDANCE_PROFILES"]["N_BINS"],
                range=(config["ABUNDANCE_PROFILES"]["MIN_RADIUS_CKPC"],
                       config["ABUNDANCE_PROFILES"]["MAX_RADIUS_CKPC"]))[0]
    # endregion

    data = pd.DataFrame(data)
    data.to_csv(f"{paths.results}/"
                f"abundance_profile{config['FILE_SUFFIX']}.csv",
                index=False)
    
    print("Ok.")
    return data


def fit_profiles(sample: list, config: dict):
    gal_data = pd.read_csv("data/iza_2022.csv")
    for simulation in sample:
        galaxy = parse(simulation)[0]
        disc_radius = gal_data["DiscRadius_kpc"][gal_data["Galaxy"] == galaxy]
        short_sim = "_".join(simulation.split("_")[:-1])
        df = pd.read_csv(f"results/{short_sim}/"
                         f"abundance_profile{config['FILE_SUFFIX']}.csv")
        mask = df["CylindricalRadius_ckpc"] <= disc_radius.values[0]
        x = df["CylindricalRadius_ckpc"][mask]
        y = df["[Fe/H]_CD_Stars"][mask]
        lreg = linregress(x=x, y=y)
        reg_dict = {"slope": lreg.slope,
                    "intercept": lreg.intercept,
                    "rvalue": lreg.rvalue,
                    "pvalue": lreg.pvalue,
                    "stderr": lreg.stderr,
                    "intercept_stderr": lreg.intercept_stderr}
        with open(f"results/{short_sim}/FeH_abundance_profile_fit.json",
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
    sample = [f"au{i}_or_l4_s127" for i in settings.groups["Included"]]
    for simulation in sample:
        calculate_profile(simulation, config)
    fit_profiles(sample, config)


if __name__ == "__main__":
    main()