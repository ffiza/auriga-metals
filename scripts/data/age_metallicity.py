"""
File:           age_metallicity.py
Author:         Federico Iza
Description:    This script reads data from a snapshot, calculated binned
                statistics over the stellar ages and saves the results by
                galaxy.
Usage:          `python scripts/data/age_metalllicity.py --simulation
                au6_or_l4_s127 --config 02`
"""
import numpy as np
from scipy.stats import binned_statistic
import pandas as pd
import yaml
import argparse

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.parser import parse
from auriga.mathematics import mad


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
    s.add_metal_abundance(of="Fe", to='H')
    s.add_circularity()
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)

    props = {
        "StellarAge_Gyr": s.stellar_age[is_real_star & is_main_obj],
        "[Fe/H]": s.metal_abundance["Fe/H"][is_real_star & is_main_obj],
        "ComponentTag": s.region_tag[is_real_star & is_main_obj]}

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def get_stats_of_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    This method returns the stats calculated on `df`.

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    config : dict
        A dictionary with the configuration values.

    Returns
    -------
    pd.DataFrame
        The statistical properties of `df`.
    """
    settings = Settings()

    binned_median, bin_edges, _ = binned_statistic(
        x=df["StellarAge_Gyr"],
        values=df["[Fe/H]"],
        statistic=np.nanmedian,
        bins=config["AGE_METALLICITY_RELATION"]["N_BINS_AGE"],
        range=(config["AGE_METALLICITY_RELATION"]["AGE_MIN"],
               config["AGE_METALLICITY_RELATION"]["AGE_MAX"]))
    binned_mad = binned_statistic(
        x=df["StellarAge_Gyr"],
        values=df["[Fe/H]"],
        statistic=mad,
        bins=config["AGE_METALLICITY_RELATION"]["N_BINS_AGE"],
        range=(config["AGE_METALLICITY_RELATION"]["AGE_MIN"],
               config["AGE_METALLICITY_RELATION"]["AGE_MAX"]))[0]
    bin_centers = bin_edges[1:] - np.diff(bin_edges)[0] / 2

    stats = {"[Fe/H]_Median": binned_median,
             "[Fe/H]_MAD": binned_mad,
             "StellarAge_BinCenters_Gyr": bin_centers}
    
    # Add median por each component
    for i in range(len(settings.components)):
        binned_median, _, _ = binned_statistic(
            x=df["StellarAge_Gyr"][df["ComponentTag"] == i],
            values=df["[Fe/H]"][df["ComponentTag"] == i],
            statistic=np.nanmedian,
            bins=config["AGE_METALLICITY_RELATION"]["N_BINS_AGE"],
            range=(config["AGE_METALLICITY_RELATION"]["AGE_MIN"],
                   config["AGE_METALLICITY_RELATION"]["AGE_MAX"]))
        binned_mad, _, _ = binned_statistic(
            x=df["StellarAge_Gyr"][df["ComponentTag"] == i],
            values=df["[Fe/H]"][df["ComponentTag"] == i],
            statistic=mad,
            bins=config["AGE_METALLICITY_RELATION"]["N_BINS_AGE"],
            range=(config["AGE_METALLICITY_RELATION"]["AGE_MIN"],
                   config["AGE_METALLICITY_RELATION"]["AGE_MAX"]))
        stats[f"[Fe/H]_Median_{settings.components[i]}"] = binned_median
        stats[f"[Fe/H]_MAD_{settings.components[i]}"] = binned_mad

    stats = pd.DataFrame(stats)
    return stats


def main():
    settings = Settings()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    for i in settings.groups["Included"]:
        simulation = f"au{i}_or_l4_s127 "
        
        # Run the analysis
        data = read_data(simulation=simulation, config=config)
        stats = get_stats_of_df(df=data, config=config)
        stats = stats.round(config["AGE_METALLICITY_RELATION"]["DECIMALS"])

        # Save data
        galaxy, rerun, resolution, _ = parse(simulation)
        paths = Paths(galaxy, rerun, resolution)
        stats.to_csv(
            f"{paths.results}/age_metallicity{config['FILE_SUFFIX']}.csv")


if __name__ == "__main__":
    main()