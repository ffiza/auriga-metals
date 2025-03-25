import numpy as np
from scipy.stats import binned_statistic
import pandas as pd
import yaml
import argparse
from multiprocessing import Pool

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
        "ParticleType": s.type[mask],
        "CylindricalRadius_ckpc": s.rho[mask],
        "[Fe/H]": s.metal_abundance["Fe/H"][mask],
        "[O/H]": s.metal_abundance["O/H"][mask],
        "[O/Fe]": s.metal_abundance["O/Fe"][mask],
        "ComponentTag": s.region_tag[mask],
        "Circularity": s.circularity[mask],
        "NormalizedPotential": s.normalized_potential[mask],
        "StellarAge_Gyr": s.stellar_age[mask]}

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def calculate_profile(simulation: str, config: dict):
    settings = Settings()
    galaxy, rerun, resolution, _ = parse(simulation)
    label = f"Au{galaxy}"
    paths = Paths(galaxy, rerun, resolution)
    abundances = [("Fe", "H"), ("O", "H"), ("O", "Fe")]

    df = read_data(simulation, config)

    data = {}

    is_star = df["ParticleType"] == 4
    is_age = (df["StellarAge_Gyr"] <= config["AB_PROF_MAX_STELL_AGE_GYR"]) \
        & (df["StellarAge_Gyr"] >= config["AB_PROF_MIN_STELL_AGE_GYR"])

    for of, to in abundances:
        binned_stat, bin_edges, _ = binned_statistic(
            x=df["CylindricalRadius_ckpc"][is_star & is_age],
            values=df[f"[{of}/{to}]"][is_star & is_age],
            statistic=np.nanmean,
            bins=config["AB_PROF_N_BINS"],
            range=(config["AB_PROF_MIN_RADIUS_CKPC"],
                   config["AB_PROF_MAX_RADIUS_CKPC"]))
        bin_centers = bin_edges[1:] - np.diff(bin_edges)[0] / 2
        data["CylindricalRadius_ckpc"] = bin_centers
        data[f"[{of}/{to}]_Stars"] = binned_stat
        data[f"[{of}/{to}]_Stars_Std"] = binned_statistic(
            x=df["CylindricalRadius_ckpc"][is_star & is_age],
            values=df[f"[{of}/{to}]"][is_star & is_age],
            statistic=np.nanstd,
            bins=config["AB_PROF_N_BINS"],
            range=(config["AB_PROF_MIN_RADIUS_CKPC"],
                config["AB_PROF_MAX_RADIUS_CKPC"]))[0]
        for i, c in enumerate(settings.components):
            is_component = df["ComponentTag"] == i
            is_finite = np.isfinite(df[f"[{of}/{to}]"])
            data[f"[{of}/{to}]_{c}_Stars"] = binned_statistic(
                x=df["CylindricalRadius_ckpc"][
                    is_star & is_age & is_component & is_finite],
                values=df[f"[{of}/{to}]"][
                    is_star & is_age & is_component & is_finite],
                statistic=np.nanmean,
                bins=config["AB_PROF_N_BINS"],
                range=(config["AB_PROF_MIN_RADIUS_CKPC"],
                    config["AB_PROF_MAX_RADIUS_CKPC"]))[0]
            data[f"[{of}/{to}]_{c}_Stars_Std"] = binned_statistic(
                x=df["CylindricalRadius_ckpc"][
                    is_star & is_age & is_component & is_finite],
                values=df[f"[{of}/{to}]"][
                    is_star & is_age & is_component & is_finite],
                statistic=np.nanstd,
                bins=config["AB_PROF_N_BINS"],
                range=(config["AB_PROF_MIN_RADIUS_CKPC"],
                    config["AB_PROF_MAX_RADIUS_CKPC"]))[0]

    data = pd.DataFrame(data)
    data = data.round(6)
    data.to_csv(f"{paths.results}/"
                f"abundance_profile{config['FILE_SUFFIX']}.csv",
                index=False)
    
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
    Pool(8).starmap(calculate_profile, args)


if __name__ == "__main__":
    main()