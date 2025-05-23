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
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    is_gas = s.type == 0
    mask = is_main_obj & (is_real_star | is_gas)

    props = {
        "ParticleType": s.type[mask],
        "CylindricalRadius_ckpc": s.rho[mask],
        "Mass_Msun": s.mass[mask],
        "ComponentTag": s.region_tag[mask]}

    df = pd.DataFrame(props)
    return df


def calculate_profile(simulation: str, config: dict):
    settings = Settings()
    galaxy, rerun, resolution, _ = parse(simulation)
    label = f"Au{galaxy}"
    paths = Paths(galaxy, rerun, resolution)

    df = read_data(simulation, config)

    data = {}

    for ptype in [0, 4]:
        is_ptype = df["ParticleType"] == ptype
        binned_stat, bin_edges, _ = binned_statistic(
            x=df["CylindricalRadius_ckpc"][is_ptype],
            values=df["Mass_Msun"][is_ptype],
            statistic="sum",
            bins=config["DENSITY_PROFILES"]["N_BINS"],
            range=(config["DENSITY_PROFILES"]["MIN_RADIUS_CKPC"],
                config["DENSITY_PROFILES"]["MAX_RADIUS_CKPC"]))
        bin_centers = bin_edges[1:] - np.diff(bin_edges)[0] / 2
        data["CylindricalRadius_ckpc"] = bin_centers
        bin_area = np.pi * np.diff(bin_edges * bin_edges) * 1000 * 1000
        # Mass surfance density in Msun / pc^2
        data[f"SurfaceDensity_Type{ptype}_Msun/ckpc^2"] \
            = binned_stat / bin_area

    for i, c in enumerate(settings.components):
        is_component = df["ComponentTag"] == i
        is_star = df["ParticleType"] == 4
        binned_stat = binned_statistic(
            x=df["CylindricalRadius_ckpc"][is_star & is_component],
            values=df["Mass_Msun"][is_star & is_component],
            statistic="sum",
            bins=config["DENSITY_PROFILES"]["N_BINS"],
            range=(config["DENSITY_PROFILES"]["MIN_RADIUS_CKPC"],
                config["DENSITY_PROFILES"]["MAX_RADIUS_CKPC"]))[0]
        data[f"SurfaceDensity_Type4_{c}_Msun/ckpc^2"] \
            = binned_stat / bin_area

    data = pd.DataFrame(data)
    data = data.round(4)
    data.to_csv(f"{paths.results}/"
                f"density_profile{config['FILE_SUFFIX']}.csv",
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