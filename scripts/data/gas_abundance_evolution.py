"""
File:           gas_abundance_evolution.py
Author:         Federico Iza
Description:    This script analizes the evolution of the gas abundance.
Usage:          `python scripts/data/gas_abundance_evolution.py --simulation
                au6_or_l4_s127 --of Fe --to H --config 02`
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
from auriga.support import make_snapshot_number


def get_gas_abundances(simulation: str, of: str, to: str) -> list:
    s = Snapshot(simulation=simulation, loadonlytype=[0])
    s.add_metal_abundance(of=of, to=to)
    abundance = s.metal_abundance[f"{of}/{to}"]
    abundance = abundance[np.isfinite(abundance)]
    return list(abundance)


def get_gas_abundance_evolution(simulation: str, of: str, to: str,
                                config: dict):
    settings = Settings()
    galaxy, rerun, resolution = parse(simulation)
    paths = Paths(galaxy, rerun, resolution)
    n_snapshots = make_snapshot_number(rerun, resolution)
    snapshot_times = pd.read_csv(
        f"{paths.results}/temporal_data.csv")["Time_Gyr"]

    accumulated_abundances = []
    data = np.nan * np.ones(
        (config["GAS_ABUNDANCE_EVOLUTION"]["N_BINS_ABUNDANCE"],
         config["GAS_ABUNDANCE_EVOLUTION"]["N_BINS_TIME"]))
    time_ref_idx = 0
    for i in range(settings.first_snap, n_snapshots):
        if snapshot_times[i] > (time_ref_idx + 1) * 0.5:
            # Create histogram
            binned_abundance = np.histogram(
                accumulated_abundances, density=True,
                range=(config["GAS_ABUNDANCE_EVOLUTION"]["ABUNDANCE_MIN"],
                       config["GAS_ABUNDANCE_EVOLUTION"]["ABUNDANCE_MAX"],),
                bins=config["GAS_ABUNDANCE_EVOLUTION"]["N_BINS_ABUNDANCE"])[0]
            data[:, time_ref_idx] = binned_abundance

            # Reset loop variables
            accumulated_abundances = []
            time_ref_idx += 1
        accumulated_abundances += get_gas_abundances(
            simulation + f"_s{i}", of, to)
    return data
            


def main():
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=str, required=True)
    parser.add_argument("--of", type=str, required=True)
    parser.add_argument("--to", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Run the analysis
    data = get_gas_abundance_evolution(args.simulation, args.of, args.to,
                                       config)

    # Save data
    galaxy, rerun, resolution = parse(args.simulation)
    paths = Paths(galaxy, rerun, resolution)
    np.save(f"{paths.results}gas_abundance_evolution", data)

if __name__ == "__main__":
    main()