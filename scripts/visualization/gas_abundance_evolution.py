"""
File:           gas_abundance_evolution.py
Author:         Federico Iza
Description:    This script creates a plot of the evolution of the gas
                abundance.
Usage:          `python scripts/visualization/age_metalllicity.py --simulation
                au6_or_l4 --config 20 --filename au6 --label Au6`
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import yaml
import argparse
import numpy as np

from auriga.images import figure_setup
from auriga.settings import Settings
from auriga.parser import parse
from auriga.paths import Paths


def make_plot(data: np.ndarray, config: dict, filename: str, label: str):
    fig = plt.figure(figsize=(3.0, 2.5))
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex=True, sharey=True)

    ax.set_xlim(0, 14)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel("Time [Gyr]")

    ax.set_ylim(-2.5, 1.5)
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_ylabel("[Fe/H]")

    ax.tick_params(which='both', direction="in")
    ax.label_outer()

    im = ax.imshow(data, cmap="viridis", extent=(0, 14, -2.5, 1.5),
                   aspect="auto", origin="lower",
                   norm=LogNorm(0.001, vmax=5.0))

    cbar = fig.colorbar(im, ax=ax, label="PDF", ticks=[0.001, 0.01, 0.1, 1.0])
    cbar.ax.set_yticklabels([0.001, 0.01, 0.1, 1.0])

    ax.text(x=0.05, y=0.95, size=8.0, s=r"$\texttt{" + label + "}$",
            ha='left', va='top', transform=ax.transAxes)

    fig.savefig(f"images/gas_abundance/{filename}.pdf")
    plt.close(fig)


def main():
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Load data
    galaxy, rerun, resolution = parse(args.simulation)
    paths = Paths(galaxy, rerun, resolution)
    data = np.load(f"{paths.results}gas_abundance_evolution.npy")

    # Create the plot
    make_plot(data, config, args.filename, args.label)


if __name__ == "__main__":
    main()