"""
File:           age_metallicity.py
Author:         Federico Iza
Description:    This script creates a plot of the age-metallicity relation
                showing all galaxies in the sample.
Usage:          `python scripts/visualization/age_metalllicity.py --config 20`
"""
from matplotlib import pyplot as plt
import pandas as pd
import yaml
import numpy as np
import argparse

from auriga.images import figure_setup
from auriga.settings import Settings
from auriga.parser import parse
from auriga.paths import Paths


def make_plot(config: dict):
    settings = Settings()
    sample = [f"au{i}_or_l4_s127" for i in settings.groups["Included"]]

    fig = plt.figure(figsize=(8.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=5, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flat:
        ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlabel("Stellar Age [Gyr]")

        ax.set_ylim(-2, 1.5)
        ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1])
        ax.set_ylabel("[Fe/H]")

        ax.tick_params(which='both', direction="in")
        ax.label_outer()
    
    sample_data = {}
    for simulation in sample:
        galaxy, rerun, resolution, _ = parse(simulation)
        paths = Paths(galaxy, rerun, resolution)
        data = pd.read_csv(
            f"{paths.results}age_metallicity{config['FILE_SUFFIX']}.csv")

        axs[0].plot(data["StellarAge_BinCenters_Gyr"], data["[Fe/H]_Median"],
                    ls="-", c="silver", zorder=10, lw=1.0)
        
        if "StellarAge_BinCenters_Gyr" not in sample_data.keys():
            sample_data["StellarAge_BinCenters_Gyr"] = \
                data["StellarAge_BinCenters_Gyr"].to_numpy()
        sample_data[f"[Fe/H]_Au{galaxy}_Median"] = \
            data["[Fe/H]_Median"].to_numpy()

        for i in range(len(settings.components)):
            component = settings.components[i]
            ax = axs[i + 1]

            ax.plot(data["StellarAge_BinCenters_Gyr"],
                    data[f"[Fe/H]_Median_{component}"],
                    ls="-", c="silver", zorder=10, lw=1.0)
            sample_data[f"[Fe/H]_Au{galaxy}_Median_{component}"] = \
                data[f"[Fe/H]_Median_{component}"].to_numpy()

    sample_data = pd.DataFrame(sample_data)

    galaxies = [i for i in settings.groups["Included"]]
    x = sample_data["StellarAge_BinCenters_Gyr"]
    y_mean = sample_data[
        [f"[Fe/H]_Au{i}_Median" for i in galaxies]].median(axis=1)
    y_std = sample_data[
        [f"[Fe/H]_Au{i}_Median" for i in galaxies]].std(axis=1)
    axs[0].plot(x, y_mean, ls="-", c="black", zorder=11, lw=1.0)
    axs[0].plot(x, y_mean - y_std, ls="--", c="black", zorder=11, lw=1.0)
    axs[0].plot(x, y_mean + y_std, ls="--", c="black", zorder=11, lw=1.0)

    print("\n== Average Standard Deviation by Component ==")
    for i, c in enumerate(settings.components):
        ax = axs[i + 1]
        color = settings.component_colors[c]
        x = sample_data["StellarAge_BinCenters_Gyr"]
        y_mean = sample_data[
            [f"[Fe/H]_Au{i}_Median_{c}" for i in galaxies]].mean(axis=1)
        y_std = sample_data[
            [f"[Fe/H]_Au{i}_Median_{c}" for i in galaxies]].std(axis=1)
        ax.plot(x, y_mean, ls="-", c=color, zorder=11, lw=1.0)
        ax.plot(x, y_mean + y_std, ls="--", c=color, zorder=11, lw=1.0)
        ax.plot(x, y_mean - y_std, ls="--", c=color, zorder=11, lw=1.0)
        print(c.rjust(5) + ": " + str(np.round(np.mean(y_std), 3)))
    print()

    axs[0].text(
        x=0.05, y=0.05, size=8.0, ha="left", va="bottom",
        s=r"$\textbf{All}$", c="black", transform=axs[0].transAxes)
    for i in range(len(settings.components)):
        ax = axs[i + 1]
        ax.text(
            x=0.05, y=0.05, size=8.0, ha="left", va="bottom",
            s=r"$\textbf{" + list(settings.component_labels.values())[i] \
                + "}$",
            c=list(settings.component_colors.values())[i],
            transform=ax.transAxes)

    fig.savefig(
        f"images/age_metallicity_by_region/"
        f"sample{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main():
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Create the plot
    make_plot(config)


if __name__ == "__main__":
    main()