import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
import argparse
import yaml
import pandas as pd

from auriga.parser import parse
from auriga.settings import Settings
from auriga.images import figure_setup


def plot_insideout_vs_disc_size(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3, 3), ncols=1)

    ax.grid(True, ls='-', lw=0.25, c='gainsboro')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.5, 4.5)
    ax.set_xlabel(r"$R_\mathrm{d}$ [kpc]")
    ax.set_ylabel(r"$\eta_\mathrm{Net}$ [Gyr]")

    data = np.nan * np.ones((len(sample), 3))

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        with open("data/iza_2024.json", "r") as f:
            d = json.load(f)
            data[i, 1] = d["InsideOutParameter_Gyr"][f"Au{galaxy}"]
            data[i, 2] = d["InsideOutParameterError_Gyr"][f"Au{galaxy}"]
        df = pd.read_csv("data/iza_2022.csv", index_col="Galaxy")
        data[i, 0] = df["DiscRadius_kpc"][galaxy]

    ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2],
                markeredgecolor="white", capsize=2, capthick=1, color="black",
                marker='o', markersize=4, linestyle='none', zorder=10)

    corr = pearsonr(data[:, 0], data[:, 2])
    ax.text(x=0.05, y=0.95, size=8.0, color="black",
            ha="left", va="center", s=r"$r$: " \
                + str(np.round(corr[0], 2)),
            transform=ax.transAxes)
    ax.text(x=0.05, y=0.9, size=8.0, color="black",
            ha="left", va="center", s=r"$p$-value: " \
                + str(np.round(corr[1], 3)),
            transform=ax.transAxes)

    fig.savefig(f"images/insideout_vs_disc_radius{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main():
    settings = Settings()
    sample = [f"au{i}_or_l4" for i in settings.groups["Included"]]

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Create figures
    figure_setup()
    plot_insideout_vs_disc_size(sample, config)


if __name__ == "__main__":
    main()