import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import argparse
from scipy.stats import pearsonr

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup


class Helpers:
    def __init__(self, config: dict):
        self.config = config
        self.settings = Settings()

    def get_stellar_age_df(self):
        return pd.read_csv(
            f"results/stellar_age{self.config['FILE_SUFFIX']}.csv",
            index_col=0)


def plot_sample_stats(sample: list, config: dict):
    settings = Settings()
    helpers = Helpers(config=config)

    xlabel = \
        r"$\mathrm{Stellar\,Age} - \mathrm{Stellar\,Age}_\mathrm{CD}$ [Gyr]"

    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(-0.25, 6.25)
    ax.set_ylim(-0.46, 0.02)
    ax.set_xlabel(xlabel)
    ax.set_yticks([- i * 0.02 for i in range(23)])
    galaxies = [parse(simulation)[0] for simulation in sample]
    ax.set_yticklabels([f"Au{i}" for i in galaxies])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    df = helpers.get_stellar_age_df()

    for i, simulation in enumerate(sample):
        galaxy, _, _ = parse(simulation)
        for j, c in enumerate(["H", "B", "WD"]):
            ax.scatter(
                df.loc[f"Au{galaxy}"][f"MedianStellarAge_{c}_Gyr"] \
                    - df.loc[f"Au{galaxy}"][f"MedianStellarAge_CD_Gyr"],
                0 - 0.02 * i,
                color=settings.component_colors[c],
                marker='o', s=6, zorder=10)
    
    for j, c in enumerate(["H", "B", "WD"]):
        age_diff = df[f"MedianStellarAge_{c}_Gyr"].to_numpy() \
            - df[f"MedianStellarAge_CD_Gyr"].to_numpy()
        ax.plot(age_diff,
                [0 - 0.02 * i for i in range(len(sample))],
                lw=0.5, color=settings.component_colors[c])
        ax.text(
            x=0.05, y=0.96 - j * 0.03, size=6, ha="left", va="top",
            s=r"$\textbf{" + settings.component_labels[c] + "}$",
            c=settings.component_colors[c],
            transform=ax.transAxes)
        ax.plot([age_diff.mean()] * 2,
                ax.get_ylim(), ls="--", lw=0.75,
                color=settings.component_colors[c],
                zorder=5)
        r = Rectangle(
            (age_diff.mean() - age_diff.std(), ax.get_ylim()[0]),
            2 * age_diff.std(),
            np.diff(ax.get_ylim()),
            fill=True, alpha=0.15, zorder=5, lw=0,
            color=settings.component_colors[c])
        ax.add_patch(r)

    ax.plot([0] * 2, ax.get_ylim(), ls="--", lw=0.75, color='k', zorder=10)

    fig.savefig(
        f"images/stellar_formation_time/"
        f"stellar_age_sample_component_comparison{config['FILE_SUFFIX']}.pdf")
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
    plot_sample_stats(sample=sample, config=config)


if __name__ == "__main__":
    main()
