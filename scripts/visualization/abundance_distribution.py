import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import json
import argparse
from decimal import Decimal
from scipy.stats import pearsonr

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup
from auriga.mathematics import round_to_1, get_decimal_places
from auriga.support import float_to_latex


def plot_sample_stats(sample: list, config: dict):
    settings = Settings()

    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(-1.0, 0.4)
    ax.set_ylim(-0.12, 0.02)
    ax.set_xlabel(r"$\mathrm{[Fe/H]}$")
    ax.set_yticks([- i * 0.02 for i in range(24)])
    ax.set_yticklabels([])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    df = pd.read_csv(
        f"results/abundance_distribution_medians{config['FILE_SUFFIX']}.csv",
        index_col=0)

    for i, simulation in enumerate(sample):
        galaxy, _, _ = parse(simulation)
        for j, c in enumerate(settings.components):
            ax.scatter(
                df.loc[f"Au{galaxy}"][f"MedianAbundance_Fe/H_{c}"],
                0 - 0.02 * i,
                color=settings.component_colors[c],
                marker='o', s=6, zorder=10)
        galaxy = parse(sample[i])[0]
        ax.text(
            x=-1.05, y=0 - 0.02 * i, size=6.0, color="gray",
            ha="right", va="center", s=f"Au{galaxy}")
    
    for j, c in enumerate(settings.components):
        ax.plot(df[f"MedianAbundance_Fe/H_{c}"].to_numpy(),
                [0 - 0.02 * i for i in range(len(sample))],
                lw=0.5, color=settings.component_colors[c])
        ax.text(
            x=0.02, y=0.98 - j * 0.03, size=6, ha="left", va="top",
            s=r"$\textbf{" + settings.component_labels[c] + "}$",
            c=settings.component_colors[c],
            transform=ax.transAxes)

    ax.plot([0] * 2, ax.get_ylim(), ls="--", lw=0.75, color='k', zorder=10)

    fig.savefig(
        f"images/metal_abundance_distribution/Fe_H/"
        f"sample_component_comparison{config['FILE_SUFFIX']}.pdf")
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
    plot_sample_stats(sample, config)


if __name__ == "__main__":
    main()