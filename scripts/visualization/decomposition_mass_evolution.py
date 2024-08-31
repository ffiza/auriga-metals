import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import argparse

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup


def plot_decomposition_mass_evolution(config: dict):
    settings = Settings()
    SAMPLE = [f"au{i}_or_l4" for i in settings.groups["Included"]]

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flat:
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim((0, 14))
        ax.set_ylim(0, 6)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel("Time [Gyr]")
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r"$M_\star$ [$10^{10} \ \mathrm{M}_\odot$]")

    for i, simulation in enumerate(SAMPLE):
        galaxy = parse(simulation)[0]
        df = pd.read_csv(
            f"results/{simulation}/"
            f"decomposition_mass_evolution{config['FILE_SUFFIX']}.csv")
        ax = axs.flatten()[i]

        for c in settings.components:
            ax.plot(df["Time_Gyr"], df[f"Mass_{c}_Msun"] / 1E10, lw=1.0,
                    color=settings.component_colors[c],
                    label=settings.component_labels[c], zorder=15)

        ax.text(
            x=0.95, y=0.95, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha='right', va='top', transform=ax.transAxes)

        if i == 0:
            ax.legend(loc="upper left", framealpha=0, fontsize=5.0)

    fig.savefig(
        f"images/decomposition_mass_evolution/"
        f"included{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main():
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Create figures
    figure_setup()
    plot_decomposition_mass_evolution(config)


if __name__ == "__main__":
    main()