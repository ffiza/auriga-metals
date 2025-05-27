"""
File:           decomposition_mass_evolution.py
Description:    Creates figures of the temporal evolution of the mass of the 
                galaxies and each of their components.

Usage:          python scripts/visualization/decomposition_mass_evolution.py \
--config CONFIG_FILE

Arguments:
    --config        Configuration filename for input parameters
                    (e.g., '02').
"""
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import argparse

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup


def create_figure_all_panels(config: dict):
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


def create_figure_panels_by_component(config: dict):
    HIGHLIGHT = 6
    settings = Settings()
    SAMPLE = [f"au{i}_or_l4" for i in settings.groups["Included"]]

    fig = plt.figure(figsize=(8.0, 3.5))
    gs = fig.add_gridspec(nrows=2, ncols=5, hspace=0.1, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0, 0].set_ylabel(r"$M_\star$ [$10^{10} \ \mathrm{M}_\odot$]")
    axs[1, 0].set_ylabel(r"$M_\star / M_\star (z=0)$")

    for ax in axs.flat:
        ax.tick_params(which='both', direction="in")
        ax.set_xlim((0, 14))
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        if ax.get_subplotspec().is_first_row():
            ax.set_ylim(0, 10)
            ax.set_yticks([1, 3, 5, 7, 9])
            ax.set_yticklabels(["1.0", "3.0", "5.0", "7.0", "9.0"])
        else:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        # ax.set_axisbelow(True)
        ax.set_xlabel("Time [Gyr]")
        ax.label_outer()

    for i, simulation in enumerate(SAMPLE):
        galaxy = parse(simulation)[0]
        label = f"Au{galaxy}" if galaxy == HIGHLIGHT else None
        zorder = 10 if galaxy != HIGHLIGHT else 11

        df = pd.read_csv(
            f"results/{simulation}/"
            f"decomposition_mass_evolution{config['FILE_SUFFIX']}.csv")
        norm = df["Mass_Msun"].iloc[-1]

        #region AllStars
        color = "silver" if galaxy != HIGHLIGHT else "black"
        axs[0, 0].plot(
            df["Time_Gyr"], df[f"Mass_Msun"] / 1E10, lw=1.0,
            color=color, label=label, zorder=zorder)
        axs[1, 0].plot(
            df["Time_Gyr"], df[f"Mass_Msun"].to_numpy() / norm, lw=1.0,
            color=color, label=label, zorder=zorder)
        #endregion

        for j, c in enumerate(settings.components):
            color = "silver" if galaxy != HIGHLIGHT else \
                settings.component_colors[c]
            axs[0, j + 1].plot(
                df["Time_Gyr"], df[f"Mass_{c}_Msun"] / 1E10, lw=1.0,
                color=color, label=label, zorder=zorder)
            axs[1, j + 1].plot(
                df["Time_Gyr"], df[f"Mass_{c}_Msun"].to_numpy() / norm, lw=1.0,
                color=color, label=label, zorder=zorder)
    
    axs[0, 0].text(
        x=0.05, y=0.9, size=7.0, s=r"$\textbf{All}$",
        ha='left', va='center', transform=axs[0, 0].transAxes)
    axs[1, 0].text(
        x=0.05, y=0.9, size=7.0, s=r"$\textbf{All}$",
        ha='left', va='center', transform=axs[1, 0].transAxes)
    for j, c in enumerate(settings.components):
        c_text = settings.component_labels[c]
        axs[0, j + 1].text(
            x=0.05, y=0.9, size=7.0, s=r"$\textbf{" + c_text + "}$",
            ha='left', va='center', transform=axs[0, j + 1].transAxes,
            color=settings.component_colors[c])
        axs[1, j + 1].text(
            x=0.05, y=0.9, size=7.0, s=r"$\textbf{" + c_text + "}$",
            ha='left', va='center', transform=axs[1, j + 1].transAxes,
            color=settings.component_colors[c])

    for ax in axs.flatten():
        ax.legend(loc="center left", framealpha=0, fontsize=5.0,
                      bbox_to_anchor=(0.02, 0.8))

    fig.savefig(
        f"images/decomposition_mass_evolution/"
        f"included_panelsbycomponent{config['FILE_SUFFIX']}.pdf")
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
    create_figure_all_panels(config)
    create_figure_panels_by_component(config)


if __name__ == "__main__":
    main()