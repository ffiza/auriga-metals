import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import argparse
import json

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup
from auriga.support import float_to_latex


def plot_density_profile(sample: list, config: dict):
    settings = Settings()

    # region ReadDiscSize
    gal_data = pd.read_csv("data/iza_2022.csv")
    # endregion

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flat:
        ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim(0, 50)
        ax.set_ylim(1, 2E4)
        ax.set_xticks([10, 20, 30, 40])
        ax.set_yscale("log")
        ax.set_yticks([1E1, 1E2, 1E3, 1E4])
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r"$r_{xy}$ [ckpc]")
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r"$\Sigma$ [$\mathrm{M}_\odot \, \mathrm{pc}^{-2}$]")

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        df = pd.read_csv(
            f"results/{simulation}/"
            f"density_profile{config['FILE_SUFFIX']}.csv")
        ax = axs.flatten()[i]

        ax.plot(df["CylindricalRadius_ckpc"],
                df["SurfaceDensity_Type0_Msun/ckpc^2"],
                lw=1.0, color="black", ls=":",
                zorder=14, label="All Gas")
        ax.plot(df["CylindricalRadius_ckpc"],
                df["SurfaceDensity_Type4_Msun/ckpc^2"],
                lw=1.0, color="black", ls="-",
                zorder=15, label="All Stars")
        ax.plot(df["CylindricalRadius_ckpc"],
                df["SurfaceDensity_Type4_CD_Msun/ckpc^2"],
                lw=1.0, color=settings.component_colors["CD"], ls="-",
                zorder=16, label="Cold Disc Stars")

        ax.text(
            x=0.95, y=0.95, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha="right", va="top", transform=ax.transAxes)

        disc_radius = gal_data["DiscRadius_kpc"][gal_data["Galaxy"] == galaxy]
        ax.text(
            x=0.05, y=0.95, size=6.0,
            s=r"$R_\mathrm{d}$: " + f"{disc_radius.values[0]} kpc",
            ha="left", va="top", transform=ax.transAxes)
        
    axs[5, 2].legend(loc="lower left", framealpha=0, fontsize=5.0,
                     bbox_to_anchor=(1.05, 0.05), borderpad=0,
                     borderaxespad=0)

    fig.savefig(
        f"images/density_profiles/density_profiles{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_density_profile_with_abundance(sample: list, config: dict):
    settings = Settings()
    gal_data = pd.read_csv("data/iza_2022.csv")

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flat:
        # ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(1, 2E4)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yscale("log")
        ax.set_yticks([1E1, 1E2, 1E3, 1E4])
        ax.set_axisbelow(True)

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        ax = axs.flatten()[i]
        disc_radius = gal_data["DiscRadius_kpc"][gal_data["Galaxy"] == galaxy]
        disc_radius = disc_radius.values[0]

        #region Grid
        for x in [0.2, 0.4, 0.6, 0.8]:
            ax.plot([x] * 2, ax.get_ylim(), lw=0.25, color="gainsboro",
                    ls="-", zorder=-20)
        #endregion

        #region Surfance Density
        df = pd.read_csv(
            f"results/{simulation}/"
            f"density_profile{config['FILE_SUFFIX']}.csv")
        ax.plot(df["CylindricalRadius_ckpc"] / disc_radius,
                df["SurfaceDensity_Type4_CD_Msun/ckpc^2"],
                lw=1.0, color=settings.component_colors["CD"], ls="-",
                zorder=16, label=r"$\Sigma_\mathrm{CD}$")
        #endregion

        #region Abundance Profile
        ax2 = ax.twinx()
        ax2.set_ylim(-0.6, 0.6)
        ax2.set_yscale("linear")
        ax2.tick_params(axis="y", direction="in")
        ax2.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        # ax2.grid(True, ls='--', lw=0.25, c="gainsboro")
        ax2.set_zorder(1.0)
        df = pd.read_csv(
            f"results/{simulation}/"
            f"abundance_profile{config['FILE_SUFFIX']}.csv")
        ax2.plot(df["CylindricalRadius_ckpc"] / disc_radius,
                 df["[Fe/H]_CD_Stars"],
                 lw=1.0, color=settings.component_colors["CD"], ls="--",
                 zorder=15, label="[Fe/H]$_\mathrm{CD}$")
        #endregion
    
        # region LinearRegression
        with open(f"results/{simulation}/FeH_abundance_profile_stars_"
                  f"fit{config['FILE_SUFFIX']}.json",
                  'r') as f:
            lreg = json.load(f)
        ax2.plot(
            df["CylindricalRadius_ckpc"] / disc_radius,
            df["CylindricalRadius_ckpc"] * lreg["slope"] + lreg["intercept"],
            color="black", ls="--", lw=0.5)
        ax2.text(x=0.05, y=0.05, size=6.0, ha="left", va="bottom",
                 s=r"$\nabla \mathrm{[Fe/H]} = $ " \
                    + float_to_latex(np.round(lreg['slope'], 3)) \
                        + " $\mathrm{kpc}^{-1}$",
                 transform=ax2.transAxes)
        # endregion

        ax2.text(
            x=0.95, y=0.95, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha="right", va="top", transform=ax.transAxes)

        ax2.text(
            x=0.05, y=0.95, size=6.0,
            s=r"$R_\mathrm{d}$: " + f"{disc_radius} kpc",
            ha="left", va="top", transform=ax.transAxes, zorder=20)
        # ax2.plot([disc_radius] * 2, ax2.get_ylim(), ls=":", lw=1.0,
        #          color="black")

        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r"$r_{xy} / R_\mathrm{d}$")
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r"$\Sigma$ [$\mathrm{M}_\odot \, \mathrm{pc}^{-2}$]")
        if ax.get_subplotspec().is_last_col() or ax == axs[5, 2]:
            ax2.set_ylabel("[Fe/H]")
            ax2.set_yticklabels([r"$-0.4$", r"$-0.2$", 0, 0.2, 0.4])
        else:
            ax2.set_yticklabels([])
        
        if i == len(sample) - 1:
            ax.legend(loc="lower left", framealpha=0, fontsize=6.0,
                      bbox_to_anchor=(1.5, 0.5), borderpad=0,
                      borderaxespad=0)
            ax2.legend(loc="lower left", framealpha=0, fontsize=6.0,
                       bbox_to_anchor=(1.5, 0.4), borderpad=0,
                       borderaxespad=0)

    fig.savefig(
        "images/density_profiles/"
        f"density_profiles_with_abundance_norm{config['FILE_SUFFIX']}.pdf")
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
    plot_density_profile(sample, config)
    plot_density_profile_with_abundance(sample, config)


if __name__ == "__main__":
    main()
