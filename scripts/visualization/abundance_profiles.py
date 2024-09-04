import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import json
import argparse

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup


def plot_abundance_profile(sample: list, config: dict, abundance: tuple):
    settings = Settings()

    # region ReadDiscSize
    gal_data = pd.read_csv("data/iza_2022.csv")
    # endregion

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flat:
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim(0, 50)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([10, 20, 30, 40])
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r"$r_{xy}$ [ckpc]")
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("[Fe/H]")

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        df = pd.read_csv(
            f"results/{simulation}/"
            f"abundance_profile{config['FILE_SUFFIX']}.csv")
        ax = axs.flatten()[i]

        ax.plot(df["CylindricalRadius_ckpc"], df["[Fe/H]_CD_Stars"],
                lw=1.0, color=settings.component_colors["CD"],
                zorder=15, label="Data")

        ax.text(
            x=1, y=-0.5, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha='left', va='center')

        disc_radius = gal_data["DiscRadius_kpc"][gal_data["Galaxy"] == galaxy]
        ax.plot([disc_radius] * 2, ax.get_ylim(), lw=1.0, ls=":", c="k")
        ax.text(
            x=disc_radius.values[0] + 2, y=0.3, size=6.0,
            s=r"$R_\mathrm{d}$" + "\n" + f"{disc_radius.values[0]} \n kpc",
            ha='left', va='top')
        
        # region LinearRegression
        with open(f"results/{simulation}/FeH_abundance_profile_fit.json",
                  'r') as f:
            lreg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"],
            df["CylindricalRadius_ckpc"] * lreg["slope"] + lreg["intercept"],
            color=settings.component_colors["CD"], ls="--", lw=1.0,
            label="This Work")
        # endregion

        # region LiteratureFits
        with open(f"data/lemasle_2018.json", 'r') as f:
            reg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"],
            df["CylindricalRadius_ckpc"] * reg["Data"]["LeastSquaresSlope"] + lreg["intercept"],
            color="blue", ls="--", lw=0.5, label="Lemasle et al. (2018)")
        with open(f"data/genovali_2014.json", 'r') as f:
            reg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"],
            df["CylindricalRadius_ckpc"] * reg["Data"]["Slope"] + lreg["intercept"],
            color="purple", ls="--", lw=0.5, label="Genovali et al. (2014)")
        with open(f"data/lemasle_2008.json", 'r') as f:
            reg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"],
            df["CylindricalRadius_ckpc"] * reg["Data"]["Slope"] + lreg["intercept"],
            color="green", ls="--", lw=0.5, label="Lemasle et al. (2008)")
        with open(f"data/lemasle_2007.json", 'r') as f:
            reg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"],
            df["CylindricalRadius_ckpc"] * reg["Data"]["Slope"] + lreg["intercept"],
            color="orange", ls="--", lw=0.5, label="Lemasle et al. (2007)")
        # endregion

    axs[1, 3].legend(loc="upper right", framealpha=0, fontsize=3.5)

    fig.savefig(
        f"images/abundance_profiles/"
        f"FeH_included{config['FILE_SUFFIX']}.pdf")
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
    plot_abundance_profile(sample, config, ("Fe", "H"))


if __name__ == "__main__":
    main()