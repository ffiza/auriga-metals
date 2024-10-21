import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import json
import argparse
from decimal import Decimal

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup
from auriga.mathematics import round_to_1, get_decimal_places
from auriga.support import float_to_latex

REF_PATHS = ["data/lemasle_2007.json",
             "data/lemasle_2008.json",
             "data/genovali_2014.json",
             "data/lemasle_2018.json"]
REF_COLORS = ["orange", "green", "purple", "blue"]


def plot_abundance_profile(sample: list, config: dict):
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
        ax.set_xlim(0, 16)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
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
                zorder=15)

        ax.text(
            x=0.95, y=0.95, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha="right", va="top", transform=ax.transAxes)

        disc_radius = gal_data["DiscRadius_kpc"][gal_data["Galaxy"] == galaxy]
        ax.text(
            x=0.05, y=0.95, size=6.0,
            s=r"$R_\mathrm{d}$: " + f"{disc_radius.values[0]} kpc",
            ha="left", va="top", transform=ax.transAxes)
        
        # region LinearRegression
        with open(f"results/{simulation}/FeH_abundance_profile_stars_"
                  f"fit{config['FILE_SUFFIX']}.json",
                  'r') as f:
            lreg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"],
            df["CylindricalRadius_ckpc"] * lreg["slope"] + lreg["intercept"],
            color=settings.component_colors["CD"], ls="--", lw=1.0,
            label="This Work")
        # endregion

        # region LiteratureFits
        for i in range(len(REF_PATHS)):
            ref_path = REF_PATHS[i]
            ref_color = REF_COLORS[i]
            with open(ref_path, 'r') as f:
                reg = json.load(f)
                # Define intercept
                min_radius = config["ABUNDANCE_PROFILE_FIT"]["MIN_RADIUS_CKPC"]
                max_radius = config["ABUNDANCE_PROFILE_FIT"]["MAX_RADIUS_CKPC"]
                xm = np.mean([min_radius, max_radius])
                ym = lreg["intercept"] + lreg["slope"] * xm
                intercept = ym - reg["SlopeValue"] * xm
                ax.plot(
                    ax.get_xlim(),
                    np.array(ax.get_xlim()) * reg["SlopeValue"] + intercept,
                    color=ref_color, ls="--", lw=0.5, label=reg["Label"])
        # endregion

    axs[1, 3].legend(loc="lower left", framealpha=0, fontsize=3.5,
                     bbox_to_anchor=(0.05, 0.05), borderpad=0,
                     borderaxespad=0)

    fig.savefig(
        f"images/abundance_profiles/FeH_included{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_fit_stats(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(-0.1, 0.02)
    ax.set_ylim(-0.53, 0.02)
    ax.set_xlabel(r"$\nabla \mathrm{[Fe/H]}$ [dex/ckpc]")
    ax.set_yticks([- i * 0.02 for i in range(28)])
    ax.set_yticklabels([])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    for i, simulation in enumerate(sample):
        with open(f"results/{simulation}/"
                  f"FeH_abundance_profile_stars_fit{config['FILE_SUFFIX']}"
                  f".json",
                  'r') as f:
            lreg = json.load(f)
            ax.errorbar(
                lreg["slope"], 0 - 0.02 * i,
                xerr=lreg["stderr"], color="gray",
                markeredgecolor="white", capsize=2, capthick=1,
                marker='o', markersize=4, linestyle='none', zorder=10)
            galaxy = parse(sample[i])[0]
            ax.text(
                x=-0.105, y=0 - 0.02 * i, size=6.0, color="gray",
                ha="right", va="center", s=f"Au{galaxy}")
            ax.annotate('', xy=(0.02, 0 - 0.02 * i),
                        xytext=(0.08, 0 - 0.02 * i),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
            slope_err = str(Decimal(str(round_to_1(lreg["stderr"]))))
            slope_val = np.round(lreg["slope"], get_decimal_places(slope_err))
            ax.text(x=0.04, y=0 - 0.02 * i, size=6.0, color="black",
                    ha="center", va="bottom",
                    s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                        + f"{slope_err}")
            pvalue_str = str(np.round(lreg["pvalue"], 2)) \
                if lreg["pvalue"] >= 0.001 else r"$<0.001$"
            ax.text(x=0.07, y=0 - 0.02 * i, size=6.0, color="black",
                    ha="center", va="bottom", s=pvalue_str)

    ax.text(x=0.04, y=0.02, size=6.0, color="black",
            ha="center", va="bottom", s="Slope [dex/ckpc]")
    ax.text(x=0.07, y=0.02, size=6.0, color="black",
            ha="center", va="bottom", s=f"$p$-value")

    # region LiteratureFits
    for i in range(len(REF_PATHS)):
        ref_path = REF_PATHS[i]
        ref_color = REF_COLORS[i]
        with open(ref_path, 'r') as f:
            reg = json.load(f)
            ax.errorbar(
                reg["SlopeValue"], - 0.02 * (23 + i),
                xerr=reg["SlopeErrValue"], markeredgecolor="white", capsize=2,
                capthick=1, color=ref_color, marker='o', markersize=4,
                linestyle='none', zorder=10)
            ax.text(x=-0.105, y=- 0.02 * (23 + i), size=6.0, color=ref_color,
                    ha="right", va="center", s=reg["Label"])
            ax.annotate('', xy=(0.02, - 0.02 * (23 + i)),
                        xytext=(0.08, - 0.02 * (23 + i)),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
            slope_str = float_to_latex(reg["SlopeValue"]) + " $\pm$ " \
                        + str(reg["SlopeErrValue"])
            ax.text(x=0.04, y=- 0.02 * (23 + i), size=6.0, color=ref_color,
                    ha="center", va="bottom", s=slope_str)
    # endregion

    ax.plot([0] * 2, ax.get_ylim(), ls="--", lw=0.75, color='k', zorder=10)

    fig.savefig(
        f"images/abundance_profiles/gradients{config['FILE_SUFFIX']}.pdf")
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
    # plot_abundance_profile(sample, config)
    plot_fit_stats(sample, config)


if __name__ == "__main__":
    main()
