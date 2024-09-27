import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import json
import argparse

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup
from auriga.mathematics import round_to_1, get_decimal_places


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


def plot_fit_stats(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(-0.1, 0.02)
    ax.set_ylim(-0.53, 0.02)
    ax.set_xlabel(r"$\nabla \mathrm{[Fe/H]}$ [dex/ckpc]")
    ax.set_yticks([- i * 0.02 for i in range(28)])
    ax.set_yticklabels([])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    for i, simulation in enumerate(sample):
        with open(f"results/{simulation}/FeH_abundance_profile_fit.json",
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
                        xytext=(0.075, 0 - 0.02 * i),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
            slope_err = round_to_1(lreg["stderr"])
            slope_val = np.round(lreg["slope"], get_decimal_places(slope_err))
            ax.text(x=0.04, y=0 - 0.02 * i, size=6.0, color="black",
                    ha="center", va="bottom",
                    s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                        + f"{slope_err}")
            pvalue_str = str(np.round(lreg["pvalue"])) \
                if lreg["pvalue"] >= 0.001 else r"$<0.001$"
            ax.text(x=0.07, y=0 - 0.02 * i, size=6.0, color="black",
                    ha="center", va="bottom", s=pvalue_str)

    ax.text(x=0.04, y=0.02, size=6.0, color="black",
            ha="center", va="bottom", s="Slope [dex/ckpc]")
    ax.text(x=0.07, y=0.02, size=6.0, color="black",
            ha="center", va="bottom", s=f"$p$-value")

    with open(f"data/genovali_2014.json", 'r') as f:
        reg = json.load(f)
        ax.errorbar(
            reg["Data"]["Slope"], - 0.02 * 23, xerr=reg["Data"]["SlopeErr"],
            markeredgecolor="white", capsize=2, capthick=1, color="purple",
            marker='o', markersize=4, linestyle='none', zorder=10)
        ax.text(x=-0.105, y=- 0.02 * 23, size=6.0, color="purple",
                ha="right", va="center", s="Genovali et al. (2014)")
        ax.annotate('', xy=(0.02, - 0.02 * 23),
                        xytext=(0.075, - 0.02 * 23),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
        slope_err = reg["Data"]["SlopeErr"]
        slope_val = reg["Data"]["Slope"]
        ax.text(x=0.04, y=- 0.02 * 23, size=6.0, color="purple",
                ha="center", va="bottom",
                s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                    + f"{slope_err}")

    with open(f"data/lemasle_2007.json", 'r') as f:
        reg = json.load(f)
        ax.errorbar(
            reg["Data"]["Slope"], - 0.02 * 24, xerr=reg["Data"]["SlopeErr"],
            markeredgecolor="white", capsize=2, capthick=1, color="orange",
            marker='o', markersize=4, linestyle='none', zorder=10)
        ax.text(x=-0.105, y=- 0.02 * 24, size=6.0, color="orange",
                ha="right", va="center", s="Lemasle et al. (2007)")
        ax.annotate('', xy=(0.02, - 0.02 * 24),
                        xytext=(0.075, - 0.02 * 24),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
        slope_err = reg["Data"]["SlopeErr"]
        slope_val = reg["Data"]["Slope"]
        ax.text(x=0.04, y=- 0.02 * 24, size=6.0, color="orange",
                ha="center", va="bottom",
                s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                    + f"{slope_err}")

    with open(f"data/lemasle_2008.json", 'r') as f:
        reg = json.load(f)
        ax.errorbar(
            reg["Data"]["Slope"], - 0.02 * 25, xerr=reg["Data"]["SlopeErr"],
            markeredgecolor="white", capsize=2, capthick=1, color="green",
            marker='o', markersize=4, linestyle='none', zorder=10)
        ax.text(x=-0.105, y=- 0.02 * 25, size=6.0, color="green",
                ha="right", va="center", s="Lemasle et al. (2008)")
        ax.annotate('', xy=(0.02, - 0.02 * 25),
                        xytext=(0.075, - 0.02 * 25),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
        slope_err = reg["Data"]["SlopeErr"]
        slope_val = reg["Data"]["Slope"]
        ax.text(x=0.04, y=- 0.02 * 25, size=6.0, color="green",
                ha="center", va="bottom",
                s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                    + f"{slope_err}")

    with open(f"data/lemasle_2018.json", 'r') as f:
        reg = json.load(f)
        ax.errorbar(
            reg["Data"]["BootstrapSlope"], - 0.02 * 26,
            xerr=reg["Data"]["BootstrapSlopeErr"], color="blue",
            markeredgecolor="white", capsize=2, capthick=1,
            marker='o', markersize=4, linestyle='none', zorder=10)
        ax.text(x=-0.105, y=- 0.02 * 26, size=6.0, color="blue",
                ha="right", va="center", s="Lemasle et al. (2018)")
        ax.annotate('', xy=(0.02, - 0.02 * 26),
                        xytext=(0.075, - 0.02 * 26),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
        slope_err = reg["Data"]["BootstrapSlopeErr"]
        slope_val = reg["Data"]["BootstrapSlope"]
        ax.text(x=0.04, y=- 0.02 * 26, size=6.0, color="blue",
                ha="center", va="bottom",
                s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                    + f"{slope_err}")

    ax.plot([0] * 2, ax.get_ylim(), ls="--", lw=0.75, color='k', zorder=10)

    fig.savefig(
        f"images/abundance_profiles/gradients.pdf")
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