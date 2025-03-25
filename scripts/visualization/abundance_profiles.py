import numpy as np
import pandas as pd
import yaml
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import argparse
from decimal import Decimal
from scipy.stats import pearsonr
from scipy.stats import binned_statistic

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup
from auriga.mathematics import round_to_1, get_decimal_places
from auriga.support import float_to_latex

REF_PATHS = ["data/lemasle_2007.json",
             "data/lemasle_2008.json",
             "data/luck_2011.json",
             "data/genovali_2014.json",
             "data/lemasle_2018.json"]
REF_COLORS = ["orange", "green", "red", "purple", "blue"]


def plot_iron_abundance_profile(sample: list, config: dict):
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
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([.2, .4, .6, .8])
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.grid(True, ls='-', lw=0.25, c="gainsboro")
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r"$r_{xy} / R_\mathrm{d}$")
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("[Fe/H]")


    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        df = pd.read_csv(
            f"results/{simulation}/"
            f"abundance_profile{config['FILE_SUFFIX']}.csv")
        ax = axs.flatten()[i]
        disc_radius = gal_data[
            "DiscRadius_kpc"][gal_data["Galaxy"] == galaxy].values[0]
        min_radius_frac = config["AB_PROF_FIT_MIN_DSC_RAD_FRAC"]
        max_radius_frac = config["AB_PROF_FIT_MAX_DSC_RAD_FRAC"]

        ax.plot(df["CylindricalRadius_ckpc"] / disc_radius,
                df["[Fe/H]_CD_Stars"],
                lw=1.0, color=settings.component_colors["CD"],
                zorder=15, label="Data")
        # ax.fill_between(
        #     x=df["CylindricalRadius_ckpc"],
        #     y1=df["[Fe/H]_CD_Stars"] - df["[Fe/H]_CD_Stars_Std"],
        #     y2=df["[Fe/H]_CD_Stars"] + df["[Fe/H]_CD_Stars_Std"],
        #     color=settings.component_colors["CD"], zorder=1, alpha=0.3, lw=0)

        ax.text(
            x=0.95, y=0.95, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha="right", va="top", transform=ax.transAxes)

        ax.text(
            x=0.05, y=0.05, size=6.0,
            s=r"$R_\mathrm{d}$: " + f"{disc_radius} kpc",
            ha="left", va="bottom", transform=ax.transAxes)
        ax.fill_between(
            x=[min_radius_frac, max_radius_frac],
            y1=[ax.get_ylim()[0]] * 2,
            y2=[ax.get_ylim()[1]] * 2,
            color="black", zorder=1, alpha=0.05, lw=0)

        # region LinearRegression
        with open(f"results/{simulation}/FeH_abundance_profile_stars_"
                  f"fit{config['FILE_SUFFIX']}.json",
                  'r') as f:
            lreg = json.load(f)
        ax.plot(
            df["CylindricalRadius_ckpc"] / disc_radius,
            df["CylindricalRadius_ckpc"] * lreg["slope"] + lreg["intercept"],
            color=settings.component_colors["CD"], ls="--", lw=0.5,
            label="Regression")
        # endregion

        # # region LiteratureFits
        # for i in range(len(REF_PATHS)):
        #     ref_path = REF_PATHS[i]
        #     ref_color = REF_COLORS[i]
        #     with open(ref_path, 'r') as f:
        #         reg = json.load(f)
        #         # Define intercept
        #         xm = np.mean([min_radius_frac * disc_radius,
        #                       max_radius_frac * disc_radius])
        #         ym = lreg["intercept"] + lreg["slope"] * xm
        #         intercept = ym - reg["SlopeValue"] * xm
        #         ax.plot(
        #             ax.get_xlim(),
        #             np.array(np.array(ax.get_xlim()) * disc_radius) \
        #                 * reg["SlopeValue"] + intercept,
        #             color=ref_color, ls="--", lw=0.5, label=reg["Label"])
        # # endregion

    axs[0, 0].legend(loc="upper left", framealpha=0, fontsize=6.0)

    fig.savefig(
        f"images/abundance_profiles/FeH_included{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_iron_abundance_profile_one_panel(sample: list, config: dict):
    settings = Settings()

    # region ReadDiscSize
    gal_data = pd.read_csv("data/iza_2022.csv")
    # endregion

    fig = plt.figure(figsize=(2.5, 2.5))
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex=True, sharey=True)

    ax.tick_params(which='both', direction="in")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 0.45)
    ax.set_xticks([.2, .4, .6, .8])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.grid(True, ls='-', lw=0.25, c="gainsboro")
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$r_{xy} / R_\mathrm{d}$")
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("[Fe/H]")

    ax.fill_between(
        x=[0.2, 1],
        y1=[ax.get_ylim()[0]] * 2,
        y2=[ax.get_ylim()[1]] * 2,
        color="black", zorder=1, alpha=0.05, lw=0)

    sample_x, sample_y = [], []
    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        df = pd.read_csv(
            f"results/{simulation}/"
            f"abundance_profile{config['FILE_SUFFIX']}.csv")
        disc_radius = gal_data[
            "DiscRadius_kpc"][gal_data["Galaxy"] == galaxy].values[0]

        if simulation == "au6_or_l4":
            zorder = 10
            color = settings.component_colors["CD"]
            label = "Au6"
        else:
            zorder = 5
            color = "gainsboro"
            label = None

        ax.plot(df["CylindricalRadius_ckpc"] / disc_radius,
                df["[Fe/H]_CD_Stars"],
                lw=1.0, color=color, zorder=zorder, label=label)

        sample_x += list(df["CylindricalRadius_ckpc"].to_numpy() / disc_radius)
        sample_y += list(df["[Fe/H]_CD_Stars"].to_list())

        # if simulation == "au6_or_l4":
        #     ax.plot(df["CylindricalRadius_ckpc"] / disc_radius,
        #             df["[Fe/H]_CD_Stars"] + df["[Fe/H]_CD_Stars_Std"],
        #             lw=1.0, color=color, zorder=zorder, ls="--")
        #     ax.plot(df["CylindricalRadius_ckpc"] / disc_radius,
        #             df["[Fe/H]_CD_Stars"] - df["[Fe/H]_CD_Stars_Std"],
        #             lw=1.0, color=color, zorder=zorder, ls="--")

        #region LinearRegression
        if simulation == "au6_or_l4":
            with open(f"results/{simulation}/FeH_abundance_profile_stars_"
                    f"fit{config['FILE_SUFFIX']}.json",
                    'r') as f:
                lreg = json.load(f)
            ax.plot(
                df["CylindricalRadius_ckpc"] / disc_radius,
                df["CylindricalRadius_ckpc"] * lreg["slope"] \
                    + lreg["intercept"],
                color=settings.component_colors["CD"], ls="--", lw=0.5,
                label="Au6 Regression", zorder=15)
        #endregion

    #region SampleStats
    median, bin_edges, _ = binned_statistic(
        x=sample_x, values=sample_y, statistic=np.median, bins=40, range=(0, 1)
    )
    bin_centers = bin_edges[1:] - np.diff(bin_edges) / 2
    ax.plot(
        bin_centers, median, color="black",
        ls="-.", lw=0.75, label="Sample Median", zorder=15)
    low_percentile, _, _ = binned_statistic(
        x=sample_x, values=sample_y, statistic=partial(np.percentile, q=25),
        bins=40, range=(0, 1)
    )
    ax.plot(
        bin_centers, low_percentile, color="black",
        ls=":", lw=0.75, label=r"Sample 25$^\mathrm{th}$ Perc.", zorder=15)
    high_percentile, _, _ = binned_statistic(
        x=sample_x, values=sample_y, statistic=partial(np.percentile, q=75),
        bins=40, range=(0, 1)
    )
    ax.plot(
        bin_centers, high_percentile, color="black",
        ls=":", lw=0.75, label=r"Sample 75$^\mathrm{th}$ Perc.", zorder=15)
    #endregion

    ax.legend(loc="upper right", framealpha=0, fontsize=5)

    fig.savefig(
        f"images/abundance_profiles/"
        f"FeH_included_onepanel{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_oxygen_abundance_profile(sample: list, config: dict):
    settings = Settings()

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
            ax.set_ylabel("Abundance")

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        df = pd.read_csv(
            f"results/{simulation}/"
            f"abundance_profile{config['FILE_SUFFIX']}.csv")
        ax = axs.flatten()[i]

        ax.plot(df["CylindricalRadius_ckpc"], df["[Fe/H]_CD_Stars"],
                lw=1.0, color=settings.component_colors["CD"],
                zorder=15, ls="-", label="[Fe/H]")        
        ax.plot(df["CylindricalRadius_ckpc"], df["[O/H]_CD_Stars"],
                lw=1.0, color=settings.component_colors["CD"],
                zorder=15, ls="--", label="[O/H]")

        ax.text(
            x=0.95, y=0.95, size=7.0,
            s=r"$\texttt{" + f"Au{galaxy}" + "}$",
            ha="right", va="top", transform=ax.transAxes)

    axs[1, 3].legend(loc="lower left", framealpha=0, fontsize=5.0,
                     bbox_to_anchor=(0.05, 0.05), borderpad=0,
                     borderaxespad=0)

    fig.savefig(
        f"images/abundance_profiles/OH_included{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_fit_stats(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(-0.09, 0)
    ax.set_ylim(-0.56, 0.02)
    ax.set_xlabel(r"$\nabla \mathrm{[Fe/H]}$ [dex/ckpc]")
    ax.set_yticks([- i * 0.02 for i in range(28)])
    ax.set_yticklabels([])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    sample_slopes = []
    for i, simulation in enumerate(sample):
        with open(f"results/{simulation}/"
                  f"FeH_abundance_profile_stars_fit{config['FILE_SUFFIX']}"
                  f".json",
                  'r') as f:
            lreg = json.load(f)
            ax.errorbar(
                lreg["slope"], 0 - 0.02 * i,
                xerr=lreg["stderr"], color="gray", markeredgewidth=0.75,
                markeredgecolor="white", capsize=2, capthick=1,
                marker='o', markersize=4, linestyle='none', zorder=10)
            galaxy = parse(sample[i])[0]
            ax.text(
                x=-0.0925, y=0 - 0.02 * i, size=6.0, color="gray",
                ha="right", va="center", s=f"Au{galaxy}")
            ax.annotate('', xy=(0, 0 - 0.02 * i),
                        xytext=(0.06, 0 - 0.02 * i),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
            slope_err = str(Decimal(str(round_to_1(lreg["stderr"]))))
            slope_val = np.round(lreg["slope"], get_decimal_places(slope_err))
            ax.text(x=0.02, y=0 - 0.02 * i, size=6.0, color="black",
                    ha="center", va="bottom",
                    s=r"$-$" + f"{np.abs(slope_val)}" + " $\pm$ " \
                        + f"{slope_err}")
            pvalue_str = str(np.round(lreg["pvalue"], 2)) \
                if lreg["pvalue"] >= 0.001 else r"$<0.001$"
            ax.text(x=0.05, y=0 - 0.02 * i, size=6.0, color="black",
                    ha="center", va="bottom", s=pvalue_str)
            sample_slopes.append(lreg["slope"])
    ax.plot([np.mean(sample_slopes)] * 2, ax.get_ylim(), ls="--",
            lw=0.75, color='gray', zorder=10)
    r = Rectangle((np.mean(sample_slopes) - np.std(sample_slopes),
                   ax.get_ylim()[0]),
                  2 * np.std(sample_slopes),
                  np.diff(ax.get_ylim()),
                  color="black", alpha=0.075, zorder=-10, lw=0)
    ax.add_patch(r)
    stat_str = str(np.abs(np.round(np.mean(sample_slopes), 4)))
    ax.text(x=-0.011, y=-0.555, size=6.0, color="gray",
            ha="left", va="bottom", rotation=90,
            s=r"$-$" + stat_str + " dex/ckpc")

    ax.text(x=0.02, y=0.02, size=6.0, color="black",
            ha="center", va="bottom", s="Slope [dex/ckpc]")
    ax.text(x=0.05, y=0.02, size=6.0, color="black",
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
                linestyle='none', zorder=10, markeredgewidth=0.75)
            ax.text(x=-0.0925, y=- 0.02 * (23 + i), size=6.0, color=ref_color,
                    ha="right", va="center", s=reg["Label"])
            ax.annotate('', xy=(0, - 0.02 * (23 + i)),
                        xytext=(0.06, - 0.02 * (23 + i)),
                        arrowprops=dict(
                            arrowstyle="-", color='gainsboro', lw=0.25))
            slope_str = float_to_latex(reg["SlopeValue"]) + " $\pm$ " \
                        + str(reg["SlopeErrValue"])
            ax.text(x=0.02, y=- 0.02 * (23 + i), size=6.0, color=ref_color,
                    ha="center", va="bottom", s=slope_str)
    # endregion

    fig.savefig(
        f"images/abundance_profiles/gradients{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_fit_vs_insideoutparam(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3, 3), ncols=1)

    ax.grid(True, ls='-', lw=0.25, c='gainsboro')
    ax.set_axisbelow(True)
    ax.set_xlim(-0.04, 0.01)
    ax.set_ylim(-1.5, 4.5)
    ax.set_xlabel(r"$\nabla \mathrm{[Fe/H]}$ [dex/ckpc]")
    ax.set_ylabel(r"$\eta_\mathrm{Net}$ [Gyr]")

    data = np.nan * np.ones((len(sample), 4))

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        with open(f"results/{simulation}/"
                  f"FeH_abundance_profile_stars_fit{config['FILE_SUFFIX']}"
                  f".json",
                  'r') as f:
            d = json.load(f)
            data[i, 0] = d["slope"]
            data[i, 1] = d["stderr"]
        with open("data/iza_2024.json", "r") as f:
            d = json.load(f)
            data[i, 2] = d["InsideOutParameter_Gyr"][f"Au{galaxy}"]
            data[i, 3] = d["InsideOutParameterError_Gyr"][f"Au{galaxy}"]

    ax.errorbar(data[:, 0], data[:, 2], xerr=data[:, 1], yerr=data[:, 3],
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

    fig.savefig(
        f"images/abundance_profiles/"
        f"gradients_vs_insideout{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_fit_vs_barfraction(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3, 3), ncols=1)

    ax.grid(True, ls='-', lw=0.25, c='gainsboro')
    ax.set_axisbelow(True)
    ax.set_xlim(-0.04, 0.01)
    ax.set_ylim(-1.5, 4.5)
    ax.set_xlabel(r"$\nabla \mathrm{[Fe/H]}$ [dex/ckpc]")
    ax.set_ylabel(r"$\eta_\mathrm{Net}$ [Gyr]")

    data = np.nan * np.ones((len(sample), 4))

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        with open(f"results/{simulation}/"
                  f"FeH_abundance_profile_stars_fit{config['FILE_SUFFIX']}"
                  f".json",
                  'r') as f:
            d = json.load(f)
            data[i, 0] = d["slope"]
            data[i, 1] = d["stderr"]
        with open("data/iza_2024.json", "r") as f:
            d = json.load(f)
            data[i, 2] = d["InsideOutParameter_Gyr"][f"Au{galaxy}"]
            data[i, 3] = d["InsideOutParameterError_Gyr"][f"Au{galaxy}"]

    ax.errorbar(data[:, 0], data[:, 2], xerr=data[:, 1], yerr=data[:, 3],
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

    fig.savefig(
        f"images/abundance_profiles/"
        f"gradients_vs_insideout{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_fit_vs_disc_radius(sample: list, config: dict):
    fig, ax = plt.subplots(figsize=(3, 3), ncols=1)

    ax.grid(True, ls='-', lw=0.25, c='gainsboro')
    ax.set_axisbelow(True)
    ax.set_xlim(-0.04, 0.01)
    ax.set_ylim(0, 40)
    ax.set_xlabel(r"$\nabla \mathrm{[Fe/H]}$ [dex/ckpc]")
    ax.set_ylabel(r"$R_\mathrm{d}$ [kpc]")

    data = np.nan * np.ones((len(sample), 4))

    for i, simulation in enumerate(sample):
        galaxy = parse(simulation)[0]
        with open(f"results/{simulation}/"
                  f"FeH_abundance_profile_stars_fit{config['FILE_SUFFIX']}"
                  f".json",
                  'r') as f:
            d = json.load(f)
            data[i, 0] = d["slope"]
            data[i, 1] = d["stderr"]
        
        df = pd.read_csv("data/iza_2022.csv", index_col="Galaxy")
        data[i, 2] = df["DiscRadius_kpc"][galaxy]

    ax.scatter(data[:, 0], data[:, 2],
               edgecolor="white", color="black",
               marker='o', s=15, lw=0, zorder=10)

    corr = pearsonr(data[:, 0], data[:, 2])
    ax.text(x=0.05, y=0.95, size=8.0, color="black",
            ha="left", va="center", s=r"$r$: " \
                + str(np.round(corr[0], 2)),
            transform=ax.transAxes)
    ax.text(x=0.05, y=0.9, size=8.0, color="black",
            ha="left", va="center", s=r"$p$-value: " \
                + str(np.round(corr[1], 3)),
            transform=ax.transAxes)

    fig.savefig(
        f"images/abundance_profiles/"
        f"gradients_vs_disc_radius{config['FILE_SUFFIX']}.pdf")
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
    # plot_iron_abundance_profile(sample, config)
    # plot_iron_abundance_profile_one_panel(sample, config)
    # plot_oxygen_abundance_profile(sample, config)
    plot_fit_stats(sample, config)
    # plot_fit_vs_insideoutparam(sample, config)
    # plot_fit_vs_disc_radius(sample, config)


if __name__ == "__main__":
    main()
