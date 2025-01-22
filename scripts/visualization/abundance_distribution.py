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

    def get_stellar_masses(self):
        gal_df = pd.read_csv(
            f"data/grand_2017.csv",
            usecols=["Run", "StellarMass_10^10Msun", "VirialMass_10^10Msun"])
        gal_df[["Galaxy", "Resolution"]] = \
            gal_df["Run"].str.extract(r'Au(\d+)_L(\d+)')
        gal_df["Galaxy"] = gal_df["Galaxy"].astype(int)
        gal_df["Resolution"] = gal_df["Resolution"].astype(int)
        gal_df = gal_df[gal_df['Galaxy'].isin(
            self.settings.groups["Included"])]
        gal_df = gal_df[gal_df["Resolution"] == 4]
        return gal_df["StellarMass_10^10Msun"].to_numpy()
    
    def get_virial_masses(self):
        gal_df = pd.read_csv(
            f"data/grand_2017.csv",
            usecols=["Run", "StellarMass_10^10Msun", "VirialMass_10^10Msun"])
        gal_df[["Galaxy", "Resolution"]] = \
            gal_df["Run"].str.extract(r'Au(\d+)_L(\d+)')
        gal_df["Galaxy"] = gal_df["Galaxy"].astype(int)
        gal_df["Resolution"] = gal_df["Resolution"].astype(int)
        gal_df = gal_df[gal_df['Galaxy'].isin(
            self.settings.groups["Included"])]
        gal_df = gal_df[gal_df["Resolution"] == 4]
        return gal_df["VirialMass_10^10Msun"].to_numpy()
    
    def get_disc_to_total(self):
        disc_to_total = pd.read_csv(
            f"data/iza_2022.csv",
            usecols=["Galaxy", "DiscToTotal"])
        disc_to_total = disc_to_total[disc_to_total['Galaxy'].isin(
            self.settings.groups["Included"])]
        return disc_to_total["DiscToTotal"].to_numpy()

    def get_abundance_gradients(self):
        slopes = []
        for simulation in [
            f"au{i}_or_l4" for i in self.settings.groups["Included"]]:
            with open(f"results/{simulation}/FeH_abundance_profile_stars_"
                                f"fit{self.config['FILE_SUFFIX']}.json",
                                'r') as f:
                            lreg = json.load(f)
            slopes.append(lreg["slope"])
        return np.array(slopes)
    
    def get_stellar_age_df(self):
        return pd.read_csv(
            f"results/stellar_age{self.config['FILE_SUFFIX']}.csv")
    
    def get_stellar_mass_fraction_df(self):
        return pd.read_csv(
            f"results/stellar_mass_fractions{self.config['FILE_SUFFIX']}.csv")


def plot_sample_stats(sample: list, of: str, to: str, config: dict,
                      xlim: tuple):
    settings = Settings()

    xlabel = r"$\mathrm{[" + f"{of}/{to}" \
        + r"]} - \mathrm{[" + f"{of}/{to}" + r"]}_\mathrm{CD}$"

    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.46, 0.02)
    ax.set_xlabel(xlabel)
    ax.set_yticks([- i * 0.02 for i in range(23)])
    galaxies = [parse(simulation)[0] for simulation in sample]
    ax.set_yticklabels([f"Au{i}" for i in galaxies])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    df = pd.read_csv(
        f"results/abundance_distribution_medians{config['FILE_SUFFIX']}.csv",
        index_col=0)

    for i, simulation in enumerate(sample):
        galaxy, _, _ = parse(simulation)
        for j, c in enumerate(["H", "B", "WD"]):
            ax.scatter(
                df.loc[f"Au{galaxy}"][f"MedianAbundance_{of}/{to}_{c}"] \
                    - df.loc[f"Au{galaxy}"][f"MedianAbundance_{of}/{to}_CD"],
                0 - 0.02 * i,
                color=settings.component_colors[c],
                marker='o', s=6, zorder=10)
        # galaxy = parse(sample[i])[0]
        # ax.text(
        #     x=ax.get_xlim()[0] - 0.01, y=0 - 0.02 * i, size=6.0, color="gray",
        #     ha="right", va="center", s=f"Au{galaxy}")
    
    for j, c in enumerate(["H", "B", "WD"]):
        abundance_diff = df[f"MedianAbundance_{of}/{to}_{c}"].to_numpy() \
            - df[f"MedianAbundance_{of}/{to}_CD"].to_numpy()
        ax.plot(abundance_diff,
                [0 - 0.02 * i for i in range(len(sample))],
                lw=0.5, color=settings.component_colors[c])
        ax.text(
            x=0.02, y=0.96 - j * 0.03, size=6, ha="left", va="top",
            s=r"$\textbf{" + settings.component_labels[c] + "}$",
            c=settings.component_colors[c],
            transform=ax.transAxes)
        ax.plot([abundance_diff.mean()] * 2,
                ax.get_ylim(), ls="--", lw=0.75,
                color=settings.component_colors[c],
                zorder=5)
        r = Rectangle(
            (abundance_diff.mean() - abundance_diff.std(), ax.get_ylim()[0]),
            2 * abundance_diff.std(),
            np.diff(ax.get_ylim()),
            fill=True, alpha=0.15, zorder=5, lw=0,
            color=settings.component_colors[c])
        ax.add_patch(r)

    ax.plot([0] * 2, ax.get_ylim(), ls="--", lw=0.75, color='k', zorder=10)

    fig.savefig(
        f"images/metal_abundance_distribution/{of}_{to}/"
        f"sample_component_comparison{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_sample_ordered_abundance(
        sample: list, of: str, to: str, config: dict, xlim: tuple) -> None:
    settings = Settings()

    xlabel = r"$\mathrm{[" + f"{of}/{to}" + r"]}$"

    fig, ax = plt.subplots(figsize=(3.5, 4.5), ncols=1)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.46, 0.02)
    ax.set_xlabel(xlabel)
    ax.set_yticks([- i * 0.02 for i in range(23)])
    galaxies = [parse(simulation)[0] for simulation in sample]
    ax.set_yticklabels([f"Au{i}" for i in galaxies])
    ax.grid(True, ls='-', lw=0.25, c='gainsboro')

    df = pd.read_csv(
        f"results/abundance_distribution_medians{config['FILE_SUFFIX']}.csv",
        index_col=0)

    for i, simulation in enumerate(sample):
        galaxy, _, _ = parse(simulation)
        for j, c in enumerate(settings.components):
            ax.scatter(
                df.loc[f"Au{galaxy}"][f"MedianAbundance_{of}/{to}_{c}"],
                0 - 0.02 * i,
                color=settings.component_colors[c],
                marker='o', s=6, zorder=10)
    
    for j, c in enumerate(settings.components):
        abundance = df[f"MedianAbundance_{of}/{to}_{c}"]
        ax.plot(abundance,
                [0 - 0.02 * i for i in range(len(sample))],
                lw=0.5, color=settings.component_colors[c])
        ax.text(
            x=0.02, y=0.96 - j * 0.03, size=6, ha="left", va="top",
            s=r"$\textbf{" + settings.component_labels[c] + "}$",
            c=settings.component_colors[c],
            transform=ax.transAxes)
        ax.plot([abundance.mean()] * 2,
                ax.get_ylim(), ls="--", lw=0.75,
                color=settings.component_colors[c],
                zorder=5)
        r = Rectangle(
            (abundance.mean() - abundance.std(), ax.get_ylim()[0]),
            2 * abundance.std(),
            np.diff(ax.get_ylim()),
            fill=True, alpha=0.15, zorder=5, lw=0,
            color=settings.component_colors[c])
        ax.add_patch(r)

    ax.plot([0] * 2, ax.get_ylim(), ls="--", lw=0.75, color='k', zorder=10)

    fig.savefig(
        f"images/metal_abundance_distribution/{of}_{to}/"
        f"sample_ordered_abundance{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_sample_stats_correlation(config: dict):
    settings = Settings()

    fig = plt.figure(figsize=(7.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs:
        ax.set_xlim(0.5, 6.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_ylim(-0.9, 0.7)
        ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_ylabel(
            r"$\mathrm{[Fe/H]} - \mathrm{[Fe/H]}_\mathrm{CD}$")
        ax.grid(True, ls='-', lw=0.25, c='gainsboro')
        ax.label_outer()

    df = pd.read_csv(
        f"results/abundance_distribution_medians{config['FILE_SUFFIX']}.csv",
        index_col=0)

    helpers = Helpers(config)

    for j, c in enumerate(["H", "B", "WD"]):
        ax = axs[j]
        x = helpers.get_stellar_age_df()[f"MedianStellarAge_{c}_Gyr"] \
            - helpers.get_stellar_age_df()["MedianStellarAge_CD_Gyr"]
        y = df[f"MedianAbundance_Fe/H_{c}"] - df[f"MedianAbundance_Fe/H_CD"]
        ax.scatter(
            x.to_numpy(), y.to_numpy(),
            color=settings.component_colors[c], s=6, zorder=10,
            marker=settings.component_markers[c])
        ax.set_xlabel(
            r"$\mathrm{Stellar~Age}_\mathrm{" + f"{c}" + "} "
            r"- \mathrm{Stellar~Age}_\mathrm{CD}$", fontsize=9)
        ax.text(
            x=0.04, y=0.9, size=7, ha="left", va="top",
            s=r"$r =" + str(np.round(pearsonr(x, y)[0], 2)) + "$",
            c=settings.component_colors[c],
            transform=ax.transAxes)
        if pearsonr(x, y)[1] >= 0.0001:
            pvalue_text = "=" + str(np.round(pearsonr(x, y)[1], 4))
        else:
            pvalue_text = "< 0.0001"
        ax.text(
            x=0.04, y=0.85, size=7, ha="left", va="top",
            s=r"$p\mathrm{-value}" + pvalue_text + "$",
            c=settings.component_colors[c],
            transform=ax.transAxes)

        ax.text(
            x=0.04, y=0.96, size=7, ha="left", va="top",
            s=r"$\textbf{" + settings.component_labels[c] + "}$",
            c=settings.component_colors[c],
            transform=ax.transAxes)

    fig.savefig(
        f"images/metal_abundance_distribution/Fe_H/"
        f"sample_component_comparison_correlation{config['FILE_SUFFIX']}.pdf")
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
    # plot_sample_stats(sample=sample, of="Fe", to="H", config=config,
    #                   xlim=(-1.0, 0.4))
    # plot_sample_stats(sample=sample, of="O", to="Fe", config=config,
    #                   xlim=(-0.025, 0.085))
    # plot_sample_stats_correlation(config)
    plot_sample_ordered_abundance(sample=sample, of="Fe", to="H",
                                  config=config, xlim=(-1.0, 0.4))
    plot_sample_ordered_abundance(sample=sample, of="O", to="H",
                                  config=config, xlim=(-0.6, 0.6))
    plot_sample_ordered_abundance(sample=sample, of="O", to="Fe",
                                  config=config, xlim=(0.2, 0.3))


if __name__ == "__main__":
    main()
