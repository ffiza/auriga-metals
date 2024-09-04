import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import argparse
import json

from auriga.settings import Settings
from auriga.parser import parse
from auriga.images import figure_setup


class Helpers:
    @staticmethod
    def load_data(simulation: str, config: dict) -> pd.DataFrame:
        df = pd.read_csv(
            f"results/{simulation}/decomposition_mass_evolution_02.csv")
        df.dropna(inplace=True)
        df["Snapshot"] = df["Snapshot"].astype(np.int8)
        return df
    
    def load_sample_data(sample: list, config: dict) -> pd.DataFrame:
        settings = Settings()
        data = np.nan * np.ones((len(sample), 4))
        for s, simulation in enumerate(sample):
            df = Helpers.load_data(simulation, config)
            for c, component in enumerate(settings.components):
                data[s, c] = df.loc[127, f"Mass_{component}_Msun"] \
                    / df.loc[127, "Mass_Msun"]
        columns = [f"Mass_{c}_Msun" for c in settings.components]
        data = pd.DataFrame(data=data, index=sample, columns=columns)
        return data


def plot_fraction_for_galaxy(simulation: str, sample: list, config: dict):
    settings = Settings()
    data = Helpers.load_sample_data(sample, config)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.tick_params(which='both', direction="in")

    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks([0, 1, 2, 3])

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel(r"$f_\star$")

    label = f"Au{parse(simulation)[0]}"

    for j, component in enumerate(settings.components):
        value = data.loc[simulation].iloc[j]
        ax.bar(x=j, height=value,
               color=list(settings.component_colors.values())[j],
               width=0.5, linewidth=0)
        for i, c in enumerate(data.columns):
            ax.boxplot(data[c], positions=[i], widths=0.25,
                       flierprops={"markersize": 2,
                                   "markerfacecolor": "black"},
                       medianprops={"color": "black"})
        ax.text(j, data[f"Mass_{component}_Msun"].max() + 0.02,
                s=r"$\textbf{" + str(int(np.round(100 * value, 0))) + "\%}$",
                c=settings.component_colors[component],
                ha="center", va="bottom", size=8.0)
        ax.text(j, -0.05, size=8.0,
                s=r"$\textbf{" + settings.components[j] + "}$",
                c=list(settings.component_colors.values())[j],
                ha="center", va="top")

    ax.set_xticklabels([])

    ax.text(
        x=0.05, y=0.95, size=8.0,
        s=r"$\texttt{" + label + "}$",
        ha="left", va="top", transform=ax.transAxes)

    fig.savefig(
        f"images/galaxy_decomposition/"
        f"stellar_mass_distribution_{simulation}{config['FILE_SUFFIX']}.pdf")
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
    plot_fraction_for_galaxy("au6_or_l4", sample, config)


if __name__ == "__main__":
    main()