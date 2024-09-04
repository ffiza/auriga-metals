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
    def transform_label(label: str) -> str:
        if label == "Halo":
            new_label = 'H'
        elif label == "Bulge":
            new_label = 'B'
        elif label == "Warm\nDisc":
            new_label = 'WD'
        elif label == "Cold\nDisc":
            new_label = 'CD'
        else:
            new_label = "Galaxy"
        return new_label
    
    @staticmethod
    def get_color(label: str):
        settings = Settings()
        if label in settings.components:
            return settings.component_colors[label]
        else:
            return "tab:gray"
    
    @staticmethod
    def read_data(sample: list, config: dict) -> pd.DataFrame:
        settings = Settings()
        with open(
            f"results/stellar_origin{config['FILE_SUFFIX']}.json") as f:
            data = json.load(f)
            arr = np.nan * np.ones((len(sample), 5))
            for s, simulation in enumerate(sample):
                this = data[simulation]
                bar_widths = this["BarWidths"]
                labels = [Helpers.transform_label(label) for label in \
                    this["ComponentLabel"]]
                this_dict = {}
                for i in range(len(bar_widths)):
                    this_dict[labels[i]] = bar_widths[i]
                for c, component in enumerate(settings.components):
                    arr[s, c] = this_dict[component]
                arr[s, -1] = this_dict["Galaxy"]
        columns = [f"InSituStarFraction_{c}" for c in settings.components]
        columns += ["InSituStarFraction_Galaxy"]
        df = pd.DataFrame(data=arr, index=sample, columns=columns)
        return df


def plot_fraction_for_galaxy(simulation: str, sample: list, config: dict):
    settings = Settings()
    df = Helpers.read_data(sample, config)
    galaxy = f"Au{parse(simulation)[0]}"
    labels = [name.split("_")[-1] for name in df.columns]
    colors = [Helpers.get_color(label) for label in labels]

    fig = plt.figure(figsize=(3.0, 2.25))
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex=True, sharey=True)

    ax.label_outer()
    ax.tick_params(which='both', direction="in",
                    bottom=True, top=False, left=False, right=False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.6, 4.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.barh(y=np.arange(df.loc[simulation].shape[0]),
            width=100 * df.loc[simulation], color=colors, linewidth=0)
    for i, c in enumerate(df.columns):
        ax.boxplot(df[c] * 100, positions=[i], vert=False, widths=0.25,
                   flierprops={"markersize": 2, "markerfacecolor": "black"},
                   medianprops={"color": "black"})
    ax.set_yticks([])
    for i, label in enumerate(labels):
        value = np.round(df.loc[simulation].iloc[i] * 100, 1)
        ax.text(x=2, y=i, size=8.0, ha="left", va="center", c="white",
                s=r"$\textbf{" + str(value) + r"\%" + "}$")
        ax.text(
            x=-2, y=i, size=8.0, ha="right", va="center",
            s=r"$\textbf{" + label + "}$",
            c=colors[i])
    ax.set_xlabel(r"$f_\mathrm{in-situ}$")


    ax.text(x=ax.get_xlim()[0], y=ax.get_ylim()[1],
            s=r"$\texttt{" + galaxy + "}$",
            size=9.0, ha='left', va='bottom',
            )

    fig.savefig(f"images/stellar_origin_by_region/"
                f"{simulation}{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main():
    settings = Settings()
    sample = [f"au{i}_or_l4_s127" for i in settings.groups["Included"]]

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Create figures
    figure_setup()
    plot_fraction_for_galaxy("au6_or_l4_s127", sample, config)


if __name__ == "__main__":
    main()