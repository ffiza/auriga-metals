from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import yaml

from auriga.snapshot import Snapshot
from auriga.images import figure_setup
from auriga.paths import Paths
from auriga.settings import Settings
from auriga.parser import parse

figure_setup()
settings = Settings()

CONFIG_FILE: str = "02"
N_BINS: int = 28
AGE_MIN: float = 0.0
AGE_MAX: float = 14.0

config = yaml.safe_load(open(f"configs/{CONFIG_FILE}.yml"))
sample = [f"au{i}_or_l4_s127" for i in settings.groups["Included"]]

fig = plt.figure(figsize=(8, 2.0))
gs = fig.add_gridspec(nrows=1, ncols=5, hspace=0.0, wspace=0.0)
axs = gs.subplots(sharex=True, sharey=True)

for ax in axs.flatten():
    ax.tick_params(which='both', direction="in")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 0.3)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel('Stellar Age [Gyr]')
    ax.set_ylabel(r"$f_{\star, \mathrm{ex-situ}}$")
    ax.grid(True, ls='-', lw=0.25, c="gainsboro")
    ax.label_outer()

for simulation in sample:
    galaxy = parse(simulation)[0]
    df = pd.read_csv(f"results/{'_'.join(simulation.split('_')[:-1])}/"
                     "stellar_formation_time_dist.csv")
    label, zorder = None, 10
    if parse(simulation)[0] == 6:
        label, zorder = f"Au{galaxy}", 11

    # Plot all stars
    color = 'k' if galaxy == 6 else "silver"
    axs[0].plot(df["TimeBinCenters_Gyr"],
                df["StellarAgeDist_All"] - df["StellarAgeDist_All_InSitu"],
                zorder=zorder, c=color, lw=1, label=label)

    for i, c in enumerate(settings.components):
        color = settings.component_colors[c] if galaxy == 6 else "silver"

        axs[i + 1].plot(
            df["TimeBinCenters_Gyr"],
            df[f"StellarAgeDist_{c}"] - df[f"StellarAgeDist_{c}_InSitu"],
            zorder=zorder, c=color, lw=1, label=label)

# region Medians
data = np.zeros((len(sample), N_BINS))
for i, simulation in enumerate(sample):
    df = pd.read_csv(f"results/{'_'.join(simulation.split('_')[:-1])}/"
                     "stellar_formation_time_dist.csv")
    data[i, :] = df["StellarAgeDist_All"] - df["StellarAgeDist_All_InSitu"]
median = np.median(data, axis=0)
axs[0].plot(df["TimeBinCenters_Gyr"], median, ls=(0, (5, 2)),
            zorder=zorder, c="black", lw=1, label="Median")

for j, c in enumerate(settings.components):
    data = np.zeros((len(sample), N_BINS))
    for i, simulation in enumerate(sample):
        df = pd.read_csv(f"results/{'_'.join(simulation.split('_')[:-1])}/"
                         "stellar_formation_time_dist.csv")
        data[i, :] = df[f"StellarAgeDist_{c}"] \
            - df[f"StellarAgeDist_{c}_InSitu"]
    median = np.median(data, axis=0)
    axs[j + 1].plot(df["TimeBinCenters_Gyr"], median, ls=(0, (5, 2)),
                    zorder=zorder, c=settings.component_colors[c],
                    lw=1, label="Median")
# endregion

axs[0].text(
    x=0.05, y=0.95, size=8.0, ha="left", va="top",
    s=r"$\textbf{" + "All" + "}$", c='k', transform=axs[0].transAxes)
for i, c in enumerate(settings.components):
    axs[i + 1].text(
        x=0.05, y=0.95, size=8.0, ha="left", va="top",
        s=r"$\textbf{" + settings.component_labels[c] + "}$",
        c=settings.component_colors[c], transform=axs[i + 1].transAxes)

for ax in axs.flatten():
    ax.legend(loc="center left", framealpha=0, fontsize=6,
              bbox_to_anchor=(0, 0.825))

fig.savefig(
    f"images/stellar_formation_time/"
    f"au6_or_l4_s127_exsitu{config['FILE_SUFFIX']}.pdf")
plt.close(fig)