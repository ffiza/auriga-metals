import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from auriga.snapshot import Snapshot
from auriga.settings import Settings
from auriga.paths import Paths
from auriga.parser import parse
from auriga.images import figure_setup
from auriga.physics import Physics

BOX_SIZE = 100.0  # ckpc
N_BINS = 200
COLORMAP = "gist_ncar"


def read_data(simulation: str) -> pd.DataFrame:
    """
    This method returns data related to this analysis.

    Parameters
    ----------
    simulation : str
        The simulation to consider.
    config : dict
        A dictionary with the configuration values.

    Returns
    -------
    pd.DataFrame
        The properties in a Pandas DataFrame.
    """
    s = Snapshot(simulation=simulation, loadonlytype=[0, 1, 2, 3, 4, 5])
    s.add_metal_abundance(of="Fe", to="H")
    s.add_metal_abundance(of="O", to="H")
    s.add_metallicity()

    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    mask = is_main_obj & is_real_star

    physics = Physics()

    props = {
        "xPosition_ckpc": s.pos[mask, 0],
        "yPosition_ckpc": s.pos[mask, 1],
        "zPosition_ckpc": s.pos[mask, 2],
        "Z_Zsun": s.metallicity[mask] / physics.solar_metallicity,
        "[Fe/H]": s.metal_abundance["Fe/H"][mask],
        "[O/H]": s.metal_abundance["O/H"][mask],
        }

    df = pd.DataFrame(props)
    df[~np.isfinite(df)] = np.nan
    df.dropna(inplace=True)
    return df


def calculate_maps(simulation: str, box_size: float, n_bins: int):
    settings = Settings()
    galaxy = parse(simulation)[0]
    label = f"Au{galaxy}"
    print(f"{label.rjust(4)}... ", end="")

    df = read_data(simulation)
    box_halfsize = box_size / 2.0
    is_box = (np.abs(df["xPosition_ckpc"]) <= box_halfsize) \
        & (np.abs(df["yPosition_ckpc"]) <= box_halfsize) \
            & (np.abs(df["zPosition_ckpc"]) <= box_halfsize)

    n_particles = np.histogram2d(
        df["xPosition_ckpc"][is_box], df["yPosition_ckpc"][is_box],
        bins=n_bins,
        range=[[-box_halfsize, box_halfsize],
               [-box_halfsize, box_halfsize]])
    h_metallicity = np.histogram2d(
        df["xPosition_ckpc"][is_box], df["yPosition_ckpc"][is_box],
        bins=n_bins, weights=df["Z_Zsun"][is_box],
        range=[[-box_halfsize, box_halfsize], [-box_halfsize, box_halfsize]])
    h_feh = np.histogram2d(
        df["xPosition_ckpc"][is_box], df["yPosition_ckpc"][is_box],
        bins=n_bins, weights=df["[Fe/H]"][is_box],
        range=[[-box_halfsize, box_halfsize], [-box_halfsize, box_halfsize]])
    h_oh = np.histogram2d(
        df["xPosition_ckpc"][is_box], df["yPosition_ckpc"][is_box],
        bins=n_bins, weights=df["[O/H]"][is_box],
        range=[[-box_halfsize, box_halfsize], [-box_halfsize, box_halfsize]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = {"XEdges": n_particles[1], "YEdges": n_particles[1],
                "Z_Hist": h_metallicity[0] / n_particles[0],
                "FeH_Hist": h_feh[0] / n_particles[0],
                "OH_Hist": h_oh[0] / n_particles[0]}

    print("Ok.")
    return data


def plot_map(data: dict, label: str, box_size: float, xedges: np.ndarray,
             yedges: np.ndarray, colormap: str):
    fig = plt.figure(figsize=(7.0, 4.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0.0, wspace=1.0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
        ax.set_xlabel(r"$x$ [ckpc]", fontsize=8.0)
        ax.set_ylabel(r"$y$ [ckpc]", fontsize=8.0)
        ax.set_xlim(-box_size / 2, box_size / 2)
        ax.set_ylim(-box_size / 2, box_size / 2)
        ax.set_aspect("equal")
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.tick_params(axis="both", which="both", labelsize=6.0)

    xx, yy = np.meshgrid(data["XEdges"], data["YEdges"])
    pcm = axs[0].pcolormesh(xx, yy, np.log10(data["Z_Hist"]).T,
                            cmap=COLORMAP, vmin=-1.0, vmax=0.5,
                            rasterized=True)
    cb = fig.colorbar(pcm, ax=axs[0], fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}\left( Z / \mathrm{Z}_\odot \right)$", fontsize=8)
    cb.set_ticks([-1, 0, -0.5, 0.5])
    cb.ax.tick_params(axis="both", which="both", labelsize=6.0)

    pcm = axs[1].pcolormesh(xx, yy, data["FeH_Hist"].T,
                            cmap=COLORMAP, vmin=-1.5, vmax=0.5,
                            rasterized=True)
    cb = fig.colorbar(pcm, ax=axs[1], fraction=0.046, pad=0.04)
    cb.set_label("[Fe/H]", fontsize=8.0)
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5])
    cb.ax.tick_params(axis="both", which="both", labelsize=6.0)

    pcm = axs[2].pcolormesh(xx, yy, data["OH_Hist"].T,
                            cmap=COLORMAP, vmin=-1.0, vmax=0.5,
                            rasterized=True)
    cb = fig.colorbar(pcm, ax=axs[2], fraction=0.046, pad=0.04)
    cb.set_label("[O/H]", fontsize=8.0)
    cb.set_ticks([-1, -0.5, 0, 0.5])
    cb.ax.tick_params(axis="both", which="both", labelsize=6.0)

    axs[0].text(x=axs[0].get_xlim()[0], y=axs[0].get_ylim()[1],
            s=r"$\texttt{" + label + r"}$",
            size=9.0, ha='left', va='bottom',
            )

    fig.savefig(f"images/projected_metallicity_maps/{label.lower()}.pdf")
    plt.close(fig)


def main():
    # settings = Settings()
    figure_setup()

    # sample = [f"au{i}_or_l4_s1127" for i in settings.groups["Included"]]
    sample = ["au6_or_l4_s127"]
    for simulation in sample:
        label = simulation.split("_")[0].capitalize()
        data = calculate_maps(simulation, BOX_SIZE, N_BINS)
        plot_map(data, label, BOX_SIZE, data["XEdges"], data["YEdges"],
                 COLORMAP)


if __name__ == "__main__":
    main()