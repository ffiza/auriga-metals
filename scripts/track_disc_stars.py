import numpy as np
import pandas as pd
import yaml
import argparse
import warnings
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from auriga.snapshot import Snapshot
from auriga.support import find_indices
from auriga.parser import parse
from auriga.paths import Paths
from auriga.settings import Settings
from auriga.images import figure_setup
from auriga.support import timer
from auriga.cosmology import Cosmology


def read_data(simulation: str,
              config: dict,
              tag_in_situ: bool = False) -> pd.DataFrame:
    s = Snapshot(simulation=simulation, loadonlytype=[0, 1, 2, 3, 4, 5])
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])
    
    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    is_target = is_real_star & is_main_obj

    props = {
        "ID": s.ids[is_target],
        "ComponentTag": s.region_tag[is_target],
        "zPosition_kpc": s.pos[is_target, 2] * s.expansion_factor,
        "Circularity": s.circularity[is_target],
        "zAngularMomentumFraction": \
            s.get_specific_angular_momentum()[is_target, 2] \
                / np.linalg.norm(
                    s.get_specific_angular_momentum()[is_target, :],
                    axis=1),
        }

    if tag_in_situ:
        s.tag_in_situ_stars()
        props["IsInSitu"] = s.is_in_situ[is_target]
        props["FormationSnapshot"] = s.stellar_formation_snapshot[is_target]

    df = pd.DataFrame(props)
    df.simulation = simulation
    df.time = s.time
    df.expansion_factor = s.expansion_factor

    return df


def read_merger_data(simulation):
    mergers = np.load(f'data/iza_2022/{simulation}/subhaloes.npy')
    r200 = np.load(f'data/iza_2022/{simulation}/R200.npy')
    return mergers, r200


def get_track_ids(simulation: str, config: dict) -> dict:
    settings = Settings()
    today_df = read_data(f"{simulation}_s127", config, tag_in_situ=True)

    galaxy, rerun, _, _ = parse(today_df.simulation)
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"

    component_at_birth = -1 * np.ones(len(today_df), dtype=np.int8)
    for i in range(40, 128, 1):
        this_df = read_data(f"{simulation}_s{i}", config)

        idx = find_indices(
            this_df["ID"].to_numpy(),
            today_df["ID"][
                (today_df["FormationSnapshot"] == i) \
                    & today_df["IsInSitu"]].to_numpy(),
            -1)
        idx = idx[idx >= 0]

        component_at_birth[(today_df["FormationSnapshot"] == i) \
            & today_df["IsInSitu"]] \
                = this_df["ComponentTag"][idx].to_numpy()

    today_df["ComponentTagAtBirth"] = component_at_birth

    track_ids = {
        "CD_to_WD": today_df["ID"][
            (today_df["ComponentTag"] == settings.component_tags["WD"]) \
                & (today_df["ComponentTagAtBirth"] \
                    == settings.component_tags["CD"])].to_numpy(),
        "CD_to_CD": today_df["ID"][
            (today_df["ComponentTag"] == settings.component_tags["CD"]) \
                & (today_df["ComponentTagAtBirth"] \
                    == settings.component_tags["CD"])].to_numpy()
        }
    
    return track_ids


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--simulation", type=str, required=False)
    parser.add_argument("--recalculate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.recalculate and args.simulation is None:
        parser.error("--simulation is required when --"
                     "recalculate is specified.")

    return vars(args)


@timer
def calculate_stats(simulation: str, config: dict) -> None:
    track_ids = get_track_ids(simulation, config)
    prop_names = [
        "Time_Gyr",
        "ExpansionFactor",
        "zPositionMedian_kpc",
        "zPosition16thPercentile_kpc",
        "zPosition84thPercentile_kpc",
        "zPositionAbsMedian_kpc",
        "zPositionAbs16thPercentile_kpc",
        "zPositionAbs84thPercentile_kpc",
        "zAngularMomFracMedian",
        "zAngularMomFrac16thPercentile",
        "zAngularMomFrac84thPercentile",
        "CircularityMedian",
        "Circularity16thPercentile",
        "Circularity84thPercentile"
        ]

    stats = {key: {} for key in track_ids.keys()}
    for key in stats.keys():
        for prop_name in prop_names:
            stats[key][prop_name] = np.nan * np.ones(128)
    
    galaxy, rerun, resolution = parse(simulation)
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"
    paths = Paths(galaxy, rerun, resolution)

    for i in range(40, 128, 1):
        this_df = read_data(f"{simulation}_s{i}", config)

        for key in stats.keys():

            idx = find_indices(this_df["ID"].to_numpy(), track_ids[key], -1)
            idx = idx[idx >= 0]

            if idx.shape[0] < 1:
                continue

            circularity = this_df["Circularity"].to_numpy()[idx]
            circularity[~np.isfinite(circularity)] = np.nan

            stats[key]["Time_Gyr"][i] = this_df.time
            stats[key]["ExpansionFactor"][i] = this_df.expansion_factor
            stats[key]["zPositionMedian_kpc"][i] = \
                np.nanmedian(this_df["zPosition_kpc"].to_numpy()[idx])
            stats[key]["zPosition16thPercentile_kpc"][i] = \
                np.nanpercentile(this_df["zPosition_kpc"].to_numpy()[idx], 16)
            stats[key]["zPosition84thPercentile_kpc"][i] = \
                np.nanpercentile(this_df["zPosition_kpc"].to_numpy()[idx], 84)
            stats[key]["zPositionAbsMedian_kpc"][i] = \
                np.nanmedian(np.abs(
                    this_df["zPosition_kpc"].to_numpy())[idx])
            stats[key]["zPositionAbs16thPercentile_kpc"][i] = \
                np.nanpercentile(np.abs(
                    this_df["zPosition_kpc"].to_numpy())[idx], 16)
            stats[key]["zPositionAbs84thPercentile_kpc"][i] = \
                np.nanpercentile(np.abs(
                    this_df["zPosition_kpc"].to_numpy())[idx], 84)
            stats[key]["zAngularMomFracMedian"][i] = \
                np.nanmedian(
                    this_df["zAngularMomentumFraction"].to_numpy()[idx])
            stats[key]["zAngularMomFrac16thPercentile"][i] = \
                np.nanpercentile(
                    this_df["zAngularMomentumFraction"].to_numpy()[idx], 16)
            stats[key]["zAngularMomFrac84thPercentile"][i] = \
                np.nanpercentile(
                    this_df["zAngularMomentumFraction"].to_numpy()[idx], 84)
            stats[key]["CircularityMedian"][i] = \
                np.nanmedian(circularity)
            stats[key]["Circularity16thPercentile"][i] = \
                np.nanpercentile(circularity, 16)
            stats[key]["Circularity84thPercentile"][i] = \
                np.nanpercentile(circularity, 84)

    for key in stats.keys():
        df = pd.DataFrame(data=stats[key], columns=prop_names)
        df.to_csv(
            f"{paths.results}track_results_{key}{config['FILE_SUFFIX']}.csv")


def plot_property(df_prop: str, config: dict,
                  ylim: tuple, yticks: list, ylabel: str,
                  filename: str) -> None:
    settings = Settings()

    DISTANCE_FRAC = 0.5
    FS_MIN = 0.01

    with open('data/lopez_et_al_2025.json', 'r') as file:
        bar_data_lopez = json.load(file)
    bar_data_fragkoudi = pd.read_csv("data/fragkoudi_et_al_2025.csv")

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r'Time [Gyr]')
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel)

    for i in range(len(settings.groups["Included"])):
        ax = axs.flatten()[i]
        galaxy = settings.groups["Included"][i]
        simulation = f"au{galaxy}_or_l4"
        paths = Paths(parse(simulation)[0],
                      parse(simulation)[1],
                      parse(simulation)[2])
        label = f"Au{galaxy}"
        dfs = [
            pd.read_csv(f"{paths.results}track_results_CD_to_CD"
                        f"{config['FILE_SUFFIX']}.csv"),
            pd.read_csv(f"{paths.results}track_results_CD_to_WD"
                        f"{config['FILE_SUFFIX']}.csv"),
        ]
        df_labels = [r"CD $\to$ CD", r"CD $\to$ WD"]
        df_colors = ["tab:red", "tab:orange"]
        for i, df in enumerate(dfs):
            is_finite = np.isfinite(df[df_prop].to_numpy())
            prop = df[df_prop].to_numpy()[is_finite]
            prop[-1] = np.mean(prop[-3:])
            ax.plot(
                df["Time_Gyr"].to_numpy()[is_finite],
                prop,
                ls="-", lw=1.0, color=df_colors[i], label=df_labels[i],
            )

        # Indicate mergers
        mergers, r200 = read_merger_data(simulation)
        for merger in mergers:
            if merger[2] < DISTANCE_FRAC * r200[int(merger[1]-15), 2] \
                and merger[4] > FS_MIN and merger[0] > 0:
                rect = patches.Rectangle((merger[0], ax.get_ylim()[0]),
                                         0.2, np.diff(ax.get_ylim())[0],
                                         linewidth=1, edgecolor='none',
                                         facecolor='#dbdbdb', zorder=-1)
                ax.add_patch(rect)

        # Indicate bar formation times from López et al. (2025)
        gals = [d["Galaxy"] for d in bar_data_lopez["Data"]]
        if parse(simulation)[0] in gals:
            idx = gals.index(parse(simulation)[0])
            bar_formation_time = Cosmology().present_time \
                - bar_data_lopez["Data"][idx]["t_bar"]
            idx_min = np.nanargmin(
                np.abs(dfs[1]["Time_Gyr"].to_numpy() - bar_formation_time))
            ax.annotate("",
                        xytext=(bar_formation_time,
                                dfs[1][df_prop].to_numpy()[idx_min] \
                                    - 0.15 * np.diff(ax.get_ylim())[0]),
                        xy=(bar_formation_time,
                            dfs[1][df_prop].to_numpy()[idx_min]),
                        arrowprops=dict(arrowstyle="->", color="tab:purple"))

        # Indicate bar formation times from Fragkoudi et al. (2025)
        gals = list(bar_data_fragkoudi["Halo"].unique())
        if parse(simulation)[0] in gals:
            idx = gals.index(parse(simulation)[0])
            if np.isfinite(bar_data_fragkoudi["tlookback_bf_Gyr"].iloc[idx]):
                bar_formation_time = Cosmology().present_time \
                    - float(bar_data_fragkoudi["tlookback_bf_Gyr"].iloc[idx])
                idx_min = np.nanargmin(
                    np.abs(dfs[0]["Time_Gyr"].to_numpy() - bar_formation_time))
                ax.annotate("",
                            xytext=(bar_formation_time,
                                    dfs[0][df_prop].to_numpy()[idx_min] \
                                        + 0.15 * np.diff(ax.get_ylim())[0]),
                            xy=(bar_formation_time,
                                dfs[0][df_prop].to_numpy()[idx_min]),
                            arrowprops=dict(arrowstyle="->",
                            color="tab:green"))

        if parse(simulation)[0] == 17:
            ax.text(x=0.05, y=0.10,
                    s="López et al. (2025)",
                    size=6.0, transform=ax.transAxes,
                    ha='left', va='center',
                    color="tab:purple")
            ax.text(x=0.05, y=0.19,
                    s="Fragkoudi et al. (2025)",
                    size=6.0, transform=ax.transAxes,
                    ha='left', va='center',
                    color="tab:green")

        ax.text(x=0.95, y=0.05,
                s=r"$\texttt{" + label + "}$",
                size=6.0, transform=ax.transAxes,
                ha='right', va='bottom',
                )

    axs[3, 0].legend(loc="lower left", framealpha=0, fontsize=5)

    fig.savefig(
        "images/warm_disc_star_tracking/"
        f"{filename}{config['FILE_SUFFIX']}.pdf")

    plt.close(fig)


def plot_circularity_change_vs_mass_change(config: dict) -> None:
    settings = Settings()

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim(-0.25, 0.25)
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_ylim(-2, 2)
        ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r'$\Delta \epsilon$')
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r"$\Delta M ~ \mathrm{[10^{9} \, M_\odot]}$")

    for i in range(len(settings.groups["Included"])):
        ax = axs.flatten()[i]
        galaxy = settings.groups["Included"][i]
        simulation = f"au{galaxy}_or_l4"
        paths = Paths(parse(simulation)[0],
                      parse(simulation)[1],
                      parse(simulation)[2])
        label = f"Au{galaxy}"
        df_circularity = pd.read_csv(
            f"{paths.results}track_results_CD_to_WD"
            f"{config['FILE_SUFFIX']}.csv")
        df_mass = pd.read_csv(
            f"{paths.results}mass_in_virial_radius.csv")

        is_time_range = (df_circularity["Time_Gyr"] >= 4.0)
        ax.scatter(
            np.diff(df_circularity["CircularityMedian"][
                is_time_range].to_numpy()),
            np.diff(df_mass["StellarMassIn0.5R200_Msun"][
                is_time_range].to_numpy()) / 1E9,
            c=df_circularity["Time_Gyr"].to_numpy()[is_time_range][1:],
            cmap="plasma", s=10, edgecolor="none", vmin=0, vmax=14,
        )
        
        ax.text(x=0.95, y=0.05,
                s=r"$\texttt{" + label + "}$",
                size=6.0, transform=ax.transAxes,
                ha='right', va='bottom',
                )

    fig.savefig(
        "images/warm_disc_star_tracking/"
        f"circ_change_vs_mass_change{config['FILE_SUFFIX']}.pdf")

    plt.close(fig)


def main() -> None:
    BLUE = "\033[94m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args['config']}.yml"))

    if not args["recalculate"] and not args["plot"]:
        warnings.warn("Either pass --recalculate and/or --plot for this "
                      "script to take effect.")

    if args["recalculate"]:
        s = f"Au{parse(args['simulation'])[0]}".rjust(4)
        print(f"{BLUE}Processing {UNDERLINE}{s}{RESET}{BLUE}... {BLUE}",
              end="", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calculate_stats(simulation=args["simulation"], config=config)
        print(f"{BLUE}Done.{BLUE}", flush=True)

    if args["plot"]:
        plot_property(df_prop="CircularityMedian", config=config,
                      ylim=(-1.5, 1.5), yticks=[-1, -0.5, 0, 0.5, 1],
                      ylabel=r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$',
                      filename="circularity")
        plot_property(df_prop="zPositionAbsMedian_kpc", config=config,
                      ylim=(0, 3), yticks=[0.5, 1, 1.5, 2, 2.5],
                      ylabel=r'$\left| z \right|$ [kpc]',
                      filename="zabs")
        plot_property(df_prop="zPositionMedian_kpc", config=config,
                      ylim=(-1, 1), yticks=[-0.5, 0, 0.5],
                      ylabel=r'$z$ [kpc]',
                      filename="z")
        plot_property(df_prop="zAngularMomFracMedian", config=config,
                      ylim=(-1.1, 1.1), yticks=[-1, -0.5, 0, 0.5, 1],
                      ylabel=r'$j_z / \left| \mathbf{j} \right|$',
                      filename="jzfrac")
        plot_circularity_change_vs_mass_change(config=config)


if __name__ == "__main__":
    main()
