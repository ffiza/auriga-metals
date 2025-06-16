import numpy as np
import pandas as pd
import yaml
import argparse
import warnings
import matplotlib.pyplot as plt

from auriga.snapshot import Snapshot
from auriga.support import find_indices
from auriga.parser import parse
from auriga.paths import Paths
from auriga.settings import Settings
from auriga.images import figure_setup
from auriga.support import timer


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
    parser.add_argument("--simulation", type=str, required=True)
    parser.add_argument("--recalculate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
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


def create_figure(dfs: list, config: dict, simulation: str,
                  df_labels: list, df_colors: list) -> None:
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.35, wspace=0.35)
    axs = gs.subplots(sharex=False, sharey=False)

    axs[0, 0].set_xlabel("Time [Gyr]")
    axs[0, 0].set_ylabel(r"$\left| z \right|$ [kpc]")
    axs[0, 0].set_xlim(0, 14)
    axs[0, 0].set_ylim(0, 5)
    for i, df in enumerate(dfs):
        axs[0, 0].plot(
            df["Time_Gyr"],
            df["zPositionAbsMedian_kpc"],
            ls="-", lw=1.0, color=df_colors[i],
        )
    axs[0, 0].text(
        x=0.95, y=0.95, s=r"$\texttt{Au" + str(parse(simulation)[0]) + "}$",
        size=7, transform=axs[0, 0].transAxes, ha='right', va='top')
    
    axs[0, 1].set_xlabel("Time [Gyr]")
    axs[0, 1].set_ylabel(r"$z$ [kpc]")
    axs[0, 1].set_xlim(0, 14)
    axs[0, 1].set_ylim(-1, 1)
    for i, df in enumerate(dfs):
        axs[0, 1].plot(
            df["Time_Gyr"],
            df["zPositionMedian_kpc"],
            ls="-", lw=1.0, color=df_colors[i], label=df_labels[i]
        )
    axs[0, 1].legend(loc="lower right", fontsize=6, framealpha=0)
    
    axs[1, 0].set_xlabel("Time [Gyr]")
    axs[1, 0].set_ylabel(r"$\epsilon = j_z / j_\mathrm{circ}$")
    axs[1, 0].set_xlim(0, 14)
    axs[1, 0].set_ylim(-1.2, 1.2)
    for i, df in enumerate(dfs):
        axs[1, 0].plot(
            df["Time_Gyr"],
            df["CircularityMedian"],
            ls="-", lw=1.0, color=df_colors[i],
        )

    axs[1, 1].set_xlabel("Time [Gyr]")
    axs[1, 1].set_ylabel(r"$j_z / \left| \mathbf{j} \right|$")
    axs[1, 1].set_xlim(0, 14)
    axs[1, 1].set_ylim(-1, 1)
    for i, df in enumerate(dfs):
        axs[1, 1].plot(
            df["Time_Gyr"],
            df["zAngularMomFracMedian"],
            ls="-", lw=1.0, color=df_colors[i],
        )

    for ax in axs.flatten():  # Make all panels square
        ax.set_aspect(np.diff(ax.get_xlim() / np.diff(ax.get_ylim())))

    fig.savefig(
        f"images/warm_disc_star_tracking/{simulation}/"
        f"temporal_evolution{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main() -> None:
    BLUE = "\033[94m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args['config']}.yml"))
    paths = paths = Paths(parse(args["simulation"])[0],
                          parse(args["simulation"])[1],
                          parse(args["simulation"])[2])

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
        dfs = [
            pd.read_csv(f"{paths.results}track_results_CD_to_CD"
                        f"{config['FILE_SUFFIX']}.csv"),
            pd.read_csv(f"{paths.results}track_results_CD_to_WD"
                        f"{config['FILE_SUFFIX']}.csv"),
        ]
        df_labels = [r"CD $\to$ CD", r"CD $\to$ WD"]
        df_colors = ["tab:red", "tab:orange"]
        create_figure(dfs, config, args["simulation"], df_labels, df_colors)


if __name__ == "__main__":
    main()
