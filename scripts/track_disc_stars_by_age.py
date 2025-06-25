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

AGE_BINS_GYR = [(2 * i, 2 * i + 2) for i in range(7)]

BLUE = "\033[94m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"
RED = "\033[31m"

def read_data(simulation: str,
              config: dict,
              tag_in_situ: bool = False) -> pd.DataFrame:
    s = Snapshot(simulation=simulation, loadonlytype=[0, 1, 2, 3, 4, 5])
    s.add_stellar_age()
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
        "StellarAge_Gyr": s.stellar_age[is_target],
        "zPosition_kpc": s.pos[is_target, 2] * s.expansion_factor,
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

    track_ids = {}
    for age_bin in AGE_BINS_GYR:
        track_ids[f"CD_to_WD_ages_{age_bin[0]}Gyr_{age_bin[1]}Gyr"] = \
            today_df["ID"][
                (today_df["ComponentTag"] == settings.component_tags["WD"]) \
                    & (today_df["ComponentTagAtBirth"] \
                        == settings.component_tags["CD"]) \
                            & (today_df["StellarAge_Gyr"] >= age_bin[0]) \
                                & (today_df["StellarAge_Gyr"] <= age_bin[1])
                                ].to_numpy(),
    
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
        "zPositionAbsMedian_kpc",
        ]

    stats = {key: {} for key in track_ids.keys()}
    for key in stats.keys():
        for prop_name in prop_names:
            stats[key][prop_name] = np.nan * np.ones(128)
    
    rerun_text = "_or" if not parse(simulation)[1] else "_re"
    simulation = f"au{parse(simulation)[0]}{rerun_text}_l4"
    paths = Paths(
        parse(simulation)[0], parse(simulation)[1], parse(simulation)[2])

    for i in range(40, 128, 1):
        this_df = read_data(f"{simulation}_s{i}", config)

        for key in stats.keys():

            idx = find_indices(this_df["ID"].to_numpy(), track_ids[key], -1)
            idx = idx[idx >= 0]

            if idx.shape[0] < 1:
                continue

            stats[key]["Time_Gyr"][i] = this_df.time
            stats[key]["ExpansionFactor"][i] = this_df.expansion_factor
            stats[key]["zPositionAbsMedian_kpc"][i] = \
                np.nanmedian(np.abs(this_df["zPosition_kpc"].to_numpy())[idx])

    for key in stats.keys():
        df = pd.DataFrame(data=stats[key], columns=prop_names)
        df.to_csv(
            f"{paths.results}track_results_{key}{config['FILE_SUFFIX']}.csv")


def create_figure(config: dict) -> None:
    settings = Settings()
    cmap = plt.get_cmap("brg_r")

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(nrows=6, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_ylim(0, 3)
        ax.set_yticks([0.5, 1, 1.5, 2, 2.5])
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r'Time [Gyr]')
            ax.tick_params(labelbottom=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r"$\left| z \right|$ [kpc]")

    for i in range(len(settings.groups["Included"])):
        galaxy = settings.groups["Included"][i]
        try:
            ax = axs.flatten()[i]
            simulation = f"au{galaxy}_or_l4"
            paths = Paths(parse(simulation)[0],
                        parse(simulation)[1],
                        parse(simulation)[2])
            label = f"Au{galaxy}"
            dfs = [
                pd.read_csv(f"{paths.results}track_results_CD_to_WD_ages_"
                            f"{age_bin[0]}Gyr_{age_bin[1]}Gyr"
                            f"{config['FILE_SUFFIX']}.csv") for 
                            age_bin in AGE_BINS_GYR]
            for j, df in enumerate(dfs):
                is_finite = np.isfinite(
                    df["zPositionAbsMedian_kpc"].to_numpy())
                ax.plot(
                    df["Time_Gyr"].to_numpy()[is_finite],
                    df["zPositionAbsMedian_kpc"].to_numpy()[is_finite],
                    ls="-", lw=1.0, color=cmap(j / len(dfs)),
                    label=f"{AGE_BINS_GYR[j][0]} - {AGE_BINS_GYR[j][1]} Gyr"
                )
            ax.text(x=0.95, y=0.95,
                    s=r"$\texttt{" + label + "}$",
                    size=6.0, transform=ax.transAxes,
                    ha='right', va='top',
                    )
        except FileNotFoundError:
            print(f"{RED}Data file for Au{galaxy} not found. Skipping.{RESET}")

    axs[0, 3].legend(loc="upper left", framealpha=0, fontsize=3.5, ncol=2,
                     title="Stellar Age [Gyr]", title_fontsize=4)

    fig.savefig(
        "images/warm_disc_star_tracking/"
        f"zabs_by_age{config['FILE_SUFFIX']}.pdf")

    plt.close(fig)


def main() -> None:


    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args['config']}.yml"))

    if not args["recalculate"] and not args["plot"]:
        warnings.warn("Either pass --recalculate and/or --plot for this "
                      "script to take effect.")

    if args["recalculate"]:
        s = f"Au{parse(args['simulation'])[0]}".ljust(4, ".")
        print(f"{BLUE}Processing {s}...... ",
              end="", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calculate_stats(simulation=args["simulation"], config=config)
        print(f"{BLUE}Done.{BLUE}", flush=True)

    if args["plot"]:
        create_figure(config=config)


if __name__ == "__main__":
    main()
