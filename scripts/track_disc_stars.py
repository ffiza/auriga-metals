import numpy as np
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from auriga.snapshot import Snapshot
from auriga.support import find_indices
from auriga.parser import parse
from auriga.paths import Paths
from auriga.settings import Settings
from auriga.images import figure_setup


def read_data(simulation: str, config: dict,
              tag_in_situ: bool = False) -> tuple:
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
        "zPosition_ckpc": s.pos[is_target, 2],
        "Circularity": s.circularity[is_target],
        "zAngularMomentum_kpckm/s": s.get_specific_angular_momentum()[
            is_target, 2],
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


def get_track_ids(simulation: str, config: dict) -> np.ndarray:
    settings = Settings()
    today_df = read_data(simulation, config, tag_in_situ=True)

    galaxy, rerun, _, _ = parse(today_df.simulation)
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"

    component_at_birth = -1 * np.ones(len(today_df), dtype=np.int8)
    for i in range(40, 127, 1):
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

    # Add component tags at birth for the last snapshot
    component_at_birth[(today_df["FormationSnapshot"] == 127) \
        & today_df["IsInSitu"]] \
            = today_df["ComponentTag"][
                today_df["FormationSnapshot"] == 127].to_numpy()

    today_df["ComponentTagAtBirth"] = component_at_birth

    track_ids = today_df["ID"][
        (today_df["ComponentTag"] == settings.component_tags["WD"]) \
            & (today_df["ComponentTagAtBirth"] \
                == settings.component_tags["CD"])].to_numpy()
    
    return track_ids


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--simulation", type=str, required=True)
    parser.add_argument("--recalculate", action="store_true")
    args = parser.parse_args()
    return args


def calculate_stats(simulation: str, config: dict) -> tuple:
    track_ids = get_track_ids(simulation, config)
    prop_names = [
        "Time_Gyr", "ExpansionFactor",
        "zAbsPositionMedian_ckpc",
        "zAbs16thPercentile_ckpc", "zAbs84thPercentile_ckpc",
        "zAbsPositionMean_ckpc", "zAbsPositionStd_ckpc",
        "zAngularMomMedian_kpckm/s",
        "zAngularMom16thPercentile_ckpc", "zAngularMom84thPercentile_ckpc",
        "zAngularMomMean_kpckm/s", "zAngularMomStd_kpckm/s",
        "CircularityMean", "CircularityMedian",
        ]

    particle_circularity = np.nan * np.ones((128, track_ids.shape[0]))
    times = np.nan * np.ones(128)

    stats = {}
    for prop_name in prop_names:
        stats[prop_name] = np.nan * np.ones(128)
    
    galaxy, rerun, resolution, _ = parse(simulation)
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"
    paths = Paths(galaxy, rerun, resolution)

    for i in range(40, 128, 1):
        this_df = read_data(f"{simulation}_s{i}", config)

        idx = find_indices(this_df["ID"].to_numpy(), track_ids, -1)
        particle_circularity[i] = this_df["Circularity"].to_numpy()[idx]
        particle_circularity[i][idx == -1] = np.nan

        idx = idx[idx >= 0]

        if idx.shape[0] < 1:
            continue

        z_pos = this_df["zPosition_ckpc"].to_numpy()[idx]
        circularity = this_df["Circularity"].to_numpy()[idx]
        circularity[~np.isfinite(circularity)] = np.nan
        jz = this_df["zAngularMomentum_kpckm/s"].to_numpy()[idx]

        stats["Time_Gyr"][i] = this_df.time
        stats["ExpansionFactor"][i] = this_df.expansion_factor
        stats["zAbsPositionMedian_ckpc"][i] = np.nanmedian(np.abs(z_pos))
        stats["zAbs16thPercentile_ckpc"][i] = np.nanpercentile(np.abs(z_pos),
                                                               16)
        stats["zAbs84thPercentile_ckpc"][i] = np.nanpercentile(np.abs(z_pos),
                                                               84)
        stats["zAbsPositionMean_ckpc"][i] = np.nanmean(np.abs(z_pos))
        stats["zAbsPositionStd_ckpc"][i] = np.nanstd(np.abs(z_pos))
        stats["zAngularMomMean_kpckm/s"][i] = np.nanmean(jz)
        stats["zAngularMomMedian_kpckm/s"][i] = np.nanmedian(jz)
        stats["zAngularMom16thPercentile_ckpc"][i] = np.nanpercentile(jz, 16)
        stats["zAngularMom84thPercentile_ckpc"][i] = np.nanpercentile(jz, 84)
        stats["zAngularMomStd_kpckm/s"][i] = np.nanstd(jz)
        stats["CircularityMean"][i] = np.nanmean(circularity)
        stats["CircularityMedian"][i] = np.nanmedian(circularity)

        times[i] = this_df.time

    stats_df = pd.DataFrame(data=stats, columns=prop_names)
    stats_df.to_csv(
        f"{paths.results}track_results{config['FILE_SUFFIX']}.csv")

    return pd.DataFrame(stats_df), times, particle_circularity


def plot_height(df: pd.DataFrame) -> None:
    label = f"Au{parse(df.simulation)[0]}"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.tick_params(which='both', direction="in")
    ax.set_axisbelow(True)
    ax.grid(True, ls='-', lw=0.25, c='silver')

    ax.set_xlim(0, 14)
    ax.set_xlabel(r"Time [Gyr]")

    ax.set_ylim(0, 6)
    ax.set_ylabel(r"$\left| z \right|$ [kpc]")

    ax.plot(df["Time_Gyr"], df["zAbsPositionMedian_ckpc"], c="tab:purple",
            label="Median [kpc]", lw=0.75)
    ax.plot(df["Time_Gyr"], df["zAbsPositionMean_ckpc"], c="tab:cyan",
            label="Mean [kpc]", lw=0.75)
    ax.plot(df["Time_Gyr"],
            df["ExpansionFactor"] * df["zAbsPositionMedian_ckpc"],
            c="tab:purple", label="Median [ckpc]", lw=0.75,
            ls=(0, (3, 1, 1, 1)))
    ax.plot(df["Time_Gyr"],
            df["ExpansionFactor"] * df["zAbsPositionMean_ckpc"],
            c="tab:cyan", ls=(0, (3, 1, 1, 1)), label="Mean [ckpc]", lw=0.75)
    ax.fill_between(
        df["Time_Gyr"],
        df["zAbs16thPercentile_ckpc"], df["zAbs84thPercentile_ckpc"],
        color="tab:purple", alpha=0.1, label="16th-84th Percentile", lw=0,
    )

    ax.legend(loc="upper right", framealpha=0, fontsize=5)

    ax.text(x=0.95, y=0.95, s=r"$\texttt{" + label + "}$",
            size=7.0, transform=ax.transAxes, ha='right', va='top')

    fig.savefig(f"images/warm_disc_star_tracking/{df.simulation}_height.pdf")
    plt.close(fig)


def plot_circularity(df: pd.DataFrame, config: dict) -> None:
    label = f"Au{parse(df.simulation)[0]}"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.tick_params(which='both', direction="in")
    ax.set_axisbelow(True)
    ax.grid(True, ls='-', lw=0.25, c='silver')

    ax.set_xlim(0, 14)
    ax.set_xlabel(r"Time [Gyr]")

    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel(r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$')

    ax.plot(df["Time_Gyr"], df["CircularityMedian"], c="tab:purple",
            label="Median [kpc]", lw=0.75)
    ax.plot(df["Time_Gyr"], df["CircularityMean"], c="tab:cyan",
            label="Mean [kpc]", lw=0.75)

    ax.plot(ax.get_xlim(),
            [config["DISC_STD_CIRC"] - config["COLD_DISC_DELTA_CIRC"]] * 2,
            ls=(0, (3, 1, 1, 1)), color="black", lw=0.5)
    ax.plot(ax.get_xlim(),
            [config["DISC_STD_CIRC"] + config["COLD_DISC_DELTA_CIRC"]] * 2,
            ls=(0, (3, 1, 1, 1)), color="black", lw=0.5)
    ax.plot(ax.get_xlim(),
            [config["DISC_MIN_CIRC"]] * 2,
            ls=(0, (3, 1, 1, 1)), color="black", lw=0.5)

    ax.legend(loc="lower right", framealpha=0, fontsize=5)

    ax.text(x=0.95, y=0.95, s=r"$\texttt{" + label + "}$",
            size=7.0, transform=ax.transAxes, ha='right', va='top')

    fig.savefig(
        f"images/warm_disc_star_tracking/{df.simulation}_circularity.pdf")
    plt.close(fig)


def plot_angular_momentum(df: pd.DataFrame) -> None:
    label = f"Au{parse(df.simulation)[0]}"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.tick_params(which='both', direction="in")
    ax.set_axisbelow(True)
    ax.grid(True, ls='-', lw=0.25, c='silver')

    ax.set_xlim(0, 14)
    ax.set_xlabel(r"Time [Gyr]")

    ax.set_ylim(1E1, 1E4)
    ax.set_yscale("log")
    ax.set_ylabel(r'$j_z$ $\mathrm{[kpc \, km \, s^{-1}]}$')

    ax.fill_between(df["Time_Gyr"], df["zAngularMom16thPercentile_ckpc"],
                    df["zAngularMom84thPercentile_ckpc"], color="tab:purple",
                    alpha=0.1, lw=0,
                    label=r"16$^\mathrm{th}$-84$^\mathrm{th}$ Perc. Region")
    ax.plot(df["Time_Gyr"], df["zAngularMomMedian_kpckm/s"], c="tab:purple",
            label="Median", lw=0.75)
    ax.plot(df["Time_Gyr"], df["zAngularMomMean_kpckm/s"], c="tab:cyan",
            label="Mean", lw=0.75)

    ax.legend(loc="upper left", framealpha=0, fontsize=5)

    ax.text(x=0.95, y=0.95, s=r"$\texttt{" + label + "}$",
            size=7.0, transform=ax.transAxes, ha='right', va='top')

    fig.savefig(
        f"images/warm_disc_star_tracking/{df.simulation}_momentum.pdf")
    plt.close(fig)


def plot_circularity_by_part(time: np.ndarray, circularity: np.ndarray,
                             config: dict, simulation: str) -> None:
    label = f"Au{parse(simulation)[0]}"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.tick_params(which='both', direction="in")
    ax.set_axisbelow(True)
    ax.grid(True, ls='-', lw=0.25, c='silver')

    ax.set_xlim(0, 14)
    ax.set_xlabel(r"Time [Gyr]")

    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel(r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$')


    for i in range(circularity.shape[1]):
        ax.plot(time, circularity[:, i], c="black", alpha=0.005, lw=0.2)

    ax.plot(ax.get_xlim(),
            [config["DISC_STD_CIRC"] - config["COLD_DISC_DELTA_CIRC"]] * 2,
            ls=(0, (3, 1, 1, 1)), color="black", lw=0.5)
    ax.plot(ax.get_xlim(),
            [config["DISC_STD_CIRC"] + config["COLD_DISC_DELTA_CIRC"]] * 2,
            ls=(0, (3, 1, 1, 1)), color="black", lw=0.5)
    ax.plot(ax.get_xlim(),
            [config["DISC_MIN_CIRC"]] * 2,
            ls=(0, (3, 1, 1, 1)), color="black", lw=0.5)

    ax.text(x=0.95, y=0.95, s=r"$\texttt{" + label + "}$",
            size=7.0, transform=ax.transAxes, ha='right', va='top')

    fig.savefig(
        "images/warm_disc_star_tracking/"
        f"{simulation}_circularity_by_particle.png")
    plt.close(fig)


def plot_density_map(data: pd.DataFrame) -> None:
    N_BINS = 100
    BOX_SIZE = 100
    settings = Settings()

    fig, axs = plt.subplots(figsize=(6.0, 6.0), ncols=3, nrows=3)
    gs = axs[1, 2].get_gridspec()
    for ax in axs[:2, :2].flatten():
        ax.remove()
    axs[-1, -1].remove()
    axbig = fig.add_subplot(gs[:2, :2])
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
    axbig.set_xlim(-50, 50)
    axbig.set_ylim(-50, 50)
    axbig.set_aspect(1)
    axbig.set_xticks([])
    axbig.set_yticks([])

    axbig.text(s="Warm Disc", x=0.5, y=0.95, size=10,
               transform=axbig.transAxes, va='top', ha="center",
               bbox=dict(facecolor='white', edgecolor='none',
                         boxstyle='round,pad=0.3', alpha=0.75))
    axs[2, 0].text(s=r"Halo $\to$ Warm Disc", x=0.5, y=0.95, size=8,
                   transform=axs[2, 0].transAxes, va='top', ha="center",
                   bbox=dict(facecolor='white', edgecolor='none',
                             boxstyle='round,pad=0.3', alpha=0.75))
    axs[2, 1].text(s=r"Bulge $\to$ Warm Disc", x=0.5, y=0.95, size=8,
                   transform=axs[2, 1].transAxes, va='top', ha="center",
                   bbox=dict(facecolor='white', edgecolor='none',
                             boxstyle='round,pad=0.3', alpha=0.75))
    axs[0, 2].text(s=r"Cold Disc $\to$ Warm Disc", x=0.5, y=0.95, size=8,
                   transform=axs[0, 2].transAxes, va='top', ha="center",
                   bbox=dict(facecolor='white', edgecolor='none',
                             boxstyle='round,pad=0.3', alpha=0.75))
    axs[1, 2].text(s=r"Warm Disc $\to$ Warm Disc", x=0.5, y=0.95, size=8,
                   transform=axs[1, 2].transAxes, va='top', ha="center",
                   bbox=dict(facecolor='white', edgecolor='none',
                             boxstyle='round,pad=0.3', alpha=0.75))

    axbig.hist2d(
        data["yPosition_ckpc"], data["zPosition_ckpc"],
        weights=data["Mass_Msun"],
        bins=N_BINS, range=[[-BOX_SIZE / 2, BOX_SIZE / 2],
                            [-BOX_SIZE / 2, BOX_SIZE / 2]],
        cmap="viridis", zorder=-10,
        norm=mcolors.LogNorm(vmin=1E4, vmax=1E9))

    component_axs = {"H": (2, 0), "B": (2, 1), "CD": (0, 2), "WD": (1, 2)}
    for c in settings.components:
        axs[component_axs[c]].hist2d(
            data["yPosition_ckpc"][
                data["ComponentAtBirth"] == settings.component_tags[c]],
            data["zPosition_ckpc"][
                data["ComponentAtBirth"] == settings.component_tags[c]],
            weights=data["Mass_Msun"][
                data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=[[-BOX_SIZE / 2, BOX_SIZE / 2],
                                [-BOX_SIZE / 2, BOX_SIZE / 2]],
            cmap=settings.component_colormaps[c], zorder=-10,
            norm=mcolors.LogNorm(vmin=1E4, vmax=1E9))

    axs[-1, -2].text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=1.05, y=0.9,
                     ha="left", va="center", transform=axs[-1, -2].transAxes)
    axs[-1, -2].text(s=f"Redshift: {np.round(data.redshift, 2)}", x=1.05,
                     y=0.8, ha="left", va="center",
                     transform=axs[-1, -2].transAxes)
    axs[-1, -2].text(s=f"Snapshot: {int(data.snapshot)}", x=1.05, y=0.7,
                     ha="left", va="center",
                     transform=axs[-1, -2].transAxes)
    axs[-1, -2].text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=1.05, y=0.6,
                     ha="left", va="center", transform=axs[-1, -2].transAxes)
    axs[-1, -2].text(s="Box: $100^3 \, \mathrm{ckpc}^3$", x=1.05, y=0.5,
                     ha="left", va="center", transform=axs[-1, -2].transAxes)

    fig.savefig(
        f"images/warm_disc_star_tracking/temp/snapshot_127.png")
    plt.close(fig)

def main() -> None:
    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))
    galaxy, rerun, resolution, _ = parse(args.simulation)
    paths = Paths(galaxy, rerun, resolution)

    # if args.recalculate:
    #     print("\n> Calculating statistics.\n")
    #     _, time, circularity = calculate_stats(args.simulation, config)
    #     plot_circularity_by_part(time, circularity, config, args.simulation)

    # stats = pd.read_csv(
    #     f"{paths.results}track_results{config['FILE_SUFFIX']}.csv")
    # stats.simulation = args.simulation
    # plot_height(stats)
    # plot_circularity(stats, config)
    # plot_angular_momentum(stats)

    data = pd.DataFrame(
        {
            "yPosition_ckpc": np.random.normal(0, 10, 1_000_000),
            "zPosition_ckpc": np.random.normal(0, 10, 1_000_000),
            "Mass_Msun": 1E5 * np.ones(1_000_000),
            "ComponentAtBirth": np.random.choice([0, 1, 2, 3], 1_000_000)
        }
    )
    data.time = 6.43252
    data.redshift = 0.97463834
    data.simulation = "au6_or_l4_s127"
    data.snapshot = 60
    plot_density_map(data)

if __name__ == "__main__":
    main()
