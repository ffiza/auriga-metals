"""
File:           track_warn_disc_stars_trajectory.py
Description:    Computes the trajectories of stars between the time of birth 
                and z=0.

Usage:          python scripts/track_warn_disc_stars_trajectory.py \
                --simulation SIMULATION_NAME --config CONFIG_FILE \
                    --recalculate

Arguments:
    --simulation    Name of the simulation to analyze (e.g., 'au6_or_l4').
    --config        Configuration filename for input parameters
                    (e.g., '02').
    --recalculate   If included, the script will perform the calculations
                    again, otherwise it will read the results.
"""
import numpy as np
import pandas as pd
import yaml
import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from auriga.snapshot import Snapshot
from auriga.support import find_indices
from auriga.parser import parse
from auriga.settings import Settings
from auriga.images import figure_setup

np.random.seed(42)


def read_data(simulation: str, config: dict,
              tag_in_situ: bool = False) -> pd.DataFrame:
    """
    Reads relevant snapshot data and, optinally, tags in-situ stars.

    Parameters
    ----------
    simulation : str
        The simulation snapshot to read.
    config : dict
        The configuration dictionary.
    tag_in_situ : bool, optional
        If True, tag in-situ stars. Note that tagging stars takes quite 
        some time. By default False.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the properties of the stars.
    """    
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
        "xPosition_kpc": s.pos[is_target, 0] * s.expansion_factor,
        "yPosition_kpc": s.pos[is_target, 1] * s.expansion_factor,
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
    df.redshift = s.redshift
    df.expansion_factor = s.expansion_factor

    return df


def compute_properties_at_birth(df: pd.DataFrame, config: dict) -> tuple:
    if parse(df.simulation)[2] != 4:
        raise ValueError("Only L4 resolution implemented.")

    galaxy = parse(df.simulation)[0]
    rerun = parse(df.simulation)[1]
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"

    component_at_birth = -1 * np.ones(len(df), dtype=np.int8)
    circularity_at_birth = np.nan * np.ones(len(df))

    for i in range(40, 127, 1):
        this_df = read_data(f"{simulation}_s{i}", config)

        idx = find_indices(
            this_df["ID"].to_numpy(),
            df["ID"][
                (df["FormationSnapshot"] == i) \
                    & df["IsInSitu"]].to_numpy(),
            -1)
        idx = idx[idx >= 0]

        component_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["ComponentTag"][idx].to_numpy()
        circularity_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["Circularity"][idx].to_numpy()
    
    # Add component tags at birth for the last snapshot
    component_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["ComponentTag"][
                df["FormationSnapshot"] == 127].to_numpy()
    circularity_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["Circularity"][
                df["FormationSnapshot"] == 127].to_numpy()

    return (component_at_birth, circularity_at_birth)


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--simulation", type=str, required=True)
    parser.add_argument("--recalculate", action="store_true")
    args = parser.parse_args()
    return vars(args)


def get_arr_from_idx(idx: np.ndarray, s: pd.Series) -> np.ndarray:
    arr = s.to_numpy()[idx]
    arr[idx == -1] = np.nan
    return arr


def track_props_of_ids(simulation: str, ids: np.ndarray,
                       props_to_track: list, config: dict) -> dict:
    props = {"Time_Gyr": [], "SnapshotNumber": []}
    for pid in ids:
        props[str(pid)] = {}
        for prop_to_track in props_to_track:
            props[str(pid)][prop_to_track] = []
    
    for i in range(40, 128, 1):
        this_df = read_data(f"{simulation}_s{i}", config)
        props["Time_Gyr"].append(np.round(this_df.time, 4))
        props["SnapshotNumber"].append(i)

        idx = find_indices(this_df["ID"].to_numpy(), ids, -1)

        for i in range(ids.shape[0]):
            for prop_to_track in props_to_track:
                if idx[i] >= 0:
                    props[str(ids[i])][
                        prop_to_track].append(
                            np.round(this_df[prop_to_track][idx[i]], 4))
                else:
                    props[str(ids[i])][prop_to_track].append(np.nan)

    return props


def windowed_average(x: np.ndarray, y: np.ndarray,
                     window_length: float) -> np.ndarray:
    y_avg = np.zeros_like(x)
    for i in range(len(x)):
        mask = (x >= x[i] - window_length / 2) \
            & (x <= x[i] + window_length / 2)
        if np.sum(mask) > 0:
            y_avg[i] = np.nanmean(y[mask])
        else:
            y_avg[i] = np.nan
    return y_avg


def plot_trajectories(time: np.ndarray, df: pd.DataFrame, config: dict,
                      simulation: str, file_label: str) -> None:

    def plot_custom_trajectory(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                               c: list):
        ax.plot(x, y, lw=0.25, color="whitesmoke", zorder=10)
        s = ax.scatter(x, y, lw=0, c=c, s=5, cmap="gnuplot", zorder=12,
                       vmin=0, vmax=14)
        return s

    r_xy = np.sqrt(np.array(df["xPosition_kpc"])**2 \
        + np.array(df["yPosition_kpc"])**2)

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(nrows=3, ncols=3, hspace=0.35, wspace=0.35)
    axs = gs.subplots(sharex=False, sharey=False)

    axs[0, 0].set_xlabel(r"$x$ [kpc]")
    axs[0, 0].set_ylabel(r"$y$ [kpc]")
    axs[0, 0].set_xlim(-20, 20)
    axs[0, 0].set_ylim(-20, 20)
    axs[0, 0].set_xticks([-20, -10, 0, 10, 20])
    axs[0, 0].set_yticks([-20, -10, 0, 10, 20])
    plot_custom_trajectory(
        ax=axs[0, 0],
        x=df["xPosition_kpc"].to_numpy(),
        y=df["yPosition_kpc"].to_numpy(),
        c=list(time))
    axs[0, 0].text(
        0.05, 0.95,
        r"$\texttt{Au" + str(parse(simulation)[0]) + "}$",
        transform=axs[0, 0].transAxes, size=6.0, ha='left',
        va='top')

    axs[0, 1].set_xlabel(r"$x$ [kpc]")
    axs[0, 1].set_ylabel(r"$z$ [kpc]")
    axs[0, 1].set_xlim(-20, 20)
    axs[0, 1].set_ylim(-20, 20)
    axs[0, 1].set_xticks([-20, -10, 0, 10, 20])
    axs[0, 1].set_yticks([-20, -10, 0, 10, 20])
    s = plot_custom_trajectory(
        ax=axs[0, 1],
        x=df["xPosition_kpc"].to_numpy(),
        y=df["zPosition_kpc"].to_numpy(), c=list(time))
    cax = inset_axes(axs[0, 1], width="90%", height="3%", loc='lower center',
                     borderpad=0.3)
    cbar = fig.colorbar(s, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(which='both', length=0)
    cbar.ax.tick_params(axis='x', pad=2)
    cbar.ax.xaxis.grid(True, which='both', color='white',
                       linestyle='-', linewidth=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_label("Time [Gyr]", size=6)
    
    axs[0, 2].set_xlabel(r"$y$ [kpc]")
    axs[0, 2].set_ylabel(r"$z$ [kpc]")
    axs[0, 2].set_xlim(-20, 20)
    axs[0, 2].set_ylim(-20, 20)
    axs[0, 2].set_xticks([-20, -10, 0, 10, 20])
    axs[0, 2].set_yticks([-20, -10, 0, 10, 20])
    plot_custom_trajectory(
        ax=axs[0, 2],
        x=df["yPosition_kpc"].to_numpy(),
        y=df["zPosition_kpc"].to_numpy(), c=list(time))
    
    axs[1, 0].set_xlabel(r"$r_{xy}$ [kpc]")
    axs[1, 0].set_ylabel(r"$z$ [kpc]")
    axs[1, 0].set_xlim(0, 20)
    axs[1, 0].set_ylim(-20, 20)
    axs[1, 0].set_xticks([0, 5, 10, 15, 20])
    axs[1, 0].set_yticks([-20, -10, 0, 10, 20])
    plot_custom_trajectory(
        ax=axs[1, 0],
        x=r_xy,
        y=df["zPosition_kpc"].to_numpy(),
        c=list(time))
    
    axs[1, 1].set_xlabel("Time [Gyr]")
    axs[1, 1].set_ylabel(r"$r_{xy}$ [kpc]")
    axs[1, 1].set_xlim(0, 14)
    axs[1, 1].set_ylim(0, 20)
    plot_custom_trajectory(
        ax=axs[1, 1],
        x=time,
        y=windowed_average(
            x=np.array(time),
            y=r_xy,
            window_length=1.0),
        c=list(time))
    
    axs[1, 2].set_xlabel("Time [Gyr]")
    axs[1, 2].set_ylabel(r"$\left| z \right|$ [kpc]")
    axs[1, 2].set_xlim(0, 14)
    axs[1, 2].set_ylim(0, 10)
    plot_custom_trajectory(
        ax=axs[1, 2],
        x=time,
        y=windowed_average(
            x=np.array(time),
            y=np.abs(df["zPosition_kpc"]),
            window_length=1.0),
        c=list(time))
    
    axs[2, 0].set_xlabel("Time [Gyr]")
    axs[2, 0].set_ylabel(r"$j_z / \left| \mathbf{j} \right|$")
    axs[2, 0].set_xlim(0, 14)
    axs[2, 0].set_ylim(-1, 1)
    plot_custom_trajectory(
        ax=axs[2, 0],
        x=time,
        y=windowed_average(
            x=np.array(time),
            y=df["zAngularMomentumFraction"].to_numpy(),
            window_length=1.0),
        c=list(time))
    
    axs[2, 1].set_xlabel("Time [Gyr]")
    axs[2, 1].set_ylabel(r"$\epsilon$")
    axs[2, 1].set_xlim(0, 14)
    axs[2, 1].set_ylim(-1.5, 1.5)
    plot_custom_trajectory(
        ax=axs[2, 1],
        x=time,
        y=windowed_average(
            x=np.array(time),
            y=df["Circularity"].to_numpy(),
            window_length=1.0),
        c=list(time))

    axs[2, 2].axis("off")

    for ax in axs.flatten():  # Make all plots square
        ax.set_aspect(np.diff(ax.get_xlim() / np.diff(ax.get_ylim())))

    fig.savefig(
        f"images/warm_disc_star_tracking_trajectory/{simulation}/"
        f"{file_label}{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_all_temporal_evolution(
        df: dict,
        config: dict,
        simulation: str) -> None:
    
    def plot_lines_with_average(
            ax: plt.Axes,
            x: np.ndarray,
            ys: list) -> None:
        for y in ys:
            ax.plot(x, y, lw=1, color="gainsboro", zorder=10)
        ax.plot(x[x >= 4.0],
                np.nanmedian(ys, axis=0)[x >= 4.0],
                lw=1, color="black", zorder=12)

    pids = list(df.keys())[2:]

    fig = plt.figure(figsize=(7, 4.5))
    gs = fig.add_gridspec(nrows=2, ncols=3, hspace=0.35, wspace=0.4)
    axs = gs.subplots(sharex=False, sharey=False)

    axs[0, 0].set_xlabel("Time [Gyr]")
    axs[0, 0].set_ylabel(r"$r_{xy}$ [kpc]")
    axs[0, 0].set_xlim(0, 14)
    axs[0, 0].set_ylim(0, 20)
    axs[0, 0].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    axs[0, 0].set_yticks([0, 5, 10, 15, 20])
    plot_lines_with_average(
        ax=axs[0, 0],
        x=np.array(df["Time_Gyr"]),
        ys=[windowed_average(
                x=np.array(df["Time_Gyr"]),
                y=np.sqrt(np.array(df[pid]["xPosition_kpc"])**2 \
                    + np.array(df[pid]["yPosition_kpc"])**2),
                window_length=1.0) for pid in pids]
    )
    axs[0, 0].text(
        0.05, 0.95,
        r"$\texttt{Au" + str(parse(simulation)[0]) + "}$",
        transform=axs[0, 0].transAxes, size=6.0, ha='left',
        va='top')

    axs[0, 1].set_xlabel("Time [Gyr]")
    axs[0, 1].set_ylabel(r"$\left| z \right|$ [kpc]")
    axs[0, 1].set_xlim(0, 14)
    axs[0, 1].set_ylim(0, 10)
    axs[0, 1].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    axs[0, 1].set_yticks([0, 2, 4, 6, 8, 10])
    plot_lines_with_average(
        ax=axs[0, 1],
        x=np.array(df["Time_Gyr"]),
        ys=[windowed_average(
                x=np.array(df["Time_Gyr"]),
                y=np.abs(df[pid]["zPosition_kpc"]),
                window_length=1.0) for pid in pids]
    )
    
    axs[0, 2].set_xlabel("Time [Gyr]")
    axs[0, 2].set_ylabel(r"$j_z / \left| \mathbf{j} \right|$")
    axs[0, 2].set_xlim(0, 14)
    axs[0, 2].set_ylim(-1, 1)
    axs[0, 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    axs[0, 2].set_yticks([-1, -0.5, 0, 0.5, 1])
    plot_lines_with_average(
        ax=axs[0, 2],
        x=np.array(df["Time_Gyr"]),
        ys=[windowed_average(
                x=np.array(df["Time_Gyr"]),
                y=np.array(df[pid]["zAngularMomentumFraction"]),
                window_length=1.0) for pid in pids]
    )
    
    axs[1, 0].set_xlabel("Time [Gyr]")
    axs[1, 0].set_ylabel(r"$\epsilon$")
    axs[1, 0].set_xlim(0, 14)
    axs[1, 0].set_ylim(-1.5, 1.5)
    axs[1, 0].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    axs[1, 0].set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    plot_lines_with_average(
        ax=axs[1, 0],
        x=np.array(df["Time_Gyr"]),
        ys=[windowed_average(
                x=np.array(df["Time_Gyr"]),
                y=np.array(df[pid]["Circularity"]),
                window_length=1.0) for pid in pids]
    )
    
    axs[1, 1].axis("off")
    axs[1, 2].axis("off")
    
    for ax in axs.flatten():  # Make all plots square
        ax.set_aspect(np.diff(ax.get_xlim() / np.diff(ax.get_ylim())))

    fig.savefig(
        f"images/warm_disc_star_tracking_trajectory/{simulation}/"
        f"temporal_evolution{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main() -> None:
    settings = Settings()
    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args['config']}.yml"))

    #region Data Processing
    if args["recalculate"]:
        df = read_data(args["simulation"] + "_s127", config, tag_in_situ=True)
        component_at_birth, \
            circularity_at_birth = compute_properties_at_birth(df, config)
        df["ComponentTagAtBirth"] = component_at_birth
        df["CircularityAtBirth"] = circularity_at_birth

        df["CircularityDeltaNorm"] = (df["Circularity"] \
            - df["CircularityAtBirth"]) / df["CircularityAtBirth"]
        df["CircularityDelta"] = df["Circularity"] - df["CircularityAtBirth"]

        df = df[df["IsInSitu"] == 1]
        df.simulation = args["simulation"]

        # Select some N_TRACK IDs but given circularity delta condition
        N_TRACK: int = 10
        BIRTH_COMPONENT: int = settings.component_tags["CD"]
        TODAY_COMPONENT: str = settings.component_tags["WD"]
        CIRCULARITY_DELTA_RANGE: tuple = (-0.4, -0.35)
        AGE_RANGE_GYR: tuple = (10.0, 11.1)
        ids_to_track = np.random.choice(
            df["ID"][
                (df["ComponentTagAtBirth"] == BIRTH_COMPONENT) \
                    & (df["ComponentTag"] == TODAY_COMPONENT) \
                    & (df["CircularityDelta"] > CIRCULARITY_DELTA_RANGE[0]) \
                    & (df["CircularityDelta"] < CIRCULARITY_DELTA_RANGE[1]) \
                    & (df["StellarAge_Gyr"] > AGE_RANGE_GYR[0]) \
                    & (df["StellarAge_Gyr"] < AGE_RANGE_GYR[1])].to_numpy(),
            N_TRACK, replace=False)

        # Track positions for the selected IDs
        track_props = track_props_of_ids(
            simulation=args["simulation"],
            ids=ids_to_track,
            props_to_track=["xPosition_kpc", "yPosition_kpc",
                            "zPosition_kpc", "zAngularMomentumFraction",
                            "Circularity"],
            config=config)

        with open(f"results/{args['simulation']}/wd_track_trajectories.json",
                "w") as f:
            json.dump(track_props , f)
    #endregion

    #region Plotting
    with open(f"results/{args['simulation']}/wd_track_trajectories.json") as f:
        track_props = json.load(f)
    for pid in list(track_props.keys())[2:]:
        plot_trajectories(time=track_props["Time_Gyr"],
                          df=pd.DataFrame(track_props[pid]),
                          config=config, simulation=args["simulation"],
                          file_label=pid)
    plot_all_temporal_evolution(
        df=track_props,
        config=config,
        simulation=args["simulation"])
    #endregion


if __name__ == "__main__":
    main()
