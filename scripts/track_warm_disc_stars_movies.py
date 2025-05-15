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
    settings = Settings()
    s = Snapshot(simulation=simulation, loadonlytype=[0, 1, 2, 3, 4, 5])
    s.tag_particles_by_region(
        disc_std_circ=config["DISC_STD_CIRC"],
        disc_min_circ=config["DISC_MIN_CIRC"],
        cold_disc_delta_circ=config["COLD_DISC_DELTA_CIRC"],
        bulge_max_specific_energy=config["BULGE_MAX_SPECIFIC_ENERGY"])
    
    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)
    is_main_obj = (s.halo == s.halo_idx) & (s.subhalo == s.subhalo_idx)
    is_target = is_real_star & is_main_obj

    jz_frac = s.get_specific_angular_momentum()[is_target, 2] \
        / np.linalg.norm(s.get_specific_angular_momentum()[is_target, :],
                         axis=1)

    props = {
        "ID": s.ids[is_target],
        "ComponentTag": s.region_tag[is_target],
        "yPosition_ckpc": s.pos[is_target, 1],
        "zPosition_ckpc": s.pos[is_target, 2],
        "Circularity": s.circularity[is_target],
        "Mass_Msun": s.mass[is_target],
        "zAngularMomentum_kpckm/s": s.get_specific_angular_momentum()[
            is_target, 2],
        "zAngularMomentumFraction": jz_frac,
        "AngularMomentumAbs_kpckm/s": np.linalg.norm(
            s.get_specific_angular_momentum()[is_target, :], axis=1),
        "SphericalRadius_ckpc": s.r[is_target],
        "CylindricalRadius_ckpc": s.rho[is_target],
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

    # Add additional data: vertical heights of cold disc
    df.z80_kpc = np.percentile(
        np.abs(df["zPosition_ckpc"][
            df["ComponentTag"] == settings.component_tags["CD"]]),
        80
    ) * df.expansion_factor

    return df


def compute_birth_component(df: pd.DataFrame, config: dict) -> np.ndarray:
    galaxy, rerun, _, _ = parse(df.simulation)
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"

    component_at_birth = -1 * np.ones(len(df), dtype=np.int8)
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

    # Add component tags at birth for the last snapshot
    component_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["ComponentTag"][
                df["FormationSnapshot"] == 127].to_numpy()

    return component_at_birth


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--simulation", type=str, required=True)
    args = parser.parse_args()
    return args


def get_arr_from_idx(idx: np.ndarray, s: pd.Series) -> np.ndarray:
    arr = s.to_numpy()[idx]
    arr[idx == -1] = np.nan
    return arr


def create_figures(simulation: str, config: dict) -> None:
    settings = Settings()

    # Get IDs to track
    print(f"Getting IDs to track... ", end="")
    today_df = read_data(simulation=simulation,
                         config=config,
                         tag_in_situ=True)
    today_df["ComponentTagAtBirth"] = compute_birth_component(df=today_df,
                                                              config=config)
    ids_to_track = today_df["ID"][
        (today_df["ComponentTag"] == settings.component_tags["WD"]) \
            & (today_df["IsInSitu"] == 1)].to_numpy()
    component_tag_at_birth = today_df["ComponentTagAtBirth"][
        (today_df["ComponentTag"] == settings.component_tags["WD"]) \
            & (today_df["IsInSitu"] == 1)].to_numpy()
    print("Done.")

    # Parse galaxy and set up paths
    galaxy, rerun, _, _ = parse(simulation)
    rerun_text = "_or" if not rerun else "_re"
    simulation = f"au{galaxy}{rerun_text}_l4"

    # Make plot for each snapshot
    for i in range(40, 128, 1):
        print(f"Analyzing snapshot {i}...")
        this_df = read_data(f"{simulation}_s{i}", config)

        idx = find_indices(this_df["ID"].to_numpy(), ids_to_track, -1)

        if idx.shape[0] < 1:
            continue

        # Create new dataframe with the properties need to create the figure
        y_pos = get_arr_from_idx(idx, this_df["yPosition_ckpc"])
        z_pos = get_arr_from_idx(idx, this_df["zPosition_ckpc"])
        mass = get_arr_from_idx(idx, this_df["Mass_Msun"])
        jz = get_arr_from_idx(idx, this_df["zAngularMomentum_kpckm/s"])
        circularity = get_arr_from_idx(idx, this_df["Circularity"])
        jzfrac = get_arr_from_idx(idx, this_df["zAngularMomentumFraction"])
        j = get_arr_from_idx(idx, this_df["AngularMomentumAbs_kpckm/s"])
        r = get_arr_from_idx(idx, this_df["SphericalRadius_ckpc"])
        rxy = get_arr_from_idx(idx, this_df["CylindricalRadius_ckpc"])
        data = pd.DataFrame({
            "yPosition_kpc": y_pos * this_df.expansion_factor,
            "zPosition_kpc": z_pos * this_df.expansion_factor,
            "Mass_Msun": mass,
            "ComponentAtBirth": component_tag_at_birth,
            "zAngularMomentum_kpckm/s": jz,
            "Circularity": circularity,
            "zAngularMomentumFraction": jzfrac,
            "AngularMomentumAbs_kpckm/s": j,
            "SphericalRadius_kpc": r * this_df.expansion_factor,
            "CylindricalRadius_kpc": rxy * this_df.expansion_factor,
        })
        data.time = this_df.time
        data.simulation = this_df.simulation
        data.redshift = this_df.redshift
        data.snapshot = i
        data.z80_kpc = this_df.z80_kpc

        # Create figure for this snapshot
        plot_density_map(data)
        plot_jz_distribution(data)
        plot_circularity_distribution(data)
        plot_jzfrac_distribution(data)
        plot_j_distribution(data)
        plot_z_distribution(data)
        plot_r_distribution(data)
        plot_rxy_distribution(data)
    print("Done.")


def plot_density_map(data: pd.DataFrame) -> None:
    N_BINS = 200
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
        data["yPosition_kpc"], data["zPosition_kpc"],
        weights=data["Mass_Msun"],
        bins=N_BINS, range=[[-BOX_SIZE / 2, BOX_SIZE / 2],
                            [-BOX_SIZE / 2, BOX_SIZE / 2]],
        cmap="viridis", zorder=-10,
        norm=mcolors.LogNorm(vmin=1E4, vmax=1E9))
    
    # Draw cold disc data
    axbig.plot(axbig.get_xlim(), [data.z80_kpc] * 2, ls="--", lw=0.5, c="k")
    axbig.plot(axbig.get_xlim(), [-data.z80_kpc] * 2, ls="--", lw=0.5, c="k")
    for ax in axs.flatten():
        ax.plot(ax.get_xlim(), [data.z80_kpc] * 2, ls="--", lw=0.5, c="k")
        ax.plot(ax.get_xlim(), [-data.z80_kpc] * 2, ls="--", lw=0.5, c="k")

    component_axs = {"H": (2, 0), "B": (2, 1), "CD": (0, 2), "WD": (1, 2)}
    for c in settings.components:
        axs[component_axs[c]].hist2d(
            data["yPosition_kpc"][
                data["ComponentAtBirth"] == settings.component_tags[c]],
            data["zPosition_kpc"][
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
    axs[-1, -2].text(s="Box [$\mathrm{kpc}^3$]: $100^3$", x=1.05, y=0.5,
                     ha="left", va="center", transform=axs[-1, -2].transAxes)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/density_maps/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_jz_distribution(data: pd.DataFrame) -> None:
    N_BINS = 50
    X_RANGE = (0, 4E3)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlabel(r'$j_z$ $\mathrm{[kpc \, km \, s^{-1}]}$')
    ax.set_ylim(0, 0.002)
    ax.set_ylabel("PDF")

    ax.hist(x=data["zAngularMomentum_kpckm/s"], color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=data["zAngularMomentum_kpckm/s"][
            data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.5, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.5,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.5, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.5, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/jz/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_j_distribution(data: pd.DataFrame) -> None:
    N_BINS = 50
    X_RANGE = (0, 4E3)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlabel(
        r'$\left| \mathbf{j} \right|$ $\mathrm{[kpc \, km \, s^{-1}]}$')
    ax.set_ylim(0, 0.002)
    ax.set_ylabel("PDF")

    ax.hist(x=data["AngularMomentumAbs_kpckm/s"], color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=data["AngularMomentumAbs_kpckm/s"][
            data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.5, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.5,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.5, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.5, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/j/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_z_distribution(data: pd.DataFrame) -> None:
    N_BINS = 50
    X_RANGE = (0, 20)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel(
        r'$\left| z \right|$ [kpc]')
    ax.set_ylim(0, 1)
    ax.set_ylabel("PDF")

    ax.hist(x=np.abs(data["zPosition_kpc"]), color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=np.abs(data["zPosition_kpc"][
            data["ComponentAtBirth"] == settings.component_tags[c]]),
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.5, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.5,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.5, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.5, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/z/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_circularity_distribution(data: pd.DataFrame) -> None:
    N_BINS = 50
    X_RANGE = (-1.5, 1.5)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xlabel(r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$')
    ax.set_ylim(0, 4)
    ax.set_ylabel("PDF")

    ax.hist(x=data["Circularity"], color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=data["Circularity"][
            data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.05, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.05,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.05, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.05, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/circularity/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_jzfrac_distribution(data: pd.DataFrame) -> None:
    N_BINS = 50
    X_RANGE = (-1, 1)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xlabel(r'$j_z / \left| \mathbf{j} \right|$')
    ax.set_ylim(0, 4)
    ax.set_ylabel("PDF")

    ax.hist(x=data["zAngularMomentumFraction"], color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=data["zAngularMomentumFraction"][
            data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.05, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.05,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.05, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.05, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/jz_frac/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_r_distribution(data: pd.DataFrame) -> None:
    N_BINS = 80
    X_RANGE = (0, 40)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylim(0, 0.4)
    ax.set_ylabel("PDF")

    ax.hist(x=data["SphericalRadius_kpc"], color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=data["SphericalRadius_kpc"][
            data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.05, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.05,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.05, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.05, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/r/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def plot_rxy_distribution(data: pd.DataFrame) -> None:
    N_BINS = 80
    X_RANGE = (0, 40)
    settings = Settings()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(X_RANGE)
    ax.set_xlabel(r'$r_{xy}$ [kpc]')
    ax.set_ylim(0, 0.4)
    ax.set_ylabel("PDF")

    ax.hist(x=data["CylindricalRadius_kpc"], color="k",
            bins=N_BINS, range=X_RANGE, density=True, histtype="step")
    for c in settings.components:
        ax.hist(x=data["CylindricalRadius_kpc"][
            data["ComponentAtBirth"] == settings.component_tags[c]],
            bins=N_BINS, range=X_RANGE, density=True, histtype="step",
            color=settings.component_colors[c],
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.text(s=f"Time [Gyr]: {np.round(data.time, 2)}", x=0.05, y=0.95,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Redshift: {np.round(data.redshift, 2)}", x=0.05,
            y=0.9, ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Snapshot: {int(data.snapshot)}", x=0.05, y=0.85,
            ha="left", va="center", transform=ax.transAxes, size=6.0)
    ax.text(s=f"Galaxy: Au{parse(data.simulation)[0]}", x=0.05, y=0.8,
            ha="left", va="center", transform=ax.transAxes, size=6.0)

    fig.savefig(
        "images/warm_disc_star_tracking/temp/rxy/"
        f"snapshot_{int(data.snapshot)}.png")
    plt.close(fig)


def main() -> None:
    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    create_figures(simulation=args.simulation, config=config)

if __name__ == "__main__":
    main()
