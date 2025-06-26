"""
File:           track_warn_disc_stars_delta.py
Description:    Computes the properties of stars at z=0 and at their birth
                time, and generates figures showing the deltas between the
                two times.

Usage:          python scripts/track_warn_disc_stars_delta.py --simulation \
SIMULATION_NAME --config CONFIG_FILE

Arguments:
    --simulation    Name of the simulation to analyze (e.g., 'au6_or_l4').
    --config        Configuration filename for input parameters
                    (e.g., '02').
"""
import numpy as np
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt

from auriga.snapshot import Snapshot
from auriga.support import find_indices
from auriga.parser import parse
from auriga.paths import Paths
from auriga.settings import Settings
from auriga.images import figure_setup


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
        "Mass_Msun": s.mass[is_target],
        "SphericalRadius_kpc": s.r[is_target] * s.expansion_factor,
        "CylindricalRadius_kpc": s.rho[is_target] * s.expansion_factor,
        "Circularity": s.circularity[is_target],
        "zAngularMomentum_kpckm/s": s.get_specific_angular_momentum()[
            is_target, 2],
        "zAngularMomentumFraction": \
            s.get_specific_angular_momentum()[is_target, 2] \
                / np.linalg.norm(
                    s.get_specific_angular_momentum()[is_target, :],
                    axis=1),
        "AngularMomentumMagnitude_kpckm/s": np.linalg.norm(
            s.get_specific_angular_momentum()[is_target, :], axis=1)
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
    r_at_birth = np.nan * np.ones(len(df))
    rxy_at_birth = np.nan * np.ones(len(df))
    z_at_birth = np.nan * np.ones(len(df))
    circularity_at_birth = np.nan * np.ones(len(df))
    jz_at_birth = np.nan * np.ones(len(df))
    jz_frac_at_birth = np.nan * np.ones(len(df))
    j_at_birth = np.nan * np.ones(len(df))

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
        r_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["SphericalRadius_kpc"][idx].to_numpy() \
                * this_df.expansion_factor
        rxy_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["CylindricalRadius_kpc"][idx].to_numpy() \
                * this_df.expansion_factor
        z_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["zPosition_kpc"][idx].to_numpy() \
                * this_df.expansion_factor
        circularity_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["Circularity"][idx].to_numpy()
        jz_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["zAngularMomentum_kpckm/s"][idx].to_numpy()
        jz_frac_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["zAngularMomentumFraction"][idx].to_numpy()
        j_at_birth[(df["FormationSnapshot"] == i) \
            & df["IsInSitu"]] \
            = this_df["AngularMomentumMagnitude_kpckm/s"][idx].to_numpy()
    
    # Add component tags at birth for the last snapshot
    component_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["ComponentTag"][
                df["FormationSnapshot"] == 127].to_numpy()
    r_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["SphericalRadius_kpc"][
                df["FormationSnapshot"] == 127].to_numpy() \
                    * df.expansion_factor
    rxy_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["CylindricalRadius_kpc"][
                df["FormationSnapshot"] == 127].to_numpy() \
                    * df.expansion_factor
    z_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["zPosition_kpc"][
                df["FormationSnapshot"] == 127].to_numpy() \
                    * df.expansion_factor
    circularity_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["Circularity"][
                df["FormationSnapshot"] == 127].to_numpy()
    jz_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["zAngularMomentum_kpckm/s"][
                df["FormationSnapshot"] == 127].to_numpy()
    jz_frac_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["zAngularMomentumFraction"][
                df["FormationSnapshot"] == 127].to_numpy()
    j_at_birth[(df["FormationSnapshot"] == 127) \
        & df["IsInSitu"]] \
            = df["AngularMomentumMagnitude_kpckm/s"][
                df["FormationSnapshot"] == 127].to_numpy()

    return (component_at_birth, r_at_birth, rxy_at_birth,
            z_at_birth, circularity_at_birth, jz_at_birth, jz_frac_at_birth,
            j_at_birth)


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--simulation", type=str, required=True)
    args = parser.parse_args()
    return vars(args)


def get_arr_from_idx(idx: np.ndarray, s: pd.Series) -> np.ndarray:
    arr = s.to_numpy()[idx]
    arr[idx == -1] = np.nan
    return arr


def plot_distribution(df: pd.DataFrame, prop: str, x_label: str,
                      x_range: tuple, y_range: tuple, filename: str,
                      **kwargs) -> None:
    settings = Settings()
    label = "Au" + str(parse(df.simulation)[0])

    fig, ax = plt.subplots(figsize=(2.5, 2.5), ncols=1, nrows=1)

    ax.set_axisbelow(True)
    ax.set_xlim(x_range)
    ax.set_xlabel(x_label)
    if "xticks" in kwargs:
        ax.set_xticks(kwargs["xticks"])
    ax.set_ylim(y_range)
    ax.set_ylabel("PDF")
    ax.grid(True, ls="-", lw=0.25, color="gainsboro")

    x_scale_factor = kwargs.get("x_scale_factor", 1.0)

    for c in settings.components:
        ax.hist(
            x=df[prop][
                (df["ComponentTag"] == settings.component_tags["WD"]) \
                    & (df["ComponentTagAtBirth"] \
                        == settings.component_tags[c])] / x_scale_factor,
            color=settings.component_colors[c], bins=100, range=x_range,
            density=True, histtype="step", lw=0.9,
            label=settings.component_labels[c] + r" $\to$ Warm Disc")

    ax.legend(loc="upper right", fontsize=5, framealpha=0)
    ax.text(x=0.05, y=0.95, s=r"$\texttt{" + label + "}$",
            size=6.0, transform=ax.transAxes, ha='left', va='top')

    fig.savefig(
        f"images/warm_disc_star_tracking_delta/{df.simulation}/{filename}.pdf")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame,
                 x_prop: str, x_label: str, x_range: tuple,
                 y_prop: str, y_label: str, y_range: tuple,
                 filename: str,
                 x_ticks: list, y_ticks: list, **kwargs) -> None:
    settings = Settings()
    label = "Au" + str(parse(df.simulation)[0])

    x_scale_factor = kwargs.get("x_scale_factor", 1.0)
    y_scale_factor = kwargs.get("y_scale_factor", 1.0)

    fig = plt.figure(figsize=(8, 2.5))
    gs = fig.add_gridspec(nrows=1, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(x_range)
        ax.set_xlabel(x_label)
        ax.set_xticks(x_ticks)
        ax.set_ylim(y_range)
        ax.set_ylabel(y_label)
        ax.set_yticks(y_ticks)
        ax.grid(True, ls="-", lw=0.25, color="gainsboro")
        ax.label_outer()
        # ax.set_aspect("equal")

    for i, c in enumerate(settings.components):
        axs[i].scatter(
            x=df[x_prop][
                (df["ComponentTag"] == settings.component_tags["WD"]) \
                    & (df["ComponentTagAtBirth"] \
                        == settings.component_tags[c])] / x_scale_factor,
            y=df[y_prop][
                (df["ComponentTag"] == settings.component_tags["WD"]) \
                    & (df["ComponentTagAtBirth"] \
                        == settings.component_tags[c])] / y_scale_factor,
            color=settings.component_colors[c], s=0.25, alpha=0.05,
            label=settings.component_labels[c] + r" $\to$ Warm Disc")
        axs[i].legend(loc="lower right", fontsize=7, framealpha=0)

    axs[0].text(x=0.05, y=0.95, s=r"$\texttt{" + label + "}$",
                size=7, transform=axs[0].transAxes, ha='left', va='top')

    fig.savefig(
        f"images/warm_disc_star_tracking_delta/{df.simulation}/{filename}.pdf")
    plt.close(fig)


def main() -> None:
    figure_setup()
    args = get_user_input()
    config = yaml.safe_load(open(f"configs/{args['config']}.yml"))

    #region Data Processing
    df = read_data(args["simulation"] + "_s127", config, tag_in_situ=True)
    component_at_birth, r_at_birth, rxy_at_birth, \
        z_at_birth, circularity_at_birth, jz_at_birth, jz_frac_at_birth, \
            j_at_birth = compute_properties_at_birth(df, config)
    df["ComponentTagAtBirth"] = component_at_birth
    df["SphericalRadiusAtBirth_kpc"] = r_at_birth
    df["CylindricalRadiusAtBirth_kpc"] = rxy_at_birth
    df["zPositionAtBirth_kpc"] = z_at_birth
    df["CircularityAtBirth"] = circularity_at_birth
    df["zAngularMomentumAtBirth_kpckm/s"] = jz_at_birth
    df["zAngularMomentumFractionAtBirth"] = jz_frac_at_birth
    df["AngularMomentumMagnitudeAtBirth_kpckm/s"] = j_at_birth

    df["SphericalRadiusDeltaNorm"] = (df["SphericalRadius_kpc"] \
        - df["SphericalRadiusAtBirth_kpc"]) / df["SphericalRadiusAtBirth_kpc"]
    df["CylindricalRadiusDeltaNorm"] = (df["CylindricalRadius_kpc"] \
        - df["CylindricalRadiusAtBirth_kpc"]) \
            / df["CylindricalRadiusAtBirth_kpc"]
    df["zPositionDeltaNorm"] = (df["zPosition_kpc"] \
        - df["zPositionAtBirth_kpc"]) / df["zPositionAtBirth_kpc"]
    df["zPositionAbsDeltaNorm"] = (np.abs(df["zPosition_kpc"]) \
        - np.abs(df["zPositionAtBirth_kpc"])) \
            / np.abs(df["zPositionAtBirth_kpc"])
    df["zPositionDelta"] = df["zPosition_kpc"] - df["zPositionAtBirth_kpc"]
    df["CircularityDeltaNorm"] = (df["Circularity"] \
        - df["CircularityAtBirth"]) / df["CircularityAtBirth"]
    df["CircularityDelta"] = df["Circularity"] - df["CircularityAtBirth"]
    df["zAngularMomentumDelta_kpckm/s"] = df["zAngularMomentum_kpckm/s"] \
        - df["zAngularMomentumAtBirth_kpckm/s"]
    df["zAngularMomentumFractionDelta"] = df["zAngularMomentumFraction"] \
        - df["zAngularMomentumFractionAtBirth"]
    df["AngularMomentumMagnitudeDelta_kpckm/s"] = \
        df["AngularMomentumMagnitude_kpckm/s"] \
        - df["AngularMomentumMagnitudeAtBirth_kpckm/s"]

    df = df[df["IsInSitu"] == 1]
    df.simulation = args["simulation"]
    #endregion

    #region Plotting
    plot_distribution(
        df=df,
        prop="SphericalRadiusDeltaNorm",
        x_label=r'$\Delta r / r^\mathrm{birth}$',
        x_range=(-15, 15),
        y_range=(0, 0.6),
        filename="r_delta_norm",
        )
    plot_distribution(
        df=df,
        prop="CylindricalRadiusDeltaNorm",
        x_label=r'$\Delta r_{xy} / r_{xy}^\mathrm{birth}$',
        x_range=(-15, 15),
        y_range=(0, 0.6),
        filename="rxy_delta_norm",
        )
    plot_distribution(
        df=df,
        prop="zPositionAbsDeltaNorm",
        x_label=r'$\Delta \left| z \right| / \left| z^\mathrm{birth} \right|$',
        x_range=(-1, 15),
        y_range=(0, 0.5),
        filename="zabs_delta_norm",
        )
    plot_distribution(
        df=df,
        prop="zPositionDeltaNorm",
        x_label=r'$\Delta z / z^\mathrm{birth}$',
        x_range=(-15, 15),
        y_range=(0, 0.25),
        filename="z_delta_norm",
        )
    plot_distribution(
        df=df,
        prop="zPositionDelta",
        x_label=r'$\Delta z$ [kpc]',
        x_range=(-15, 15),
        y_range=(0, 0.8),
        filename="z_delta",
        )
    plot_distribution(
        df=df,
        prop="zAngularMomentumFractionDelta",
        x_label=r'$\Delta \left( j_z / \left| \mathbf{j} \right| \right)$',
        x_range=(-1, 1),
        y_range=(0, 10),
        filename="jz_frac_delta",
        )
    plot_distribution(
        df=df,
        prop="AngularMomentumMagnitudeDelta_kpckm/s",
        x_label=r'$\Delta \left| \mathbf{j} \right|$ '
                r'[$10^{3} \mathrm{kpc \, km \, s^{-1}}$]',
        x_range=(-2, 2),
        y_range=(0, 4),
        filename="j_delta",
        x_scale_factor=1E3,
        xticks=[-2, -1, 0, 1, 2],
        )
    plot_distribution(
        df=df,
        prop="CircularityDeltaNorm",
        x_label=r'$\Delta \epsilon / \epsilon^\mathrm{birth}$',
        x_range=(-1.5, 1.5),
        y_range=(0, 3),
        filename="circularity_delta_norm",
        )
    plot_distribution(
        df=df,
        prop="CircularityDelta",
        x_label=r'$\Delta \epsilon$',
        x_range=(-1.5, 1.5),
        y_range=(0, 3),
        filename="circularity_delta",
        )
    plot_distribution(
        df=df,
        prop="Circularity",
        x_label=r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$',
        x_range=(-1.5, 1.5),
        y_range=(0, 5),
        filename="circularity_today",
        )
    plot_distribution(
        df=df,
        prop="CircularityAtBirth",
        x_label=r'$\epsilon = j_z \, j_\mathrm{circ}^{-1}$',
        x_range=(-1.5, 1.5),
        y_range=(0, 5),
        filename="circularity_at_birth",
        )
    plot_scatter(
        df=df,
        x_prop="AngularMomentumMagnitudeDelta_kpckm/s",
        x_label=r"$\Delta \left| \mathbf{j} \right|$ "
                r"[$10^{3} \, \mathrm{kpc \, km \, s^{-1}}$]",
        x_range=(-2, 2),
        x_ticks=[-1, 0, 1],
        y_prop="zAngularMomentumDelta_kpckm/s",
        y_label=r"$\Delta j_z$ [$10^{3} \, \mathrm{kpc \, km \, s^{-1}}$]",
        y_range=(-2, 2),
        y_ticks=[-1, 0, 1],
        filename="jz_delta_vs_j_delta",
        x_scale_factor=1E3, y_scale_factor=1E3,
        )
    plot_scatter(
        df=df,
        x_prop="zPositionDeltaNorm",
        x_label=r"$\Delta z / z^\mathrm{birth}$",
        x_range=(-15, 15),
        x_ticks=[-10, -5, 0, 5, 10],
        y_prop="zPositionAtBirth_kpc",
        y_label=r"$z^\mathrm{birth}$ [kpc]",
        y_range=(-5, 5),
        y_ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        filename="z_birth_vs_z_delta_norm",
        )
    #endregion


if __name__ == "__main__":
    main()
