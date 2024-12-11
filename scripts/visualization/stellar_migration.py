from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
import argparse

from auriga.images import figure_setup
from auriga.settings import Settings
from auriga.parser import parse


def plot_migration_matrix(simulation: str, config: dict):
    galaxy, _, _ = parse(simulation)
    label = f"Au{galaxy}"
    settings = Settings()

    migration_matrix = np.load(
        f"results/{simulation}/migration_matrix{config['FILE_SUFFIX']}.npy")

    fig = plt.figure(figsize=(3.0, 3.0))
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex=True, sharey=False)

    ax.matshow(migration_matrix, cmap="viridis", vmin=0, vmax=100)

    ax.set_xlabel(r"Component at Birth")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(settings.components)
    ax.tick_params(axis='x', bottom=True, top=False,
                   labelbottom=True, labeltop=False)

    ax.set_ylabel(r"Component at $z=0$")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(settings.components)

    for i in range(migration_matrix.shape[0]):
        for j in range(migration_matrix.shape[1]):
            color = "white" if migration_matrix[j, i] < 50 else "black"
            ax.text(i, j, r"$\mathbf{" \
                + f"{np.round(migration_matrix[j, i], 1)}" + r"\%}$",
                c=color, ha="center", va="center", size=10)

    ax.text(x=0.01, y=1.01, size=7.0, s=r"$\texttt{" + label + "}$",
            ha="left", va="bottom", transform=ax.transAxes)

    fig.savefig(f"images/stellar_migration/{simulation}/"
                f"migration_matrix{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_stellar_age_distribution(simulation: str, config: dict):
    settings = Settings()
    label = f"Au{parse(simulation)[0]}"

    fig = plt.figure(figsize=(8.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.grid(True, ls=(0, (5, 1)), lw=0.1, c='silver')
        ax.set_xlabel("Stellar Age [Gyr]")
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_ylabel(f"$f_\star$")
        ax.set_ylim(0, 0.12)
        ax.set_yticks([0.02 * i for i in range(6)])
        ax.label_outer()
        ax.set_axisbelow(True)

    data = pd.read_csv(
        f"results/{simulation}/"
        f"stellar_migration_age_distribution{config['FILE_SUFFIX']}.csv")
    for i, c0 in enumerate(settings.components):
        s = r"$\textbf{" + settings.component_labels[c0] + r"}$"
        for _ in range(2):
            axs[_, i].text(x=0.05, y=0.95, s=s, transform=axs[_, i].transAxes,
                        ha="left", va="top", size=8.0,
                        color=settings.component_colors[c0])
        l1, = axs[0, i].plot(
            data["StellarAge_Gyr"],
            data[f"StellarMassFraction_{i}"],
            c='k', ls="-", label="All")
        l2, = axs[0, i].plot(
            data["StellarAge_Gyr"], data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l2, = axs[1, i].plot(
            data["StellarAge_Gyr"], data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l3, = axs[0, i].plot(
            data["StellarAge_Gyr"], data[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)), label="Ex Situ")
        lines = []
        for j, c1 in enumerate(settings.components):
            l, = axs[1, i].plot(
                data["StellarAge_Gyr"],
                data[f"StellarMassFraction_{j}to{i}"],
                c=settings.component_colors[c1], ls=(0, (5, 1)),
                label=settings.component_labels[c1])
            lines.append(l)

        if i == 0:
            legend1 = axs[0, 0].legend(
                handles=[l1, l2, l3], loc="upper right",
                fontsize=6.5, framealpha=0)
            legend2 = axs[1, 0].legend(
                handles=lines, loc="upper right", fontsize=6.5, framealpha=0,
                title="Origin", title_fontsize=7.0)

    axs[0, 0].text(x=0.05, y=0.85, size=9.0,
                s=r"$\texttt{" + str(label) + "}$",
                ha="left", va="top", transform=axs[0, 0].transAxes)

    fig.savefig(f"images/stellar_migration/{simulation}/"
                f"age_distribution{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_stellar_age_distribution_for_sample_median(sample: list,
                                                    config: dict):
    settings = Settings()

    # Get median data
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_age_distribution{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    data = df_concat.groupby(df_concat.index).median()

    fig = plt.figure(figsize=(8.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.grid(True, ls=(0, (5, 1)), lw=0.1, c='silver')
        ax.set_xlabel("Stellar Age [Gyr]")
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_ylabel(f"$f_\star$")
        ax.set_ylim(0, 0.12)
        ax.set_yticks([0.02 * i for i in range(6)])
        ax.label_outer()
        ax.set_axisbelow(True)

    for i, c0 in enumerate(settings.components):
        s = r"$\textbf{" + settings.component_labels[c0] + r"}$"
        for _ in range(2):
            axs[_, i].text(x=0.05, y=0.95, s=s, transform=axs[_, i].transAxes,
                        ha="left", va="top", size=8.0,
                        color=settings.component_colors[c0])
        l1, = axs[0, i].plot(
            data["StellarAge_Gyr"],
            data[f"StellarMassFraction_{i}"],
            c='k', ls="-", label="All")
        l2, = axs[0, i].plot(
            data["StellarAge_Gyr"], data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l2, = axs[1, i].plot(
            data["StellarAge_Gyr"], data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l3, = axs[0, i].plot(
            data["StellarAge_Gyr"], data[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)), label="Ex Situ")
        lines = []
        for j, c1 in enumerate(settings.components):
            l, = axs[1, i].plot(
                data["StellarAge_Gyr"],
                data[f"StellarMassFraction_{j}to{i}"],
                c=settings.component_colors[c1], ls=(0, (5, 1)),
                label=settings.component_labels[c1])
            lines.append(l)

        if i == 0:
            legend1 = axs[0, 0].legend(
                handles=[l1, l2, l3], loc="upper right",
                fontsize=6.5, framealpha=0)
            legend2 = axs[1, 0].legend(
                handles=lines, loc="upper right", fontsize=6.5, framealpha=0,
                title="Origin", title_fontsize=7.0)

    fig.savefig(f"images/stellar_migration/statistics/"
                f"age_distribution_median{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_abundance_distribution(simulation: str, abundance: tuple,
                                config: dict,
                                xlim: tuple, ylim: tuple,
                                xticks: list, yticks: list):
    of, to = abundance
    settings = Settings()
    label = f"Au{parse(simulation)[0]}"

    fig = plt.figure(figsize=(8.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.grid(True, ls=(0, (5, 1)), lw=0.1, c='silver')
        ax.set_xlabel(f"[{of}/{to}]")
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylabel(f"$f_\star$")
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.label_outer()
        ax.set_axisbelow(True)

    data = pd.read_csv(
        f"results/{simulation}/"
        f"stellar_migration_abundance_{of}{to}_"
        f"dist{config['FILE_SUFFIX']}.csv")
    for i, c0 in enumerate(settings.components):
        s = r"$\textbf{" + settings.component_labels[c0] + r"}$"
        for _ in range(2):
            axs[_, i].text(x=0.05, y=0.95, s=s, transform=axs[_, i].transAxes,
                        ha="left", va="top", size=8.0,
                        color=settings.component_colors[c0])
        l1, = axs[0, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}"],
            c='k', ls="-", label="All")
        l2, = axs[0, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l2, = axs[1, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l3, = axs[0, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)), label="Ex Situ")
        lines = []
        for j, c1 in enumerate(settings.components):
            l, = axs[1, i].plot(data[f"Abundance_[{of}/{to}]"],
                                data[f"StellarMassFraction_{j}to{i}"],
                                ls=(0, (5, 1)),
                                c=settings.component_colors[c1],
                                label=settings.component_labels[c1])
            lines.append(l)

        if i == 0:
            axs[0, 0].legend(handles=[l1, l2, l3], loc="upper right",
                             fontsize=6.5, framealpha=0)
            axs[1, 0].legend(handles=lines, loc="upper right",
                             fontsize=6.5, framealpha=0, title="Origin",
                             title_fontsize=7.0)

    label = f"Au{parse(simulation)[0]}"
    axs[0, 0].text(x=0.05, y=0.85, size=9.0,
                s=r"$\texttt{" + str(label) + "}$",
                ha="left", va="top", transform=axs[0, 0].transAxes)

    fig.savefig(f"images/stellar_migration/{simulation}/"
                f"{of}{to}_distribution{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_migration_matrix_for_sample(sample: list, config: dict):
    settings = Settings()

    fig, axs = plt.subplots(figsize=(8.0, 8.0), nrows=6, ncols=4,
                            sharey=True, sharex=True,
                            gridspec_kw={"hspace": 0.15, "wspace": -0.1})

    for i, simulation in enumerate(sample):
        label = f"Au{parse(simulation)[0]}"
        ax = axs.flatten()[i]
        migration_matrix = np.load(
            f"results/{simulation}/"
            f"migration_matrix{config['FILE_SUFFIX']}.npy")
        ax.matshow(migration_matrix, cmap="viridis", vmin=0, vmax=100)
        ax.text(x=0.01, y=1.01, size=6.0,
                s=r"$\texttt{" + label + "}$",
                ha="left", va="bottom", transform=ax.transAxes)

        for a in range(migration_matrix.shape[0]):
            for b in range(migration_matrix.shape[1]):
                color = "white" if migration_matrix[b, a] < 50 else "black"
                ax.text(
                    a, b,
                    r"$\mathbf{" + f"{np.round(migration_matrix[b, a], 1)}" \
                        + r"\%}$",
                    c=color, ha="center", va="center", size=3.5)

    for ax in axs.flatten():
        ax.tick_params(which='both', direction="in")
        if ax == axs[-1, -1]: ax.axis("off")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(settings.components)
        ax.set_axisbelow(True)
        if ax.get_subplotspec().is_last_row() or ax == axs[-2, -1]:
            ax.set_xlabel(r"Comp. at Birth")
            ax.tick_params(axis='x', bottom=True, top=False,
                           labelbottom=True, labeltop=False)
        if ax.get_subplotspec().is_first_col():
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(settings.components)
            ax.set_ylabel(r"Comp. at $z=0$")
        else:
            ax.tick_params(axis='y', left=False, right=False,
                           labelbottom=False, labeltop=False)
        ax.xaxis.label.set_size(8.0)
        ax.yaxis.label.set_size(8.0)

    fig.savefig(
        f"images/stellar_migration/sample/included{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_migration_matrix_for_sample_median(sample: list, config: dict):
    settings = Settings()

    migration_matrices = np.nan * np.ones((4, 4, len(sample)))
    for i, simulation in enumerate(sample):
        migration_matrices[:, :, i] = np.load(
            f"results/{simulation}/"
            f"migration_matrix{config['FILE_SUFFIX']}.npy")
    
    median = np.median(migration_matrices, axis=2)
    min_val = np.min(migration_matrices, axis=2)
    max_val = np.max(migration_matrices, axis=2)

    fig = plt.figure(figsize=(3.0, 3.0))
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.0, wspace=0.0)
    ax = gs.subplots(sharex=True, sharey=False)

    ax.matshow(median, cmap="viridis", vmin=0, vmax=100)

    ax.set_xlabel(r"Component at Birth")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(settings.components)
    ax.tick_params(axis='x', bottom=True, top=False,
                   labelbottom=True, labeltop=False)

    ax.set_ylabel(r"Component at $z=0$")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(settings.components)

    for i in range(median.shape[0]):
        for j in range(median.shape[1]):
            color = "white" if median[j, i] < 50 else "black"
            ax.text(i, j, r"$\mathbf{" \
                + f"{np.round(median[j, i], 1)}" + r"\%}$",
                c=color, ha="center", va="center", size=10.0)
            ax.text(i, j - 0.25, r"$(" \
                + f"{np.round(max_val[j, i], 1)}" + r"\%)$",
                c=color, ha="center", va="center", size=5.0)
            ax.text(i, j + 0.25, r"$(" \
                + f"{np.round(min_val[j, i], 1)}" + r"\%)$",
                c=color, ha="center", va="center", size=5.0)

    fig.savefig(f"images/stellar_migration/statistics/"
                f"migration_matrix_median{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_abundance_distribution_for_sample_median(
        sample: list, abundance: tuple, config: dict,
        xlim: tuple, ylim: tuple, xticks: list, yticks: list):
    of, to = abundance
    settings = Settings()

    # Get median data
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_abundance_{of}{to}_"
            f"dist{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    data = df_concat.groupby(df_concat.index).median()

    fig = plt.figure(figsize=(8.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=4, hspace=0.0, wspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.grid(True, ls=(0, (5, 1)), lw=0.1, c='silver')
        ax.set_xlabel(f"[{of}/{to}]")
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_ylabel(f"$f_\star$")
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.label_outer()
        ax.set_axisbelow(True)

    for i, c0 in enumerate(settings.components):
        s = r"$\textbf{" + settings.component_labels[c0] + r"}$"
        for _ in range(2):
            axs[_, i].text(x=0.05, y=0.95, s=s, transform=axs[_, i].transAxes,
                        ha="left", va="top", size=8.0,
                        color=settings.component_colors[c0])
        l1, = axs[0, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}"],
            c='k', ls="-", label="All")
        l2, = axs[0, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l2, = axs[1, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l3, = axs[0, i].plot(
            data[f"Abundance_[{of}/{to}]"],
            data[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)), label="Ex Situ")
        lines = []
        for j, c1 in enumerate(settings.components):
            l, = axs[1, i].plot(data[f"Abundance_[{of}/{to}]"],
                                data[f"StellarMassFraction_{j}to{i}"],
                                ls=(0, (5, 1)),
                                c=settings.component_colors[c1],
                                label=settings.component_labels[c1])
            lines.append(l)

        if i == 0:
            axs[0, 0].legend(handles=[l1, l2, l3], loc="upper right",
                             fontsize=6.5, framealpha=0)
            axs[1, 0].legend(handles=lines, loc="upper right",
                             fontsize=6.5, framealpha=0, title="Origin",
                             title_fontsize=7.0)

    fig.savefig(f"images/stellar_migration/statistics/"
                f"{of}{to}_distribution_median{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_present_day_distribution_for_sample_median(sample: list,
                                                    config: dict):
    settings = Settings()

    # Stellar ages
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_age_distribution{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    stellar_age = df_concat.groupby(df_concat.index).median()

    # Iron abundance
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_abundance_FeH_"
            f"dist{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    fe_abundance = df_concat.groupby(df_concat.index).median()

    # Oxygen abundance
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_abundance_OFe_"
            f"dist{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    o_abundance = df_concat.groupby(df_concat.index).median()

    fig = plt.figure(figsize=(8.0, 6.0))
    gs = fig.add_gridspec(nrows=3, ncols=4, hspace=0.3, wspace=0.0)
    axs = gs.subplots(sharex=False, sharey=False)

    for ax in axs.flatten():
        ax.grid(True, ls=(0, (5, 1)), lw=0.1, c='silver')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f"$f_\star$")
        else:
            ax.set_yticklabels([])
        ax.set_axisbelow(True)

    for ax in axs.flatten():
        if ax.get_subplotspec().is_first_row():
            ax.set_xlabel("Stellar Age [Gyr]")
            ax.set_ylim(0, 0.06)
            ax.set_xlim(0, 14)
            ax.set_xticks([2, 4, 6, 8, 10, 12])
        elif ax.get_subplotspec().is_last_row():
            ax.set_xlabel("[O/Fe]")
            ax.set_ylim(0, 0.08)
            ax.set_xlim(0.1, 0.4)
            ax.set_xticks([0.1, 0.2, 0.3])
        else:
            ax.set_xlabel("[Fe/H]")
            ax.set_ylim(0, 0.1)
            ax.set_xlim(-2.5, 1)
            ax.set_xticks([-2, -1, 0])

    for i, c0 in enumerate(settings.components):
        s = r"$\textbf{" + settings.component_labels[c0] + r"}$"
        for _ in range(3):
            axs[_, i].text(x=0.05, y=0.95, s=s, transform=axs[_, i].transAxes,
                        ha="left", va="top", size=8.0,
                        color=settings.component_colors[c0])
        l1, = axs[0, i].plot(
            stellar_age["StellarAge_Gyr"],
            stellar_age[f"StellarMassFraction_{i}"],
            c='k', ls="-", label="All")
        l2, = axs[0, i].plot(
            stellar_age["StellarAge_Gyr"],
            stellar_age[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)), label="In Situ")
        l3, = axs[0, i].plot(
            stellar_age["StellarAge_Gyr"],
            stellar_age[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)), label="Ex Situ")
        axs[1, i].plot(
            fe_abundance["Abundance_[Fe/H]"],
            fe_abundance[f"StellarMassFraction_{i}"],
            c='k', ls="-")
        axs[1, i].plot(
            fe_abundance["Abundance_[Fe/H]"],
            fe_abundance[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)))
        axs[1, i].plot(
            fe_abundance["Abundance_[Fe/H]"],
            fe_abundance[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)))
        axs[2, i].plot(
            o_abundance["Abundance_[O/Fe]"],
            o_abundance[f"StellarMassFraction_{i}"],
            c='k', ls="-")
        axs[2, i].plot(
            o_abundance["Abundance_[O/Fe]"],
            o_abundance[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)))
        axs[2, i].plot(
            o_abundance["Abundance_[O/Fe]"],
            o_abundance[f"StellarMassFraction_{i}_ExSitu"],
            c='k', ls=(0, (1, 1)))

        if i == 0:
            axs[0, 0].legend(
                handles=[l1, l2, l3], loc="center left",
                fontsize=6.5, framealpha=0)

    fig.savefig(f"images/stellar_migration/statistics/"
                f"present_day_distribution_median{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def plot_origin_distribution_for_sample_median(sample: list, config: dict):
    settings = Settings()

    # Stellar ages
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_age_distribution{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    stellar_age = df_concat.groupby(df_concat.index).median()

    # Iron abundance
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_abundance_FeH_"
            f"dist{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    fe_abundance = df_concat.groupby(df_concat.index).median()

    # Oxygen abundance
    dfs = []
    for simulation in sample:
        df = pd.read_csv(
            f"results/{simulation}/"
            f"stellar_migration_abundance_OFe_"
            f"dist{config['FILE_SUFFIX']}.csv")
        dfs.append(df)
    df_concat = pd.concat(dfs)
    o_abundance = df_concat.groupby(df_concat.index).median()

    fig = plt.figure(figsize=(8.0, 6.0))
    gs = fig.add_gridspec(nrows=3, ncols=4, hspace=0.3, wspace=0.0)
    axs = gs.subplots(sharex=False, sharey=False)

    for ax in axs.flatten():
        ax.grid(True, ls=(0, (5, 1)), lw=0.1, c='silver')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f"$f_\star$")
        else:
            ax.set_yticklabels([])
        ax.set_axisbelow(True)

    for ax in axs.flatten():
        if ax.get_subplotspec().is_first_row():
            ax.set_xlabel("Stellar Age [Gyr]")
            ax.set_ylim(0, 0.04)
            ax.set_xlim(0, 14)
            ax.set_xticks([2, 4, 6, 8, 10, 12])
        elif ax.get_subplotspec().is_last_row():
            ax.set_xlabel("[O/Fe]")
            ax.set_ylim(0, 0.08)
            ax.set_xlim(0.1, 0.4)
            ax.set_xticks([0.1, 0.2, 0.3])
        else:
            ax.set_xlabel("[Fe/H]")
            ax.set_ylim(0, 0.1)
            ax.set_xlim(-2.5, 1)
            ax.set_xticks([-2, -1, 0])

    for i, c0 in enumerate(settings.components):
        s = r"$\textbf{" + settings.component_labels[c0] + r"}$"
        for _ in range(3):
            axs[_, i].text(x=0.05, y=0.95, s=s, transform=axs[_, i].transAxes,
                        ha="left", va="top", size=8.0,
                        color=settings.component_colors[c0])
        axs[0, i].plot(
            stellar_age["StellarAge_Gyr"],
            stellar_age[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)))
        axs[1, i].plot(
            fe_abundance["Abundance_[Fe/H]"],
            fe_abundance[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)))
        axs[2, i].plot(
            o_abundance["Abundance_[O/Fe]"],
            o_abundance[f"StellarMassFraction_{i}_InSitu"],
            c='k', ls=(0, (5, 1)))
        lines = []
        for j, c1 in enumerate(settings.components):
            l, = axs[0, i].plot(stellar_age["StellarAge_Gyr"],
                                stellar_age[f"StellarMassFraction_{j}to{i}"],
                                ls=(0, (5, 1)),
                                c=settings.component_colors[c1],
                                label=settings.component_labels[c1])
            lines.append(l)
            axs[1, i].plot(fe_abundance[f"Abundance_[Fe/H]"],
                           fe_abundance[f"StellarMassFraction_{j}to{i}"],
                           ls=(0, (5, 1)),
                           c=settings.component_colors[c1],
                           label=settings.component_labels[c1])
            axs[2, i].plot(o_abundance[f"Abundance_[O/Fe]"],
                           o_abundance[f"StellarMassFraction_{j}to{i}"],
                           ls=(0, (5, 1)),
                           c=settings.component_colors[c1],
                           label=settings.component_labels[c1])

        if i == 0:
            axs[1, 0].legend(
                handles=lines, loc="upper right", fontsize=6.5, framealpha=0,
                title="Origin", title_fontsize=7.0)

    fig.savefig(f"images/stellar_migration/statistics/"
                f"origin_distribution_median{config['FILE_SUFFIX']}.pdf")
    plt.close(fig)


def main():
    settings = Settings()
    sample=[f"au{i}_or_l4" for i in settings.groups["Included"]]

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # Create figures
    figure_setup()
    # plot_migration_matrix(simulation="au6_or_l4", config=config)
    # plot_migration_matrix_for_sample(sample=sample, config=config)
    # plot_migration_matrix_for_sample_median(sample=sample, config=config)
    # plot_stellar_age_distribution(simulation="au6_or_l4", config=config)
    # plot_stellar_age_distribution_for_sample_median(sample=sample,
    #                                                 config=config)
    # plot_abundance_distribution(
    #     simulation="au6_or_l4", abundance=("Fe", "H"), config=config,
    #     xlim=(-2.5, 1), ylim=(0, 0.12),
    #     xticks=[-2, -1, 0], yticks=[0.02 * i for i in range(6)])
    # plot_abundance_distribution(
    #     simulation="au6_or_l4", abundance=("O", "H"), config=config,
    #     xlim=(-2, 1.5), ylim=(0, 0.12),
    #     xticks=[-2, -1, 0, 1], yticks=[0.02 * i for i in range(6)])
    # plot_abundance_distribution(
    #     simulation="au6_or_l4", abundance=("O", "Fe"), config=config,
    #     xlim=(0.1, 0.4), ylim=(0, 0.12),
    #     xticks=[0.1, 0.2, 0.3], yticks=[0.02 * i for i in range(6)])
    # plot_abundance_distribution_for_sample_median(
    #     sample=sample, abundance=("Fe", "H"), config=config,
    #     xlim=(-2.5, 1), ylim=(0, 0.12),
    #     xticks=[-2, -1, 0], yticks=[0.02 * i for i in range(6)])
    # plot_abundance_distribution_for_sample_median(
    #     sample=sample, abundance=("O", "H"), config=config,
    #     xlim=(-2, 1.5), ylim=(0, 0.12),
    #     xticks=[-2, -1, 0, 1], yticks=[0.02 * i for i in range(6)])
    # plot_abundance_distribution_for_sample_median(
    #     sample=sample, abundance=("O", "Fe"), config=config,
    #     xlim=(0.1, 0.4), ylim=(0, 0.12),
        # xticks=[0.1, 0.2, 0.3], yticks=[0.02 * i for i in range(6)])
    
    # plot_present_day_distribution_for_sample_median(
    #     sample=sample, config=config)
    plot_origin_distribution_for_sample_median(
        sample=sample, config=config)


if __name__ == "__main__":
    main()
