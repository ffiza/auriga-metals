import numpy as np
import os
from source.auriga.settings import Settings


def find_indices(a: np.array, b: np.array, invalid_specifier=-1) -> np.array:
    """"
    Returns an array with the indices of the elements of b
    in a. If an element of b is not in a, it returns
    invalid_specifier for that element.

    Parameters
    ----------
    a : np.array
        An array in which to search.
    b : np.array
        An array that contains the values to be searched for.
    invalid_specifier : optional
        A value to use in case the element is not found.

    Returns
    -------
    np.array
        An array with the indices.
    """

    sidx = a.argsort()
    idx = np.searchsorted(a, b, sorter=sidx)
    idx[idx == len(a)] = 0
    idx0 = sidx[idx]
    return np.where(a[idx0] == b, idx0, invalid_specifier)


def make_paths(galaxy: int, rerun: bool,
               resolution: int) -> tuple:
    """"
    Returns the correct absolute path to this repository data, figures and
    to the location of the simulation snapshots.

    Parameters
    ----------
    galaxy : int
        The number of the halo.
    rerun : bool
        A bool to indicate if this is a original run or a rerun.
    resolution : int
        The resolution level of the simulation.

    Returns
    -------
    tuple
        The three paths of interest (snapshots, data, figures).
    """

    settings = Settings()
    rerun_str = '_reruns' if rerun is True else ''
    resolution_str = f'_l{resolution}' if resolution != 4 else ''
    node_name = os.uname()[1]

    if node_name == 'virgo':
        snapshot_path = f'/{node_name}/simulations/Auriga/' + \
            f'level{resolution}_MHD{rerun_str}/halo_{galaxy}/output/'
        data_path = f'/u/fiza/{settings.repo_name}/data/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
        figure_path = f'/u/fiza/{settings.repo_name}/images/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
    elif node_name == 'neuromancer':
        snapshot_path = '/media/federico/Elements1/Simulations/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
        data_path = f'/home/federico/{settings.repo_name}/data/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
        figure_path = f'/home/federico/{settings.repo_name}/images/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
    elif node_name == 'wintermute':
        snapshot_path = '/home/fiza/simulations/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
        data_path = f'/home/fiza/{settings.repo_name}/data/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
        figure_path = f'/home/fiza/{settings.repo_name}/images/' + \
            f'au{galaxy}{resolution_str}{rerun_str}/'
    else:
        raise ValueError("Node name not implemented.")

    return snapshot_path, data_path, figure_path
