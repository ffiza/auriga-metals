import numpy as np
import time
from typing import Callable
import os


def find_indices(a: np.array, b: np.array,
                 invalid_specifier: int = -1) -> np.array:
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


def snapshot_path(galaxy: int, rerun: bool, resolution: int) -> str:
    """
    This method creates the path to the snapshot files based on the
    simulation.

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
    str
        The path to the snapshot files.
    """

    if os.uname()[1] == 'virgo':
        if rerun:
            dir_name = 'RerunsHighFreqStellarSnaps'
        else:
            dir_name = 'Original'

        snapshot_path = f'/virgotng/mpa/Auriga/level{resolution}/' + \
            f'{dir_name}/halo_{galaxy}/output/'
        return snapshot_path
    elif os.uname()[1] == 'neuromancer':
        if rerun:
            rerun_text = '_rerun'
        else:
            rerun_text = ''

        if (galaxy != 6) or (resolution != 4):
            raise Exception('Only halo 6 with resolution 4 is stored locally.')

        snapshot_path = '/media/federico/Elements1/Simulations/' + \
            f'au{galaxy}{rerun_text}/'
        return snapshot_path


def timer(method: Callable) -> Callable:
    """
    A decorator to monitor the time it takes to run a function.

    Parameters
    ----------
    method : func
        A method to decorate.

    Returns
    -------
    wrapper : Callable
        A wrapper.
    """
    def wrapper(*args, **kw):
        start_time = int(round(time.time()))
        result = method(*args, **kw)
        end_time = int(round(time.time()))

        print(f'Timer: {end_time-start_time} s.')
        return result

    return wrapper
