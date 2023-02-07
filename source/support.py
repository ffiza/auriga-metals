import numpy as np
import time
from typing import Callable
from sys import stdout
import pandas as pd
from os.path import exists


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
    invalid_specifier : int, optional
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


def timer(method: Callable) -> Callable:
    """
    A decorator to monitor the time it takes to run a function.

    Parameters
    ----------
    method : Callable
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

        stdout.write(f"Timer: {end_time-start_time} s.")
        return result

    return wrapper


def get_bool_input(msg: str) -> bool:
    """This method get the input from the user by displaying a message
    and returns the result as a boolean.

    Parameters
    ----------
    msg : str
        A message to display.

    Returns
    -------
    bool
        The user input as a boolean.
    """

    rsp = None
    while rsp is None:
        rsp = input(f"{msg} [y/n]")
        if rsp == 'y':
            rsp = True
        elif rsp == 'n':
            rsp = False
        else:
            rsp = None
    return rsp


def find_idx_ksmallest(arr: np.ndarray, k: int) -> np.ndarray:
    """This method find the indizes of the k smallest numbers in arr.

    Parameters
    ----------
    arr : np.ndarray
        An array to search.
    k : int
        The amount of values to find.

    Returns
    -------
    np.ndarray
        The indizes of the k smallest numbers in the array.
    """

    idx = np.argpartition(arr, k)
    return idx[:k]


def make_snapshot_number(rerun: bool, resolution: int) -> int:
    """
    This method calculates the number of snapshots in a given simulation.

    Parameters
    ----------
    rerun : bool
        If the simulation is a rerun or not.
    resolution : int
        The resolution of the simulation.

    Returns
    -------
    int
        The number of snapshots in this simulation.

    Raises
    ------
    ValueError
        If the resolution is not implemented.
    """

    if resolution == 2 and rerun is False:
        n_snapshots = 127 + 1
        return n_snapshots
    if resolution == 3:
        n_snapshots = 63 + 1 if rerun is False else 127 + 1
        return n_snapshots
    elif resolution == 4:
        n_snapshots = 127 + 1 if rerun is False else 251 + 1
        return n_snapshots
    elif resolution == 5 and rerun is False:
        n_snapshots = 63 + 1
        return n_snapshots
    else:
        raise ValueError("Resolution value not implemented.")


def create_or_load_dataframe(df_path: str) -> pd.DataFrame:
    """
    This method loads the dataframe of the given path or, if it does not
    exist, it creates a new (empty) dataframe.

    Parameters
    ----------
    df_path : str
        The path to the dataframe.

    Returns
    -------
    pd.DataFrame
        The dataframe.
    """

    if exists(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame()
    return df
