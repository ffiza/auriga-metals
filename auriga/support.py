import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection
import time
from typing import Callable
from sys import stdout
import pandas as pd
from os.path import exists

from auriga.parser import parse


def find_indices(a: np.array,
                 b: np.array,
                 invalid_specifier: int = -1,
                 ) -> np.array:
    """
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
        BLUE = "\033[94m"
        RESET = "\033[0m"
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        delta_time = int((end_time - start_time) / 60)
        delta_time_str = str(delta_time).rjust(3)

        stdout.write(f"{BLUE}Timer: {delta_time_str} min. {RESET}")
        return result

    return wrapper


def get_bool_input(msg: str) -> bool:
    """
    This method get the input from the user by displaying a message
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


def find_idx_ksmallest(arr: np.ndarray,
                       k: int,
                       ) -> np.ndarray:
    """
    This method find the indizes of the k smallest numbers in arr.

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


def make_snapshot_number(rerun: bool,
                         resolution: int,
                         ) -> int:
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

    if resolution == 2 and not rerun:
        return 127 + 1
    if resolution == 3:
        return 127 + 1 if rerun else 63 + 1
    elif resolution == 4:
        return 251 + 1 if rerun else 127 + 1
    elif resolution == 5 and not rerun:
        return 63 + 1
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

    return pd.read_csv(df_path) if exists(df_path) else pd.DataFrame()


def multi_color_line(x: np.ndarray, y: np.ndarray,
                     c: np.ndarray, vmax: float, vmin: float,
                     lw: float, cmap: str, return_params: bool = False):
    """
    Create a line with different colors using a color map.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of each point.
    y : np.ndarray
        The y-coordinates of each point.
    c : np.ndarray
        The value to define the colors of the segments.
    vmax : float
        The max value for the color map normalisation.
    vmin : float
        The min value for the color map normalisation.
    lw : float
        The line width.
    cmap : str
        The color map.
    return_params : bool, optional
        If True, return the normalisation and the color map., by default False.

    Returns
    -------
    lc : LineCollection
        The line collection.
    norm
        The normalisation of the color map. Only if `return_params` is True.
    cmap
        The color map. Only if `return_params` is True.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = mpl.colormaps[cmap]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = list()
    for val in c:
        c = cmap(norm(val))
        colors.append(c)

    lc = LineCollection(segments, linewidth=lw, colors=colors,
                        capstyle="round")

    if not return_params:
        return lc
    else:
        return lc, norm, cmap


def get_name_of_previous_snapshot(simulation: str) -> str:
    """
    This method takes a simulation using the format auW_XX_LY_SZZZ and returns
    the preceding snapshot.

    Parameters
    ----------
    simulation : str
        The simulation name.

    Returns
    -------
    str
        The previous snapshot of the simulation.
    """

    try:
        galaxy, rerun, resolution, snapshot = parse(simulation)
    except ValueError:
        raise ValueError(f"{simulation} is an invalid format for this "
                         f"method. Format should include snapshot number, "
                         f"for example: `au6_or_l4_s127`.")

    if snapshot == 0:
        raise ValueError(f"{simulation} is the first snapshot of this "
                         f"simulation.")

    type_txt = "or" if not rerun else "re"
    return f"au{galaxy}_{type_txt}_l{resolution}_s{snapshot - 1}"


def get_present_day_disc_radius_of_galaxy(simulation: str) -> float:
    """
    This method returns the disc radius at z=0 in kpc for the given galaxy.

    Paramters
    ---------
    simulation : str
        The simulation name.
    
    Returns
    -------
    float
        The disc radius at z=0 in kpc.
    """

    try:
        galaxy, rerun, resolution = parse(simulation)
    except ValueError:
        raise ValueError(f"{simulation} is an invalid format for this "
                         f"method. Format should include snapshot number, "
                         f"for example: `au6_or_l4`.")
    
    if rerun is True:
        raise ValueError("No data available for reruns.")
    
    if resolution != 4:
        raise ValueError("No data available for resolutions other than 4.")

    data = pd.read_csv("../data/iza_2022.csv")
    return data["DiscRadius_kpc"][data["Galaxy"] == galaxy].iloc[0]


def float_to_latex(x: float) -> str:
    """
    Return a LaTeX string with the correctly formatted minus sign.

    Paramters
    ---------
    x : float
        A float to convert to string.
    
    Returns
    -------
    float
        The LaTeX string.
    """
    if x < 0.0:
        return "$-$" + str(np.abs(x))
    else:
        return str(x)
