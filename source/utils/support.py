import numpy as np
import time
from typing import Callable


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

        print(f'Timer: {end_time-start_time} s.')
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
