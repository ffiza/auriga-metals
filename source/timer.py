from typing import Callable
import time as tm


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
        start_time = int(round(tm.time()))
        result = method(*args, **kw)
        end_time = int(round(tm.time()))

        print(f'Timer: {end_time-start_time} s.', end='')
        return result

    return wrapper
