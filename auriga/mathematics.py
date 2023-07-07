import numpy as np


def linear(x: np.array, slope: float, intercept: float) -> np.array:
    """
    This method returns a linear function with the given data and parameters.

    Parameters
    ----------
    x : np.array
        The values on which to compute the function.
    slope : float
        The slope of the linear function.
    intercept : float
        The intercept of the linear function.

    Returns
    -------
    np.array
        The result of the linear function.
    """

    f = slope*x + intercept
    return f


def exponential(x: np.array, amplitude: float, scale: float) -> np.array:
    """
    This method returns an exponential function with the given data and
    parameters.

    Parameters
    ----------
    x : np.array
        The values on which to compute the function.
    amplitude : float
        The amplitude of the exponential function.
    scale : float
        The scale of the exponential function.

    Returns
    -------
    np.array
        The result of the linear function.
    """

    f = amplitude * np.exp(-x/scale)
    return f
