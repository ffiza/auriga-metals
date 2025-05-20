import numpy as np
from math import log10, floor
from decimal import Decimal


def weighted_percentile(data: np.ndarray, weights: np.ndarray,
                        percentile: float) -> float:
    """
    This function performs a weighted percentile calculation using linear
    interpolation. It assumes that all input arrays are 1D and of equal length.

    Parameters
    ----------
    data: np.ndarray
        1D array of values (e.g., positions of particles).
    weights: np.ndarray
        1D array of weights (e.g., masses of particles), same shape as `data`.
    percentile: float
        Percentile to compute, must be in the range [0, 100].

    Returns
    -------
    float
        The value below which the specified percentage of the weighted
        data lies.
    """
    if data.shape != weights.shape:
        raise ValueError("`data` and `weights` must be the same shape.")
    if data.ndim != 1:
        raise ValueError("`data` and `weights` must be 1D arrays.")
    if not (0 <= percentile <= 100):
        raise ValueError("`percentile` must be between 0 and 100.")

    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]
    normalized_cumsum = cumulative_weights / total_weight

    return np.interp(percentile / 100, normalized_cumsum, sorted_data)


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

    return slope * x + intercept


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

    return amplitude * np.exp(-x / scale)


def pdf_gaussian(x: np.ndarray, mean: float, sigma: float):
    """
    Return a gaussian function for the given x values and parameters. This
    function has the form of a normal distribution (Note that no
    amplitude is needed).

    Parameters
    ----------
    x : np.ndarray
        An array with the input values.
    mean : float
        Mean value.
    sigma : float
        Standard deviation.

    Returns
    -------
    res : np.ndarray
        An array with the results.
    """
    amplitude = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = - 0.5 * ((x - mean) / sigma)**2
    return amplitude * np.exp(exponent)


def double_pdf_gaussian(x: np.ndarray,
                        mean1: float, sigma1: float,
                        mean2: float, sigma2: float):
    """
    Return a double gaussian function for the given x values and parameters.

    Parameters
    ----------
    x : np.ndarray
        An array with the input values.
    mean1 : float
        Mean value of the first component.
    sigma1 : float
        Standard deviation of the first component.
    mean2 : float
        Mean value of the second component.
    sigma2 : float
        Standard deviation of the second component.

    Returns
    -------
    res : np.ndarray
        An array with the results.
    """
    gauss1 = pdf_gaussian(x, mean1, sigma1)
    gauss2 = pdf_gaussian(x, mean2, sigma2)
    return gauss1 + gauss2


def gaussian(x: np.ndarray, amplitude: float, mean: float, scale: float):
    """
    Return a gaussian function for the given x values and parameters.

    Parameters
    ----------
    x : np.ndarray
        An array with the input values.
    amplitude : float
        Amplitude.
    mean : float
        Mean value.
    scale : float
        Scale factor.

    Returns
    -------
    res : np.ndarray
        An array with the results.
    """
    return amplitude * np.exp(- ((x - mean) / scale)**2)


def double_gaussian(x: np.ndarray,
                    amplitude1: float, mean1: float, scale1: float,
                    amplitude2: float, mean2: float, scale2: float):
    """
    Return a double gaussian function for the given x values and parameters.

    Parameters
    ----------
    x : np.ndarray
        An array with the input values.
    amplitude1 : float
        The amplitude of the first component.
    mean1 : float
        Mean value of the first component.
    scale1 : float
        Standard deviation of the first component.
    amplitude2 : float
        The amplitude of the second component.
    mean2 : float
        Mean value of the second component.
    scale2 : float
        Standard deviation of the second component.

    Returns
    -------
    res : np.ndarray
        An array with the results.
    """
    gauss1 = gaussian(x, amplitude1, mean1, scale1)
    gauss2 = gaussian(x, amplitude2, mean2, scale2)
    return gauss1 + gauss2


def mad(x: np.ndarray) -> float:
    """
    Return a median absolute deviation.

    Parameters
    ----------
    x : np.ndarray
        An array with the input values.

    Returns
    -------
    float
        The median absolute deviation.
    """
    return np.median(np.abs(x - np.median(x)))


def round_to_1(x: float) -> float:
    """
    Rounds a number to one significant digit.

    Parameters
    ----------
    x : float
        The number to round.
    
    Returns
    -------
    float
        The result of rounding.
    """
    return round(x, -int(floor(log10(abs(x)))))


def get_decimal_places(x: float) -> int:
    """
    Returns the number of decimal places in a number. Note that an integer
    will have at least one decimal place by default.

    Parameters
    ----------
    x : float
        A float to calculate decimal places.
    
    Returns
    -------
    int
        The number of decimal places.
    """
    if isinstance(x, int): x = float(x)
    return len(str(Decimal(str(x))).split(".")[1])

