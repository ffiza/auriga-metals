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
