from scipy.stats import binned_statistic
import numpy as np


def iqr(x: np.ndarray) -> float:
    """
    Return the interquartile range (IQR), the difference between the 75th and
    25th percentiles of the data

    Parameters
    ----------
    x : np.ndarray
        An array with the data values.

    Returns
    -------
    float
        The IQR.
    """
    return np.subtract(*np.percentile(x, [75, 25]))


def std_1dof(x: np.ndarray) -> float:
    """
    Return the standard deviation considering one degree of freedom of
    the data.

    Parameters
    ----------
    x : np.ndarray
        An array with the data values.

    Returns
    -------
    float
        The standard deviation.
    """
    return np.std(x, ddof=1)


def get_binned_statistic(x: np.ndarray,
                         values: np.ndarray,
                         bin_edges: np.ndarray,
                         xrange: tuple) -> tuple:
    """
    Compute binned statistics over the `values`, binned according to `x` in
    the bins defined by `bin_edges` in the range `xrange`. The statistics
    return are the mean, the standard deviation with one degree of freedom,
    the median, the interquartile range plus the bin centers.

    Parameters
    ----------
    x : np.ndarray
        A sequence of values to be binned.
    values : np.ndarray
        The data on which the statistic will be computed.
    bin_edges : np.ndarray
        The bin edges.
    xrange : tuple
        The lower and upper range of the bins.

    Returns
    -------
    tuple
        A tuple with the statistics and the bin centers.
    """
    mean, _, _ = binned_statistic(
        x=x,
        values=values,
        statistic="mean",
        bins=bin_edges,
        range=xrange,
    )
    std, _, _ = binned_statistic(
        x=x,
        values=values,
        statistic=std_1dof,
        bins=bin_edges,
        range=xrange,
    )
    bin_centers = bin_edges[1:] - np.diff(bin_edges)[1]
    median, _, _ = binned_statistic(
        x=x,
        values=values,
        statistic="median",
        bins=bin_edges,
        range=xrange,
    )
    iqrange, _, _ = binned_statistic(
        x=x,
        values=values,
        statistic=iqr,
        bins=bin_edges,
        range=xrange,
    )

    return mean, std, median, iqrange, bin_centers
