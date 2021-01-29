"""
Created on 2020

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2020, Paris
#
# This file contains the main functions concerning the color magnitude
# diagram (CMD).
# It provides a Kernel Density Estimation (KDE) of the CMD distribution.
#
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

from scipy.stats import gaussian_kde
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"KDE approach"
# ------------------------------------------------------------------------------


def kde(x, y, ww=None, bw="silverman", confidence=None):
    """
    This function computes the KDE estimative of the PDF of the color
    magnitude diagram (CMD).

    Parameters
    ----------
    x : array_like
        Color from the CMD.
    y : array_like
        Magnitude.
    ww : array_like, optional
        Weight passed to the KDE estimation. The default is None.
    bw : string or float, optional
        Argument "bw_method" from the "gaussian_kde" method from scipy.stats.
        The default is 'silverman'.
    confidence : float, optional
        Confidence level (in percent) of arrays x and y to be considered.
        Default is None.

    Returns
    -------
    Z : array of size [2, Nsamples]
        KDE estimative of the PDF.

    """

    if ww is None:
        ww = np.ones(len(x))

    if confidence is not None:
        conf = confidence / 100
        qx_i, qx_f = quantile(x, [conf, 1 - conf])
        qy_i, qy_f = quantile(y, [conf, 1 - conf])
    else:
        qx_i, qx_f = np.nanmin(x), np.nanmax(x)
        qy_i, qy_f = np.nanmin(y), np.nanmax(y)

    ranges = [[qx_i, qx_f], [qy_i, qy_f]]

    # fit an array of size [Ndim, Nsamples]
    data = np.vstack([x, y])
    kernel = gaussian_kde(data, bw_method=bw, weights=ww)

    q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50
    nbins_x = int(10 * (np.amax(x) - np.amin(x)) / (min(q_m, q_p)))

    q_16, q_50, q_84 = quantile(y, [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50
    nbins_y = int(10 * (np.amax(y) - np.amin(y)) / (min(q_m, q_p)))

    # evaluate on a regular grid
    xgrid = np.linspace(ranges[0][0], ranges[0][1], nbins_x)
    ygrid = np.linspace(ranges[1][0], ranges[1][1], nbins_y)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Zhist = kernel.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    Z = Zhist.reshape(Xgrid.shape)

    return Z


def bw_silver(x, y):
    """
    Returns the Silverman bandwidth factor.

    Parameters
    ----------
    x : array_like
        Color from the CMD.
    y : array_like
        Magnitude.

    Returns
    -------
    bw : float
        Silverman's bandwidth factor.

    """

    d = 2
    n = len(x)

    bw = (n * (d + 2) / 4) ** (-1 / (d + 4))

    return bw


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"General functions"
# ---------------------------------------------------------------------------


def quantile(x, q):
    """
    Compute sample quantiles.

    Parameters
    ----------
    x : array_like
        Array containing set of values.
    q : array_like, float
        Quantile values to be derived (must be between 0 and 1).

    Raises
    ------
    ValueError
        Quantiles must be between 0 and 1.

    Returns
    -------
    percentile: array_like (same shape as q), float
        Percentile value(s).

    """

    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    return np.percentile(x, 100.0 * q)


def inside_contour(x, y, contours, idx=None, idx_level=0):
    """
    Returns indexes from a subset of previous indexes corresponding
    to the points in (x,y) inside a given contour.

    Parameters
    ----------
    x : array_like
        Data in x-direction.
    y : array_like
        Data in y-direction.
    contours : QuadContourSet
        Contour object from pyplot.
    idx : array of integers, optional
        Array containing the indexes in (x,y) from which to search points
        inside a given contour. The default is None.
    idx_level : int, optional
        Which contour lines to consider from contour object.
        The default is 0 (wider contour).

    Returns
    -------
    idx_inside : array of integers
        Array of indexes inside the contour.

    """

    lcontours = list()
    inside = list()

    p = contours.collections[0].get_paths()
    npoly = len(p)
    for i in range(0, npoly):
        v = p[i].vertices
        contourx = v[:, 0]
        contoury = v[:, 1]

        contour_array = np.array([contourx, contoury])

        lcontours.append(contour_array)
        inside.append(
            Polygon([(contourx[i], contoury[i]) for i in range(0, len(contourx))])
        )

    idx_inside = list()
    for i in idx:
        for j in range(0, npoly):
            if inside[j].contains(Point(x[i], y[i])):
                idx_inside.append(i)

    idx_inside = np.asarray(idx_inside).astype(int)

    return idx_inside
