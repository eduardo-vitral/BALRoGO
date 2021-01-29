"""
Created on 2020

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2020, Paris
#
# This file contains the main functions concerning parallax information.
# It provides a kernel density estimation of the distance distribution,
# as well as a fit of the mode of this distribution.
#
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import gaussian_kde
import numdifftools as ndt

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Mode estimation"
# ---------------------------------------------------------------------------


def kde(data, bw=None):
    """
    Computes the Kernel Density Estimation of the data.

    Parameters
    ----------
    data : array_like
        Data to be analyzed.
    bw : float, optional
        bandwidth method from the kde routine from scipy. The default is None.

    Returns
    -------
    kde object
        kde object of the data, using a specific bandwith.

    """

    if bw is None:
        bw = 0.5 * (len(data) * (1 + 2) / 4) ** (-1 / (1 + 4))

    return gaussian_kde(data, bw_method=bw)


def refined_mode(data, bw=None, use_log=False):
    """
    Computes the mode of data, by a KDE non-parametric fit.

    Parameters
    ----------
    data : array_like
        Data to be analyzed.
    bw : float, optional
        bandwidth method from the kde routine from scipy. The default is None.
    use_log : boolean, optional
        True is the user wants to find the mode on a logarithmic spaced data.
        The default is False.

    Returns
    -------
    mode : float
        Mode of the analyzed distribution.
    uncertainty : float
        Uncertainty on the mode measurement.

    """

    if bw is None:
        bw = 0.5 * (len(data) * (1 + 2) / 4) ** (-1 / (1 + 4))

    if use_log is False:
        mode, uncertainty = mode_lin(data, bw)
    else:
        mode, uncertainty = mode_log(data, bw)

    return mode, uncertainty


def mode_lin(data, bw):
    """
    Computes the mode from linear spaced data.

    Parameters
    ----------
    data : array_like
        Data to be analyzed.
    bw : float
        bandwidth method from the kde routine from scipy.

    Returns
    -------
    mode : float
        Mode of the analyzed distribution.
    uncertainty : float
        Uncertainty on the mode measurement.

    """

    kernel = kde(data, bw=bw)
    pdf = kernel.pdf(data)
    x0 = data[np.argmax(pdf)]
    span = data.max() - data.min()
    dx = span / 10
    bounds = np.array([[x0 - dx, x0 + dx]])

    results = differential_evolution(lambda x: -kernel(x), bounds)

    hfun = ndt.Hessian(lambda x: -kernel(x), full_output=True)
    hessian_ndt, info = hfun(results.x[0])
    var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    return results.x[0], var[0]


def mode_log(data, bw):
    """
    Computes the mode from logarithmic spaced data.

    Parameters
    ----------
    data : array_like
        Data to be analyzed.
    bw : float
        bandwidth method from the kde routine from scipy.

    Returns
    -------
    mode : float
        Mode of the analyzed distribution.
    uncertainty : 2D array
        Lower and upper uncertainties on the mode measurement.

    """

    logdata = np.log(data)
    kernel = kde(logdata, bw=bw)
    height = kernel.pdf(logdata)
    x0 = logdata[np.argmax(height)]
    span = logdata.max() - logdata.min()
    dx = span / 10
    bounds = np.array([[x0 - dx, x0 + dx]])

    results = differential_evolution(lambda x: -kernel(x), bounds)

    hfun = ndt.Hessian(lambda x: -kernel(x), full_output=True)
    hessian_ndt, info = hfun(results.x[0])
    var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    mode = np.exp(results.x[0])
    unc1 = np.exp(results.x[0]) - np.exp(results.x[0] - var[0])
    unc2 = np.exp(results.x[0] + var[0]) - np.exp(results.x[0])

    uncertainty = np.asarray([unc1, unc2])

    return mode, uncertainty


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"General functions"
# ---------------------------------------------------------------------------


def parallax_to_distance(par, remove_far=True):
    """
    Converts parallax into distance [kpc].

    Parameters
    ----------
    par : array_like
        Array containing parallaxes.
    remove_far : boolean, optional
        Set True is the user want to remove upper outliers.
        The default is True.

    Returns
    -------
    dd : array_like
        Distance in kpc.

    """

    dd = 1 / par

    dd = dd[np.where(dd > 0)]

    if remove_far is True:
        q975 = quantile(dd, 0.975)
        idx_far = np.where(dd < q975)
        dd = dd[idx_far]

    return dd


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
