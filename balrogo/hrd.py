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
# Documentation is provided on Vitral & Macedo, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

from scipy.stats import gaussian_kde
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"KDE approach"
# ------------------------------------------------------------------------------


def kde(x, y, ww=None, bw="silverman"):
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

    Returns
    -------
    Z : array of size [2, Nsamples]
        KDE estimative of the PDF.

    """

    if ww is None:
        ww = np.ones(len(x))

    ranges = [[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]]

    # fit an array of size [Ndim, Nsamples]
    data = np.vstack([x, y])
    kernel = gaussian_kde(data, bw_method=bw, weights=ww)

    # evaluate on a regular grid
    xgrid = np.linspace(ranges[0][0], ranges[0][1], 200)
    ygrid = np.linspace(ranges[1][0], ranges[1][1], 200)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Zhist = kernel.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    Z = Zhist.reshape(Xgrid.shape)

    return Z
