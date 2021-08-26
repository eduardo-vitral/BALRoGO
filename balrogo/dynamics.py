"""
Created on 2021

@author: Eduardo Vitral
"""

###############################################################################
#
# June 2021, Paris
#
# This file contains the main functions concerning the dispersion functions,
# (i.e., velocity disperstion and anisotropy). It also converts plane of sky
# velocities (and uncertainties) from (RA,Dec) to polar coordinates.
#
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

from . import position
from . import angle

import numpy as np
import operator
import math
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import uncertainties as unc

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Proper motions and conversions"
# ------------------------------------------------------------------------------


def v_sky_to_polar(a, d, pma, pmd, a0, d0, pma0, pmd0):
    """
    Transforms proper motions in RA Dec into polar coordinates
    (radial and tangential).

    Parameters
    ----------
    a : array_like
        RA of the source.
    d : array_like
        Dec of the source.
    pma : array_like
        PMRA of the source.
    pmd : array_like
        PMDec of the source.
    a0 : float
        Bulk RA.
    d0 : float
        Bulk Dec.
    pma0 : float
        Bulk PMRA.
    pmd0 : float
        Bulk PMDec.

    Returns
    -------
    pmr : array_like
        PM in radial direction of the source.
    pmt : array_like
        PM in tangential direction of the source.

    """

    a = a * np.pi / 180
    d = d * np.pi / 180
    a0 = a0 * np.pi / 180
    d0 = d0 * np.pi / 180

    sinda = np.sin(a - a0)
    cosda = np.cos(a - a0)
    sind = np.sin(d)
    sind0 = np.sin(d0)
    cosd = np.cos(d)
    cosd0 = np.cos(d0)

    dx = sinda * cosd
    dy = cosd0 * sind - sind0 * cosd * cosda
    rho = np.sqrt(dx * dx + dy * dy)
    theta = np.arccos(sind0 * sind + cosd0 * cosd * cosda)

    cost = np.cos(theta)

    dmux = cosda * (pma - pma0 * cosd / cosd0) - sind * sinda * pmd
    dmuy = (
        (cosd * cosd0 + sind * sind0 * cosda) * pmd
        - cost * pmd0
        + (pma - pma0 * cosd / cosd0) * sind0 * sinda
    )

    pmr = (dx * dmux + dy * dmuy) / rho
    pmt = (-dx * dmuy + dy * dmux) / rho

    return pmr, pmt


def unc_sky_to_polar(a, d, epma, epmd, epmad, a0, d0, epma0, epmd0):
    """
    Transforms proper motions uncertainties in RA Dec into polar coordinates
    uncertainties (radial and tangential).

    Parameters
    ----------
    a : array_like
        RA of the source.
    d : array_like
        Dec of the source.
    epma : array_like
        Uncertainty in PMRA of the source.
    epmd : array_like
        Uncertainty in PMDec of the source.
    epmad : array_like
        Correlation between epma and epmd.
    a0 : float
        Bulk RA.
    d0 : float
        Bulk Dec.
    epma0 : float
        Uncertainty in Bulk PMRA.
    epmd0 : float
        Uncertainty in Bulk PMDec.

    Returns
    -------
    uncpmr : array_like
        Uncertainty in PM in radial direction.
    uncpmt : array_like
        Uncertainty in PM in tangential direction.

    """

    a = a * np.pi / 180
    d = d * np.pi / 180
    a0 = a0 * np.pi / 180
    d0 = d0 * np.pi / 180

    sina = np.sin(a)
    cosa = np.cos(a)
    sina0 = np.sin(a0)
    cosa0 = np.cos(a0)
    sind = np.sin(d)
    cosd = np.cos(d)
    sind0 = np.sin(d0)
    cosd0 = np.cos(d0)
    sinda = np.sin(a - a0)
    cosda = np.cos(a - a0)

    dentheta = np.sqrt(
        cosd ** 2 * sinda ** 2 + (cosd0 * sind - cosda * cosd * sind0) ** 2
    )

    dvdpma = (cosd0 * sinda * (cosda * cosd * cosd0 + sind * sind0)) / dentheta
    dvdpmd = (
        -cosd * sinda ** 2 * sind
        + (cosd0 * sind - cosda * cosd * sind0) * (cosd * cosd0 + cosda * sind * sind0)
    ) / dentheta
    dvdpma0 = -cosd * sinda * (cosda * cosd * cosd0 + sind * sind0) / dentheta
    dvdpmd0 = (
        -(cosd0 * sind - cosda * cosd * sind0)
        * (cosda * cosd * cosd0 + sind * sind0)
        / dentheta
    )

    uncpmr = np.sqrt(
        (dvdpma * epma) ** 2
        + (dvdpmd * epmd) ** 2
        + (dvdpma0 * epma0) ** 2
        + (dvdpmd0 * epmd0) ** 2
        + 2 * (dvdpma * dvdpmd * epma * epmd * epmad)
    )

    dvdpma = (
        cosa * cosa0 * cosd0 * sind + cosd0 * sina * sina0 * sind - cosd * sind0
    ) / dentheta
    dvdpmd = (-cosd0 * sinda) / dentheta
    dvdpma0 = (
        cosd
        * (-cosa * cosa0 * sind - sina * sina0 * sind + cosd * sind0 / cosd0)
        / dentheta
    )
    dvdpmd0 = cosd * sinda * (cosda * cosd * cosd0 + sind * sind0) / dentheta

    uncpmt = np.sqrt(
        (dvdpma * epma) ** 2
        + (dvdpmd * epmd) ** 2
        + (dvdpma0 * epma0) ** 2
        + (dvdpmd0 * epmd0) ** 2
        + 2 * (dvdpma * dvdpmd * epma * epmd * epmad)
    )

    return uncpmr, uncpmt


def pmr_corr(vlos, r, d):
    """
    Correction on radial proper motion due to apparent contraction/expansion
    of the cluster.

    Parameters
    ----------
    vlos : float
        Line of sight velocity, in km/s.
    r : array_like, float
        Projected radius, in degrees.
    d : float
        Cluster distance from the Sun, in kpc.

    Returns
    -------
    pmr : array_like, float
        Correction in the radial component of the proper motion, in mas/yr.

    """
    r = r * 60
    # Equation 4 from Bianchini et al. 2018.
    pmr = -6.1363 * 1e-5 * vlos * r / d

    return pmr


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Dispersion functions"
# ------------------------------------------------------------------------------


def aux_disp(idx, y, ey, dimy, robust_sig):
    """
    Auxiliary function used by hexbin to compute the dispersion.

    Parameters
    ----------
    idx : array_like
        Array of indexes to consider in the data.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    robust_sig : boolean
        True if the user wants to compute  a dispersion
        less sensible to outliers.

    Returns
    -------
    disp : array_like
        Dispersion over the selected values.

    """

    disp = np.zeros((dimy, 1))

    if robust_sig is True:

        for i in range(0, dimy):

            disp[i, 0] = np.median(np.abs(y[i][idx] - np.median(y[i][idx]))) / 0.6745
            disp[i, 0] = np.sqrt(disp[i, 0] ** 2 - np.nanmean(ey[i][idx] ** 2))

    else:

        for i in range(0, dimy):

            if len(idx) < 30:
                disp[i, 0] = np.sqrt(
                    np.nanstd(y[i][idx], ddof=1) ** 2 - np.nanmean(ey[i][idx] ** 2)
                )
            else:
                disp[i, 0] = np.sqrt(
                    np.nanstd(y[i][idx], ddof=0) ** 2 - np.nanmean(ey[i][idx] ** 2)
                )

    disp = np.sqrt(np.sum(disp * disp, axis=0)) / np.sqrt(dimy)

    return disp


def aux_err(idx, y, ey, dimy, robust_sig, bootp):
    """
    Auxiliary function used by hexbin to compute the dispersion.

    Parameters
    ----------
    idx : array_like
        Array of indexes to consider in the data.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    robust_sig : boolean
        True if the user wants to compute  a dispersion
        less sensible to outliers.
    bootp : boolean
        True if the errors are drawn from a Bootstrap method.

    Returns
    -------
    err : array_like
        Uncertainty on the dispersion over the selected values.

    """

    disp = np.zeros((dimy, 1))
    err = np.zeros((dimy, 1))

    if robust_sig is True:

        for i in range(0, dimy):

            disp[i, 0] = np.median(np.abs(y[i][idx] - np.median(y[i][idx]))) / 0.6745
            disp[i, 0] = np.sqrt(disp[i, 0] ** 2 - np.nanmean(ey[i][idx] ** 2))

            if bootp is True:
                err[i, 0] = bootstrap(y[i][idx])
            else:
                err[i, 0] = disp[i, 0] / np.sqrt(2 * (len(y[i][idx]) - 1))

    else:

        for i in range(0, dimy):

            if len(idx) < 30:
                disp[i, 0] = np.sqrt(
                    np.nanstd(y[i][idx], ddof=1) ** 2 - np.nanmean(ey[i][idx] ** 2)
                )
            else:
                disp[i, 0] = np.sqrt(
                    np.nanstd(y[i][idx], ddof=0) ** 2 - np.nanmean(ey[i][idx] ** 2)
                )

            if bootp is True:
                err[i, 0] = bootstrap(y[i][idx])
            else:
                err[i, 0] = disp[i, 0] / np.sqrt(2 * (len(y[i][idx]) - 1))

    err = np.sqrt(np.nansum(err * err, axis=0)) / np.sqrt(dimy)

    return err


def bin_disp1d(x, y, ey, dimy, bins, bootp=True, logx=True):
    """
    Computes the dispersion of y (and its uncertainty) as a function of x.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    bins : int
        Number of bins.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is True.

    Returns
    -------
    r : array_like
        Binned version of x.
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    disp = np.zeros((dimy, bins - 1))
    err = np.zeros((dimy, bins - 1))
    r = np.zeros(bins - 1)

    if logx is True:
        rbin = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), bins)
    else:
        rbin = np.linspace(np.nanmin(x), np.nanmax(x), bins)

    for i in range(0, dimy):
        for j in range(0, bins - 1):
            cond1 = np.where(x < rbin[j + 1])
            cond2 = np.where(x >= rbin[j])
            cond = np.intersect1d(cond1, cond2)
            if len(cond) > 30:
                dof = 0
            else:
                dof = 1

            if logx is True:
                r[j] = np.sqrt(rbin[j] * rbin[j + 1])
            else:
                r[j] = 0.5 * (rbin[j] * rbin[j + 1])

            disp[i, j] = np.sqrt(
                np.nanstd(y[i][cond], ddof=dof) ** 2 - np.nanmean(ey[i][cond] ** 2)
            )

            if bootp is True:
                err[i, j] = bootstrap(y[i][cond])
            else:
                err[i, j] = disp[i, j] / np.sqrt(2 * (len(y[i][cond]) - 1))

    disp = np.sqrt(np.sum(disp * disp, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.sum(err * err, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def moving_grid1d(x, y, ey, dimy, bootp=True, logx=True, bins=10, ngrid=10):
    """
    Calculates the dispersion in a moving grid.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    bins : int, optional
        Number of bins.
        The default is None.
    ngrid : int, optional
        Number of grids per bin.
        The default is 10.

    Returns
    -------
    r : array_like
        Binned version of x.
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    if bins is None:
        bins = int(0.5 * position.good_bin(x))

    if logx is True:
        rbin = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), bins + 1)
        rbin = np.log10(rbin)
    else:
        rbin = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)

    for i in range(1, ngrid):

        add_x = (rbin[1] - rbin[0]) * i / (1 + ngrid)

        if logx is True:
            x_ini = 10 ** (rbin[0] + add_x)
            x_fin = 10 ** (rbin[len(rbin) - 2] + add_x)
        else:
            x_ini = rbin[0] + add_x
            x_fin = rbin[len(rbin) - 2] + add_x

        idx_x = np.intersect1d(np.where(x >= x_ini), np.where(x < x_fin))
        idx_x = idx_x.astype(int)
        xi = x[idx_x]
        yi = y[:, idx_x]
        eyi = ey[:, idx_x]

        if i == 1:
            r, disp, err = bin_disp1d(
                xi, yi, eyi, dimy, bins=bins, bootp=bootp, logx=logx
            )
        else:
            ri, dispi, erri = bin_disp1d(
                xi, yi, eyi, dimy, bins=bins, bootp=bootp, logx=logx
            )

            r = np.append(r, ri)
            disp = np.append(disp, dispi)
            err = np.append(err, erri)

    r = np.asarray(r)
    disp = np.asarray(disp)
    err = np.asarray(err)

    # Sorts the values according to R_proj
    L = sorted(zip(r, disp, err), key=operator.itemgetter(0))
    r, disp, err = zip(*L)

    r = np.asarray(r)
    disp = np.asarray(disp)
    err = np.asarray(err)

    return r, disp, err


def equal_size(x, y, ey, dimy, bootp=True, logx=True, nbin=10):
    """
    Calculates the dispersion in a fixed bin size grid.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    nbin : int, optional
        Number of tracers per bin. The default is 10.

    Returns
    -------
    r : array_like
        Binned version of x.
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    if nbin > 30:
        dof = 0
    else:
        dof = 1

    size = math.floor(len(x) / nbin)
    disp = np.zeros((dimy, size))
    err = np.zeros((dimy, size))
    r = np.zeros(size)

    for i in range(0, size):
        if logx is True:
            r[i] = np.sqrt(x[nbin * (1 + i) - 1] * x[nbin * i])
        else:
            r[i] = (x[nbin * (1 + i) - 1] + x[nbin * i]) * 0.5

        for j in range(0, dimy):

            disp[j, i] = np.sqrt(
                np.nanstd(y[j][nbin * i : nbin * (i + 1) - 1], ddof=dof) ** 2
                - np.nanmean(ey[j][nbin * i : nbin * (i + 1) - 1] ** 2)
            )

            if bootp is True:
                err[j, i] = bootstrap(y[j][nbin * i : nbin * (i + 1) - 1])
            else:
                err[j, i] = disp[j, i] / np.sqrt(
                    2
                    * (
                        len(
                            y[j][nbin * i : nbin * (i + 1) - 1][
                                np.logical_not(
                                    np.isnan(y[j][nbin * i : nbin * (i + 1) - 1])
                                )
                            ]
                        )
                        - 1
                    )
                )

    disp = np.sqrt(np.sum(disp * disp, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.sum(err * err, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def perc_bins(x, y, ey, dimy, bootp=True, logx=True, nnodes=5):
    """
    Calculates the dispersion in a grid whose size varies geometrically with
    the parcentile of x.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    nnodes : int, optional
        Number of grids. The default is 5.

    Returns
    -------
    r : array_like
        Binned version of x.
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    size = nnodes
    disp = np.zeros((dimy, size))
    err = np.zeros((dimy, size))
    r = np.zeros(size)

    for i in range(0, nnodes):

        z = 100 ** (1 / nnodes)
        qi = max(0, 0.01 * z ** i)
        qf = min(0.01 * z ** (i + 1), 1)

        ri = position.quantile(x, qi)
        rf = position.quantile(x, qf)

        idxr = np.intersect1d(np.where(x >= ri), np.where(x <= rf))

        if len(idxr) > 30:
            dof = 0
        else:
            dof = 1

        if logx is True:
            r[i] = np.sqrt(np.nanmin(x[idxr]) * np.nanmax(x[idxr]))
        else:
            r[i] = (np.nanmin(x[idxr]) + np.nanmax(x[idxr])) * 0.5

        for j in range(0, dimy):

            disp[j, i] = np.sqrt(
                np.nanstd(y[j][idxr], ddof=dof) ** 2 - np.nanmean(ey[j][idxr] ** 2)
            )

            if bootp is True:
                err[j, i] = bootstrap(y[j][idxr])
            else:
                err[j, i] = disp[j, i] / np.sqrt(
                    2 * (len(y[j][idxr][np.logical_not(np.isnan(y[j][idxr]))]) - 1)
                )

    disp = np.sqrt(np.sum(disp * disp, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.sum(err * err, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def closest_points(x, y, ey, dimy, bootp=True, logx=True, nbin=5):
    """
    Calculates the dispersion in a grid of the closest nbin tracers.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    nbin : int, optional
        Number of closest points. The default is 5.

    Returns
    -------
    r : array_like
        Binned version of x.
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    size = len(x)
    disp = np.zeros((dimy, size))
    err = np.zeros((dimy, size))
    r = np.zeros(size)

    for i in range(0, size):

        dist_r = np.abs(x[i] - x)
        idxr = (np.argpartition(dist_r, nbin)[:nbin]).astype(int)

        if len(idxr) > 30:
            dof = 0
        else:
            dof = 1

        r[i] = x[i]

        for j in range(0, dimy):

            disp[j, i] = np.sqrt(
                np.nanstd(y[j][idxr], ddof=dof) ** 2 - np.nanmean(ey[j][idxr] ** 2)
            )

            if bootp is True:
                err[j, i] = bootstrap(y[j][idxr])
            else:
                err[j, i] = disp[j, i] / np.sqrt(
                    2 * (len(y[j][idxr][np.logical_not(np.isnan(y[j][idxr]))]) - 1)
                )

    disp = np.sqrt(np.sum(disp * disp, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.sum(err * err, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def disp1d(
    x,
    y,
    ey,
    dimy,
    bins="percentile",
    smooth=True,
    bootp=True,
    logx=True,
    nbin=None,
    polorder=None,
    return_fits=False,
):
    """
    Computes the dispersion of y.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    bins : int, string, optional
        Number of bins or method used to bin the data.
        "moving" stands for a moving grid, later interpolated with a
        cubic spline.
        The default is "percentile".
    smooth : boolean, optional
        True if the dispersion calculated from the "moving" method should
        be smoothed.
        The default is True.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    nbin : Auxiliar value for binning when bins is not an integer.
        The default is None.
    polorder : int, optional
        Order of smoothing polynomial.
        The default is None.
    return_fits: Whether the user wants to return the polynomial
        smoothing fits.
        The default is False.

    Returns
    -------
    r : array_like
        Reference array (possibly binned).
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    if isinstance(bins, int) is True:
        r, disp, err = bin_disp1d(x, y, ey, dimy, bins=bins, bootp=bootp, logx=logx)

    if bins == "moving":
        if nbin is None:
            bins = int(0.5 * position.good_bin(x))
        else:
            bins = nbin
        ngrid = 2
        r, disp, err = moving_grid1d(
            x, y, ey, dimy, bootp=bootp, logx=logx, bins=bins, ngrid=ngrid
        )

    if bins == "fix-size":
        if nbin is None:
            bins = int(position.good_bin(x))
            nbin = int(len(x) / bins)
        else:
            nbin = nbin
        r, disp, err = equal_size(x, y, ey, dimy, bootp=bootp, logx=logx, nbin=nbin)

    if bins == "percentile":
        if nbin is None:
            nnodes = int(2 * position.good_bin(x))
        else:
            nnodes = nbin
        r, disp, err = perc_bins(x, y, ey, dimy, bootp=bootp, logx=logx, nnodes=nnodes)

    if bins == "closest":
        if nbin is None:
            bins = int(position.good_bin(x))
            nbin = int(len(x) / bins)
        else:
            nbin = nbin
        r, disp, err = closest_points(x, y, ey, dimy, bootp=bootp, logx=logx, nbin=nbin)

    nonan1 = np.logical_not(np.isnan(disp))
    nonan2 = np.logical_not(np.isnan(err))
    nonan = nonan1 * nonan2

    rmin = np.nanmin(r)
    rmax = np.nanmax(r)
    idxrange = np.intersect1d(np.where(x > rmin), np.where(x < rmax))

    if smooth is True:

        disp = disp[nonan]
        err = err[nonan]
        r = r[nonan]

        if polorder is None:
            pold = int(0.2 * position.good_bin(disp))
        else:
            pold = polorder

        if logx is False:

            poly_disp, cov_disp = np.polyfit(r, disp, pold, w=1 / err, cov=True)

            # Do the interpolation for plotting:
            t = x[idxrange]
            # Matrix with rows 1, t, t**2, ...:
            TT = np.vstack([t ** (pold - i) for i in range(pold + 1)]).T
            yi = np.dot(
                TT, poly_disp
            )  # matrix multiplication calculates the polynomial values
            C_yi = np.dot(TT, np.dot(cov_disp, TT.T))  # C_y = TT*C_z*TT.T
            sig_yi = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal

            disp = yi
            err = sig_yi
            r = x[idxrange]
        else:
            poly_disp, cov_disp = np.polyfit(
                np.log10(r), disp, pold, w=1 / err, cov=True
            )

            # Do the interpolation for plotting:
            t = np.log10(x[idxrange])
            # Matrix with rows 1, t, t**2, ...:
            TT = np.vstack([t ** (pold - i) for i in range(pold + 1)]).T
            yi = np.dot(
                TT, poly_disp
            )  # matrix multiplication calculates the polynomial values
            C_yi = np.dot(TT, np.dot(cov_disp, TT.T))  # C_y = TT*C_z*TT.T
            sig_yi = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal

            disp = yi
            err = sig_yi
            r = x[idxrange]

        if return_fits is True:
            return r, disp, err, poly_disp

    return r, disp, err


def disp2d(
    x,
    y,
    ey,
    dimy,
    smooth=True,
    bootp=True,
    a0=None,
    d0=None,
    robust_sig=False,
    nbin=None,
    nmov=None,
):
    """
    Calculates a dispersion map in the x plane.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like
        Uncertainty on the quantity from which the dispersion is calculated.
    dimy : int
        Dimension of y.
    smooth : boolean, optional
        True if the dispersion calculated from the "moving" method should
        be smoothed.
        The default is True.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is True.
    a0 : float, optional.
        Bulk RA. The default is None.
    d0 : float, optional.
        Bulk Dec. The default is None.
    robust_sig : boolean, optional
        True if the user wants to compute  a dispersion
        less sensible to outliers.
        The default is False.
    nbin : int, optional
        Auxiliar value for binning when bins is not an integer.
        The default is None.
    nmov : int, optional
        Auxiliar value for moving grids.
        The default is None.

    Returns
    -------
    r : array_like
        Reference array (possibly binned).
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    if (a0 is None) or (d0 is None):
        center, unc = position.find_center(x[0], y[0], method="iterative")
        a0 = center[0]
        d0 = center[1]

    allPoints = np.column_stack((x[0], x[1]))
    hullPoints = ConvexHull(allPoints)
    idx_lims = hullPoints.vertices
    rlims = angle.sky_distance_deg(x[0, idx_lims], x[1, idx_lims], a0, d0)
    rlim = np.nanmin(rlims)

    alim0, dlim0 = angle.get_circle_sph_trig(rlim, a0, d0)

    if nbin is None:
        nbin = int(0.25 * (position.good_bin(x[0]) + position.good_bin(x[1])))
    if nmov is None:
        nmov = int(0.7 * nbin)
    rm = rlim * 0.8

    shift = (2 * rm) / (nbin * nmov)

    amin = list()
    amax = list()

    dmin = list()
    dmax = list()

    a0 = a0 - 0.5 * nmov * shift
    d0 = d0 - 0.5 * nmov * shift

    for i in range(0, nmov):
        for j in range(0, nmov):

            alim1, dlim1 = angle.get_circle_sph_trig(rm, a0 + i * shift, d0 + j * shift)

            amin.append(np.amin(alim1))
            dmin.append(np.amin(dlim1))

            amax.append(np.amax(alim1))
            dmax.append(np.amax(dlim1))

    raux_disp = partial(aux_disp, y=y, ey=ey, dimy=dimy, robust_sig=robust_sig)
    raux_err = partial(
        aux_err, y=y, ey=ey, dimy=dimy, robust_sig=robust_sig, bootp=bootp
    )

    for i in range(0, len(amin)):

        index = np.arange(len(x[0]))

        hex_disp = plt.hexbin(
            x[0],
            x[1],
            C=index,
            gridsize=nbin,
            reduce_C_function=raux_disp,
            extent=(amin[i], amax[i], dmin[i], dmax[i]),
        )

        zax0 = hex_disp.get_array()
        verts0 = hex_disp.get_offsets()
        xax0 = np.zeros(verts0.shape[0])
        yax0 = np.zeros(verts0.shape[0])
        disp = np.zeros(verts0.shape[0])

        for offc in range(verts0.shape[0]):
            binx, biny = verts0[offc][0], verts0[offc][1]
            if zax0[offc]:
                xax0[offc], yax0[offc], disp[offc] = binx, biny, zax0[offc]

        if i == 0:
            ddisp = disp
            pointsd = np.zeros((len(xax0), 2))
            for j in range(0, len(xax0)):
                pointsd[j] = np.asarray([xax0[j], yax0[j]])
        else:
            pointsnew = np.zeros((len(xax0), 2))
            for j in range(0, len(xax0)):
                pointsnew[j] = np.asarray([xax0[j], yax0[j]])
            pointsd = np.append(pointsd, pointsnew, axis=0)
            ddisp = np.append(ddisp, disp, axis=0)

        hex_err = plt.hexbin(
            x[0],
            x[1],
            C=index,
            gridsize=nbin,
            reduce_C_function=raux_err,
            extent=(amin[i], amax[i], dmin[i], dmax[i]),
        )

        zax0 = hex_err.get_array()
        verts0 = hex_err.get_offsets()
        xax0 = np.zeros(verts0.shape[0])
        yax0 = np.zeros(verts0.shape[0])
        err = np.zeros(verts0.shape[0])

        for offc in range(verts0.shape[0]):
            binx, biny = verts0[offc][0], verts0[offc][1]
            if zax0[offc]:
                xax0[offc], yax0[offc], err[offc] = binx, biny, zax0[offc]

        if i == 0:
            eerr = err
            pointse = np.zeros((len(xax0), 2))
            for j in range(0, len(xax0)):
                pointse[j] = np.asarray([xax0[j], yax0[j]])
        else:
            pointsnew = np.zeros((len(xax0), 2))
            for j in range(0, len(xax0)):
                pointsnew[j] = np.asarray([xax0[j], yax0[j]])
            pointse = np.append(pointse, pointsnew, axis=0)
            eerr = np.append(eerr, err, axis=0)

    grid_x, grid_y = np.mgrid[
        np.amin(alim0) : np.amax(alim0) : 300j, np.amin(dlim0) : np.amax(dlim0) : 300j
    ]

    grid_disp = griddata(pointsd, ddisp, (grid_x, grid_y), method="cubic")
    grid_err = griddata(pointse, eerr, (grid_x, grid_y), method="cubic")

    grid_disp[np.isnan(grid_disp)] = 0
    grid_err[np.isnan(grid_err)] = 0

    disp = grid_disp
    err = grid_err
    r = np.asarray([pointsd, pointse])

    plt.clf()

    return r, disp, err


def dispersion(
    x,
    y,
    ey=None,
    bins=None,
    smooth=True,
    bootp=False,
    logx=False,
    nbin=None,
    polorder=None,
    return_fits=False,
    robust_sig=False,
    a0=None,
    d0=None,
    nmov=None,
):
    """
    Calculates the dispersion.

    Parameters
    ----------
    x : array_like
        Reference array.
    y : array_like
        Quantity from which the dispersion is calculated.
    ey : array_like, optional
        Uncertainty on the quantity from which the dispersion is calculated.
        Default is None.
    bins : int, string, optional
        Number of bins or method used to bin the data.
        "moving" stands for a moving grid, later interpolated with a
        cubic spline.
        The default is None.
    smooth : boolean, optional
        True if the dispersion calculated from the "moving" method should
        be smoothed.
        The default is True.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is False.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    nbin : int, optional
        Auxiliar value for binning when bins is not an integer.
        The default is None.
    polorder : int, optional
        Order of smoothing polynomial.
        The default is None.
    return_fits: Whether the user wants to return the polynomial
        smoothing fits.
        The default is False.
    robust_sig : boolean, optional
        True if the user wants to compute  a dispersion
        less sensible to outliers.
        The default is False.
    a0 : float, optional.
        Bulk RA. The default is None.
    d0 : float, optional.
        Bulk Dec. The default is None.
    nmov : int, optional
        Auxiliar value for moving grids.
        The default is None.

    Returns
    -------
    r : array_like
        Reference array (possibly binned).
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    if len(np.shape(x)) == 1:
        dimx = 1
        if bins is None:
            bins = "percentile"
    else:
        dimx = 2
        x = np.asarray(x)

    if bins is None:
        bins = "moving"

    if len(np.shape(y)) == 1:
        dimy = 1
        if ey is None:
            ey = np.asarray([np.zeros(len(y))])
        else:
            ey = np.asarray([ey])
        y = np.asarray([y])
    else:
        y = np.asarray(y)
        if ey is None:
            ey = np.zeros(np.shape(y))
        else:
            ey = np.asarray(ey)
        dimy = np.shape(y)[0]

    if dimx == 1:
        r, disp, err = disp1d(
            x,
            y,
            ey,
            dimy,
            bins=bins,
            smooth=smooth,
            bootp=bootp,
            polorder=polorder,
            logx=logx,
            nbin=nbin,
            return_fits=return_fits,
        )
    else:
        r, disp, err = disp2d(
            x,
            y,
            ey,
            dimy,
            smooth=smooth,
            bootp=bootp,
            a0=a0,
            d0=d0,
            robust_sig=robust_sig,
            nbin=nbin,
            nmov=nmov,
        )

    err[np.where(err <= 0)] = np.nan

    return r, disp, err


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"General functions"
# ------------------------------------------------------------------------------


def btsp(x):
    """
    Selects randomly a sub array of elements from x, by shuffling it.

    Parameters
    ----------
    x : array_like
        Elements to be shuffled.

    Returns
    -------
    x_new : array_like
        Shuffled array.

    """
    q = np.random.rand(len(x))
    ind = np.rint(len(x) * q - 0.5).astype(int)
    x_new = x[ind]
    return x_new


def bootstrap(array):
    """
    Bootstrap method: Computes a dispersion whitin an array of values.

    Parameters
    ----------
    array : array_like
        Array to calculate the dispersion.

    Returns
    -------
    unc : float
        Uncertainty associated to array.

    """

    if len(array) > 30:
        dof = 0
    else:
        dof = 1

    sig = np.zeros(100)
    for i in range(100):
        xb = btsp(array)
        sig[i] = np.std(xb, ddof=dof)

    unc = np.std(sig)

    return unc


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Handles anisotropy"
# ------------------------------------------------------------------------------


def bgOM(r, ra, b0, bi):
    """
    Generalized (Osipkov1979; Merritt 1985) anisotropy profile.

    Parameters
    ----------
    r : array_like, float
        Distance from center of the system.
    ra : float
        Anisotropy radius.
    b0 : float
        Anisotropy at r = 0.
    bi : float
        Anisotropy at r -> Infinity.

    Returns
    -------
    b : array_like, float
        Anisotropy profile.

    """

    b = b0 + (bi - b0) / (1 + (ra / r) * (ra / r))

    return b


def bgTiret(r, ra, b0, bi):
    """
    Generalized Tiret et al. 2007 anisotropy profile.

    Parameters
    ----------
    r : array_like, float
        Distance from center of the system.
    ra : float
        Anisotropy radius.
    b0 : float
        Anisotropy at r = 0.
    bi : float
        Anisotropy at r -> Infinity.

    Returns
    -------
    b : array_like, float
        Anisotropy profile.

    """

    b = b0 + (bi - b0) / (1 + (ra / r))

    return b


def bCOM(r, ra, b0):
    """
    Anisotropy profile from (Cuddeford 1991; Osipkov 1979; Merritt 1985) inversion.

    Parameters
    ----------
    r : array_like, float
        Distance from center of the system.
    ra : float
        Anisotropy radius.
    b0 : float
        Anisotropy at r = 0.

    Returns
    -------
    b : array_like, float
        Anisotropy profile.

    """

    b = (b0 + (r / ra) * (r / ra)) / (1 + (r / ra) * (r / ra))

    return b


def get_anisotropy(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    method="moving",
    nbin=30,
    polorder=10,
    logx=False,
    bootp=False,
    smooth=True,
):
    """
    Computes the velocity anisotropy.

     Parameters
    ----------
    x : array_like, float
        x-axis.
    y : array_like, float
        y-axis.
    z : array_like, float
        z-axis.
    vx : array_like, float
        x-axis velocity.
    vy : array_like, float
        y-axis velocity.
    vz : array_like, float
        z-axis velocity.
    method : string, optional
        Method used to compute the dispersion.
        The default is "moving".
    nbin : int, optional
        Number of bins/tracers used in the dispersion calculation.
        The default is 30.
    polorder : int, optional
        Order of smoothing polynomial.
        The default is 10.
    logx : boolean, optional
        If the x axis should be binned logarithmly.
        The default is False.
    bootp : boolean, optional
        Use bootstrap to compute errors. The default is False.
    smooth : boolean, optional
        Smooth the data with a polynomial. The default is True.

    Returns
    -------
    rr : array_like
        Radius.
    beta : array_like
        Velocity anisotropy.
    ebeta : array_like
        Uncertainty on velocity anisotropy.

    """

    r, phi, theta, vr, vphi, vtheta = angle.cart_to_sph(x, y, z, vx, vy, vz)

    L = sorted(zip(r, vr, vtheta, vphi), key=operator.itemgetter(0))
    r, vr, vtheta, vphi = zip(*L)

    r = np.asarray(r)
    vr = np.asarray(vr)
    vtheta = np.asarray(vtheta)
    vphi = np.asarray(vphi)

    rr, dr, err = dispersion(
        r,
        vr,
        bins=method,
        nbin=nbin,
        polorder=polorder,
        bootp=bootp,
        logx=logx,
        smooth=smooth,
    )

    rr, dt, ert = dispersion(
        r,
        vtheta,
        bins=method,
        nbin=nbin,
        polorder=polorder,
        bootp=bootp,
        logx=logx,
        smooth=smooth,
    )

    rr, dp, erp = dispersion(
        r,
        vphi,
        bins=method,
        nbin=nbin,
        polorder=polorder,
        bootp=bootp,
        logx=logx,
        smooth=smooth,
    )

    beta = 1 - (dt * dt + dp * dp) / (2 * dr * dr)

    ebeta = (
        0.5
        * (dt * dt + dp * dp)
        / (2 * dr * dr)
        * np.sqrt(
            2 * err * err / (dr * dr)
            + 2 * (dt * dt * ert * ert + dp * dp * erp * erp) / (dt * dt + dp * dp) ** 2
        )
    )

    return rr, beta, ebeta


def get_beta(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    method="moving",
    nbin=30,
    polorder=10,
    logx=False,
    bootp=False,
    smooth=True,
    model="gOM",
):
    """
    Computes the velocity anisotropy.

     Parameters
    ----------
    x : array_like, float
        x-axis.
    y : array_like, float
        y-axis.
    z : array_like, float
        z-axis.
    vx : array_like, float
        x-axis velocity.
    vy : array_like, float
        y-axis velocity.
    vz : array_like, float
        z-axis velocity.
    method : string, optional
        Method used to compute the dispersion.
        The default is "moving".
    nbin : int, optional
        Number of bins/tracers used in the dispersion calculation.
        The default is 30.
    polorder : int, optional
        Order of smoothing polynomial.
        The default is 10.
    logx : boolean, optional
        If the x axis should be binned logarithmly.
        The default is False.
    bootp : boolean, optional
        Use bootstrap to compute errors. The default is False.
    smooth : boolean, optional
        Smooth the data with a polynomial. The default is True.
    model : string, optional
        Model used to fit (or not fit) the data. The default is "gOM".

    Raises
    ------
    ValueError
        Anisotropy model is not one of the following:
            - 'gOM'
            - 'gTiret'
            - 'gCOM'
            - 'polynomial'
            - 'discrete'

    Returns
    -------
    rr : array_like
        Radius.
    beta : array_like
        Velocity anisotropy.
    ebeta : array_like
        Uncertainty on velocity anisotropy.

    """

    if model not in ["gOM", "gTiret", "gCOM", "polynomial", "discrete"]:
        raise ValueError("Does not recognize surface density model.")

    if model == "discrete":
        smooth = False

    if smooth is False or model == "polynomial":
        rr, beta, ebeta = get_anisotropy(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            method=method,
            nbin=nbin,
            polorder=polorder,
            bootp=bootp,
            logx=logx,
            smooth=smooth,
        )
    else:
        rr, beta, ebeta = get_anisotropy(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            method=method,
            nbin=nbin,
            polorder=polorder,
            bootp=bootp,
            logx=logx,
            smooth=False,
        )

        r, phi, theta, vr, vphi, vtheta = angle.cart_to_sph(x, y, z, vx, vy, vz)
        r = np.sort(r)

        nonan1 = np.logical_not(np.isnan(beta))
        nonan2 = np.logical_not(np.isnan(ebeta))
        nonan = nonan1 * nonan2

        rr = rr[nonan]
        beta = beta[nonan]
        ebeta = ebeta[nonan]

        # https://stackoverflow.com/questions/24633664/confidence-interval-for-exponential-curve-fit/26042460#26042460

        if model == "gOM":

            popt, pcov = curve_fit(
                bgOM,
                rr,
                beta,
                sigma=ebeta,
                p0=[np.nanmean(r), 0, 0],
                absolute_sigma=False,
                bounds=([0, -np.inf, -np.inf], [np.inf, 1.0, 1.0]),
            )

            b_params = unc.correlated_values(popt, pcov)

            px = r
            py = bgOM(px, *b_params)

        if model == "gTiret":

            popt, pcov = curve_fit(
                bgTiret,
                rr,
                beta,
                sigma=ebeta,
                p0=[np.nanmean(r), 0, 0],
                absolute_sigma=False,
                bounds=([0, -np.inf, -np.inf], [np.inf, 1.0, 1.0]),
            )

            b_params = unc.correlated_values(popt, pcov)

            px = r
            py = bgTiret(px, *b_params)

        if model == "gCOM":

            popt, pcov = curve_fit(
                bCOM,
                rr,
                beta,
                sigma=ebeta,
                p0=[np.nanmean(r), 0],
                absolute_sigma=False,
                bounds=([0, -np.inf], [np.inf, 1.0]),
            )

            b_params = unc.correlated_values(popt, pcov)

            px = r
            py = bCOM(px, *b_params)

        rr = px
        beta = unp.nominal_values(py)
        ebeta = unp.std_devs(py)

    return rr, beta, ebeta


def get_betasym(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    method="moving",
    nbin=30,
    polorder=10,
    logx=False,
    bootp=False,
    smooth=True,
    model="gOM",
):
    """
    Computes the symmetric velocity anisotropy, defined as:
        beta_sym(r) = beta(r) / (1 - beta(r)/2)

    Parameters
    ----------
    x : array_like, float
        x-axis.
    y : array_like, float
        y-axis.
    z : array_like, float
        z-axis.
    vx : array_like, float
        x-axis velocity.
    vy : array_like, float
        y-axis velocity.
    vz : array_like, float
        z-axis velocity.
    method : string, optional
        Method used to compute the dispersion.
        The default is "moving".
    nbin : int, optional
        Number of bins/tracers used in the dispersion calculation.
        The default is 30.
    polorder : int, optional
        Order of smoothing polynomial.
        The default is 10.
    logx : boolean, optional
        If the x axis should be binned logarithmly.
        The default is False.
    bootp : boolean, optional
        Use bootstrap to compute errors. The default is False.
    smooth : boolean, optional
        Smooth the data with a polynomial. The default is True.
    model : string, optional
            Model used to fit (or not fit) the data. The default is "gOM".

    Returns
    -------
    rr : array_like
        Radius.
    beta : array_like
        Symmetric velocity anisotropy.
    ebeta : array_like
        Uncertainty on the symmetric velocity anisotropy.

    """
    r, beta, ebeta = get_beta(
        x,
        y,
        z,
        vx,
        vy,
        vz,
        method=method,
        nbin=nbin,
        polorder=polorder,
        bootp=bootp,
        logx=logx,
        smooth=smooth,
        model=model,
    )

    beta_sym = beta / (1 - beta * 0.5)

    ebeta_sym = np.abs(beta_sym * ebeta) * np.sqrt(
        1 / (beta * beta) + 0.25 / ((1 - beta * 0.5) * (1 - beta * 0.5))
    )

    return r, beta_sym, ebeta_sym


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Plotting functions"
# ------------------------------------------------------------------------------


def plot_disp1D(
    ra,
    dec,
    pmra,
    epmra,
    pmdec,
    epmdec,
    corrpm,
    ra0,
    dec0,
    pmra0,
    epmra0,
    pmdec0,
    epmdec0,
    vlos0,
    d0,
    logx=False,
    logy=False,
    bootp=False,
    polorder=None,
    return_fits=False,
):
    """
    Plots a 1D dispersion map.

    Parameters
    ----------
    ra : array_like
        Gaia designation: ra.
    dec : array_like
        Gaia designation: dec.
    pmra : array_like
        Gaia designation: pmra.
    pmdec : array_like
        Gaia designation: pmdec.
    epmra : array_like
        Gaia designation: pmra_error.
    epmdec : array_like
        Gaia designation: pmdec_error.
    corrpm : array_like
        Gaia designation: pmra_pmdec_corr.
    ra0 : float
        Bulk RA, in degrees.
    dec0 : float
        Bulk Dec, in degrees.
    pmra0 : float
        Bulk PMRA, in mas/yr.
    epmra0 : float
        Bulk PMRA uncertainty, in mas/yr.
    pmdec0 : float
        Bulk PMDec, in mas/yr.
    epmdec0 : float
        Bulk PMDec uncertainty, in mas/yr.
    vlos0 : float
        Bulk line-of-sight velocity, in km/s.
    d0 : float
        Bulk distance, in kpc.
        The default is True.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is False.
    logx : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of x.
        The default is False.
    logy : boolean, optional
        True if the dispersion is evaluated in a logarithm-spaced grid of y.
        The default is False.
    polorder : int, optional
        Order of smoothing polynomial.
        The default is None.
    return_fits: Whether the user wants to return the polynomial
        smoothing fits.
        The default is False.

    Returns
    -------
    r : array_like
        Reference array (possibly binned).
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    pmr, pmt = v_sky_to_polar(ra, dec, pmra, pmdec, ra0, dec0, pmra0, pmdec0)
    rproj = angle.sky_distance_deg(ra, dec, ra0, dec0)
    pmr = pmr - pmr_corr(vlos0, rproj, d0)

    uncpmr, uncpmt = unc_sky_to_polar(
        ra, dec, epmra, epmdec, corrpm, ra0, dec0, epmra0, epmdec0
    )

    # Sorts the values according to R_proj
    L = sorted(zip(rproj, pmr, pmt, uncpmr, uncpmt), key=operator.itemgetter(0))
    rproj, pmr, pmt, uncpmr, uncpmt = zip(*L)

    rproj = np.asarray(rproj)
    pmr = np.asarray(pmr)
    pmt = np.asarray(pmt)
    uncpmr = np.asarray(uncpmr)
    uncpmt = np.asarray(uncpmt)

    rr, dd, err = dispersion(
        rproj,
        [pmr, pmt],
        [uncpmr, uncpmt],
        bins="percentile",
        polorder=polorder,
        bootp=bootp,
        logx=logx,
        return_fits=return_fits,
    )

    rr2, dd2, err2 = dispersion(
        rproj,
        [pmr, pmt],
        [uncpmr, uncpmt],
        bins=int(0.5 * position.good_bin(rproj)),
        bootp=False,
        logx=True,
        smooth=False,
    )

    if logx is False and logy is False:
        plt.plot(rr, dd, ls="-", color="red", lw=3)
        plt.plot(rr, dd + err, ls="--", color="red")
        plt.plot(rr, dd - err, ls="--", color="red")
        plt.plot(rr2, dd2, "bo")

    if logx is True and logy is False:
        plt.semilogx(rr, dd, ls="-", color="red", lw=3)
        plt.semilogx(rr, dd + err, ls="--", color="red")
        plt.semilogx(rr, dd - err, ls="--", color="red")
        plt.semilogx(rr2, dd2, "bo")

    if logx is False and logy is True:
        plt.semilogy(rr, dd, ls="-", color="red", lw=3)
        plt.semilogy(rr, dd + err, ls="--", color="red")
        plt.semilogy(rr, dd - err, ls="--", color="red")
        plt.semilogy(rr2, dd2, "bo")

    if logx is True and logy is True:
        plt.loglog(rr, dd, ls="-", color="red", lw=3)
        plt.loglog(rr, dd + err, ls="--", color="red")
        plt.loglog(rr, dd - err, ls="--", color="red")
        plt.loglog(rr2, dd2, "bo")

    plt.errorbar(rr2, dd2, yerr=err2, color="b", ls="none", barsabove=True, zorder=10)

    plt.show()

    if return_fits is True:
        return rr2, dd2, err2
    else:
        return


def plot_disp2D(
    ra,
    dec,
    pmra,
    epmra,
    pmdec,
    epmdec,
    corrpm,
    ra0,
    dec0,
    pmra0,
    epmra0,
    pmdec0,
    epmdec0,
    vlos0,
    d0,
    rlim=None,
    bootp=False,
    robust_sig=False,
    nbin=None,
    nmov=None,
    return_fits=False,
):
    """
    Plots a 2D dispersion map, along with a map of respective uncertainties.

    Parameters
    ----------
    ra : array_like
        Gaia designation: ra.
    dec : array_like
        Gaia designation: dec.
    pmra : array_like
        Gaia designation: pmra.
    pmdec : array_like
        Gaia designation: pmdec.
    epmra : array_like
        Gaia designation: pmra_error.
    epmdec : array_like
        Gaia designation: pmdec_error.
    corrpm : array_like
        Gaia designation: pmra_pmdec_corr.
    ra0 : float
        Bulk RA, in degrees.
    dec0 : float
        Bulk Dec, in degrees.
    pmra0 : float
        Bulk PMRA, in mas/yr.
    epmra0 : float
        Bulk PMRA uncertainty, in mas/yr.
    pmdec0 : float
        Bulk PMDec, in mas/yr.
    epmdec0 : float
        Bulk PMDec uncertainty, in mas/yr.
    vlos0 : float
        Bulk line-of-sight velocity, in km/s.
    d0 : float
        Bulk distance, in kpc.
    rlim : float, optional
        Maximum radius in the image, in degrees.
        The default is None.
    bootp : boolean, optional
        True if the errors are drawn from a Bootstrap method.
        The default is False.
    robust_sig : boolean, optional
        True if the user wants to compute  a dispersion
        less sensible to outliers.
        The default is False.
    nbin : int, optional
        Auxiliar value for binning when bins is not an integer.
        The default is None.
    nmov : int, optional
        Auxiliar value for moving grids.
        The default is None.
    return_fits : boolean, optional
        True is user wants to return the dispersion profile.
        The default is False.

    Returns
    -------
    r : array_like
        Reference array (possibly binned).
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.

    """

    pmr, pmt = v_sky_to_polar(ra, dec, pmra, pmdec, ra0, dec0, pmra0, pmdec0)
    rproj = angle.sky_distance_deg(ra, dec, ra0, dec0)
    pmr = pmr - pmr_corr(vlos0, rproj, d0)

    uncpmr, uncpmt = unc_sky_to_polar(
        ra, dec, epmra, epmdec, corrpm, ra0, dec0, epmra0, epmdec0
    )

    rr, dd, err = dispersion(
        [ra, dec],
        [pmr, pmt],
        [uncpmr, uncpmt],
        bootp=bootp,
        a0=ra0,
        d0=dec0,
        robust_sig=robust_sig,
        nbin=nbin,
        nmov=nmov,
    )

    if rlim is None:
        rlim = position.quantile(rproj, 0.5)
    xlim, ylim = angle.get_circle_sph_trig(rlim, ra0, dec0)

    allPoints = np.column_stack((ra, dec))
    hullPoints = ConvexHull(allPoints)
    idx_lims = hullPoints.vertices
    rlims = angle.sky_distance_deg(ra[idx_lims], dec[idx_lims], ra0, dec0)
    rmax = np.nanmin(rlims)

    alim0, dlim0 = angle.get_circle_sph_trig(rmax, ra0, dec0)

    qmin = position.quantile(dd.flatten()[np.where(dd.flatten() > 0)], 0.05)
    qmax = position.quantile(dd.flatten()[np.where(dd.flatten() > 0)], 0.95)

    fig, ax = plt.subplots(figsize=(5, 4))
    plt.title(r"Dispersion")
    c = plt.imshow(
        dd.T,
        origin="lower",
        aspect="auto",
        vmin=qmin,
        vmax=qmax,
        cmap="jet",
        extent=[np.nanmin(alim0), np.nanmax(alim0), np.nanmin(dlim0), np.nanmax(dlim0)],
    )
    cbar = fig.colorbar(c)
    cbar.ax.tick_params(labelsize=13)
    plt.xlim([np.amin(xlim), np.amax(xlim)])
    plt.ylim([np.amin(ylim), np.amax(ylim)])
    plt.gca().invert_xaxis()
    plt.show()

    qmin = position.quantile(err.flatten()[np.where(err.flatten() > 0)], 0.05)
    qmax = position.quantile(err.flatten()[np.where(err.flatten() > 0)], 0.95)

    fig, ax = plt.subplots(figsize=(5, 4))
    plt.title(r"Uncertainty in dispersion")
    c = plt.imshow(
        err.T,
        origin="lower",
        aspect="auto",
        vmin=qmin,
        vmax=qmax,
        cmap="jet",
        extent=[np.nanmin(alim0), np.nanmax(alim0), np.nanmin(dlim0), np.nanmax(dlim0)],
    )
    cbar = fig.colorbar(c)
    cbar.ax.tick_params(labelsize=13)
    plt.xlim([np.amin(xlim), np.amax(xlim)])
    plt.ylim([np.amin(ylim), np.amax(ylim)])
    plt.gca().invert_xaxis()
    plt.show()

    if return_fits is True:
        return rr, dd, err
    else:
        return
