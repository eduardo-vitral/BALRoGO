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
# If you have any further questions please email evitral@stsci.edu
#
###############################################################################

from . import position
from . import angle

import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.special import gamma, erf, erfc, erfcx, log_ndtr
from scipy.optimize import differential_evolution, curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from functools import partial
import uncertainties.unumpy as unp
import uncertainties as unc
import numdifftools as ndt
import emcee
from multiprocessing import Pool
from multiprocessing import cpu_count
import warnings
import operator
import time

ncpu = cpu_count()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Global variables"
# ------------------------------------------------------------------------------

# Gravitational constant, in N m^2 kg^-2
G = 6.67430 * 1e-11

# Multiplying factor to pass from solar mass to kg
msun_to_kg = 1.98847 * 1e30

# Multiplying factor to pass from kpc to km
kpc_to_km = 3.086 * 10**16

# Multiplying factor to pass from mas to radians
mas_to_rad = 1e-3 * (1 / 3600) * (np.pi / 180)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Generical statistics"
# ------------------------------------------------------------------------------


def weight_mean(x, dx, w):
    """
    x: Averaged quantity
    dx: Uncertainty on x
    w: weight
    """
    if np.isscalar(w):
        w = np.ones_like(x) * w

    wmean = np.nansum(x * w) / np.nansum(w)

    dmudx = w / np.nansum(w)
    dmu2 = (dmudx * dx) ** 2
    dwmean = np.sqrt(np.nansum(dmu2))

    return wmean, dwmean


# Numpy implementation
def weighted_median(x, w):
    # Sort the values and weights by the values
    sorted_indices = np.argsort(x)
    sorted_values = x[sorted_indices]
    sorted_weights = w[sorted_indices]

    # Compute the cumulative sum of the weights
    cumsum_weights = np.cumsum(sorted_weights)

    # Find the cutoff for the median
    cutoff = np.sum(sorted_weights) / 2.0

    # Find the first value where the cumulative weight
    # exceeds or equals the cutoff
    median_index = np.searchsorted(cumsum_weights, cutoff)

    return sorted_values[median_index]


def weighted_std(x, w):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(x, weights=w)
    # Fast and numerically precise:
    variance = np.average((x - average) ** 2, weights=w)
    return np.sqrt(variance)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Proper motions and conversions"
# ------------------------------------------------------------------------------


def pos_sky_to_cart(
    a,
    d,
    a0,
    d0,
):
    """
    Transforms sky positions in cartesian projected ones.

    Parameters
    ----------
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
    """
    a = np.copy(a) * (np.pi / 180)
    d = np.copy(d) * (np.pi / 180)
    a0 = np.copy(a0) * (np.pi / 180)
    d0 = np.copy(d0) * (np.pi / 180)

    sinda = np.sin(a - a0)
    cosda = np.cos(a - a0)
    sind = np.sin(d)
    sind0 = np.sin(d0)
    cosd = np.cos(d)
    cosd0 = np.cos(d0)

    dx = sinda * cosd
    dy = cosd0 * sind - sind0 * cosd * cosda

    return dx, dy


def pos_cart_to_sky(
    dx,
    dy,
    a0,
    d0,
):
    """
    Transforms cartesian projected positions in sky ones.

    Parameters
    ----------
    dx : array_like
        x coordinate of the source, in radians.
    dy : array_like
        y coordinate of the source, in radians.
    a0 : float
        Bulk RA, in radians.
    d0 : float
        Bulk Dec, in radians.

    Returns
    -------
    a : array_like
        Right ascention in degrees.
    d : array_like
        Declination in degrees.
    """
    dr = np.arcsin(np.sqrt(dx**2 + dy**2))
    dp = np.arctan2(dx, dy)

    a, d = angle.polar_to_sky(
        dr,
        dp,
        a0,
        d0,
    )

    a = np.copy(a) * 180 / np.pi
    d = np.copy(d) * 180 / np.pi

    return a, d


def v_sky_to_cart(
    a,
    d,
    pma,
    pmd,
    a0,
    d0,
    pma0,
    pmd0,
):
    """
    Transforms proper motions in RA Dec into projected cartesian.

    Parameters
    ----------
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    pma : array_like
        PMRA of the source.
    pmd : array_like
        PMDec of the source.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
    pma0 : float
        Bulk PMRA.
    pmd0 : float
        Bulk PMDec.
    """
    a = np.copy(a) * (np.pi / 180)
    d = np.copy(d) * (np.pi / 180)
    a0 = np.copy(a0) * (np.pi / 180)
    d0 = np.copy(d0) * (np.pi / 180)

    sinda = np.sin(a - a0)
    cosda = np.cos(a - a0)
    sind = np.sin(d)
    sind0 = np.sin(d0)
    cosd = np.cos(d)
    cosd0 = np.cos(d0)

    theta = np.arccos(sind0 * sind + cosd0 * cosd * cosda)
    cost = np.cos(theta)

    pmx = cosda * (pma - pma0 * cosd / cosd0) - sind * sinda * pmd
    pmy = (
        (cosd * cosd0 + sind * sind0 * cosda) * pmd
        - cost * pmd0
        + (pma - pma0 * cosd / cosd0) * sind0 * sinda
    )

    return pmx, pmy


def v_sky_to_polar(
    a,
    d,
    pma,
    pmd,
    a0,
    d0,
    pma0,
    pmd0,
):
    """
    Transforms proper motions in RA Dec into polar coordinates
    (radial and tangential).

    Parameters
    ----------
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    pma : array_like
        PMRA of the source.
    pmd : array_like
        PMDec of the source.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
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
    dx, dy = pos_sky_to_cart(a, d, a0, d0)
    pmx, pmy = v_sky_to_cart(a, d, pma, pmd, a0, d0, pma0, pmd0)

    rho = np.sqrt(dx * dx + dy * dy)

    pmr = (dx * pmx + dy * pmy) / rho
    pmt = (-dx * pmy + dy * pmx) / rho

    return pmr, pmt


def unc_sky_to_cart(
    a,
    d,
    epma,
    epmd,
    a0,
    d0,
    epma0,
    epmd0,
):
    """
    Transforms proper motions uncertainties in RA Dec into projected
    cartesian uncertainties.

    Parameters
    ----------
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    epma : array_like
        Uncertainty in PMRA of the source.
    epmd : array_like
        Uncertainty in PMDec of the source.
    epmad : array_like
        Correlation between epma and epmd.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
    epma0 : float
        Uncertainty in Bulk PMRA.
    epmd0 : float
        Uncertainty in Bulk PMDec.

    Returns
    -------
    uncpmx : array_like
        Uncertainty in PM in radial direction.
    uncpmy : array_like
        Uncertainty in PM in tangential direction.

    """
    a = np.copy(a) * (np.pi / 180)
    d = np.copy(d) * (np.pi / 180)
    a0 = np.copy(a0) * (np.pi / 180)
    d0 = np.copy(d0) * (np.pi / 180)

    sinda = np.sin(a - a0)
    cosda = np.cos(a - a0)
    sind = np.sin(d)
    sind0 = np.sin(d0)
    cosd = np.cos(d)
    cosd0 = np.cos(d0)

    dvdpma = cosda
    dvdpmd = -sinda * sind
    dvdpma0 = -cosda * cosd / cosd0
    dvdpmd0 = 0

    uncpmx = np.sqrt(
        (dvdpma * epma) ** 2
        + (dvdpmd * epmd) ** 2
        + (dvdpma0 * epma0) ** 2
        + (dvdpmd0 * epmd0) ** 2
    )

    dvdpma = sinda * sind0
    dvdpmd = cosd * cosd0 + cosda * sind * sind0
    dvdpma0 = -cosd * sinda * sind0 / cosd0
    dvdpmd0 = -cosda * cosd * cosd0 - sind * sind0
    uncpmy = np.sqrt(
        (dvdpma * epma) ** 2
        + (dvdpmd * epmd) ** 2
        + (dvdpma0 * epma0) ** 2
        + (dvdpmd0 * epmd0) ** 2
    )

    return uncpmx, uncpmy


def unc_sky_to_polar(
    a,
    d,
    epma,
    epmd,
    epmad,
    a0,
    d0,
    epma0,
    epmd0,
):
    """
    Transforms proper motions uncertainties in RA Dec into polar coordinates
    uncertainties (radial and tangential).

    Parameters
    ----------
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    epma : array_like
        Uncertainty in PMRA of the source.
    epmd : array_like
        Uncertainty in PMDec of the source.
    epmad : array_like
        Correlation between epma and epmd.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
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
    a = np.copy(a) * (np.pi / 180)
    d = np.copy(d) * (np.pi / 180)
    a0 = np.copy(a0) * (np.pi / 180)
    d0 = np.copy(d0) * (np.pi / 180)

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
        cosd**2 * sinda**2 + (cosd0 * sind - cosda * cosd * sind0) ** 2,
    )

    dvdpma = (cosd0 * sinda * (cosda * cosd * cosd0 + sind * sind0)) / dentheta
    dvdpmd = (
        -cosd * sinda**2 * sind
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


def pmr_corr(v0, ev0, a, d, a0, d0, dist):
    """
    Correction on radial proper motion due to apparent contraction/expansion
    of the cluster.

    One should perform pmr_new = pmr_old - pmr_corr.
    Uncertainties should be added quadratically.

    Reference:
    van der Marel, R. P., Alves, D. R., Hardy, E., & Suntzeff, N. B.
    2002, AJ, 124, 2639
    - Equation (13).

    Parameters
    ----------
    v0: array-like
        Bulk line-of-sight velocity, in km/s
    ev0: array-like
        Uncertainty in Bulk line-of-sight velocity, in km/s
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
    dist: float
        Cluster distance from the Sun, in kpc.

    Returns
    -------
    pmrcorr : array_like, float
        Correction in the radial component of the proper motion, in mas/yr.
    epmrcorr : array_like, float
        Uncertainty in the Correction in the radial component of the
        proper motion, in mas/yr.

    """
    conv = 1 / (4.7405 * dist)

    dx, dy = pos_sky_to_cart(a, d, a0, d0)
    rho = np.sqrt(dx * dx + dy * dy)

    pmrcorr = -conv * v0 * np.sin(rho)
    epmrcorr = conv * ev0 * np.sin(rho)

    return pmrcorr, epmrcorr


def vlos_corr(
    v0,
    ev0,
    a,
    d,
    a0,
    d0,
    pma0,
    pmd0,
    epma0,
    epmd0,
    dist,
):
    """
    Correction on line-of-sight velocity due to apparent
    contraction/expansion of the cluster.

    One should perform vlos_new = vlos_old - vlos_corr.
    Uncertainties should be added quadratically.

    Reference:
    van der Marel, R. P., Alves, D. R., Hardy, E., & Suntzeff, N. B.
    2002, AJ, 124, 2639
    - Equation (13).

    Parameters
    ----------
    v0: array-like
        Bulk line-of-sight velocity, in km/s
    ev0: array-like
        Uncertainty in Bulk line-of-sight velocity, in km/s
    a : array_like
        RA of the source, in degrees.
    d : array_like
        Dec of the source, in degrees.
    a0 : float
        Bulk RA, in degrees.
    d0 : float
        Bulk Dec, in degrees.
    pma0 : float
        Bulk PMRA, in mas/yr.
    pmd0 : float
        Bulk PMDec, in mas/yr.
    epma0 : float
        Uncertainty in Bulk PMRA, , in mas/yr.
    epmd0 : float
        Uncertainty in Bulk PMDec, , in mas/yr.
    dist: float
        Cluster distance from the Sun, in kpc.

    Returns
    -------
    vcorr : array_like, float
        Correction in the vlos, in km/s.
    evcorr : array_like, float
        Uncertainty in the Correction in the vlos, in km/s.

    """
    conv = 4.7405 * dist

    dx, dy = pos_sky_to_cart(a, d, a0, d0)

    a0 = np.copy(a0) * (np.pi / 180)
    d0 = np.copy(d0) * (np.pi / 180)

    at = mas_to_rad * pma0 / np.cos(d0) + a0
    dt = mas_to_rad * pmd0 + d0

    dxt, dyt = pos_sky_to_cart(
        at * (180 / np.pi),
        dt * (180 / np.pi),
        a0 * (180 / np.pi),
        d0 * (180 / np.pi),
    )

    rho = np.sqrt(dx * dx + dy * dy)
    phi = np.arctan2(dy, dx)
    thetat = np.arctan2(dyt, dxt)

    vt = np.sqrt(pma0**2 + pmd0**2) * conv
    evt = (
        np.sqrt(
            (pma0 * epma0 / vt) ** 2 + (pmd0 * epmd0 / vt) ** 2,
        )
        * conv**2
    )

    vcorr = vt * np.sin(rho) * np.cos(phi - thetat) + v0 * (np.cos(rho) - 1)

    evcorr = np.sqrt(
        evt**2 * (np.sin(rho) * np.cos(phi - thetat)) ** 2
        + ev0**2 * (np.cos(rho) - 1) ** 2
    )

    return vcorr, evcorr


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Higher order moment functions"
# ------------------------------------------------------------------------------


def log1mexp(x):
    """log(1 - exp(-x)).

    Taken from pymc3.math

    This function is numerically more stable than the naive approach.
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    with np.errstate(divide="ignore"):
        return np.where(
            x < 0.683,
            np.log(-np.expm1(-x)),
            np.log1p(-np.exp(-x)),
        )


def logdiffexp(a, b):
    """log(exp(a) - exp(b))"""
    return a + log1mexp(a - b)


def alpha(y):
    return np.exp(-(y**2) / 2.0) / np.sqrt(2.0 * np.pi)


def lnalpha(y):
    return -(y**2) / 2.0 - 0.5 * np.log(2.0 * np.pi)


def lnerfc(x):
    """ln erfc(x) = ln (1-erf(x)) = ln sqrt{2/pi}int_x^infty e^{-t^2} dt

    For positive arguments we use the identity
    ln erfc(x) = ln erfcx(x)-x^2

    """
    return (x > 0.0) * (np.log(erfcx(np.abs(x))) - x**2) + (x <= 0.0) * np.log(
        erfc(-np.abs(x))
    )


def lnerfcx(x, LOWERLIM=1e-300):
    """ln erfcx(x) = ln (exp(x^2)erfc(x))

    For negative arguments we use the identity
    ln erfcx(x) = ln erfc(x)+x^2

    """
    return (x < 0.0) * (np.log(erfc(-np.abs(x)) + LOWERLIM) + x**2) + (
        x >= 0.0
    ) * np.log(erfcx(np.abs(x)))


# ==============================================================================
#
# Negative kurtosis family of models
# ----------------------------------
# These models are formed from the convolution of a Gaussian with a uniform
# kernel. To introduce skewness, the uniform kernel has a different
# width/height on either side of the axis
#
# K(y) = 1/(2a_+) for 0<y<a_+; 1/(2a_-) for -a_-<y<=0
#
# See Section 4.1 of Sanders & Evans (2020) for more details
#
# ==============================================================================


def _uniform_kernel_parameters(h3, h4):
    """
    Converts the Gauss-Hermite coefficients (h3, h4) for uniform kernel model
    into the corresponding (a,Delta,b,w_0) as outlined in Table 1 of
    Sanders & Evans (2020)

    Parameters
    ----------
    h3 : array_like
        3rd GH coefficient
    h4 : array_like
        4th GH coefficient

    Returns
    -------
    (a, delta, b, w0) : tuple of array_like
        Parameters of pdf, width a, skewness Delta (note here capital Delta),
        variance scale b, mean scale w_0.
    """

    if np.any(h4 > 0):
        warnings.warn(
            "h4<0 passed to _uniform_kernel_parameters "
            "-- implicitly converting to -|h4|"
        )

    h40 = -0.187777
    if np.any(h4 < h40):
        warnings.warn(
            "h4<-0.187777 passed to _uniform_kernel_parameters "
            "-- limiting value of h4 is -0.187777, will return nan"
        )

    delta_h3 = 0.82
    delta_h4 = 4.3
    kinf = 1.3999852768764105
    k0 = np.sqrt(3.0)
    scl_a = 2.0
    scl = 3.3

    h4_3 = np.abs(h4 / (h3 + 1e-20)) / (-h40)
    delta = (
        np.sign(h3)
        * (-delta_h3 * h4_3 + np.sqrt((delta_h3 * h4_3) ** 2 + 4 * delta_h4))
        / (2 * delta_h4)
    )

    # The following assumes the index i = 4 in Table 1 from Sanders & Evans 20
    a = scl_a / np.sqrt(
        np.sqrt((1 - delta_h4 * delta**2) * np.abs(h40 / (h4 + 1e-20))) - 1
    )
    kinf = kinf * np.sqrt(1 + delta**2 + 3 * delta**4)
    delta *= a
    b = np.sqrt(1.0 + a**2 / (k0 - (k0 - kinf) * np.tanh(a / scl)) ** 2)
    w0 = (-(delta / 2.0) + (delta / 3.0) * np.tanh(a / scl)) / b

    return a, delta, b, w0


def uniform_kernel_pdf(x, err, mean, sigma, h3, h4):
    """

    Probability density function for the uniform kernel
    model from Sanders & Evans (2020)

    f_{sigma_e}(x) = f_s(w)/sigma

    where w = (x-mean)/sigma, s = sigma_e/sigma

    f_s(w) = b/(2a_+a_-)(
        a_+ Phi((bw'+a_-)/t) - a_- Phi((bw'-a_+)/t)
        -2 Delta Phi(bw'/t))

    see equation (38) of Sanders & Evans (2020)

    Phi(x) is the cumulative of the unit normal.

    The parameters of the model (a, delta, b, w_0) are
    chosen such that h_1~h_2~0 and reproduce the required
    h_3, h_4. See Table 1 of Sanders & Evans (2020). The
    transformations are computed by
    _uniform_kernel_parameters. These models are only valid
    if h4<0. If h4>0 is passed, the code will use -h4 and give
    a warning.

    w' = w-w_0
    t = 1 + b^2 s^2
    a_pm = a pm delta

    Parameters
    ----------
    x : array_like
        input coordinate (velocity)
    err : array_like
        input coordinate uncertainties
    mean : array_like
        mean velocity
    sigma : array_like
        dispersion parameter (not standard deviation)
    h3 : array_like
        3rd Gauss-Hermite coefficient
    h4 : array_like
        4th Gauss-Hermite coefficient

    Returns
    -------
    pdf: array_like
        probability density function

    """
    w = (x - mean) / sigma
    werr = err / sigma

    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    t = np.sqrt(1.0 + b * b * werr * werr)

    am, ap = a - delta, a + delta
    it = 1.0 / (np.sqrt(2.0) * t)
    bw = b * (w - w0)
    if type(delta) is not np.ndarray:
        if delta == 0:
            pdf = (
                0.25
                * b
                / a
                * (
                    erf(
                        (a - bw) * it,
                    )
                    + erf(
                        (a + bw) * it,
                    )
                )
                / sigma
            )
            return pdf
    pdf = (
        0.25
        * b
        * (
            am * erf((ap - bw) * it)
            + ap * erf((am + bw) * it)
            - 2 * delta * erf(bw * it)
        )
        / (ap * am)
        / sigma
    )

    return pdf


def ln_uniform_kernel_pdf(x, err, mean, sigma, h3, h4):
    """

    Natural logarithm of the probability density function
    for the uniform kernel model from Sanders &
    Evans (2020). Full details are given in
    uniform_kernel_pdf. This function is optimized for
    numerical stability to avoid under/overflow (see
    Appendix E of Sanders & Evans, 2020)

    Parameters
    ----------
    x : array_like
        input coordinate (velocity)
    err : array_like
        input coordinate uncertainties
    mean : array_like
        mean velocity
    sigma : array_like
        dispersion parameter (not standard deviation)
    h3 : array_like
        3rd Gauss-Hermite coefficient
    h4 : array_like
        4th Gauss-Hermite coefficient

    Returns
    -------
    ln_pdf: array_like
        probability density function

    """

    w = (x - mean) / sigma
    werr = err / sigma

    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    t = np.sqrt(1.0 + b * b * werr * werr)

    am, ap = a - delta, a + delta
    it = 1.0 / t
    bw = b * (w - w0)

    if type(delta) is not np.ndarray:
        if delta == 0.0:
            ln_pdf = np.log(0.5 * b / a) + np.where(
                (b * w + a) * it < 0.0,
                logdiffexp(log_ndtr((bw + a) * it), log_ndtr((bw - a) * it)),
                logdiffexp(log_ndtr(-(bw - a) * it), log_ndtr(-(bw + a) * it)),
            )
            ln_pdf -= np.log(sigma)

            return ln_pdf

    ln_pdf = np.log(0.5 * b / (ap * am)) + np.logaddexp(
        np.log(am)
        + np.where(
            (ap - bw) * it < 0.0,
            logdiffexp(log_ndtr((ap - bw) * it), log_ndtr(-bw * it)),
            logdiffexp(log_ndtr(bw * it), log_ndtr(-(ap - bw) * it)),
        ),
        np.log(ap)
        + np.where(
            (bw + am) * it < 0.0,
            logdiffexp(log_ndtr((am + bw) * it), log_ndtr(bw * it)),
            logdiffexp(log_ndtr(-bw * it), log_ndtr(-(am + bw) * it)),
        ),
    )
    ln_pdf -= np.log(sigma)

    return ln_pdf


def uniform_kernel_variance_kurtosis(sigma, h3, h4, mean=None):
    """
    Evaluate the variance and excess kurtosis of the
    uniform kernel model from Sanders & Evans (2020).
    See Table D2 of Sanders & Evans (2020) for more
    information.

    Parameters
    ----------
    sigma : array_like
        Dispersion parameter.
    h3 : array_like
        3rd Gauss-Hermite coefficient.
    h4 : array_like
        4th Gauss-Hermite coefficient.

    Returns
    -------
     res : tuple of array_like
         (variance, excess kurtosis) of uniform kernel
         model.

    """

    a, delta, b, w0 = _uniform_kernel_parameters(h3, h4)
    variance = (1.0 + a * a / 3.0 + delta**2 / 12.0) / b / b * sigma**2
    kurtosis = (
        -1.0
        / 120.0
        * (16.0 * a**4 - 4 * a**2 * delta**2 + delta**4)
        / (1.0 + a * a / 3.0 + delta**2 / 12.0) ** 2
    )

    if mean is not None:
        stat_mean = mean + 0.5 * delta * sigma / b
        skewness = (delta * a * a * 0.25) / (
            1.0 + a * a / 3.0 + delta**2 / 12.0
        ) ** 1.5
        res = stat_mean, variance, skewness, kurtosis
    else:
        res = variance, kurtosis

    return res


# ==============================================================================
#
# Positive kurtosis family of models
# ----------------------------------
# These models are formed from the convolution of a Gaussian with a Laplace
# kernel. To introduce skewness, the Laplace kernel has a different width
# on either side of the axis.
#
# K(y) = exp(-y/a_+)/(2a_+) for y>=0; exp(y/a_-) for y<0
#
# See Section 4.2 of Sanders & Evans (2020) for more details
#
# ==============================================================================


def _laplace_kernel_parameters(h3, h4):
    """
    Converts the Gauss-Hermite coefficients (h3, h4) into the corresponding
    (a,Delta,b,w_0) for Laplace kernel model as outlined in Table 1 of
    Sanders & Evans (2020)

    Parameters
    ----------
    h3 : array_like
        3rd GH coefficient
    h4 : array_like
        4th GH coefficient

    Returns
    -------
    (a, delta, b, w0) : tuple of array_like
        Parameters of pdf, width a, skewness Delta (note here capital Delta),
        variance scale b, mean scale w_0.
    """

    if np.any(h4 < 0):
        warnings.warn(
            "h4>0 passed to _laplace_kernel_parameters "
            "-- implicitly converting to -|h4|"
        )

    h40 = 0.145461
    if np.any(h4 > h40):
        warnings.warn(
            "h4>0.145461 passed to _laplace_kernel_parameters "
            "-- limiting value of h4 is 0.145461, will return nan"
        )

    delta_h4 = 2.0
    delta_h3 = 0.37
    scl = 2.25
    scl_a = 1.6
    scl_a3 = 1.1
    k0 = 1.0 / np.sqrt(2.0)
    kinf = 1.0806510105505178

    acoeff = delta_h4 * h40 / (np.abs(h4 + 1e-10))
    bcoeff = -delta_h3 / np.abs(h3 + 1e-10) * (scl_a / scl_a3) ** 2
    ccoeff = h40 / np.abs(h4 + 1e-10) - 1 + (scl_a / scl_a3) ** 2
    delta = (
        np.sign(h3)
        * (-bcoeff - np.sqrt(bcoeff**2 - 4 * acoeff * ccoeff))
        / (2 * acoeff)
    )
    a = scl_a / np.sqrt(
        h40 * (1 + delta_h4 * delta**2) / np.abs(h4 + 1e-10) - 1,
    )

    kinf = kinf * np.sqrt(1 + 3 * delta**2)
    b = np.sqrt(1.0 + a**2 / (k0 - (k0 - kinf) * np.tanh(a / scl)) ** 2)
    delta *= a
    w0 = (-delta + (8.0 * delta / 7.0) * np.tanh(5.0 * a / scl / 4.0)) / b

    return a, delta, b, w0


def laplace_kernel_pdf(x, err, mean, sigma, h3, h4):
    """
    Probability density function for the Laplace kernel
    model from Sanders & Evans (2020).

    This implementation follows Eq. (41) of Sanders & Evans (2020)
    and supports vectorized evaluation on 1D or broadcasted 2D grids.

    Parameters
    ----------
    x : array_like
        Input coordinate (velocity).
    err : array_like
        Input coordinate uncertainties.
    mean : float
        Mean velocity.
    sigma : float
        Dispersion parameter (not standard deviation).
    h3 : float
        3rd Gauss–Hermite coefficient.
    h4 : float
        4th Gauss–Hermite coefficient.

    Returns
    -------
    pdf : array_like
        Probability density function evaluated at `x`.
    """

    # ------------------------------------------------------------------
    # Dimensionless variables
    # ------------------------------------------------------------------
    w = (x - mean) / sigma
    werr = err / sigma

    # Kernel parameters from Sanders & Evans (2020)
    a, delta, b, mean_w = _laplace_kernel_parameters(h3, h4)
    t = np.sqrt(1.0 + b * b * werr * werr)
    ap = a + delta
    am = a - delta

    # ==================================================================
    # Positive branch: a_+
    # ==================================================================
    argU = t * t - 2.0 * ap * b * (w - mean_w)

    # IMPORTANT FIX:
    # Allocate arrays with the same shape as argU (not x),
    # because argU is what defines the boolean masks.
    positive_term = np.zeros_like(argU)

    # ---- argU < 0 -----------------------------------------------------
    prefactor = b / (4.0 * ap)
    mask = argU < 0.0
    positive_term[mask] = (
        prefactor
        * np.exp((argU / (2.0 * ap**2))[mask])
        * erfc(
            (
                (t * t - ap * b * (w - mean_w))
                / (
                    np.sqrt(
                        2.0,
                    )
                    * t
                    * ap
                )
            )[mask]
        )
    )

    # ---- argU > 0 -----------------------------------------------------
    prefactor = b / ap
    mask = argU > 0.0
    positive_term[mask] = (
        np.sqrt(np.pi / 8.0)
        * prefactor
        * alpha((b * (w - mean_w) / t)[mask])
        * erfcx(
            (
                (t * t - ap * b * (w - mean_w))
                / (
                    np.sqrt(
                        2.0,
                    )
                    * t
                    * ap
                )
            )[mask]
        )
    )

    # ==================================================================
    # Negative branch: a_-
    # ==================================================================
    argU = t * t + 2.0 * am * b * (w - mean_w)

    # IMPORTANT FIX:
    # Same shape discipline as for positive_term.
    negative_term = np.zeros_like(argU)

    # ---- argU < 0 -----------------------------------------------------
    prefactor = b / (4.0 * am)
    mask = argU < 0.0
    negative_term[mask] = (
        prefactor
        * np.exp((argU / (2.0 * am**2))[mask])
        * erfc(
            (
                (t * t + am * b * (w - mean_w))
                / (
                    np.sqrt(
                        2.0,
                    )
                    * t
                    * am
                )
            )[mask]
        )
    )

    # ---- argU > 0 -----------------------------------------------------
    prefactor = b / am
    mask = argU > 0.0
    negative_term[mask] = (
        np.sqrt(np.pi / 8.0)
        * prefactor
        * alpha((b * (w - mean_w) / t)[mask])
        * erfcx(
            (
                (t * t + am * b * (w - mean_w))
                / (
                    np.sqrt(
                        2.0,
                    )
                    * t
                    * am
                )
            )[mask]
        )
    )

    # ------------------------------------------------------------------
    # Final PDF (Eq. 41, divided by sigma)
    # ------------------------------------------------------------------
    pdf = (positive_term + negative_term) / sigma

    return pdf


def ln_laplace_kernel_pdf(x, err, mean, sigma, h3, h4):
    """

    Natural logarithm of the probability density function
    for the Laplace kernel model from Sanders &
    Evans (2020). Full details are given in
    laplace_kernel_pdf. This function is optimized for
    numerical stability to avoid under/overflow (see
    Appendix E of Sanders & Evans, 2020)

    Parameters
    ----------
    x : array_like
        input coordinate (velocity)
    err : array_like
        input coordinate uncertainties
    mean : array_like
        mean velocity
    sigma : array_like
        dispersion parameter (not standard deviation)
    h3 : array_like
        3rd Gauss-Hermite coefficient
    h4 : array_like
        4th Gauss-Hermite coefficient

    Returns
    -------
    ln_pdf: array_like
        probability density function

    """
    w = (x - mean) / sigma
    werr = err / sigma
    a, delta, b, mean_w = _laplace_kernel_parameters(h3, h4)
    t = np.sqrt(1.0 + b * b * werr * werr)

    ap = a + delta
    am = a - delta

    argU = t * t - 2 * ap * b * (w - mean_w)
    positive_term = np.zeros_like(x)

    prefactor = np.log(b / (4.0 * ap))
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU < 0.0]
    positive_term[argU < 0.0] = (
        prefactor
        + (argU / 2.0 / ap**2)[argU < 0.0]
        + lnerfc(
            (
                (t * t - ap * b * (w - mean_w))
                / np.sqrt(
                    2,
                )
                / t
                / ap
            )[argU < 0.0]
        )
    )

    prefactor = np.log(b / ap)
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU > 0.0]
    positive_term[argU > 0.0] = (
        0.5 * np.log(np.pi / 8.0)
        + prefactor
        + lnalpha((b * (w - mean_w) / t)[argU > 0.0])
        + lnerfcx(
            (
                (t * t - ap * b * (w - mean_w))
                / np.sqrt(
                    2,
                )
                / t
                / ap
            )[argU > 0.0]
        )
    )

    argU = t * t + 2 * am * b * (w - mean_w)
    negative_term = np.zeros_like(x)

    prefactor = np.log(b / (4.0 * am))
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU < 0.0]
    negative_term[argU < 0.0] = (
        prefactor
        + (argU / 2.0 / am**2)[argU < 0.0]
        + lnerfc(
            (
                (t * t + am * b * (w - mean_w))
                / np.sqrt(
                    2,
                )
                / t
                / am
            )[argU < 0.0]
        )
    )
    prefactor = np.log(b / am)
    if type(h4) is np.ndarray:
        prefactor = prefactor[argU > 0.0]
    negative_term[argU > 0.0] = (
        0.5 * np.log(np.pi / 8.0)
        + prefactor
        + lnalpha((b * (w - mean_w) / t)[argU > 0.0])
        + lnerfcx(
            (
                (t * t + am * b * (w - mean_w))
                / np.sqrt(
                    2,
                )
                / t
                / am
            )[argU > 0.0]
        )
    )

    ln_pdf = np.logaddexp(positive_term, negative_term) - np.log(sigma)

    return ln_pdf


def laplace_kernel_variance_kurtosis(sigma, h3, h4, mean=None):
    """
    Evaluate the variance and excess kurtosis of the
    Laplace kernel model from Sanders & Evans (2020).
    See Table D2 of Sanders & Evans (2020) for more
    information.

    Parameters
    ----------
    sigma : array_like
        Dispersion parameter.
    h3 : array_like
        3rd Gauss-Hermite coefficient.
    h4 : array_like
        4th Gauss-Hermite coefficient.

    Returns
    -------
     res : tuple of array_like
         (variance, excess kurtosis) of Laplace kernel
         model.

    """

    a, delta, b, w0 = _laplace_kernel_parameters(h3, h4)
    variance = (1.0 + a * a * 2 + delta**2) / b / b * sigma**2
    kurtosis = (
        6
        * (2 * a**4 + 12 * a**2 * delta**2 + delta**4)
        / (1.0 + a * a * 2 + delta**2) ** 2
    )
    if mean is not None:
        stat_mean = mean + delta * sigma / b
        skewness = (
            2 * delta * (6 * a * a + delta**2) / (1.0 + a * a * 2 + delta**2) ** 1.5
        )
        res = stat_mean, variance, skewness, kurtosis
    else:
        res = variance, kurtosis

    return res


def mom_likelihood_func(params, x, ex, ww=None, mode="lnlik"):
    """
    Compute the negative log-likelihood for a Gauss–Hermite–based
    moment model with noise convolution.

    The model describes the intrinsic distribution using a
    Gauss–Hermite expansion parameterized by (mean, sigma, h3, h4),
    convolved with observational uncertainties. The likelihood is
    evaluated using either a Laplace or uniform kernel, depending on
    the sign of h4.

    Parameters
    ----------
    params : array_like, shape (4,)
        Model parameters:
        - params[0] : mean of the intrinsic distribution
        - params[1] : intrinsic dispersion (sigma)
        - params[2] : third Gauss–Hermite moment (h3; skewness)
        - params[3] : fourth Gauss–Hermite moment (h4; kurtosis)
    x : array_like
        Observed data values.
    ex : array_like
        Measurement uncertainties associated with `x`.
    ww : array_like
        Weights applied to each data point in the likelihood.

    Returns
    -------
    lnlik : float
        Negative log-likelihood value. Returns `np.inf` if the model
        produces non-physical moments (e.g., negative variance or
        invalid kurtosis).

    Notes
    -----
    - For h4 >= 0, a Laplace kernel is used.
    - For h4 < 0, a uniform kernel is used.
    - Models yielding non-positive variance or kurtosis outside the
      interval [1.8, 6] are rejected.
    - NaN contributions to the likelihood are removed before summation.
    """

    if ww is None:
        ww = np.ones_like(x)

    mean, sigma, h3, h4 = params

    if h4 >= 0:
        variance, kurtosis = laplace_kernel_variance_kurtosis(sigma, h3, h4)
    else:
        variance, kurtosis = uniform_kernel_variance_kurtosis(sigma, h3, h4)

    if variance <= 0 or np.isnan(variance) or np.isnan(kurtosis):
        return np.inf

    if mode == "lnlik":
        if h4 >= 0:
            lnlik_i = ln_laplace_kernel_pdf(x, ex, mean, sigma, h3, h4)
        else:
            lnlik_i = ln_uniform_kernel_pdf(x, ex, mean, sigma, h3, h4)
    else:
        if h4 >= 0:
            curve = laplace_kernel_pdf(x, ex, mean, sigma, h3, h4)
        else:
            curve = uniform_kernel_pdf(x, ex, mean, sigma, h3, h4)
        return curve

    mask = ~np.isnan(lnlik_i)

    if not np.any(mask):
        return np.inf

    lnlik = -np.sum(lnlik_i[mask] * ww[mask])
    return lnlik


def mom_likelihood_call(x, ex, ww):
    """
    Perform maximum-likelihood estimation of Gauss–Hermite moments
    (mean, sigma, h3, h4) using global optimization.

    This function initializes reasonable starting values and bounds,
    then minimizes the negative log-likelihood defined in
    `mom_likelihood_func` using differential evolution.

    Parameters
    ----------
    x : array_like
        Observed data values.
    ex : array_like
        Measurement uncertainties associated with `x`.
    ww : array_like
        Weights applied to each data point in the likelihood.

    Returns
    -------
    results : ndarray, shape (4,)
        Maximum-likelihood estimates of:
        - mean
        - intrinsic dispersion (sigma)
        - h3 (skewness)
        - h4 (kurtosis)

    Notes
    -----
    - Initial guesses for mean and dispersion are computed using
      weighted statistics.
    - Parameter bounds are scaled relative to the initial dispersion
      estimate to ensure numerical stability.
    - Optimization is performed using `scipy.optimize.differential_evolution`.
    """

    # Initial parameter guess: mean, sigma, h3, h4
    ini = np.asarray(
        [
            weighted_median(x, ww),
            weighted_std(x, ww),
            0.0,
            0.0,
        ]
    )

    # Parameter bounds for optimization
    bounds = [
        (ini[0] - 3.0 * ini[1], ini[0] + 3.0 * ini[1]),
        (0.2 * ini[1], 5.0 * ini[1]),
        (-0.2, 0.2),
        (-0.187, 0.145),
    ]

    mle_model = differential_evolution(
        lambda c: mom_likelihood_func(c, x, ex, ww, mode="lnlik"),
        bounds,
    )

    return mle_model.x


def mom_sample_generator(mom_stats, eps=None, nsig=10, debug=False):
    """
    Generate random samples from a Gauss–Hermite–based PDF
    via inverse-CDF sampling.

    The intrinsic distribution is defined by Gauss–Hermite moments
    (mean, sigma, h3, h4). Depending on the sign of h4, either a Laplace
    or uniform kernel is used to construct the PDF. Sampling is performed
    numerically using inverse transform sampling.

    Parameters
    ----------
    mom_stats : array_like, shape (4, 2)
        Moment estimates and uncertainties. Only the first column
        (moment values) is used, ordered as:
        - mean
        - sigma
        - h3
        - h4
    eps : array_like or None, optional
        Measurement uncertainties associated with each sample.
        If None, an informational message is printed and no sampling
        is performed.
    nsig : int, optional
        Extent of the sampling grid in units of sigma around the mean.
        Default is 10.
    debug : boolean, optional
        Whether to print debugging statements.
        Default is False.

    Returns
    -------
    samples : ndarray or None
        Random samples drawn from the specified PDF. Returns None if
        input parameters are invalid or incomplete.

    Notes
    -----
    - Sampling is performed on a fixed grid spanning
      [mean − nsig·sigma, mean + nsig·sigma].
    - The PDF is normalized numerically before constructing the CDF.
    - Physical validity of the moments is enforced via variance and
      kurtosis constraints.
    """

    if eps is None:
        print(
            "mom_sample_generator: Measurement uncertainties `eps` were not "
            "provided. Please supply an array of uncertainties matching the "
            "desired sample size."
        )
        return None

    mean, sigma, h3, h4 = mom_stats[:, 0]

    # ---------------------------------------------------------
    # 1. Compute variance and kurtosis from kernel moments
    # ---------------------------------------------------------
    if h4 >= 0:
        variance, kurtosis = laplace_kernel_variance_kurtosis(sigma, h3, h4)
    else:
        variance, kurtosis = uniform_kernel_variance_kurtosis(sigma, h3, h4)

    # ---------------------------------------------------------
    # 2. Reject non-physical parameter combinations
    # ---------------------------------------------------------

    if debug:
        print(
            "mom_sample_generator: ",
            f"(variance={variance:.3g}, kurtosis={kurtosis:.3g}).",
        )
    if variance <= 0 or np.isnan(variance) or np.isnan(kurtosis):
        print(
            "mom_sample_generator: Provided parameters yield ",
            "non-physical moments ",
            f"(variance={variance:.3g}, kurtosis={kurtosis:.3g}).",
        )
        return None

    # ---------------------------------------------------------
    # 3. Construct sampling grid
    # ---------------------------------------------------------
    xgrid = np.linspace(
        mean - nsig * sigma,
        mean + nsig * sigma,
        2 * nsig * 100 + 1,
    )
    # ---------------------------------------------------------
    # 4. Evaluate PDF on grid (with uncertainty marginalization)
    # ---------------------------------------------------------
    x2d = xgrid[:, None]
    e2d = eps[None, :]

    if h4 >= 0:
        pdf_2d = laplace_kernel_pdf(x2d, e2d, mean, sigma, h3, h4)
    else:
        pdf_2d = uniform_kernel_pdf(x2d, e2d, mean, sigma, h3, h4)

    # Marginalize over uncertainties
    pdf_vals = np.nanmean(pdf_2d, axis=1)

    # ---------------------------------------------------------
    # 5. Inverse CDF sampling
    # ---------------------------------------------------------
    cdf = np.cumsum(pdf_vals)
    cdf /= cdf[-1]

    inv_cdf = interp1d(
        cdf,
        xgrid,
        bounds_error=False,
        fill_value="extrapolate",
    )

    uni = np.random.rand(len(eps))
    samples = inv_cdf(uni)

    return samples


def mom_monte_carlo(
    ex,
    ww,
    mom_stats,
    nsamples,
    output="full",
):
    """
    Perform Monte Carlo bias estimation and correction for
    Gauss–Hermite moments.

    This function repeatedly draws synthetic samples based on an input set
    of moment estimates, re-fits the moments via maximum likelihood, and
    derives a multiplicative bias correction and uncertainty for each moment.

    Parameters
    ----------
    ex : array_like
        Measurement uncertainties associated with the data.
    ww : array_like
        Weights applied to each data point in the likelihood.
    mom_stats : array_like, shape (N, 2)
        Initial moment estimates and uncertainties.

        The first four rows are assumed to correspond to:
        - index 0 : mean
        - index 1 : sigma
        - index 2 : h3
        - index 3 : h4

        Additional rows (if present) correspond to derived quantities
        and are propagated but NOT used to generate samples.
    nsamples : int
        Number of Monte Carlo realisations.
    output : {"basic", "full"}, optional
        Level of output detail. If "full", additional derived quantities
        are recomputed and included in the returned array.

    Returns
    -------
    mom_corrected : ndarray, shape (N, 2)
        Bias-corrected estimates and uncertainties for all quantities
        present in `mom_stats`.

        Bias correction is applied multiplicatively using the ratio
        between recovered and input values.
    """

    # ---------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------
    nrows = mom_stats.shape[0]
    mom_samples = np.full((nrows, nsamples), np.nan)

    # Only the first four moments define the intrinsic distribution
    mom_params = mom_stats[:4, :]

    # ---------------------------------------------------------
    # 2. Monte Carlo resampling loop
    # ---------------------------------------------------------
    for k in range(nsamples):
        # Generate synthetic sample from intrinsic moments
        sample = mom_sample_generator(mom_params, eps=ex)

        # Re-fit Gauss–Hermite moments
        mom_samples[:4, k] = mom_likelihood_call(sample, ex, ww)

        if output == "full":
            # ---------------------------------------------
            # Compute derived quantities for this iteration
            # ---------------------------------------------
            mean_k, sigma_k, h3_k, h4_k = mom_samples[:4, k]

            if np.isnan(h4_k):
                continue

            if h4_k >= 0.0:
                stm_k, var_k, skew_k, kurt_k = laplace_kernel_variance_kurtosis(
                    sigma_k,
                    h3_k,
                    h4_k,
                    mean=mean_k,
                )
            else:
                stm_k, var_k, skew_k, kurt_k = uniform_kernel_variance_kurtosis(
                    sigma_k,
                    h3_k,
                    h4_k,
                    mean=mean_k,
                )

            mom_samples[4, k] = stm_k
            mom_samples[5, k] = var_k
            mom_samples[6, k] = skew_k
            mom_samples[7, k] = kurt_k
            mom_samples[8, k] = np.sqrt(var_k)
            mom_samples[9, k] = np.sqrt(var_k + stm_k**2)

    # ---------------------------------------------------------
    # 3. Bias correction
    # ---------------------------------------------------------
    mom_corrected = np.zeros((nrows, 2))

    # Multiplicative bias ratio (safe against NaNs)
    ratio = np.nanmean(mom_samples, axis=1) / mom_stats[:, 0]

    mom_corrected[:, 0] = mom_stats[:, 0] / ratio
    mom_corrected[:, 1] = mom_stats[:, 1] / ratio

    # ---------------------------------------------------------
    # 4. Physicality check on corrected (mean, sigma, h3, h4)
    # ---------------------------------------------------------
    mean_c, sigma_c, h3_c, h4_c = mom_corrected[:4, 0]

    # Compute variance and kurtosis from kernel moments, mirroring your logic
    if h4_c >= 0:
        # You indicated this returns (variance, kurtosis) in this usage
        variance_c, kurtosis_c = laplace_kernel_variance_kurtosis(
            sigma_c,
            h3_c,
            h4_c,
        )
    else:
        variance_c, kurtosis_c = uniform_kernel_variance_kurtosis(
            sigma_c,
            h3_c,
            h4_c,
        )

    if (variance_c <= 0) or np.isnan(variance_c) or np.isnan(kurtosis_c):
        print(
            "mom_monte_carlo: Corrected parameters yield non-physical moments "
            f"(variance={variance_c:.3g}, kurtosis={kurtosis_c:.3g}). "
            "Returning original mom_stats without correction."
        )
        return mom_stats

    return mom_corrected


def print_vdm_franx_consistency(mom_stats):
    """
    Print consistency diagnostics between measured moments and
    vdM & Franx (1993) approximations.

    Parameters
    ----------
    mom_stats : ndarray, shape (N, 2)
        Moment statistics array. The following rows are assumed:
        - index 1 : sigma
        - index 2 : h3
        - index 3 : h4
        - index 6 : kurtosis (SE)
        - index 7 : sigma (SE)
    """
    sigma = mom_stats[1, 0]
    h3 = mom_stats[2, 0]
    h4 = mom_stats[3, 0]

    # vdM & Franx lambda parameter
    lam = 1.0 / (1.0 + np.sqrt(0.375) * h4)

    # Second moment (variance proxy)
    val2 = sigma**2 * (
        1.0 + lam**2 * (h4 * (2.0 * np.sqrt(6.0) + 3.0 * h4) - 3.0 * h3**2)
    )

    # Fourth moment (kurtosis proxy)
    val4 = (
        0.5
        * lam**4
        * (
            16.0 * np.sqrt(6.0) * h4
            - 9.0 * h4**2 * (8.0 + 6.0 * np.sqrt(6.0) * h4 + 5.0 * h4**2)
            + 12.0 * h3**2 * (15.0 * h4**2 + 8.0 * np.sqrt(6.0) * h4 - 8.0)
            - 108.0 * h3**4
        )
    )

    print("\nConsistency with vdM & Franx (1993) approximations:")
    print(f"  lambda      = {lam:.3g}")
    print(f"  kurt_vdm    = {val4:.3g}")
    print(f"  kurt_se     = {mom_stats[6, 0]:.3g}")
    print(f"  sigma_vdm   = {np.sqrt(val2):.3g}")
    print(f"  sigma_se    = {mom_stats[7, 0]:.3g}")

    return


def fit_1d_moments(
    x,
    ex,
    ww=None,
    method="monte-carlo",
    output="full",
    nsamples=100,
    debug=False,
):
    """
    Fit 1D Gauss–Hermite moments using Monte Carlo likelihood resampling.

    Estimates the first four Gauss–Hermite moments (mean, sigma, h3, h4)
    by repeatedly maximizing the likelihood and applying a Monte Carlo
    bias correction.

    Parameters
    ----------
    x : array_like
        Observed data values.
    ex : array_like
        Measurement uncertainties associated with `x`.
    ww : array_like or None, optional
        Weights applied to each data point in the likelihood.
        If None, uniform weights are used.
    method : str, optional
        If method == "monte-carlo", apply Monte Carlo bias correction.
        Otherwise, return uncorrected moment estimates from the Monte Carlo
        sample distribution.
    output : {"basic", "full"}, optional
        Level of output detail. If "full", additional derived
        quantities are computed and stored internally.
    nsamples : int, optional
        Number of Monte Carlo realisations. Default is 100.
    debug : bool, optional
        Whether to print diagnostic output. Default is False.

    Returns
    -------
    mom_corrected : ndarray
        Array of shape (N, 2), where N depends on `output` and `method`.

        - For output == "basic": N = 4 (mean, sigma, h3, h4)
        - For output == "full":  N > 4, including derived quantities:
            variance, kurtosis, root mean square and standard deviation.

        The first column contains mean estimates, the second column
        contains uncertainties (standard deviations).

    """

    # ---------------------------------------------------------
    # 0. Default weights
    # ---------------------------------------------------------
    if ww is None:
        ww = np.ones_like(x)

    if output == "full":
        labels = [
            "mean",
            "sigma",
            "h3",
            "h4",
            "stat-mean",
            "variance",
            "skewness",
            "kurtosis",
            "standard-deviation",
            "root-mean-square",
        ]
    else:
        labels = ["mean", "sigma", "h3", "h4"]

    # ---------------------------------------------------------
    # 1. Monte Carlo likelihood sampling
    # ---------------------------------------------------------
    mom_samples = np.zeros((4, nsamples))

    if debug:
        time_start = time.time()

    for k in range(nsamples):
        mom_samples[:, k] = mom_likelihood_call(x, ex, ww)

    if debug:
        lapse = round(time.time() - time_start, 2)
        print(
            "fit_1d_moments:",
            "Recovered likelihood samples.",
            "\nTook",
            lapse,
            "seconds.",
        )

    # ---------------------------------------------------------
    # 2. Optional: compute derived physical quantities
    # ---------------------------------------------------------
    if output == "full":
        stat_mean = np.full(nsamples, np.nan)
        variance = np.full(nsamples, np.nan)
        skewness = np.full(nsamples, np.nan)
        kurtosis = np.full(nsamples, np.nan)

        # Masks based on sign of h4
        mask_pos = mom_samples[3, :] >= 0.0
        mask_neg = ~mask_pos

        # Positive h4 → Laplace kernel
        if np.any(mask_pos):
            stm, var, skew, kurt = laplace_kernel_variance_kurtosis(
                mom_samples[1, mask_pos],  # sigma
                mom_samples[2, mask_pos],  # h3
                mom_samples[3, mask_pos],  # h4
                mean=mom_samples[0, mask_pos],  # mean
            )
            stat_mean[mask_pos] = stm
            variance[mask_pos] = var
            skewness[mask_pos] = skew
            kurtosis[mask_pos] = kurt

        # Negative h4 → Uniform kernel
        if np.any(mask_neg):
            stm, var, skew, kurt = uniform_kernel_variance_kurtosis(
                mom_samples[1, mask_neg],  # sigma
                mom_samples[2, mask_neg],  # h3
                mom_samples[3, mask_neg],  # h4
                mean=mom_samples[0, mask_neg],  # mean
            )
            stat_mean[mask_neg] = stm
            variance[mask_neg] = var
            skewness[mask_neg] = skew
            kurtosis[mask_neg] = kurt

        # Additional derived quantities
        x_std = np.sqrt(variance)
        x2_mom = np.sqrt(variance + stat_mean**2)

        # Append as extra rows (internal use only)
        mom_samples = np.vstack(
            (
                mom_samples,
                stat_mean,
                variance,
                skewness,
                kurtosis,
                x_std,
                x2_mom,
            )
        )

    # ---------------------------------------------------------
    # 3. Compute raw moment statistics
    # ---------------------------------------------------------
    nrows = mom_samples.shape[0]
    mom_stats = np.zeros((nrows, 2))

    mom_stats[:, 0] = np.nanmean(mom_samples, axis=1)
    mom_stats[:, 1] = np.nanstd(mom_samples, axis=1)

    if debug:
        print("Initial fit (value ± uncertainty):")
        for name, val, err in zip(labels, mom_stats[:, 0], mom_stats[:, 1]):
            print(f"  {name:>5s} = {val:.3g} ± {err:.3g}")

        if output == "full":
            print_vdm_franx_consistency(mom_stats)

    # ---------------------------------------------------------
    # 4. Monte Carlo bias correction (if requested)
    # ---------------------------------------------------------
    if method == "monte-carlo":
        if debug:
            time_start = time.time()

        mom_corrected = mom_monte_carlo(
            ex,
            ww,
            mom_stats,
            nsamples,
            output=output,
        )

        if debug:
            lapse = round(time.time() - time_start, 2)
            print(
                "fit_1d_moments:",
                "Applied Monte Carlo bias correction.",
                "\nTook",
                lapse,
                "seconds.",
            )

            print("Final fit (value ± uncertainty):")
            for name, val, err in zip(
                labels,
                mom_corrected[:, 0],
                mom_corrected[:, 1],
            ):
                print(f"  {name:>5s} = {val:.3g} ± {err:.3g}")
    else:
        # Fallback: no bias correction
        mom_corrected = mom_stats.copy()

    return mom_corrected


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Mean functions"
# ------------------------------------------------------------------------------


def bin_mean1d(
    x,
    y,
    ey,
    dimy,
    bins,
    ww=None,
    logx=True,
    method=None,
    nsamples=100,
):
    """
    Compute the binned mean and dispersion of a set of 1D measurements.

    Parameters
    ----------
    x : array_like
        Independent variable (binning variable).
    y : array_like, shape (dimy, N)
        Observed quantities whose dispersion/mean are computed.
    ey : array_like, shape (dimy, N)
        Measurement uncertainties of `y`.
    dimy : int
        Number of components in `y` (e.g., vx, vy, vz → dimy=3).
    bins : int
        Number of bin edges to use (final number of bins = bins - 1).
    ww : array_like or None, shape (dimy, N), optional
        Weights for each data point. If None, uniform weights are used.
    logx : bool, optional
        Whether to bin in logarithmic spacing. Default: True.
    method : {"vdv+", "vdma", None}, optional
        Method for dispersion estimation:
        - None     : Standard MLE with noise correction
        - "vdv+"   : Van der Ven et al. corrected estimator
        - "vdma"   : Monte Carlo VDMA estimator
    nsamples : int, optional
        Number of Monte Carlo realisations for VDMA. Default: 100.

    Returns
    -------
    r : ndarray, shape (bins - 1,)
        Effective radius of each bin (geometric or arithmetic mean).
    mean : ndarray, shape (bins - 1,)
        Mean value of `y` in each bin.
    err : ndarray, shape (bins - 1,)
        Uncertainty of the mean in each bin.

    Notes
    -----
    - Empty bins or bins containing only NaNs are assigned NaN results.
    - The function assumes `mle_mean`, `mle_disp`, and `monte_carlo_bias`
      exist in the surrounding namespace.
    """

    # ---------------------------------------------------------
    # 0. Initialise containers
    # ---------------------------------------------------------
    nb = bins - 1
    mean = np.zeros((dimy, nb))
    disp = np.zeros((dimy, nb))
    err = np.zeros((dimy, nb))
    r = np.zeros(nb)

    # Default weights → uniform if None
    if ww is None:
        ww = np.ones_like(y)

    # ---------------------------------------------------------
    # 1. Define bin edges in x
    # ---------------------------------------------------------
    xmin, xmax = np.nanmin(x), np.nanmax(x)

    if logx:
        rbin = np.logspace(np.log10(xmin), np.log10(xmax), bins)
    else:
        rbin = np.linspace(xmin, xmax, bins)

    # ---------------------------------------------------------
    # 2. Loop over components i and bins j
    # ---------------------------------------------------------
    for i in range(dimy):
        for j in range(nb):
            # Indices belonging to the bin
            cond = np.where((x >= rbin[j]) & (x < rbin[j + 1]))[0]

            # Bin center (geometric or arithmetic)
            r[j] = (
                np.sqrt(rbin[j] * rbin[j + 1])
                if logx
                else 0.5 * (rbin[j] + rbin[j + 1])
            )

            # Empty bin → output NaNs
            if len(cond) == 0:
                mean[i, j] = np.nan
                disp[i, j] = np.nan
                err[i, j] = np.nan
                continue

            y_bin = y[i][cond]
            ey_bin = ey[i][cond]
            ww_bin = ww[i][cond]

            # -------------------------------------------------
            # 2A. Standard MLE dispersion (default)
            # -------------------------------------------------
            if method is None:
                # Correct variance for measurement errors
                variance = np.nanstd(y_bin) ** 2 - np.nanmean(ey_bin**2)
                variance = max(variance, 0.0)  # numerical safety
                disp[i, j] = np.sqrt(variance)

                # Mean & its uncertainty
                mean[i, j], err[i, j] = mle_mean(
                    y_bin,
                    ey_bin,
                    disp[i, j],
                    ww=ww_bin,
                )

            # -------------------------------------------------
            # 2B. Van der Ven et al. (VDV+) dispersion
            # -------------------------------------------------
            elif method == "vdv+":
                n = len(cond)
                # Analytical correction factor
                bn = np.sqrt(2.0 / n) * gamma(n / 2) / gamma((n - 1) / 2)

                sig_mle = np.nanstd(y_bin)
                esig2_mle = np.nanmean(ey_bin**2)

                disp_val = sig_mle**2 - (bn**2) * esig2_mle
                disp_val = max(disp_val, 0.0)
                disp[i, j] = np.sqrt(disp_val) / bn

                mean[i, j], err[i, j] = mle_mean(
                    y_bin,
                    ey_bin,
                    disp[i, j],
                    ww=ww_bin,
                )

            # -------------------------------------------------
            # 2C. VDMA Monte Carlo dispersion
            # -------------------------------------------------
            elif method == "vdma":
                disp_samples = np.zeros(nsamples)
                # Draw Monte Carlo dispersion distribution
                for k in range(nsamples):
                    disp_samples[k] = mle_disp(y_bin, ey_bin, ww=ww_bin)

                disp[i, j] = np.nanmean(disp_samples)
                err[i, j] = np.nanstd(disp_samples)

                # Apply bias correction via Monte Carlo
                disp[i, j], err[i, j] = monte_carlo_bias(
                    y_bin, ey_bin, disp[i, j], nsamples, ww=ww_bin
                )

                # Compute mean using corrected dispersion
                mean[i, j], err[i, j] = mle_mean(
                    y_bin,
                    ey_bin,
                    disp[i, j],
                    ww=ww_bin,
                )

            else:
                raise ValueError(f"Unknown method: {method}")

    # ---------------------------------------------------------
    # 3. Combine dimensions (quadrature average)
    # ---------------------------------------------------------
    if dimy > 1:
        mean = np.sqrt(np.sum(mean**2, axis=0)) / np.sqrt(dimy)
    else:
        mean = mean[0]

    # Combine mean errors from all components
    err = np.sqrt(np.sum(err**2, axis=0)) / np.sqrt(dimy)

    return r, mean, err


def mean1d(
    x,
    y,
    ey,
    dimy,
    ww=None,
    bins=2,
    smooth=True,
    bootp=True,  # kept but unused
    logx=True,
    nbin=None,  # kept but unused
    polorder=None,
    return_fits=False,
    method=None,
    nsamples=100,
):
    """
    Compute the 1D mean profile of a quantity `y(x)` with optional
    binning and polynomial smoothing.

    Parameters
    ----------
    x : array_like, shape (N,)
        Independent variable.
    y : array_like, shape (dimy, N)
        Observed values to compute means from.
    ey : array_like, shape (dimy, N)
        Uncertainties on `y`.
    dimy : int
        Number of components in `y` (e.g., vx, vy, vz → dimy=3).
    ww : array_like or None, shape (dimy, N), optional
        Weights applied to the data. If None → unity weights.
    bins : int
        Number of bins (final number of bins = bins - 1).
    smooth : bool, optional
        Whether to apply polynomial smoothing to the binned data.
    bootp : bool, optional
        Currently unused (kept for backward compatibility).
    logx : bool, optional
        Bin in logarithmic scale if True.
    nbin : optional
        Unused (kept for backward compatibility).
    polorder : int or None
        Polynomial order for smoothing. If None → auto-determined.
    return_fits : bool, optional
        If True, return the polynomial coefficients.
    method : {"vdv+", "vdma", None}, optional
        Method for dispersion estimation passed to `bin_mean1d`.
    nsamples : int, optional
        Monte-Carlo samples when method="vdma".

    Returns
    -------
    r : ndarray
        Radii (bin centers or interpolated values).
    mean : ndarray
        Mean value of y at each radius.
    err : ndarray
        Standard error of mean.
    poly_mean : ndarray (optional)
        Polynomial fit coefficients if return_fits=True.
    """

    # ---------------------------------------------------------
    # 1. Basic binned mean (no smoothing yet)
    # ---------------------------------------------------------
    if isinstance(bins, int):
        r, mean, err = bin_mean1d(
            x,
            y,
            ey,
            dimy,
            bins=bins,
            ww=ww,
            logx=logx,
            method=method,
            nsamples=nsamples,
        )
    else:
        raise ValueError("`bins` must be an integer.")

    # Identify finite bins only
    good = (~np.isnan(mean)) & (~np.isnan(err))

    # Restrict smoothing to finite values
    r_clean = r[good]
    mean_clean = mean[good]
    err_clean = err[good]

    # Range allowed for smoothed evaluation on original x-grid
    rmin, rmax = np.nanmin(r_clean), np.nanmax(r_clean)
    idxrange = np.where((x > rmin) & (x < rmax))[0]

    # If no smoothing requested → return binned results
    if not smooth:
        return r, mean, err

    # ---------------------------------------------------------
    # 2. Polynomial smoothing setup
    # ---------------------------------------------------------
    # If polynomial order not given → choose automatically
    if polorder is None:
        pold = int(0.2 * position.good_bin(mean_clean))
    else:
        pold = polorder

    if pold < 0:
        raise ValueError("Polynomial order must be >= 0.")

    # ---------------------------------------------------------
    # 3. Fit polynomial (log or linear in x)
    # ---------------------------------------------------------
    if not logx:
        # Linear scale fit
        poly_coeff, cov = np.polyfit(
            r_clean, mean_clean, pold, w=1.0 / err_clean, cov=True
        )

        t = x[idxrange]
        TT = np.vstack([t ** (pold - i) for i in range(pold + 1)]).T

    else:
        # Logarithmic fit
        log_r = np.log10(r_clean)

        poly_coeff, cov = np.polyfit(
            log_r, mean_clean, pold, w=1.0 / err_clean, cov=True
        )

        t = np.log10(x[idxrange])
        TT = np.vstack([t ** (pold - i) for i in range(pold + 1)]).T

    # ---------------------------------------------------------
    # 4. Evaluate polynomial + propagate covariance
    # ---------------------------------------------------------
    smoothed_mean = TT @ poly_coeff  # polynomial evaluated
    C_yi = TT @ cov @ TT.T  # error propagation
    smoothed_err = np.sqrt(np.diag(C_yi))  # std dev

    # Output results
    r_out = x[idxrange]
    mean_out = smoothed_mean
    err_out = smoothed_err

    if return_fits:
        return r_out, mean_out, err_out, poly_coeff

    return r_out, mean_out, err_out


def mean(
    x,
    y,
    ey=None,
    ww=None,
    bins=None,
    smooth=True,
    bootp=False,  # kept for backward compatibility
    logx=False,
    nbin=None,  # unused, kept for compatibility
    polorder=None,
    return_fits=False,
    robust_sig=False,  # unused, kept for compatibility
    a0=None,  # unused, kept for compatibility
    d0=None,  # unused, kept for compatibility
    nmov=None,  # unused, kept for compatibility
    method="vdma",
    nsamples=100,
):
    """
    Generic 1D or (limited) 2D mean estimator for binned data.

    Parameters
    ----------
    x : array_like, shape (N,) or (2, N)
        Independent variable(s). If 1D → standard profile. If 2D → currently
        still treated as 1D by flattening logic in mean1d.
    y : array_like, shape (N,) or (dimy, N)
        Quantities whose mean is computed.
    ey : array_like, same shape as y, optional
        Errors on `y`. If None → zeros.
    ww : array_like, same shape as y, optional
        Weights on y. If None → zeros.
    bins : int or str
        - integer number of bins
        - "moving" not implemented here (kept for legacy compatibility)
    smooth : bool
        Apply polynomial smoothing.
    bootp : bool
        Bootstrap mode (not implemented here, kept for API compatibility)
    logx : bool
        Logarithmic binning.
    polorder : int, optional
        Polynomial smoothing degree.
    return_fits : bool
        Return polynomial coefficients.
    method : str
        Passed to bin_mean1d ("vdma", "vdv+", or None).
    nsamples : int
        Monte-Carlo iterations for “vdma” method.

    Returns
    -------
    r : array_like
        Binned (or interpolated) x values.
    mean : array_like
        Mean values.
    err : array_like
        Errors on the mean.
    """

    # ---------------------------------------------------------
    # 1. Input shape handling
    # ---------------------------------------------------------
    x = np.asarray(x)

    # Determine dimensionality of x
    if x.ndim == 1:
        dimx = 1
    elif x.ndim == 2 and x.shape[0] == 2:
        dimx = 2
    else:
        raise ValueError("`x` must be shape (N,) or (2, N).")

    # If no binning chosen → default to 2 bins
    if bins is None:
        bins = 2

    # ---------------------------------------------------------
    # 2. Force y, ey, ww to consistent shapes (dimy, N)
    # ---------------------------------------------------------
    y = np.asarray(y)

    if y.ndim == 1:
        # Promote to (1, N)
        y = y[np.newaxis, :]
        dimy = 1

        if ey is None:
            ey = np.zeros_like(y)
        else:
            ey = np.asarray(ey)[np.newaxis, :]

        if ww is None:
            ww = np.ones_like(y)
        else:
            ww = np.asarray(ww)[np.newaxis, :]

    else:
        # 2D input: shape must be (dimy, N)
        dimy = y.shape[0]

        if ey is None:
            ey = np.zeros_like(y)
        else:
            ey = np.asarray(ey)

        if ww is None:
            ww = np.ones_like(y)
        else:
            ww = np.asarray(ww)

        # Safety checks
        if ey.shape != y.shape:
            raise ValueError("`ey` must have same shape as `y`.")
        if ww.shape != y.shape:
            raise ValueError("`ww` must have same shape as `y`.")

    # ---------------------------------------------------------
    # 3. 1D mean-profile computation
    # ---------------------------------------------------------
    if dimx == 1:
        r, mean_val, err = mean1d(
            x,
            y,
            ey,
            dimy,
            ww=ww,
            bins=bins,
            smooth=smooth,
            bootp=bootp,
            polorder=polorder,
            logx=logx,
            nbin=nbin,
            return_fits=return_fits,
            method=method,
            nsamples=nsamples,
        )
    else:
        # 2D x is not handled differently in this code base.
        # It is likely meant for future expansion.
        # For now: flatten implicitly via mean1d interface.
        r, mean_val, err = mean1d(
            x[0],  # treat x-axis as primary coordinate
            y,
            ey,
            dimy,
            ww=ww,
            bins=bins,
            smooth=smooth,
            bootp=bootp,
            polorder=polorder,
            logx=logx,
            nbin=nbin,
            return_fits=return_fits,
            method=method,
            nsamples=nsamples,
        )

    # ---------------------------------------------------------
    # 4. Clean up nonphysical values
    # ---------------------------------------------------------
    err = np.where(err <= 0, np.nan, err)

    return r, mean_val, err


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Dispersion functions"
# ------------------------------------------------------------------------------


def aux_disp(idx, y, ey, dimy, robust_sig, method=None, nsamples=100):
    """
    Compute the (possibly robust) dispersion of y over a set of indices.

    Parameters
    ----------
    idx : array_like
        Indices selecting the data subset.
    y : array_like, shape (dimy, N)
        Values whose dispersion is computed.
    ey : array_like, same shape as y
        Measurement uncertainties on y.
    dimy : int
        Number of components in y (e.g. for vector quantities).
    robust_sig : bool
        If True, uses median absolute deviation (MAD) as a robust estimator.
    method : {"vdv+", None}, optional
        Dispersion estimator:
        - None: standard std^2 - mean(err^2)
        - "vdv+": van de Ven et al. 2006 correction
    nsamples : int
        (Unused here, placeholder for compatibility)

    Returns
    -------
    disp : float
        The aggregated dispersion over all dimy components.
    """

    # ------------------------------------------------------------------
    # Safety check: empty index → return NaN
    # ------------------------------------------------------------------
    if idx is None or len(idx) == 0:
        return np.nan

    # Output container for each component
    disp_comp = np.zeros(dimy, dtype=float)

    # ------------------------------------------------------------------
    # Loop over components (y dimension)
    # ------------------------------------------------------------------
    for i in range(dimy):
        yi = y[i][idx]
        eyi = ey[i][idx]

        # Skip if no valid data
        if yi.size == 0:
            disp_comp[i] = np.nan
            continue

        # --------------------------------------------------------------
        # Case 1: Robust median-based dispersion
        # --------------------------------------------------------------
        if robust_sig:
            # Median absolute deviation (MAD)
            mad = np.median(np.abs(yi - np.median(yi))) / 0.6745
            noise = np.nanmean(eyi**2)
            disp = mad**2 - noise

        # --------------------------------------------------------------
        # Case 2: Classical dispersion minus noise
        # --------------------------------------------------------------
        elif method is None:
            sig = np.nanstd(yi)
            noise = np.nanmean(eyi**2)
            disp = sig**2 - noise

        # --------------------------------------------------------------
        # Case 3: vdv+ (van de Ven et al. 2006) estimator
        # --------------------------------------------------------------
        elif method == "vdv+":
            n = yi.size
            if n < 2:
                disp_comp[i] = np.nan
                continue

            bn = np.sqrt(2.0 / n) * gamma(n / 2) / gamma((n - 1) / 2)
            sig_mle = np.nanstd(yi)
            esig2_mle = np.nanmean(eyi**2)
            disp = (sig_mle**2 - bn * bn * esig2_mle) / (bn * bn)

        # --------------------------------------------------------------
        # Unknown method
        # --------------------------------------------------------------
        else:
            raise ValueError(f"Unknown dispersion method: {method}")

        # Numerical safety: avoid negative values inside sqrt
        disp_comp[i] = np.sqrt(max(disp, 0.0))

    # ----------------------------------------------------------------------
    # Aggregate across all components (quadrature mean)
    # ----------------------------------------------------------------------
    # disp = sqrt(sum_i disp_i^2) / sqrt(dimy)
    valid = np.isfinite(disp_comp)
    if not np.any(valid):
        return np.nan

    disp_total = np.sqrt(np.sum(disp_comp[valid] ** 2)) / np.sqrt(
        np.sum(valid),
    )

    return disp_total


def aux_err(idx, y, ey, dimy, robust_sig, bootp, method=None, nsamples=100):
    """
    Compute the uncertainty on the dispersion estimate for a
    given subset of data.

    This is used inside hexbin maps to assign an error bar on the velocity
    dispersion inside each bin (or hexagon).

    Parameters
    ----------
    idx : array_like
        Indices of data points included in the bin.
    y : array_like, shape (dimy, N)
        Data values (e.g., velocities).
    ey : array_like, shape (dimy, N)
        Measurement uncertainties associated with y.
    dimy : int
        Number of y-components being combined.
    robust_sig : bool
        If True, use a robust median-based dispersion (MAD).
    bootp : bool
        If True, compute uncertainties via bootstrap resampling.
    method : str, optional
        Dispersion method: None, "vdv+" (van der Ven), etc.
    nsamples : int, optional
        Number of bootstrap samples if bootp=True.

    Returns
    -------
    err : float
        Uncertainty on the dispersion estimate for the bin.
    """

    # Storage for per-dimension dispersion and errors
    disp_dim = np.zeros((dimy, 1))
    err_dim = np.zeros((dimy, 1))

    # ------------------------------------------------------------------
    # Per-dimension dispersion and uncertainty
    # ------------------------------------------------------------------
    for i in range(dimy):
        yi = y[i][idx]
        eyi = ey[i][idx]

        if robust_sig:
            # ----------------------------------------------------------
            # Case 1: Robust MAD-based dispersion
            # ----------------------------------------------------------
            mad = np.median(np.abs(yi - np.median(yi))) / 0.6745
            noise = np.nanmean(eyi**2)
            disp = mad**2 - noise
            disp = np.sqrt(max(disp, 0))  # ensure non-negative

            disp_dim[i, 0] = disp

            # Error estimate
            if bootp:
                err_dim[i, 0] = bootstrap(yi, eyi, method="robust")
            else:
                n = len(yi)
                err_dim[i, 0] = disp / np.sqrt(max(2 * (n - 1), 1))

        else:
            # ----------------------------------------------------------
            # Case 2: Classical dispersion estimators
            # ----------------------------------------------------------
            sig = np.nanstd(yi)
            noise = np.nanmean(eyi**2)

            if method is None:
                # Classical sqrt(sigma^2 - noise)
                disp = sig**2 - noise
                disp = np.sqrt(max(disp, 0))

            elif method == "vdv+":
                # Van der Ven et al. maximum-likelihood correction
                n = len(yi)
                if n > 2:
                    bn = np.sqrt(2 / n) * gamma(n / 2) / gamma((n - 1) / 2)
                    disp = (sig**2 - bn**2 * noise) / bn
                    disp = np.sqrt(max(disp, 0))
                else:
                    disp = 0.0

            else:
                raise ValueError(f"Unknown dispersion method: {method}")

            disp_dim[i, 0] = disp

            # Error estimate
            if bootp:
                err_dim[i, 0] = bootstrap(
                    yi,
                    eyi,
                    method=method,
                    nsamples=nsamples,
                )
            else:
                n = len(yi)
                err_dim[i, 0] = disp / np.sqrt(max(2 * (n - 1), 1))

    # ------------------------------------------------------------------
    # Combine all dimensions (quadrature)
    # ------------------------------------------------------------------
    err_total = np.sqrt(np.nansum(err_dim**2, axis=0)) / np.sqrt(dimy)

    return err_total


def bin_montecarlo_1d(x, y, ey, dimy, bins, ww=None, logx=True, nsamples=100):
    """
    Compute the 1D Monte Carlo–based dispersion profile.

    For each bin in x, the dispersion of y is computed using nsamples
    Monte Carlo realizations of mle_disp. A bias correction is then
    applied via monte_carlo_bias.

    Parameters
    ----------
    x : array_like
        Independent variable (e.g., radius).
    y : array_like, shape (dimy, N)
        Values whose dispersion is computed.
    ey : array_like, shape (dimy, N)
        Measurement uncertainties associated with y.
    dimy : int
        Number of y-components to combine.
    bins : int
        Number of x-bins.
    ww : array_like or None
        Weights for y (same shape as y). If None, zero-weights assumed.
    logx : bool
        Use logarithmic binning in x.
    nsamples : int
        Number of Monte Carlo realizations per bin.

    Returns
    -------
    r : array_like, shape (bins-1,)
        Bin centers.
    disp : array_like, shape (bins-1,)
        Estimated dispersion in each bin.
    err : array_like, shape (bins-1,)
        Uncertainty on the dispersion (from MC sampling).
    """
    # Allocate output arrays
    disp_dim = np.zeros((dimy, bins - 1))
    err_dim = np.zeros((dimy, bins - 1))
    r = np.zeros(bins - 1)

    # Prepare weights
    if ww is None:
        ww = np.ones_like(y)

    # --------------------------------------------------------------
    # Define bin edges in log-space or linear-space
    # --------------------------------------------------------------
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if logx:
        r_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)
    else:
        r_edges = np.linspace(xmin, xmax, bins)

    # --------------------------------------------------------------
    # Loop over y dimensions and x-bins
    # --------------------------------------------------------------
    for i in range(dimy):
        yi = y[i]
        eyi = ey[i]
        w_i = ww[i]

        for j in range(bins - 1):
            # Select points in bin j
            idx = np.where((x >= r_edges[j]) & (x < r_edges[j + 1]))[0]

            # Bin center
            if logx:
                r[j] = np.sqrt(r_edges[j] * r_edges[j + 1])
            else:
                r[j] = 0.5 * (r_edges[j] + r_edges[j + 1])

            # If bin is empty → assign NaN and continue
            if len(idx) == 0:
                disp_dim[i, j] = np.nan
                err_dim[i, j] = np.nan
                continue

            yi_bin = yi[idx]
            eyi_bin = eyi[idx]
            w_bin = w_i[idx]

            # ------------------------------------------------------
            # Monte Carlo sampling of dispersion estimate
            # ------------------------------------------------------
            samples = np.zeros(nsamples)
            for k in range(nsamples):
                samples[k] = mle_disp(yi_bin, eyi_bin, ww=w_bin)

            # Raw MC mean and standard deviation
            disp_mc = np.nanmean(samples)
            # err_mc = np.nanstd(samples) # Unused.

            # ------------------------------------------------------
            # Apply bias correction via monte_carlo_bias
            # ------------------------------------------------------
            disp_corr, err_corr = monte_carlo_bias(
                yi_bin,
                eyi_bin,
                disp_mc,
                nsamples,
                ww=w_bin,
            )

            disp_dim[i, j] = disp_corr
            err_dim[i, j] = err_corr

    # --------------------------------------------------------------
    # Combine dimensions in quadrature
    # --------------------------------------------------------------
    disp = np.sqrt(np.nansum(disp_dim**2, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.nansum(err_dim**2, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def bin_disp1d(
    x,
    y,
    ey,
    dimy,
    bins,
    bootp=True,
    logx=True,
    method=None,
    nsamples=100,
):
    """
    Compute the dispersion of y as a function of x using simple binning.

    Each bin computes the dispersion using either:
      - classical sqrt( std^2 - mean(err^2) )
      - the van de Ven et al. 2006 correction ("vdv+")

    Errors come from bootstrap resampling if bootp=True.

    Parameters
    ----------
    x : array_like
        Independent variable (e.g. radius).
    y : array_like, shape (dimy, N)
        Quantity whose dispersion is computed.
    ey : array_like, shape (dimy, N)
        Measurement uncertainties on y.
    dimy : int
        Number of components in y.
    bins : int
        Number of x-bins.
    bootp : bool
        If True, uncertainties are estimated by bootstrap; otherwise analytic.
    logx : bool
        Use logarithmic binning in x.
    method : str or None
        Dispersion estimator: None (standard) or 'vdv+'.
    nsamples : int
        Number of MC bootstrap samples.

    Returns
    -------
    r : array_like, shape (bins-1,)
        Bin centers.
    disp : array_like, shape (bins-1,)
        Combined dispersion over y dimensions.
    err : array_like, shape (bins-1,)
        Uncertainty on the dispersion.
    """

    # Output arrays
    disp_dim = np.zeros((dimy, bins - 1))
    err_dim = np.zeros((dimy, bins - 1))
    r = np.zeros(bins - 1)

    # --------------------------------------------------------------
    # Define bin edges
    # --------------------------------------------------------------
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if logx:
        r_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)
    else:
        r_edges = np.linspace(xmin, xmax, bins)

    # --------------------------------------------------------------
    # Loop over y components and bins
    # --------------------------------------------------------------
    for i in range(dimy):
        yi = y[i]
        eyi = ey[i]

        for j in range(bins - 1):
            # Boolean mask for the bin
            mask = (x >= r_edges[j]) & (x < r_edges[j + 1])
            idx = np.where(mask)[0]

            # Bin center
            if logx:
                r[j] = np.sqrt(r_edges[j] * r_edges[j + 1])
            else:
                r[j] = 0.5 * (r_edges[j] + r_edges[j + 1])

            # Handle empty bin
            if len(idx) == 0:
                disp_dim[i, j] = np.nan
                err_dim[i, j] = np.nan
                continue

            yi_bin = yi[idx]
            eyi_bin = eyi[idx]

            # ------------------------------------------------------
            # Compute dispersion estimate
            # ------------------------------------------------------
            if method is None:
                # Standard estimator: sqrt( std^2 - mean(err^2) )
                sig = np.nanstd(yi_bin)
                noise = np.nanmean(eyi_bin**2)
                disp = sig**2 - noise

            elif method == "vdv+":
                # van de Ven et al. 2006 correction
                n = len(idx)
                bn = np.sqrt(2 / n) * gamma(n / 2) / gamma((n - 1) / 2)
                sig = np.nanstd(yi_bin)
                noise = np.nanmean(eyi_bin**2)
                disp = (sig**2 - bn**2 * noise) / (bn**2)

            else:
                raise ValueError(f"Unknown method: {method}")

            # Ensure no negative variance
            disp = np.nan if disp < 0 else np.sqrt(disp)
            disp_dim[i, j] = disp

            # ------------------------------------------------------
            # Compute uncertainty
            # ------------------------------------------------------
            if bootp:
                # Bootstrap via your existing bootstrap() routine
                err_dim[i, j] = bootstrap(
                    yi_bin,
                    eyi_bin,
                    method=method,
                    nsamples=nsamples,
                )
            else:
                # Analytic uncertainty σ / sqrt(2(N-1))
                err_dim[i, j] = (
                    np.nan
                    if len(idx) <= 1
                    else disp
                    / np.sqrt(
                        2 * (len(idx) - 1),
                    )
                )

    # --------------------------------------------------------------
    # Combine over dimensions (quadrature average)
    # --------------------------------------------------------------
    disp = np.sqrt(np.nansum(disp_dim**2, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.nansum(err_dim**2, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def moving_grid1d(
    x,
    y,
    ey,
    dimy,
    bootp=True,
    logx=True,
    bins=10,
    ngrid=10,
    method=None,
    nsamples=100,
):
    """
    Compute dispersion in a *moving grid* of x.

    For each of `ngrid` shifted grids, the function bins x into `bins` bins,
    computes the dispersion of y, and merges all grid results.

    Parameters
    ----------
    x : array_like, shape (N,)
        Reference variable (e.g. radius).
    y : array_like, shape (dimy, N)
        Values whose dispersion is computed.
    ey : array_like, shape (dimy, N)
        Uncertainties on y.
    dimy : int
        Number of y components.
    bootp : bool
        If True, error is computed via bootstrap.
    logx : bool
        If True, bins are logarithmic in x.
    bins : int
        Number of bins per grid.
    ngrid : int
        Number of grid shifts per bin.
    method : str or None
        Dispersion estimator: None (standard) or 'vdv+'.
    nsamples : int
        Number of Monte Carlo samples for bootstrap.

    Returns
    -------
    r : array_like, shape (ngrid * (bins - 1),)
        Bin centers (sorted).
    disp : array_like
        Dispersion values (sorted by r).
    err : array_like
        Errors on the dispersion.
    """

    # --------------------------------------------------------------
    # Bin edges for the *first* grid
    # --------------------------------------------------------------
    xmin, xmax = np.nanmin(x), np.nanmax(x)

    if logx:
        r_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
        r_edges = np.log10(r_edges)  # leaves r_edges in log-space
    else:
        r_edges = np.linspace(xmin, xmax, bins + 1)

    # --------------------------------------------------------------
    # Accumulators for all grids
    # --------------------------------------------------------------
    r_all = []
    disp_all = []
    err_all = []

    # --------------------------------------------------------------
    # Generate ngrid shifted grids
    # --------------------------------------------------------------
    # Shift step between moving grids in LOG or LINEAR space
    full_step = (r_edges[1] - r_edges[0]) / (1 + ngrid)

    for g in range(1, ngrid):
        # skipping g = 0 because base grid usually unused
        # Amount to shift the bin grid
        shift = g * full_step

        # Shifted bin edges
        if logx:
            x_min_shift = 10 ** (r_edges[0] + shift)
            x_max_shift = 10 ** (r_edges[-2] + shift)
        else:
            x_min_shift = r_edges[0] + shift
            x_max_shift = r_edges[-2] + shift

        # ----------------------------------------------------------
        # Select points belonging to this grid
        # ----------------------------------------------------------
        mask = (x >= x_min_shift) & (x < x_max_shift)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            # skip entirely empty grid
            continue

        xi = x[idx]
        yi = y[:, idx]
        eyi = ey[:, idx]

        # ----------------------------------------------------------
        # Apply binning/dispersion to this grid
        # ----------------------------------------------------------
        r_i, disp_i, err_i = bin_disp1d(
            xi,
            yi,
            eyi,
            dimy,
            bins=bins,
            bootp=bootp,
            logx=logx,
            method=method,
            nsamples=nsamples,
        )

        # Accumulate results
        r_all.append(r_i)
        disp_all.append(disp_i)
        err_all.append(err_i)

    # --------------------------------------------------------------
    # Combine all grids and sort by radius
    # --------------------------------------------------------------
    if len(r_all) == 0:
        # No valid grids → return empty output
        return np.array([]), np.array([]), np.array([])

    r_all = np.concatenate(r_all)
    disp_all = np.concatenate(disp_all)
    err_all = np.concatenate(err_all)

    # Sort all results by increasing r
    order = np.argsort(r_all)
    r_all = r_all[order]
    disp_all = disp_all[order]
    err_all = err_all[order]

    return r_all, disp_all, err_all


def equal_size(
    x,
    y,
    ey,
    dimy,
    bootp=True,
    logx=True,
    nbin=10,
    method=None,
    nsamples=100,
):
    """
    Compute the dispersion of y in bins of EQUAL NUMBER OF POINTS.

    Each bin contains exactly `nbin` tracers (except possibly the last one,
    which is discarded to keep all bins equal-sized).

    Parameters
    ----------
    x : array_like, shape (N,)
        Independent variable.
    y : array_like, shape (dimy, N)
        Dependent variable whose dispersion is evaluated.
    ey : array_like, shape (dimy, N)
        Uncertainty on y.
    dimy : int
        Number of y components.
    bootp : bool
        If True → errors estimated via bootstrap.
    logx : bool
        If True → radial coordinate stored in geometric mean of bin edges.
    nbin : int
        Number of tracers per bin.
    method : str or None
        Dispersion estimator: None or 'vdv+'.
    nsamples : int
        Number of Monte Carlo samples for bootstrap.

    Returns
    -------
    r : array_like, shape (nbins,)
        Bin centers.
    disp : array_like, shape (nbins,)
        Dispersion per bin.
    err : array_like, shape (nbins,)
        Error on dispersion.
    """

    # --------------------------------------------------------------
    # Number of full bins we can form
    # --------------------------------------------------------------
    npts = len(x)
    nbins = npts // nbin  # integer number of full bins

    if nbins == 0:
        return np.array([]), np.array([]), np.array([])

    # Output arrays
    r = np.zeros(nbins)
    disp = np.zeros((dimy, nbins))
    err = np.zeros((dimy, nbins))

    # --------------------------------------------------------------
    # Loop over bins
    # --------------------------------------------------------------
    for b in range(nbins):
        i0 = b * nbin
        i1 = i0 + nbin
        xi = x[i0:i1]

        # bin-center in geometric or linear mean
        if logx:
            r[b] = np.sqrt(xi[0] * xi[-1])
        else:
            r[b] = 0.5 * (xi[0] + xi[-1])

        for j in range(dimy):
            yi = y[j, i0:i1]
            eyi = ey[j, i0:i1]

            # --------------------------------------------------
            # Compute dispersion in this bin
            # --------------------------------------------------
            if method is None:
                # Classical σ^2 - mean(err^2)
                sig = np.nanstd(yi)
                noise = np.nanmean(eyi**2)
                disp[j, b] = np.sqrt(max(sig**2 - noise, 0.0))

            elif method == "vdv+":
                # Van de Ven correction
                n = len(yi)
                bn = np.sqrt(2.0 / n) * gamma(n / 2) / gamma((n - 1) / 2)
                sig_mle = np.nanstd(yi)
                esig2_mle = np.nanmean(eyi**2)
                inside = sig_mle**2 - bn * bn * esig2_mle
                disp[j, b] = (1.0 / bn) * np.sqrt(max(inside, 0.0))

            # --------------------------------------------------
            # Compute error in this bin
            # --------------------------------------------------
            if bootp:
                err[j, b] = bootstrap(
                    yi,
                    eyi,
                    method=method,
                    nsamples=nsamples,
                )
            else:
                # Analytical error estimate
                n_eff = np.count_nonzero(~np.isnan(yi))
                err[j, b] = disp[j, b] / np.sqrt(max(2 * (n_eff - 1), 1))

    # --------------------------------------------------------------
    # Combine results across dimensions
    # --------------------------------------------------------------
    disp = np.sqrt(np.nansum(disp**2, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.nansum(err**2, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def perc_bins(
    x,
    y,
    ey,
    dimy,
    bootp=True,
    logx=True,
    nnodes=5,
    method=None,
    nsamples=100,
):
    """
    Compute dispersion in *percentile-based bins*, where each bin spans a
    geometric progression in the percentiles of x.

    The percentile edges are:
        q_i  = 0.01 * z**i
        q_f  = 0.01 * z**(i+1)
    where z = 100^(1/nnodes).

    Parameters
    ----------
    x : array_like, shape (N,)
        Reference variable.
    y : array_like, shape (dimy, N)
        Dependent variable whose dispersion is computed.
    ey : array_like, shape (dimy, N)
        Uncertainty on y.
    dimy : int
        Number of y components.
    bootp : bool
        If True → bootstrap dispersion errors.
    logx : bool
        If True → geometric bin center.
    nnodes : int
        Number of percentile bins.
    method : str or None
        Dispersion estimator ('vdv+' or None).
    nsamples : int
        Bootstrap Monte Carlo samples.

    Returns
    -------
    r : array_like, shape (nnodes,)
        Bin centers.
    disp : array_like, shape (nnodes,)
        Dispersion per bin.
    err : array_like, shape (nnodes,)
        Error on dispersion.

    """

    # ------------------------------------------------------------------
    # Prepare outputs
    # ------------------------------------------------------------------
    r = np.zeros(nnodes)
    disp = np.zeros((dimy, nnodes))
    err = np.zeros((dimy, nnodes))

    # geometric progression factor for percentile spacing
    z = 100.0 ** (1.0 / nnodes)

    # ------------------------------------------------------------------
    # Loop over bins
    # ------------------------------------------------------------------
    for i in range(nnodes):
        # percentile boundaries scaled into [0,1]
        qi = max(0.0, 0.01 * z**i)
        qf = min(1.0, 0.01 * z ** (i + 1))

        # Convert percentiles → actual bin edges in x
        ri = position.quantile(x, qi)
        rf = position.quantile(x, qf)

        # Select indices in this percentile interval
        idx = np.where((x >= ri) & (x <= rf))[0]

        # If bin is empty → skip safely
        if len(idx) == 0:
            r[i] = np.nan
            disp[:, i] = np.nan
            err[:, i] = np.nan
            continue

        # ------------------------------------------------------------------
        # Radial bin center (geometric or linear)
        # ------------------------------------------------------------------
        xmin = np.nanmin(x[idx])
        xmax = np.nanmax(x[idx])

        if logx:
            r[i] = np.sqrt(xmin * xmax)
        else:
            r[i] = 0.5 * (xmin + xmax)

        # ------------------------------------------------------------------
        # Compute dispersion for each component
        # ------------------------------------------------------------------
        for j in range(dimy):
            yi = y[j, idx]
            eyi = ey[j, idx]

            # Classical estimator
            if method is None:
                sig = np.nanstd(yi)
                noise = np.nanmean(eyi**2)
                disp_j = sig**2 - noise
                disp[j, i] = np.sqrt(max(disp_j, 0.0))

            # Van de Ven+ estimator
            elif method == "vdv+":
                n = len(yi)
                bn = np.sqrt(2.0 / n) * gamma(n / 2) / gamma((n - 1) / 2)
                sig_mle = np.nanstd(yi)
                esig2 = np.nanmean(eyi**2)
                inside = sig_mle**2 - bn * bn * esig2
                disp[j, i] = (1.0 / bn) * np.sqrt(max(inside, 0.0))

            # ------------------------------------------------------------------
            # Compute errors
            # ------------------------------------------------------------------
            if bootp:
                err[j, i] = bootstrap(
                    yi,
                    eyi,
                    method=method,
                    nsamples=nsamples,
                )
            else:
                n_eff = np.count_nonzero(~np.isnan(yi))
                err[j, i] = disp[j, i] / np.sqrt(max(2 * (n_eff - 1), 1))

    # ----------------------------------------------------------------------
    # Combine components: sqrt(sum(disp_i^2)) / sqrt(dimy)
    # ----------------------------------------------------------------------
    disp = np.sqrt(np.nansum(disp**2, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.nansum(err**2, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def closest_points(
    x,
    y,
    ey,
    dimy,
    bootp=True,
    logx=True,  # (not actually used for bin center here)
    nbin=5,
    method=None,
    nsamples=100,
):
    """
    Compute the dispersion at each x[i] using *nbin nearest neighbors*
    in x-space, for each component of y.

    Parameters
    ----------
    x : array_like, shape (N,)
        Reference coordinate.
    y : array_like, shape (dimy, N)
        Values whose dispersion is measured.
    ey : array_like, shape (dimy, N)
        Uncertainties on y.
    dimy : int
        Number of components in y.
    bootp : bool
        If True, compute errors using bootstrap.
    logx : bool
        Included for API consistency (no effect on centers).
    nbin : int
        Number of nearest neighbors per evaluation point.
    method : str or None
        Dispersion estimator ('vdv+' or None).
    nsamples : int
        Bootstrap Monte Carlo samples.

    Returns
    -------
    r : array_like, shape (N,)
        Centers (here simply x itself).
    disp : array_like, shape (N,)
        Dispersion at each x[i].
    err : array_like, shape (N,)
        Uncertainty on dispersion.
    """

    N = len(x)

    # Output arrays
    r = np.asarray(x).copy()  # bin centers = original x positions
    disp = np.zeros((dimy, N))
    err = np.zeros((dimy, N))

    # ------------------------------------------------------------------
    # Loop over each evaluation point
    # ------------------------------------------------------------------
    for i in range(N):
        # Nearest nbin neighbors in |x[i] - x|
        dist = np.abs(x - x[i])
        idx = np.argpartition(dist, nbin)[:nbin]

        # Safety: remove NaN indices (in case x has NaN)
        idx = idx[np.isfinite(x[idx])]
        if len(idx) == 0:
            disp[:, i] = np.nan
            err[:, i] = np.nan
            continue

        # ------------------------------------------------------------------
        # Compute dispersion for every component
        # ------------------------------------------------------------------
        for j in range(dimy):
            yi = y[j, idx]
            eyi = ey[j, idx]

            # Classical estimator
            if method is None:
                sig = np.nanstd(yi)
                noise = np.nanmean(eyi**2)
                inside = sig**2 - noise
                disp[j, i] = np.sqrt(max(inside, 0.0))

            # van de Ven et al. 2006 estimator
            elif method == "vdv+":
                n = len(yi)
                bn = np.sqrt(2.0 / n) * gamma(n / 2) / gamma((n - 1) / 2)
                sig_mle = np.nanstd(yi)
                esig2 = np.nanmean(eyi**2)
                inside = sig_mle**2 - bn * bn * esig2
                disp[j, i] = (1.0 / bn) * np.sqrt(max(inside, 0.0))

            # ------------------------------------------------------------------
            # Bootstrap uncertainty
            # ------------------------------------------------------------------
            if bootp:
                err[j, i] = bootstrap(
                    yi,
                    eyi,
                    method=method,
                    nsamples=nsamples,
                )
            else:
                n_eff = np.count_nonzero(~np.isnan(yi))
                err[j, i] = disp[j, i] / np.sqrt(max(2 * (n_eff - 1), 1))

    # ----------------------------------------------------------------------
    # Combine dimy components → sqrt(sum(disp^2)) / sqrt(dimy)
    # ----------------------------------------------------------------------
    disp = np.sqrt(np.nansum(disp**2, axis=0)) / np.sqrt(dimy)
    err = np.sqrt(np.nansum(err**2, axis=0)) / np.sqrt(dimy)

    return r, disp, err


def disp1d(
    x,
    y,
    ey,
    dimy,
    ww=None,
    bins="percentile",
    smooth=True,
    bootp=True,
    logx=True,
    nbin=None,
    polorder=None,
    return_fits=False,
    method=None,
    nsamples=100,
):
    """
    Compute the 1D dispersion of y as a function of x using multiple
    possible binning methods.

    Parameters
    ----------
    x : array_like, shape (N,)
        Reference array.
    y : array_like, shape (dimy, N)
        Values whose dispersion is computed.
    ey : array_like, shape (dimy, N)
        Uncertainties on y.
    dimy : int
        Number of components in y.
    ww : array_like or None
        Optional weights.
    bins : int or str
        Binning strategy:
        - integer → fixed number of bins
        - "moving" → moving grid
        - "fix-size" → fixed number of points per bin
        - "percentile" → percentile–based bins
        - "closest" → n nearest neighbors
    smooth : bool
        If True, smooth the final result with a polynomial fit.
    bootp : bool
        If True, compute errors via bootstrap.
    logx : bool
        If True, log-scale grid in x.
    nbin : int or None
        Auxiliary parameter for some binning modes.
    polorder : int or None
        Polynomial order for smoothing.
    return_fits : bool
        Return polynomial coefficients as fourth output.
    method : str or None
        Dispersion estimator ("vdv+" or None).
    nsamples : int
        Monte Carlo samples for bootstrap or VDMA.

    Returns
    -------
    r : array_like
        Bin centers or fitting x-grid.
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on the dispersion.
    """

    # ------------------------------------------------------------------
    # 1. Choose binning strategy
    # ------------------------------------------------------------------

    # --- Case A: integer number of bins --------------------------------
    if isinstance(bins, int):
        if method == "vdma":
            r, disp, err = bin_montecarlo_1d(
                x,
                y,
                ey,
                dimy,
                bins=bins,
                logx=logx,
                nsamples=nsamples,
                ww=ww,
            )
        else:
            r, disp, err = bin_disp1d(
                x,
                y,
                ey,
                dimy,
                bins=bins,
                bootp=bootp,
                logx=logx,
                method=method,
                nsamples=nsamples,
            )

    # --- Case B: moving bins -------------------------------------------
    elif bins == "moving":
        # Default bin number
        bins_m = int(0.5 * position.good_bin(x)) if nbin is None else nbin
        ngrid = 2
        r, disp, err = moving_grid1d(
            x,
            y,
            ey,
            dimy,
            bootp=bootp,
            logx=logx,
            bins=bins_m,
            ngrid=ngrid,
            method=method,
            nsamples=nsamples,
        )

    # --- Case C: fixed number of points per bin -------------------------
    elif bins == "fix-size":
        if nbin is None:
            nb = int(position.good_bin(x))
            nbin = int(len(x) / nb)
        r, disp, err = equal_size(
            x,
            y,
            ey,
            dimy,
            bootp=bootp,
            logx=logx,
            nbin=nbin,
            method=method,
            nsamples=nsamples,
        )

    # --- Case D: percentile bins (default) ------------------------------
    elif bins == "percentile":
        nnodes = int(2 * position.good_bin(x)) if nbin is None else nbin
        r, disp, err = perc_bins(
            x,
            y,
            ey,
            dimy,
            bootp=bootp,
            logx=logx,
            nnodes=nnodes,
            method=method,
            nsamples=nsamples,
        )

    # --- Case E: closest n neighbors ------------------------------------
    elif bins == "closest":
        if nbin is None:
            nb = int(position.good_bin(x))
            nbin = int(len(x) / nb)
        r, disp, err = closest_points(
            x,
            y,
            ey,
            dimy,
            bootp=bootp,
            logx=logx,
            nbin=nbin,
            method=method,
            nsamples=nsamples,
        )

    else:
        raise ValueError(f"Unknown binning option: {bins}")

    # ------------------------------------------------------------------
    # 2. Mask invalid values
    # ------------------------------------------------------------------
    valid = np.isfinite(disp) & np.isfinite(err)
    if not np.any(valid):
        return r * np.nan, disp * np.nan, err * np.nan

    r_valid = r[valid]
    disp_valid = disp[valid]
    err_valid = err[valid]

    # Determine valid interpolation range for smoothing
    rmin, rmax = np.nanmin(r), np.nanmax(r)
    idxrange = np.where((x > rmin) & (x < rmax))[0]

    # ------------------------------------------------------------------
    # 3. Optional polynomial smoothing
    # ------------------------------------------------------------------
    if smooth:
        # Choose polynomial order
        if polorder is None:
            pold = int(0.2 * position.good_bin(disp_valid))
        else:
            pold = polorder

        # Safety: polynomial order cannot exceed number of points - 1
        pold = max(0, min(pold, len(r_valid) - 1))

        if logx:
            xx = np.log10(r_valid)
            t = np.log10(x[idxrange])
        else:
            xx = r_valid
            t = x[idxrange]

        # Weighted polynomial fit
        poly, cov = np.polyfit(xx, disp_valid, pold, w=1 / err_valid, cov=True)

        # Build Vandermonde matrix
        TT = np.vstack([t ** (pold - i) for i in range(pold + 1)]).T

        # Smooth values
        yi = TT @ poly
        Cyi = TT @ cov @ TT.T
        sig_yi = np.sqrt(np.diag(Cyi))

        # Replace output arrays
        disp_out = yi
        err_out = sig_yi
        r_out = x[idxrange]

        if return_fits:
            return r_out, disp_out, err_out, poly

        return r_out, disp_out, err_out

    # ------------------------------------------------------------------
    # 4. No smoothing → return raw binned results
    # ------------------------------------------------------------------
    return r_valid, disp_valid, err_valid


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
    method=None,
):
    """
    Compute a 2D dispersion map of y across the sky plane defined by x.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Sky-plane coordinates (RA, Dec).
    y : array_like, shape (dimy, N)
        Values whose dispersion is computed.
    ey : array_like, shape (dimy, N)
        Uncertainties on y.
    dimy : int
        Number of components in y.
    smooth : bool
        If True, smooth the dispersion map after binning.
    bootp : bool
        If True, compute errors with bootstrap.
    a0, d0 : float
        Reference center (RA, Dec). If None, found iteratively.
    robust_sig : bool
        Use robust MAD-based dispersion if True.
    nbin : int or None
        Number of bins for hexbin.
    nmov : int or None
        Number of moving-grid shifts.
    method : str or None
        Dispersion estimator ("vdv+" or None).

    Returns
    -------
    r : array_like
        Array of sampled coordinates used for interpolation.
    disp : 2D array
        Interpolated dispersion map.
    err : 2D array
        Interpolated uncertainty map.
    """

    # ------------------------------------------------------------
    # 1. Determine center if not provided
    # ------------------------------------------------------------
    if (a0 is None) or (d0 is None):
        center, _ = position.find_center(x[0], x[1], method="iterative")
        a0, d0 = center

    # ------------------------------------------------------------
    # 2. Compute maximum radius from convex hull of tracer positions
    # ------------------------------------------------------------
    all_points = np.column_stack([x[0], x[1]])
    hull = ConvexHull(all_points)
    idx_hull = hull.vertices

    r_hull = angle.sky_distance_deg(x[0, idx_hull], x[1, idx_hull], a0, d0)
    rlim = np.nanmin(r_hull)

    # Circle containing the map region
    alim0, dlim0 = angle.get_circle_sph_trig(rlim, a0, d0)

    # ------------------------------------------------------------
    # 3. Binning/grid parameter setup
    # ------------------------------------------------------------
    if nbin is None:
        nbin = int(0.25 * (position.good_bin(x[0]) + position.good_bin(x[1])))

    if nmov is None:
        nmov = int(0.7 * nbin)

    rm = 0.8 * rlim
    shift = (2 * rm) / (nbin * nmov)

    # Lists of window boundary limits
    amin, amax = [], []
    dmin, dmax = [], []

    # Shifted starting point
    a_start = a0 - 0.5 * nmov * shift
    d_start = d0 - 0.5 * nmov * shift

    # ------------------------------------------------------------
    # 4. Construct moving hexbin grid windows
    # ------------------------------------------------------------
    for i in range(nmov):
        for j in range(nmov):
            a_c = a_start + i * shift
            d_c = d_start + j * shift

            a_win, d_win = angle.get_circle_sph_trig(rm, a_c, d_c)

            amin.append(np.min(a_win))
            amax.append(np.max(a_win))
            dmin.append(np.min(d_win))
            dmax.append(np.max(d_win))

    # Partial reduction functions for hexbin
    raux_disp = partial(
        aux_disp,
        y=y,
        ey=ey,
        dimy=dimy,
        robust_sig=robust_sig,
        method=method,
    )

    raux_err = partial(
        aux_err,
        y=y,
        ey=ey,
        dimy=dimy,
        robust_sig=robust_sig,
        bootp=bootp,
        method=method,
    )

    # Storage arrays for all hexbin output
    points_disp = []
    values_disp = []
    points_err = []
    values_err = []

    N = len(x[0])
    idx_all = np.arange(N)

    # ------------------------------------------------------------
    # 5. Loop through all moving-grid patches and perform hexbin
    # ------------------------------------------------------------
    for amin_i, amax_i, dmin_i, dmax_i in zip(amin, amax, dmin, dmax):
        # ----- DISPERSION -----
        hb_disp = plt.hexbin(
            x[0],
            x[1],
            C=idx_all,
            gridsize=nbin,
            reduce_C_function=raux_disp,
            extent=(amin_i, amax_i, dmin_i, dmax_i),
        )

        zvals = hb_disp.get_array()
        verts = hb_disp.get_offsets()

        # Keep only bins with values
        m = zvals != 0
        if np.any(m):
            points_disp.append(verts[m])
            values_disp.append(zvals[m])

        # ----- ERROR -----
        hb_err = plt.hexbin(
            x[0],
            x[1],
            C=idx_all,
            gridsize=nbin,
            reduce_C_function=raux_err,
            extent=(amin_i, amax_i, dmin_i, dmax_i),
        )

        zvals = hb_err.get_array()
        verts = hb_err.get_offsets()

        m = zvals != 0
        if np.any(m):
            points_err.append(verts[m])
            values_err.append(zvals[m])

        plt.close()  # prevent buildup

    # ------------------------------------------------------------
    # 6. Consolidate collections
    # ------------------------------------------------------------
    points_disp = np.vstack(points_disp)
    values_disp = np.concatenate(values_disp)

    points_err = np.vstack(points_err)
    values_err = np.concatenate(values_err)

    # ------------------------------------------------------------
    # 7. Interpolate onto a regular grid
    # ------------------------------------------------------------
    grid_x, grid_y = np.mgrid[
        np.min(alim0) : np.max(alim0) : 300j,
        np.min(dlim0) : np.max(dlim0) : 300j,
    ]

    grid_disp = griddata(
        points_disp,
        values_disp,
        (grid_x, grid_y),
        method="cubic",
    )
    grid_err = griddata(
        points_err,
        values_err,
        (grid_x, grid_y),
        method="cubic",
    )

    # Replace invalid (NaN) interpolation with 0
    grid_disp = np.nan_to_num(grid_disp, nan=0.0)
    grid_err = np.nan_to_num(grid_err, nan=0.0)

    # ------------------------------------------------------------
    # 8. Package return values
    # ------------------------------------------------------------
    r = np.array([points_disp, points_err])
    return r, grid_disp, grid_err


def dispersion(
    x,
    y,
    ey=None,
    ww=None,
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
    method="vdma",
    nsamples=100,
):
    """
    Master routine that computes the dispersion of y with respect to x,
    automatically choosing between 1D and 2D estimators.

    Parameters
    ----------
    x : array_like
        Reference coordinate(s). Shape (N,) for 1D or (2, N) for 2D.
    y : array_like
        Quantity whose dispersion is estimated. Shape (dimy, N) allowed.
    ey : array_like, optional
        Uncertainties on y. If None, assumed zero.
    ww : array_like, optional
        Weights. If None, defaults to unity weights.
    bins : int or str, optional
        Binning strategy for 1D dispersion:
            - int → fixed number of radial bins
            - "moving", "fix-size", "percentile", "closest"
        For 2D dispersion, ignored (always uses hexbin).
    smooth : bool
        Smooth the 1D or 2D dispersion results.
    bootp : bool
        If True, errors are obtained by bootstrap.
    logx : bool
        Use log-spaced bins for 1D dispersion.
    nbin : int, optional
        Auxiliary bin parameter for 1D or 2D binning.
    polorder : int, optional
        Polynomial order used for smoothing fits.
    return_fits : bool
        If True, return polynomial coefficients (1D only).
    robust_sig : bool
        Use robust dispersion estimator in 2D maps.
    a0, d0 : float, optional
        Reference sky position for 2D dispersion. Used to determine footprint.
    nmov : int, optional
        Number of moving windows in 2D hexbin smoothing.
    method : str
        Dispersion estimator ("vdma", "vdv+", or None).
    nsamples : int
        Monte Carlo samples for vdma method.

    Returns
    -------
    r : array_like
        Effective radial coordinates (1D) or sample points (2D).
    disp : array_like
        Dispersion.
    err : array_like
        Uncertainty on dispersion.
    """

    # ------------------------------------------------------------
    # 1. Determine dimensionality: 1D or 2D coordinate system
    # ------------------------------------------------------------

    x = np.asarray(x)

    if x.ndim == 1:
        dimx = 1
        if bins is None:
            bins = "percentile"  # reasonable default for 1D
    else:
        # Expect shape (2, N) for 2D coordinates
        dimx = 2
        if bins is None:
            bins = "moving"  # 2D uses moving grid inside disp2d()

    # ------------------------------------------------------------
    # 2. Standardize y, ey, ww shapes → always shape (dimy, N)
    # ------------------------------------------------------------

    y = np.asarray(y)

    if y.ndim == 1:
        # Single-component field
        dimy = 1
        y = np.asarray([y])

        if ey is None:
            ey = np.zeros_like(y)
        else:
            ey = np.asarray([ey])

        if ww is None:
            ww = np.ones_like(y)
        else:
            ww = np.asarray([ww])

    else:
        # Multi-component field
        dimy = y.shape[0]

        if ey is None:
            ey = np.zeros_like(y)
        else:
            ey = np.asarray(ey)

        if ww is None:
            ww = np.ones_like(y)
        else:
            ww = np.asarray(ww)

    # ------------------------------------------------------------
    # 3. Compute dispersion: choose 1D or 2D engine
    # ------------------------------------------------------------

    if dimx == 1:
        r, disp, err = disp1d(
            x,
            y,
            ey,
            dimy,
            ww=ww,
            bins=bins,
            smooth=smooth,
            bootp=bootp,
            logx=logx,
            nbin=nbin,
            polorder=polorder,
            return_fits=return_fits,
            method=method,
            nsamples=nsamples,
        )

    else:
        # 2D dispersion map via moving-window hexbin
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
            method=method,
        )

    # ------------------------------------------------------------
    # 4. Post-processing: invalid errors → NaN
    # ------------------------------------------------------------

    err = np.asarray(err)
    err[err <= 0] = np.nan

    return r, disp, err


def disp_plummer(r, d0, a=2.0, b=0.25):
    """
    Compute the normalized velocity-dispersion profile for an anisotropic
    Plummer model. This uses the generalized form of the Dejonghe (1987)
    parametrization:

        σ(r) = d0 / (1 + r^a)^b

    The normalization matches the usual convention:
        dd = σ_real / sqrt(G * M_tot / a)

    Parameters
    ----------
    r : array_like
        Radius values normalized by the Plummer scale radius.
    d0 : float
        Central velocity dispersion (σ at r=0).
    a : float, optional
        Exponent controlling the radial rise inside the denominator.
        Default is 2 (standard Plummer-like).
    b : float, optional
        Exponent controlling the outer slope.
        Default is 0.25.

    Returns
    -------
    dd : array_like
        Velocity-dispersion profile evaluated at each r.

    Notes
    -----
    - This function is vectorized: r may be a scalar or array.
    - No assumptions are made about units except that r is already
      normalized by the Plummer scale radius.
    """

    r = np.asarray(r, dtype=float)  # ensure array and numeric

    # Plummer-like dispersion law
    dd = d0 / (1.0 + r**a) ** b

    return dd


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"General functions"
# ------------------------------------------------------------------------------


def btsp(x, ex):
    """
    Draw a bootstrap resample of (x, ex) by sampling with replacement.

    Parameters
    ----------
    x : array_like
        Data values to be resampled.
    ex : array_like
        Associated uncertainties, same shape as `x`.

    Returns
    -------
    x_new : array_like
        Bootstrap-resampled values of x.
    ex_new : array_like
        Bootstrap-resampled uncertainties.

    Notes
    -----
    - Bootstrap sampling is done *with replacement* using uniform random
      indices in [0, len(x)-1].
    - x and ex must have the same length.
    """
    x = np.asarray(x)
    ex = np.asarray(ex)

    if len(x) != len(ex):
        raise ValueError(
            "x and ex must have the same length for bootstrapping.",
        )

    # Sample indices uniformly with replacement
    idx = np.random.randint(0, len(x), size=len(x))

    # Apply the index selection
    x_new = x[idx]
    ex_new = ex[idx]

    return x_new, ex_new


def bootstrap(array, earray, method=None, nsamples=100):
    """
    Estimate the uncertainty of a dispersion measurement via bootstrap
    resampling with replacement.

    Parameters
    ----------
    array : array_like
        Data values for which the dispersion is computed.
    earray : array_like
        Measurement uncertainties associated with `array`.
    method : str, optional
        Method used for the dispersion estimate:
            - None     : classical sqrt(std^2 - mean(err^2))
            - "vdv+"   : van der Marel & Franx bias-corrected estimator
            - "robust" : MAD-based estimator
    nsamples : int, optional
        Number of bootstrap realizations. Default is 100.

    Returns
    -------
    unc : float
        Bootstrap uncertainty (standard deviation of bootstrapped dispersions).
    """
    array = np.asarray(array)
    earray = np.asarray(earray)

    if len(array) != len(earray):
        raise ValueError("array and earray must have the same length.")

    # Stores dispersion values from bootstrap realizations
    sig = np.zeros(nsamples)

    for i in range(nsamples):
        xb, exb = btsp(array, earray)

        if method is None:
            # Classical noise-corrected dispersion
            sig[i] = np.sqrt(np.nanstd(xb) ** 2 - np.nanmean(exb**2))

        elif method == "vdv+":
            # van der Marel (1993) style bias correction
            n = len(xb)
            bn = np.sqrt(2 / n) * gamma(n / 2) / gamma((n - 1) / 2)
            sig_mle = np.nanstd(xb)
            esig2 = np.nanmean(exb**2)
            sig[i] = (1 / bn) * np.sqrt(sig_mle**2 - (bn**2) * esig2)

        elif method == "robust":
            # Robust MAD-based dispersion
            mad = np.median(np.abs(xb - np.median(xb))) / 0.6745
            sig[i] = np.sqrt(mad**2 - np.nanmean(exb**2))

        else:
            raise ValueError(f"Unknown method: {method}")

    # Bootstrap uncertainty is the std deviation of bootstrapped dispersions
    unc = np.nanstd(sig)

    return unc


def lgaussian(x, mu, sig):
    """
    Compute the natural logarithm of a Gaussian (normal) probability density.

    The Gaussian PDF is:
        G(x | μ, σ) = (1 / (σ * sqrt(2π))) * exp[ -0.5 * ((x - μ) / σ)^2 ]

    This function returns ln(G).

    Parameters
    ----------
    x : array_like or float
        Value(s) at which to evaluate the log-Gaussian.
    mu : float
        Mean of the Gaussian distribution.
    sig : float
        Standard deviation (must be > 0).

    Returns
    -------
    lg : array_like or float
        Natural logarithm of the Gaussian PDF.
        If sig <= 0, returns -np.inf.
    """

    x = np.asarray(x)

    # Standardized variable
    arg = (x - mu) / sig

    # Log Gaussian PDF
    lg = -0.5 * arg**2 - 0.5 * np.log(2 * np.pi) - np.log(sig)

    return lg


def likelihood_1gauss1d(params, Ux, ex, ww=None):
    """
    Computes the negative log-likelihood for a single Gaussian model in 1D,
    following van der Marel & Anderson (2010).

    Parameters
    ----------
    params : array_like
        Model parameters. For this function:
            params[0] = intrinsic dispersion sigma.
    Ux : array_like
        Observed data values.
    ex : array_like
        Measurement uncertainties for each observation.
    ww : array_like, float, optional
        Weights to apply to each data point. If None, all weights = 1.

    Returns
    -------
    L : float
        Negative logarithm of the likelihood function.
    """

    # --------------------------------------------------------------
    # Ensure weight vector exists
    # --------------------------------------------------------------
    if ww is None:
        ww = np.ones_like(Ux)

    # --------------------------------------------------------------
    # Extract intrinsic dispersion from parameters
    # --------------------------------------------------------------
    sig_intrinsic = params[0]

    # Total dispersion includes measurement errors (per star)
    sig_tot = np.sqrt(sig_intrinsic * sig_intrinsic + ex * ex)

    # --------------------------------------------------------------
    # Analytical MLE estimate of the mean given sigma
    # --------------------------------------------------------------
    s1 = np.sum(ww / (sig_tot * sig_tot))
    s2 = np.sum(Ux * ww / (sig_tot * sig_tot))

    mu = s2 / s1

    # --------------------------------------------------------------
    # Log-PDF for each data point under the model
    # --------------------------------------------------------------
    f_i = lgaussian(Ux, mu, sig_tot)

    # Calculates the likelihood, taking out NaN's
    f_i = f_i[np.logical_not(np.isnan(f_i))]
    L = -np.sum(f_i * ww)

    return L


def mle_mean(x, ex, sig, ww=None):
    """
    Maximum-likelihood estimate (MLE) of the mean of a Gaussian distribution,
    accounting for individual measurement uncertainties ex.
    Follows van der Marel & Anderson (2010).

    Parameters
    ----------
    x : array_like, float
        Observed values.
    ex : array_like, float
        Measurement uncertainties for each observed value.
    sig : float
        Intrinsic dispersion assumed for the population.
    ww : array_like, float, optional
        Weights to apply to each data point. If None, all weights = 1.

    Returns
    -------
    mu : float
        MLE estimate of the mean.
    emu : float
        Uncertainty of the MLE mean estimate.
    """

    # --------------------------------------------------------------
    # Ensure weight vector exists
    # --------------------------------------------------------------
    if ww is None:
        ww = np.ones_like(x)

    # --------------------------------------------------------------
    # Total variance = intrinsic variance + measurement variance
    # --------------------------------------------------------------
    sig_tot = np.sqrt(sig * sig + ex * ex)

    # --------------------------------------------------------------
    # Weighted sums for analytical MLE solution
    # --------------------------------------------------------------
    s1 = np.sum(ww / (sig_tot * sig_tot))
    s2 = np.sum(x * ww / (sig_tot * sig_tot))

    # MLE mean and its uncertainty
    mu = s2 / s1
    emu = 1.0 / np.sqrt(s1)

    return mu, emu


def mle_disp(x, ex, ww=None):
    """
    Maximum-likelihood estimate (MLE) of the intrinsic dispersion
    of a Gaussian distribution, following van der Marel & Anderson (2010).

    Parameters
    ----------
    x : array_like, float
        Observed values.
    ex : array_like, float
        Measurement uncertainties for each observed value.
    ww : array_like, float, optional
        Weights to apply to each data point. If None, all weights = 1.

    Returns
    -------
    results : float
        MLE estimate of the intrinsic dispersion.
    """

    # --------------------------------------------------------------
    # Ensure weight vector exists
    # --------------------------------------------------------------
    if ww is None:
        ww = np.ones_like(x)

    # --------------------------------------------------------------
    # Initial guess: weighted median and weighted standard deviation
    # --------------------------------------------------------------
    ini = np.asarray([weighted_median(x, ww), weighted_std(x, ww)])

    # Bounds for intrinsic dispersion during optimization
    bounds = [(0.5 * ini[1], 2.0 * ini[1])]

    # Restrict search to data within ±3σ of the median
    ranges = [ini[0] - 3 * ini[1], ini[0] + 3 * ini[1]]
    idx_x = np.intersect1d(np.where(x < ranges[1]), np.where(x > ranges[0]))

    # Trim arrays to the restricted region
    x = x[idx_x]
    ex = ex[idx_x]
    ww = ww[idx_x]

    # --------------------------------------------------------------
    # Run differential evolution on the likelihood function
    # --------------------------------------------------------------
    mle_model = differential_evolution(
        lambda c: likelihood_1gauss1d(c, x, ex, ww=ww),
        bounds,
    )

    # Extract intrinsic dispersion
    results = mle_model.x[0]

    return results


def monte_carlo_bias(x, ex, sig_mle, nsamples, ww=None):
    """
    Performs a Monte Carlo bias correction of the velocity dispersion.
    Implements the procedure described in van der Marel & Anderson (2010).

    Parameters
    ----------
    x : array_like, float
        Observed values.
    ex : array_like, float
        Measurement uncertainties for each observed value.
    sig_mle : float
        Maximum-likelihood estimate of the intrinsic dispersion.
    nsamples : int
        Number of Monte Carlo realizations to draw.
    ww : array_like, float, optional
        Weights to be applied to data.  If None, weights = 1 for all points.

    Returns
    -------
    corrected_dispersion : float
        Bias-corrected velocity dispersion.
    error_dispersion : float
        Uncertainty of the corrected dispersion based on MC samples.
    """

    # ------------------------------------------------------------------
    # Setup: ensure weight vector exists
    # ------------------------------------------------------------------
    if ww is None:
        ww = np.ones_like(x)

    # ------------------------------------------------------------------
    # Total (intrinsic + measurement) broadening for each data point
    # sig_tot[i] = sqrt(sig_mle^2 + ex[i]^2)
    # ------------------------------------------------------------------
    sig = np.sqrt(sig_mle * sig_mle + ex * ex)

    # ------------------------------------------------------------------
    # Weighted mean under the model N(mu, sig_tot)
    # ------------------------------------------------------------------
    s1 = np.sum(ww / (sig * sig))
    s2 = np.sum(x * ww / (sig * sig))
    mu = s2 / s1

    # ------------------------------------------------------------------
    # Monte-Carlo loop: generate nsamples realizations and compute MLE
    # dispersion for each synthetic dataset
    # ------------------------------------------------------------------
    sample_sig = np.zeros(nsamples)

    for k in range(nsamples):
        # Draw a Monte-Carlo sample from the model distribution.
        # Each point is drawn as N(mu, sig_tot[i]).
        sample = np.random.normal(mu, sig)

        # Compute MLE dispersion for this synthetic dataset
        sample_sig[k] = mle_disp(sample, ex, ww=ww)

    # ------------------------------------------------------------------
    # Bias correction:
    # ratio = (mean of MC dispersions) / (input sig_mle)
    # corrected = sig_mle / ratio
    # ------------------------------------------------------------------
    ratio = np.nanmean(sample_sig) / sig_mle
    corrected_dispersion = sig_mle / ratio

    # Error propagation for the corrected dispersion
    error_dispersion = np.nanstd(sample_sig) / ratio**2

    return corrected_dispersion, error_dispersion


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
    Anisotropy profile from (Cuddeford 1991; Osipkov 1979; Merritt 1985)
    inversion.

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

    term1 = dt * dt * ert * ert + dp * dp * erp * erp
    ebeta = (
        0.5
        * (dt * dt + dp * dp)
        / (2 * dr * dr)
        * np.sqrt(
            2 * err * err / (dr * dr) + 2 * term1 / (dt * dt + dp * dp) ** 2,
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

        r, phi, theta, vr, vphi, vtheta = angle.cart_to_sph(
            x,
            y,
            z,
            vx,
            vy,
            vz,
        )
        r = np.sort(r)

        nonan1 = np.logical_not(np.isnan(beta))
        nonan2 = np.logical_not(np.isnan(ebeta))
        nonan = nonan1 * nonan2

        rr = rr[nonan]
        beta = beta[nonan]
        ebeta = ebeta[nonan]

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


def likelihood_om(params, w):
    """
    Likelihood function of the velocity anisotropy for the model from
    Osipkov 1979; Merritt 1985.

    Parameters
    ----------
    params: array_lie
        Parameters to be fitted, the logarithm of the anisotropy radius and
        the anisotropy value at infinity, and the radial velocity dispersion
        at r = 0.
    w : array_like
        Array containing the ensemble of data, i.e., r, vr, vphi and vtheta.

    Returns
    -------
    L : float
       Likelihood.


    """

    ra = 10 ** params[0]
    b0 = params[1] / (1 + 0.5 * params[1])
    bi = params[2] / (1 + 0.5 * params[2])
    d0 = 10 ** params[3]
    rp = 10 ** params[4]

    r, vr, vt1, vt2 = w

    beta = bgOM(r, ra, b0, bi)

    dr = disp_plummer(r / rp, d0, params[5], params[6])

    lGr = lgaussian(vr, np.nanmean(vr), dr)
    lGt1 = lgaussian(vt1, np.nanmean(vt1), dr * np.sqrt(1 - beta))
    lGt2 = lgaussian(vt2, np.nanmean(vt2), dr * np.sqrt(1 - beta))

    lf = lGr + lGt1 + lGt2

    idx_valid = np.logical_not(np.isnan(lf))

    L = -np.sum(lf[idx_valid])

    return L


def fit_beta_bayes(r, vr, vphi, vtheta, model="OM"):
    """
    Performs a Bayesian fit of the velocity anisotropy.

    Parameters
    ----------
    r : array_like
        r-axis.
    vr : array_like
        r-axis velocity.
    vphi : array_like
        phi-angle velocity.
    vtheta : array_like
        theta-angle velocity.
    model : string, optional
        Velocity anisotropy model. The default is 'OM'.

    Raises
    ------
    ValueError
        Velocity anisotropy model is not one of the following:
            - 'OM'
        No data is provided.

    Returns
    -------
    results : array
        Best fit parameters of the velocity anisotropy model.
    var : array
        Uncertainty of the fits.


    """
    if model not in ["OM"]:
        raise ValueError("Does not recognize  velocity anisotropy model.")

    size_data = len(r)

    wi = np.vstack([r, vr, vphi, vtheta])

    for i in range(np.shape(wi)[0]):
        if len(wi[i]) != size_data:
            raise ValueError("The arrays do not have the same length.")

    # Defines initial guesses for the parameters (i.e., r_a and beta_inf).
    lra = np.log10(np.nanmedian(r) / 1.305)
    ld = np.log10(np.nanstd(vr))

    if model == "OM":
        bounds = [
            (
                max(lra - 2, np.nanquantile(np.log10(r), 0.05)),
                min(lra + 2, np.nanquantile(np.log10(r), 0.95)),
            ),
            (-1.99, 1.999),
            (-1.99, 1.999),
            (ld - 0.3, ld + 0.3),
            (
                max(lra - 2, np.nanquantile(np.log10(r), 0.05)),
                min(lra + 2, np.nanquantile(np.log10(r), 0.95)),
            ),
            (1, 8),
            (0.05, 2),
        ]
        mle_model = differential_evolution(
            lambda c: likelihood_om(c, wi),
            bounds,
        )
        results = mle_model.x
        hfun = ndt.Hessian(lambda c: likelihood_om(c, wi), full_output=True)

    hessian_ndt, info = hfun(results)
    var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    return results, var


def likelihood_prior_beta(params, bounds, gauss):
    """
    This function sets the prior probabilities for the MCMC.

    Parameters
    ----------
    params : array_like
        Array containing the fitted values.
    bounds : array_like
        Array containing the interval of variation of the parameters.
    gauss : array_like
        Gaussian priors.
    Returns
    -------
    log-prior probability: float
        0, if the values are inside the prior limits
        - Infinity, if one of the values are outside the prior limits.

    """

    if (
        (bounds[0, 0] < params[0] < bounds[0, 1])
        and (bounds[1, 0] < params[1] < bounds[1, 1])
        and (bounds[2, 0] < params[2] < bounds[2, 1])
        and (bounds[3, 0] < params[3] < bounds[3, 1])
        and (bounds[4, 0] < params[4] < bounds[4, 1])
        and (bounds[5, 0] < params[5] < bounds[5, 1])
        and (bounds[6, 0] < params[6] < bounds[6, 1])
    ):
        lprior = 0
        for i in range(len(params)):
            if gauss[i, 1] > 0:
                nutmp = (params[i] - gauss[i, 0]) / gauss[i, 1]
                lprior = lprior - 0.5 * nutmp * nutmp
        return lprior
    else:
        return -np.inf


def likelihood_prob_beta(params, w, bounds, gauss):
    """
    This function gets the prior probability for MCMC.

    Parameters
    ----------
    params: array_lie
        Parameters to be fitted, the logarithm of the anisotropy radius and
        the anisotropy value at infinity, and the radial velocity dispersion
        at r = 0.
    w : array_like
        Array containing the ensemble of data, i.e., r, vr, vphi and vtheta.
    bounds : array_like
        Array containing the interval of variation of the parameters.
    gauss : array_like
        Gaussian priors.

    Returns
    -------
    log probability: float
        log-probability for the respective params.
    """

    lp = likelihood_prior_beta(params, bounds, gauss)
    if not np.isfinite(lp):
        return -np.inf
    return lp - likelihood_om(params, w)


def mcmc_anisotropy(
    r,
    vr,
    vphi,
    vtheta,
    model="OM",
    nwalkers=None,
    steps=1000,
    ini=None,
    bounds=None,
    gaussp=False,
    use_pool=False,
):
    """
    MCMC routine based on the emcee package (Foreman-Mackey et al, 2013).

    Parameters
    ----------
    r : array_like
        r-axis.
    vr : array_like
        r-axis velocity.
    vphi : array_like
        phi-angle velocity.
    vtheta : array_like
        theta-angle velocity.
    model : string, optional
        Velocity anisotropy model. The default is 'OM'.
    nwalkers : int, optional
        Number of Markov chains. The default is None.
    steps : int, optional
        Number of steps for each chain. The default is 1000.
    ini : array_like, optional
        Array containing the initial guess of the parameters.
        The order of parameters should be the same returned by the method
        "likelihood_om".
        The default is None.
    bounds : array_like, optional
        Array containing the allowed range of variation for the parameters.
        The order of parameters should be the same returned by the method
        "likelihood_om".
        The default is None.
    gaussp : boolean, optional
        "True", if the user wishes Gaussian priors to be considered.
        The default is False.
    use_pool : boolean, optional
        "True", if the user whises to use full CPU power of the machine.
        The default is False.

    Raises
    ------
    ValueError
        Velocity anisotropy model is not one of the following:
            - 'OM'
        No data is provided.

    Returns
    -------
    chain : array_like
        Set of chains from the MCMC.

    """
    if model not in ["OM"]:
        raise ValueError("Does not recognize  velocity anisotropy model.")

    size_data = len(r)

    wi = np.vstack([r, vr, vphi, vtheta])

    for i in range(np.shape(wi)[0]):
        if len(wi[i]) != size_data:
            raise ValueError("The arrays do not have the same length.")

    # Defines initial guesses for the parameters (i.e., r_a and beta_inf).
    lra = np.log10(np.nanmedian(r) / 1.305)
    ld = np.log10(np.nanstd(vr))

    if model == "OM":
        if bounds is None:
            bounds = [
                (
                    max(lra - 2, np.nanquantile(np.log10(r), 0.05)),
                    min(lra + 2, np.nanquantile(np.log10(r), 0.95)),
                ),
                (-1.99, 1.999),
                (-1.99, 1.999),
                (ld - 0.3, ld + 0.3),
                (
                    max(lra - 2, np.nanquantile(np.log10(r), 0.05)),
                    min(lra + 2, np.nanquantile(np.log10(r), 0.95)),
                ),
                (1, 8),
                (0.05, 2),
            ]

            mle_model = differential_evolution(
                lambda c: likelihood_om(c, wi),
                bounds,
            )
            results = mle_model.x
            hfun = ndt.Hessian(
                lambda c: likelihood_om(c, wi),
                full_output=True,
            )

            hessian_ndt, info = hfun(results)
            var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

            ini = np.asarray(results)
            bounds = np.asarray(bounds)

            if ini is None:
                ini = np.asarray(results)

    ndim = len(ini)  # number of dimensions.
    if gaussp is True:
        gaussp = np.zeros((ndim, 2))
        for i in range(ndim):
            if np.logical_not(np.isnan(var[i])):
                gaussp[i, 0] = ini[i]
                gaussp[i, 1] = var[i]
    else:
        gaussp = np.zeros((ndim, 2))

    if nwalkers is None or nwalkers < 2 * ndim:
        nwalkers = int(2 * ndim + 1)

    pos = [ini + 1e-3 * ini * np.random.randn(ndim) for i in range(nwalkers)]

    if use_pool:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                likelihood_prob_beta,
                args=(wi, bounds, gaussp),
                pool=pool,
            )
            sampler.run_mcmc(pos, steps)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, likelihood_prob_beta, args=(wi, bounds, gaussp)
        )
        sampler.run_mcmc(pos, steps)

    chain = sampler.chain

    return chain


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
    L = sorted(
        zip(rproj, pmr, pmt, uncpmr, uncpmt),
        key=operator.itemgetter(0),
    )
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

    plt.errorbar(
        rr2,
        dd2,
        yerr=err2,
        color="b",
        ls="none",
        barsabove=True,
        zorder=10,
    )

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
        extent=[
            np.nanmin(alim0),
            np.nanmax(alim0),
            np.nanmin(dlim0),
            np.nanmax(dlim0),
        ],
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
        extent=[
            np.nanmin(alim0),
            np.nanmax(alim0),
            np.nanmin(dlim0),
            np.nanmax(dlim0),
        ],
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
