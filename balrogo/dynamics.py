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

from . import angle

import numpy as np
from scipy.special import gamma, erf, erfc, erfcx, log_ndtr
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count
import warnings
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


def weighted_median(x, w):
    """
    Robust weighted median.
    Returns np.nan if inputs are invalid, empty, mismatched,
    or have non-positive total weight.
    """
    if x is None or w is None:
        return np.nan

    x = np.asarray(x).ravel()
    w = np.asarray(w).ravel()

    if x.size == 0 or w.size == 0 or x.size != w.size:
        return np.nan

    # Finite + non-negative weights only
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not np.any(mask):
        return np.nan

    x = x[mask]
    w = w[mask]

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
    Robust weighted standard deviation.
    Returns np.nan if inputs are invalid or total weight is non-positive.
    """
    if x is None or w is None:
        return np.nan

    x = np.asarray(x).ravel()
    w = np.asarray(w).ravel()

    if x.size == 0 or w.size == 0 or x.size != w.size:
        return np.nan

    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not np.any(mask):
        return np.nan

    x = x[mask]
    w = w[mask]

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
    # Hard fail-safe: never let None propagate
    if x is None or ex is None:
        return np.full(4, np.nan, dtype=float), np.nan

    x = np.asarray(x).ravel()
    ex = np.asarray(ex).ravel()

    if x.size == 0 or ex.size == 0 or x.size != ex.size:
        return np.full(4, np.nan, dtype=float), np.nan

    if ww is None:
        ww = np.ones_like(x, dtype=float)
    else:
        ww = np.asarray(ww).ravel()
        if ww.size != x.size:
            return np.full(4, np.nan, dtype=float), np.nan

    # Sanitize: finite x/ex/ww and non-negative weights
    mask = np.isfinite(x) & np.isfinite(ex) & np.isfinite(ww) & (ww >= 0)
    if np.count_nonzero(mask) < 5:
        return np.full(4, np.nan, dtype=float), np.nan

    x = x[mask]
    ex = ex[mask]
    ww = ww[mask]

    if np.sum(ww) <= 0 or not np.isfinite(np.sum(ww)):
        return np.full(4, np.nan, dtype=float), np.nan

    # Initial guesses; can still be nan if something is off
    m0 = weighted_median(x, ww)
    s0 = weighted_std(x, ww)

    if (not np.isfinite(m0)) or (not np.isfinite(s0)) or (s0 <= 0):
        return np.full(4, np.nan, dtype=float), np.nan

    # Initial parameter guess: mean, sigma, h3, h4
    ini = np.asarray([m0, s0, 0.0, 0.0], dtype=float)

    # Parameter bounds for optimization
    bounds = [
        (ini[0] - 3.0 * ini[1], ini[0] + 3.0 * ini[1]),
        (0.2 * ini[1], 5.0 * ini[1]),
        (-0.2, 0.2),
        (-0.187, 0.145),
    ]

    # Differential evolution can occasionally throw if bounds are degenerate
    try:
        mle_model = differential_evolution(
            lambda c: mom_likelihood_func(c, x, ex, ww, mode="lnlik"),
            bounds,
        )
        params = np.asarray(mle_model.x, dtype=float)
        if params.shape != (4,) or not np.all(np.isfinite(params)):
            return np.full(4, np.nan, dtype=float), np.nan
        # Evaluate objective at returned params to confirm feasibility
        nll = mom_likelihood_func(params, x, ex, ww, mode="lnlik")
        if not np.isfinite(nll):
            # Non-physical (np.inf) or invalid (nan)
            return np.full(4, np.nan, dtype=float), np.nan
        logL = -float(nll)
        return params, logL
    except Exception:
        return np.full(4, np.nan, dtype=float), np.nan


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
    estimates the bias of each recovered quantity. Multiplicative correction
    is used only for intrinsically positive quantities; signed quantities are
    corrected additively.

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

        Intrinsically positive quantities are corrected multiplicatively.
        Signed quantities are corrected by subtracting their additive bias.
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
        if sample is None:
            continue  # non-physical draw; skip safely
        # Re-fit Gauss–Hermite moments
        mom_samples[:4, k], logL = mom_likelihood_call(sample, ex, ww)

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
    mom_corrected = mom_stats.copy()
    recovered_mean = np.nanmean(mom_samples, axis=1)

    # Quantities that are intrinsically positive:
    # sigma, variance, standard deviation, and root-mean-square.
    positive_idx = np.array(
        [idx for idx in (1, 5, 8, 9) if idx < nrows],
        dtype=int,
    )
    signed_idx = np.setdiff1d(
        np.arange(nrows),
        positive_idx,
    )

    # Multiplicative correction for intrinsically positive quantities.
    if positive_idx.size:
        positive_input = mom_stats[positive_idx, 0]
        positive_recovered = recovered_mean[positive_idx]

        if (
            np.any(~np.isfinite(positive_input))
            or np.any(~np.isfinite(positive_recovered))
            or np.any(positive_input <= 0)
            or np.any(positive_recovered <= 0)
        ):
            print(
                "mom_monte_carlo: Invalid positive quantities encountered "
                "during multiplicative bias correction. Returning original "
                "mom_stats without correction."
            )
            return mom_stats

        ratio = positive_recovered / positive_input
        mom_corrected[positive_idx, 0] = positive_input / ratio
        mom_corrected[positive_idx, 1] = mom_stats[positive_idx, 1] / ratio

    # Additive correction for quantities that may be zero or negative:
    # mean, h3, h4, statistical mean, skewness, and kurtosis.
    if signed_idx.size:
        signed_recovered = recovered_mean[signed_idx]

        if np.any(~np.isfinite(signed_recovered)):
            print(
                "mom_monte_carlo: Invalid signed quantities encountered "
                "during additive bias correction. Returning original "
                "mom_stats without correction."
            )
            return mom_stats

        bias = signed_recovered - mom_stats[signed_idx, 0]
        mom_corrected[signed_idx, 0] = mom_stats[signed_idx, 0] - bias
        # An additive shift does not rescale the uncertainty.
        mom_corrected[signed_idx, 1] = mom_stats[signed_idx, 1]

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

    # --- Guard 1: physicality of corrected first four moments
    if (variance_c <= 0) or np.isnan(variance_c) or np.isnan(kurtosis_c):
        print(
            "mom_monte_carlo: Corrected parameters yield non-physical moments "
            f"(variance={variance_c:.3g}, kurtosis={kurtosis_c:.3g}). "
            "Returning original mom_stats without correction."
        )
        return mom_stats

    # --- Guard 2: all corrected uncertainties must be strictly positive and finite
    unc = mom_corrected[:, 1]
    bad_unc = (~np.isfinite(unc)) | (unc <= 0)

    if np.any(bad_unc):
        # Provide a concise but useful diagnostic
        bad_idx = np.where(bad_unc)[0]
        print(
            "mom_monte_carlo: Corrected uncertainties are not strictly positive/finite "
            f"at indices {bad_idx.tolist()}."
            + " Returning original mom_stats without correction."
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
    mom_logl = np.zeros(nsamples)

    if debug:
        time_start = time.time()

    for k in range(nsamples):
        mom_samples[:, k], mom_logl[k] = mom_likelihood_call(x, ex, ww)

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

    mom_stats[:, 0] = mom_samples[:, np.nanargmax(mom_logl)]
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
"Gaussian 1D fits"
# ------------------------------------------------------------------------------


def _prepare_gaussian_data(x, ex, ww=None):
    """
    Validate and clean 1D data, uncertainties, and weights.
    """
    if x is None or ex is None:
        return None

    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return None

    if np.isscalar(ex):
        ex = np.full_like(x, ex, dtype=float)
    else:
        ex = np.asarray(ex, dtype=float).ravel()

    if ww is None:
        ww = np.ones_like(x)
    elif np.isscalar(ww):
        ww = np.full_like(x, ww, dtype=float)
    else:
        ww = np.asarray(ww, dtype=float).ravel()

    if ex.size != x.size or ww.size != x.size:
        return None

    mask = np.isfinite(x) & np.isfinite(ex) & np.isfinite(ww) & (ex >= 0) & (ww >= 0)

    if np.count_nonzero(mask) < 3:
        return None

    x = x[mask]
    ex = ex[mask]
    ww = ww[mask]

    if not np.isfinite(np.sum(ww)) or np.sum(ww) <= 0:
        return None

    return x, ex, ww


def gaussian_likelihood_func(
    params,
    x,
    ex,
    ww=None,
    mode="lnlik",
):
    """
    Gaussian likelihood convolved with Gaussian measurement errors.
    """
    if ww is None:
        ww = np.ones_like(x, dtype=float)

    mean, sigma = params

    if not np.isfinite(mean) or not np.isfinite(sigma) or sigma < 0:
        if mode == "lnlik":
            return np.inf
        return np.full_like(x, np.nan, dtype=float)

    variance = sigma**2 + ex**2

    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        if mode == "lnlik":
            return np.inf
        return np.full_like(x, np.nan, dtype=float)

    ln_pdf = -0.5 * (np.log(2.0 * np.pi * variance) + (x - mean) ** 2 / variance)

    if mode != "lnlik":
        return np.exp(ln_pdf)

    return -np.sum(ww * ln_pdf)


def gaussian_likelihood_call(x, ex, ww=None):
    """
    Fit the intrinsic Gaussian mean and dispersion.

    Returns
    -------
    params : ndarray, shape (2,)
        Mean and intrinsic dispersion.
    logl : float
        Maximum log-likelihood.
    """
    data = _prepare_gaussian_data(x, ex, ww)

    if data is None:
        return np.full(2, np.nan), np.nan

    x, ex, ww = data

    mean0 = weighted_median(x, ww)
    std0 = weighted_std(x, ww)

    if not np.isfinite(mean0) or not np.isfinite(std0):
        return np.full(2, np.nan), np.nan

    if std0 <= 0 and np.all(ex == 0):
        return np.full(2, np.nan), np.nan

    sigma0 = np.sqrt(
        max(
            std0**2 - np.average(ex**2, weights=ww),
            0.0,
        )
    )

    scale = max(
        std0,
        np.nanmedian(ex),
        np.sqrt(np.finfo(float).eps) * max(1.0, np.abs(mean0)),
    )

    bounds = [
        (
            mean0 - 5.0 * scale,
            mean0 + 5.0 * scale,
        ),
        (
            0.0,
            5.0 * scale,
        ),
    ]

    try:
        result = differential_evolution(
            lambda p: gaussian_likelihood_func(
                p,
                x,
                ex,
                ww,
            ),
            bounds,
            x0=np.array([mean0, sigma0]),
            polish=True,
            seed=0,
        )

        params = np.asarray(result.x, dtype=float)

        nll = gaussian_likelihood_func(
            params,
            x,
            ex,
            ww,
        )

        if (
            params.shape != (2,)
            or not np.all(np.isfinite(params))
            or not np.isfinite(nll)
        ):
            return np.full(2, np.nan), np.nan

        return params, -float(nll)

    except Exception:
        return np.full(2, np.nan), np.nan


def gaussian_monte_carlo(
    ex,
    ww,
    gauss_stats,
    nsamples=100,
    random_state=None,
):
    """
    Estimate uncertainties and correct Gaussian-fit bias using
    parametric Monte Carlo sampling.

    The mean is corrected additively because it may be zero or negative.
    The intrinsically positive dispersion is corrected multiplicatively.
    """
    if nsamples < 2:
        raise ValueError("nsamples must be at least 2")

    rng = np.random.default_rng(random_state)

    mean, sigma = gauss_stats[:, 0]
    recovered = np.full((2, nsamples), np.nan)

    observed_sigma = np.sqrt(sigma**2 + ex**2)

    for k in range(nsamples):
        x_mc = rng.normal(
            mean,
            observed_sigma,
        )

        recovered[:, k], _ = gaussian_likelihood_call(
            x_mc,
            ex,
            ww,
        )

    valid = np.all(np.isfinite(recovered), axis=0)

    minimum_valid = max(
        2,
        int(np.ceil(0.5 * nsamples)),
    )

    if np.count_nonzero(valid) < minimum_valid:
        return gauss_stats

    recovered = recovered[:, valid]

    recovered_mean = np.mean(recovered, axis=1)
    uncertainties = np.std(
        recovered,
        axis=1,
        ddof=1,
    )

    raw_stats = gauss_stats.copy()
    raw_stats[:, 1] = uncertainties

    if np.any(~np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
        return gauss_stats

    corrected = raw_stats.copy()

    # Mean: additive bias correction.
    mean_bias = recovered_mean[0] - gauss_stats[0, 0]
    corrected[0, 0] = gauss_stats[0, 0] - mean_bias
    corrected[0, 1] = uncertainties[0]

    # Dispersion: multiplicative bias correction.
    sigma_input = gauss_stats[1, 0]
    sigma_recovered = recovered_mean[1]

    if (
        not np.isfinite(sigma_input)
        or not np.isfinite(sigma_recovered)
        or sigma_input <= 0
        or sigma_recovered <= 0
    ):
        return raw_stats

    sigma_ratio = sigma_recovered / sigma_input
    corrected[1, 0] = sigma_input / sigma_ratio
    corrected[1, 1] = uncertainties[1] / sigma_ratio

    if (
        not np.all(np.isfinite(corrected))
        or corrected[1, 0] <= 0
        or corrected[1, 1] <= 0
    ):
        return raw_stats

    return corrected


def fit_1d_gaussian(
    x,
    ex,
    ww=None,
    method="monte-carlo",
    nsamples=100,
    random_state=None,
    debug=False,
):
    """
    Fit the mean and intrinsic dispersion of a 1D Gaussian.

    Measurement uncertainties are incorporated through Gaussian
    convolution. With ``method="monte-carlo"``, parametric Monte Carlo
    sampling provides uncertainties, an additive bias correction for the
    mean, and a multiplicative bias correction for the dispersion.

    Parameters
    ----------
    x : array_like
        Observed data.
    ex : array_like or float
        Measurement uncertainties.
    ww : array_like, float or None, optional
        Likelihood weights.
    method : str or None, optional
        Use ``"monte-carlo"`` to estimate uncertainties and correct bias.
        Any other value returns only the maximum-likelihood estimates.
    nsamples : int, optional
        Number of Monte Carlo realisations.
    random_state : int or None, optional
        Seed controlling the Monte Carlo sampling.
    debug : bool, optional
        Print fitted values.

    Returns
    -------
    stats : ndarray, shape (2, 2)
        Rows correspond to mean and intrinsic dispersion.
        Columns correspond to fitted value and uncertainty.
    """
    data = _prepare_gaussian_data(x, ex, ww)

    if data is None:
        return np.full((2, 2), np.nan)

    x, ex, ww = data

    params, logl = gaussian_likelihood_call(
        x,
        ex,
        ww,
    )

    stats = np.full((2, 2), np.nan)
    stats[:, 0] = params

    if method == "monte-carlo" and np.all(np.isfinite(params)):
        stats = gaussian_monte_carlo(
            ex,
            ww,
            stats,
            nsamples=nsamples,
            random_state=random_state,
        )

    if debug:
        print(f"fit_1d_gaussian: logL = {logl:.3f}")
        print(f"  mean  = {stats[0, 0]:.6g} " f"+/- {stats[0, 1]:.3g}")
        print(f"  sigma = {stats[1, 0]:.6g} " f"+/- {stats[1, 1]:.3g}")

    return stats


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
