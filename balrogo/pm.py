"""
Created on 2020

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2020, Paris
#
# This file contains the main functions concerning proper motion data.
# It provides MCMC and maximum likelihood fits of proper motions data,
# as well as robust initial guesses for those fits.
#
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy import integrate
from scipy.special import gamma
from scipy.signal import find_peaks
import operator
from skimage.feature import peak_local_max
import emcee
import numdifftools as ndt
from multiprocessing import Pool
from multiprocessing import cpu_count

ncpu = cpu_count()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Probability distribution functions and likelihoods"
# ---------------------------------------------------------------------------


def proj_mw_pdf(Ux, sr_pm, slp_pm):
    """
    Projection of the Milky Way field stars, with respect to the
    field stars ellipse's main directions.

    Parameters
    ----------
    Ux : array_like
        Array containting the data to be fitted.
    sr_pmx : float
        Scale radius.
    slp_pm : float
        Slope of the PDF.

    Returns
    -------
    proj : array_like
        Array containing the projection of the PDF's from
        the Milky Way field stars.

    """

    f = (1 + (Ux / sr_pm) * (Ux / sr_pm)) ** (0.5 + slp_pm / 2) / sr_pm
    fac = gamma(-0.5 - slp_pm / 2) / gamma(-1 - slp_pm / 2)

    proj = fac * f / np.sqrt(np.pi)

    return proj


def proj_global_pdf(Ux, mu_pmx, sig_pm, sr_pmx, slp_pm, frc_go_mw):
    """
    Projection of the sum of the PDF's from the galactic object
    (2D Gaussian) and Milky Way field stars, with respect to the
    field stars ellipse's main directions.

    It considers the origin (0,0) to be centered in the Milky Way field
    stars mean proper motion.

    Parameters
    ----------
    Ux : array_like
        Array containting the data to be fitted.
    mu_pmx : float
        Mean proper motion.
    sig_pm : float
        Gaussian standard deviation.
    sr_pmx : float
        Scale radius.
    slp_pm : float
        Slope of the PDF.
    frc_go_mw : float
        Fraction of galactic objects by Milky Way stars.

    Returns
    -------
    array_like
        Array containing the projection of the PDF's from the galactic object
        (2D Gaussian) and Milky Way field stars.

    """

    dx = (Ux - mu_pmx) / sig_pm
    f1 = frc_go_mw * np.exp(-0.5 * dx * dx) / np.sqrt(2 * np.pi * sig_pm * sig_pm)

    f2 = (1 - frc_go_mw) * proj_mw_pdf(Ux, sr_pmx, slp_pm)

    return f1 + f2


def pdf_field_stars(Ux, Uy, mu_pmx, mu_pmy, sr_pmx, sr_pmy, rot_pm, slp_pm):
    """
    PDF of Milky Way field stars according to Vitral (2021).

    Parameters
    ----------
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    Uy : array_like
        Array containting the data to be fitted, in y-direction.
    mu_pmx : float
        Mean proper motion in x-direction.
    mu_pmy :  float
        Mean proper motion in y-direction.
    sr_pmx : float
        Scale radius in x-direction.
    sr_pmy : float
        Scale radius in y-direction.
    rot_pm : float
        Angle of rotation of the major axis' ellipse,
        with respect to the x-direction.
    slp_pm : float
        Slope of the PDF.

    Returns
    -------
    pdf : array_like
        Array containing the PDF the Milky Way field stars.

    """

    x = (Ux - mu_pmx) * np.cos(rot_pm) + (Uy - mu_pmy) * np.sin(rot_pm)
    y = -(Ux - mu_pmx) * np.sin(rot_pm) + (Uy - mu_pmy) * np.cos(rot_pm)

    f1 = 1 + (x / sr_pmx) * (x / sr_pmx)
    f2 = 1 + (y / sr_pmy) * (y / sr_pmy)

    f = (f1 * f2) ** (0.5 + slp_pm / 2) / (sr_pmx * sr_pmy)

    fac = gamma(-0.5 - slp_pm / 2) ** 2 / gamma(-1 - slp_pm / 2) ** 2

    pdf = fac * f / np.pi

    return pdf


def gauss_1d(Ux, mu_pmx, sig_pm):
    """
    1D Gaussian PDF in cartesian coordinates.

    Parameters
    ----------
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    mu_pmx : float
        Mean proper motion in x-direction.
    sig_pm : float
        Gaussian standard deviation.

    Returns
    -------
    pdf : array_like
        Array containing the PDF the galactic object (1D Gaussian).

    """

    dx = Ux - mu_pmx

    f1 = -0.5 * (dx / sig_pm) * (dx / sig_pm)
    den = np.sqrt(2 * np.pi * sig_pm * sig_pm)

    pdf = np.exp(f1) / den

    return pdf


def gauss_2d(Ux, Uy, mu_pmx, mu_pmy, sig_pm):
    """
    2D Gaussian PDF in cartesian coordinates.

    Parameters
    ----------
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    Uy : array_like
        Array containting the data to be fitted, in y-direction.
    mu_pmx : float
        Mean proper motion in x-direction.
    mu_pmy : float
        Mean proper motion in y-direction.
    sig_pm : float
        Gaussian standard deviation.

    Returns
    -------
    pdf : array_like
        Array containing the PDF the galactic object (2D Gaussian).

    """

    dx = Ux - mu_pmx
    dy = Uy - mu_pmy

    f1 = -0.5 * (dx / sig_pm) * (dx / sig_pm)
    f2 = -0.5 * (dy / sig_pm) * (dy / sig_pm)
    den = 2 * np.pi * sig_pm * sig_pm

    pdf = np.exp(f1 + f2) / den

    return pdf


def global_pdf(
    Ux,
    Uy,
    ex,
    ey,
    exy,
    mu_pmx_go,
    mu_pmy_go,
    sig_pm_go,
    mu_pmx_mw,
    mu_pmy_mw,
    sr_pmx_mw,
    sr_pmy_mw,
    rot_pm_mw,
    slp_pm_mw,
    frc_go_mw,
):
    """
    This function gives the sum of the PDF's from the galactic object
    (2D Gaussian) and Milky Way field stars.

    Parameters
    ----------
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    Uy : array_like
        Array containting the data to be fitted, in y-direction.
    ex : array_like
        Data uncertainty in x-direction.
    ey : array_like
        Data uncertainty in y-direction.
    exy : array_like
        Correlation of data uncertainty in x and y-direction.
    mu_pmx_go : float
        Mean proper motion of galactic object in x-direction.
    mu_pmy_go : float
        Mean proper motion of galactic object in y-direction.
    sig_pm_go : float
        Gaussian standard deviation of galactic object.
    mu_pmx_mw : float
        Mean proper motion of Milky Way field stars in x-direction.
    mu_pmy_mw :  float
        Mean proper motion of Milky Way field stars in y-direction.
    sr_pmx_mw : float
        Scale radius of Milky Way field stars in x-direction.
    sr_pmy_mw : float
        Scale radius of Milky Way field stars in y-direction.
    rot_pm_mw : float
        Angle of rotation of the Milky Way field stars' major axis' ellipse,
        with respect to the x-direction.
    slp_pm_mw : float
        Slope of the Milky Way field stars PDF.
    frc_go_mw : float
        Fraction of galactic objects by Milky Way stars.

    Returns
    -------
    array_like
        Sum of the PDF's from the galactic object
        (2D Gaussian) and Milky Way field stars.

    """
    # Deals with a semi-convolution, only accounting for the galactic object.
    pm_mod = np.sqrt((Ux - mu_pmx_go) ** 2 + (Uy - mu_pmy_go) ** 2)
    err = (
        (ex * (Ux - mu_pmx_go) / pm_mod) ** 2
        + (ey * (Uy - mu_pmy_go) / pm_mod) ** 2
        + 2 * ex * ey * exy * (Ux - mu_pmx_go) * (Uy - mu_pmy_go) / pm_mod ** 2
    )
    sig_pm_go = np.sqrt(sig_pm_go * sig_pm_go + err)

    # PDF from galactic object
    pdf_go = frc_go_mw * gauss_2d(Ux, Uy, mu_pmx_go, mu_pmy_go, sig_pm_go)

    # PDF from Milky Way stars
    pdf_mw = (1 - frc_go_mw) * pdf_field_stars(
        Ux, Uy, mu_pmx_mw, mu_pmy_mw, sr_pmx_mw, sr_pmy_mw, rot_pm_mw, slp_pm_mw
    )

    return pdf_go + pdf_mw


def likelihood_function(params, Ux, Uy, ex, ey, exy, values=None):
    """
    Computes minus the likelihood of the PM model from Vitral (2021).

    Parameters
    ----------
    params : array_like
        Array of parameters from the model.
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    Uy : array_like
        Array containting the data to be fitted, in y-direction.
    ex : array_like
        Data uncertainty in x-direction.
    ey : array_like
        Data uncertainty in y-direction.
    exy : array_like
        Correlation of data uncertainty in x and y-direction.
    values : array_like, optional
        Array containing some of the parameters already fitted. If not fitted,
        they are filled with np.nan.
        The default is None.

    Returns
    -------
    L : float
        minus the logarithm of the likelihood function.

    """

    if values is None:
        values = np.zeros(10)
        values[:] = np.nan

    for i in range(len(values)):
        if np.logical_not(np.isnan(values[i])):
            params[i] = values[i]

    mu_pmx_go = params[0]  # mean pmra from galactic object
    mu_pmy_go = params[1]  # mean pmdec from galactic object
    sig_pm_go = params[2]  # pm dispersion from galactic object

    mu_pmx_mw = params[3]  # mean pmra from Milky Way stars
    mu_pmy_mw = params[4]  # mean pmdec from Milky Way stars
    sr_pmx_mw = params[5]  # scale radius (pmra) from Milky Way stars
    sr_pmy_mw = params[6]  # scale radius (pmdec) from Milky Way stars
    rot_pm_mw = params[7]  # rotation angle from Milky Way field stars
    slp_pm_mw = params[8]  # slope from Milky Way field stars

    frc_go_mw = params[9]  # fraction of galactic objects by Milky Way stars

    # Gets the PDF
    f_i = global_pdf(
        Ux,
        Uy,
        ex,
        ey,
        exy,
        mu_pmx_go,
        mu_pmy_go,
        sig_pm_go,
        mu_pmx_mw,
        mu_pmy_mw,
        sr_pmx_mw,
        sr_pmy_mw,
        rot_pm_mw,
        slp_pm_mw,
        frc_go_mw,
    )

    # Transforms zero's in NaN
    f_i[f_i <= 0] = np.nan

    # Calculates the likelihood, taking out NaN's
    f_i = f_i[np.logical_not(np.isnan(f_i))]
    L = -np.sum(np.log(f_i))

    return L


def likelihood_gauss2d(params, Ux, Uy, ex, ey, exy, values=None):
    """
    Computes minus the likelihood of a 2D Gaussian.

    Parameters
    ----------
    params : array_like
        Array of parameters from the model.
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    Uy : array_like
        Array containting the data to be fitted, in y-direction.
    ex : array_like
        Data uncertainty in x-direction.
    ey : array_like
        Data uncertainty in y-direction.
    exy : array_like
        Correlation of data uncertainty in x and y-direction.
    values : array_like, optional
        Array containing some of the parameters already fitted. If not fitted,
        they are filled with np.nan.
        The default is None.

    Returns
    -------
    L : float
        minus the logarithm of the likelihood function.

    """

    if values is None:
        values = np.zeros(10)
        values[:] = np.nan

    for i in range(len(values)):
        if np.logical_not(np.isnan(values[i])):
            params[i] = values[i]

    mu_pmx_go = params[0]  # mean pmra from galactic object
    mu_pmy_go = params[1]  # mean pmdec from galactic object
    sig_pm_go = params[2]  # pm dispersion from galactic object

    # Deals with a semi-convolution, only accounting for the galactic object.
    pm_mod = np.sqrt((Ux - mu_pmx_go) ** 2 + (Uy - mu_pmy_go) ** 2)
    err = (
        (ex * (Ux - mu_pmx_go) / pm_mod) ** 2
        + (ey * (Uy - mu_pmy_go) / pm_mod) ** 2
        + 2 * ex * ey * exy * (Ux - mu_pmx_go) * (Uy - mu_pmy_go) / pm_mod ** 2
    )
    sig_pm_go = np.sqrt(sig_pm_go * sig_pm_go + err)

    # Gets the PDF
    f_i = gauss_2d(Ux, Uy, mu_pmx_go, mu_pmy_go, sig_pm_go)

    # Transforms zero's in NaN
    f_i[f_i <= 0] = np.nan

    # Calculates the likelihood, taking out NaN's
    f_i = f_i[np.logical_not(np.isnan(f_i))]
    L = -np.sum(np.log(f_i))

    return L


def likelihood_2gauss1d(params, Ux, ex):
    """
    Computes minus the likelihood of two Gaussians.

    Parameters
    ----------
    params : array_like
        Array of parameters from the model.
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    ex : array_like
        Data uncertainty in x-direction.

    Returns
    -------
    L : float
        minus the logarithm of the likelihood function.
    """

    mu_pmx_go = params[0]  # mean from galactic object
    sig_pm_go = params[1]  # dispersion from galactic object

    mu_pmx_mw = params[2]  # mean from Milky Way stars
    sr_pmx_mw = params[3]  # scale radius from Milky Way stars

    frc_go_mw = params[4]  # fraction of galactic objects by Milky Way stars

    sig_pm_go = np.sqrt(sig_pm_go * sig_pm_go + ex * ex)
    sr_pmx_mw = np.sqrt(sr_pmx_mw * sr_pmx_mw + ex * ex)

    # PDF from galactic object
    pdf_go = frc_go_mw * gauss_1d(Ux, mu_pmx_go, sig_pm_go)

    # PDF from Milky Way stars
    pdf_mw = (1 - frc_go_mw) * gauss_1d(Ux, mu_pmx_mw, sr_pmx_mw)

    # Gets the PDF
    f_i = pdf_go + pdf_mw

    # Transforms zero's in NaN
    f_i[f_i <= 0] = np.nan

    # Calculates the likelihood, taking out NaN's
    f_i = f_i[np.logical_not(np.isnan(f_i))]
    L = -np.sum(np.log(f_i))

    return L


def likelihood_1gauss1d(params, Ux, ex):
    """
    Computes minus the likelihood of one Gaussian.

    Parameters
    ----------
    params : array_like
        Array of parameters from the model.
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    ex : array_like
        Data uncertainty in x-direction.

    Returns
    -------
    L : float
        minus the logarithm of the likelihood function.
    """

    mu_pmx_go = params[0]  # mean from galactic object
    sig_pm_go = params[1]  # dispersion from galactic object

    sig_pm_go = np.sqrt(sig_pm_go * sig_pm_go + ex * ex)

    # PDF from galactic object
    pdf_go = gauss_1d(Ux, mu_pmx_go, sig_pm_go)

    # Gets the PDF
    f_i = pdf_go

    # Transforms zero's in NaN
    f_i[f_i <= 0] = np.nan

    # Calculates the likelihood, taking out NaN's
    f_i = f_i[np.logical_not(np.isnan(f_i))]
    L = -np.sum(np.log(f_i))

    return L


def likelihood_prior(params, guess, bounds):
    """
    This function sets the prior probabilities for the MCMC.

    Parameters
    ----------
    params : array_like
        Array containing the fitted values.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.

    Returns
    -------
    log-prior probability: float
        0, if the values are inside the prior limits
        - Infinity, if one of the values are outside the prior limits.

    """

    if (
        (guess[0] - bounds[0] < params[0] < guess[0] + bounds[0])
        and (guess[1] - bounds[1] < params[1] < guess[1] + bounds[1])
        and (guess[2] - bounds[2] < params[2] < guess[2] + bounds[2])
        and (guess[3] - bounds[3] < params[3] < guess[3] + bounds[3])
        and (guess[4] - bounds[4] < params[4] < guess[4] + bounds[4])
        and (guess[5] - bounds[5] < params[5] < guess[5] + bounds[5])
        and (guess[6] - bounds[6] < params[6] < guess[6] + bounds[6])
        and (guess[7] - bounds[7] < params[7] < guess[7] + bounds[7])
        and (guess[8] - bounds[8] < params[8] < guess[8] + bounds[8])
        and (guess[9] - bounds[9] < params[9] < guess[9] + bounds[9])
    ):
        return 0.0
    else:
        return -np.inf


def likelihood_prob(params, Ux, Uy, ex, ey, exy, guess, bounds):
    """
    This function gets the prior probability for MCMC.

    Parameters
    ----------
    params : array_like
        Array containing the fitted values.
    Ux : array_like
        Array containting the data to be fitted, in x-direction.
    Uy : array_like
        Array containting the data to be fitted, in y-direction.
    ex : array_like
        Data uncertainty in x-direction.
    ey : array_like
        Data uncertainty in y-direction.
    exy : array_like
        Correlation of data uncertainty in x and y-direction.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.

    Returns
    -------
    log probability: float
        log-probability for the respective params.
    """

    lp = likelihood_prior(params, guess, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp - likelihood_function(params, Ux, Uy, ex, ey, exy)


def prob(Ux, Uy, ex, ey, exy, params, conv=True):
    """
    This function gives the probability of a certain star to belong to
    a galactic object. This is computed with distribution functions in
    proper motion space.

    Parameters
    ----------
    Ux : array_like
        Array containting the data in the x-direction.
    Uy : array_like
        Array containting the data in the y-direction.
    ex : array_like
        Data uncertainty in x-direction.
    ey : array_like
        Data uncertainty in y-direction.
    exy : array_like
        Correlation of data uncertainty in x and y-direction.
    params : array_like
        Array containing the fitted values.
    conv : boolean, optional
        True, if the user wants to convolve the galactic object PDF with
        Gaussian errors. The defualt is True.

    Returns
    -------
    probability : array_like
        Probability of a each star to belong to the respective
        a galactic object (considering only proper motions).

    """

    mu_pmx_go = params[0]  # mean pmra from galactic object
    mu_pmy_go = params[1]  # mean pmdec from galactic object
    sig_pm_go = params[2]  # pm dispersion from galactic object

    mu_pmx_mw = params[3]  # mean pmra from Milky Way stars
    mu_pmy_mw = params[4]  # mean pmdec from Milky Way stars
    sr_pmx_mw = params[5]  # scale radius (pmra) from Milky Way stars
    sr_pmy_mw = params[6]  # scale radius (pmdec) from Milky Way stars
    rot_pm_mw = params[7]  # rotation angle from Milky Way field stars
    slp_pm_mw = params[8]  # slope from Milky Way field stars

    frc_go_mw = params[9]  # fraction of galactic objects by Milky Way stars

    if conv is True:
        pm_mod = np.sqrt((Ux - mu_pmx_go) ** 2 + (Uy - mu_pmy_go) ** 2)
        err = (
            (ex * (Ux - mu_pmx_go) / pm_mod) ** 2
            + (ey * (Uy - mu_pmy_go) / pm_mod) ** 2
            + 2 * ex * ey * exy * (Ux - mu_pmx_go) * (Uy - mu_pmy_go) / pm_mod ** 2
        )
        sig_pm_go = sig_pm_go = np.sqrt(sig_pm_go * sig_pm_go + err)

    pdf_go = gauss_2d(Ux, Uy, mu_pmx_go, mu_pmy_go, sig_pm_go)
    pdf_mw = pdf_field_stars(
        Ux, Uy, mu_pmx_mw, mu_pmy_mw, sr_pmx_mw, sr_pmy_mw, rot_pm_mw, slp_pm_mw
    )

    f1 = frc_go_mw * pdf_go
    f2 = (1 - frc_go_mw) * pdf_mw

    probability = f1 / (f1 + f2)

    return probability


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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Guess initial parameters"
# ---------------------------------------------------------------------------


def gauss_sig(x_axis, gauss, peak):
    """
    This function estimates the dispersion of the GC Gaussian in PM space.

    Parameters
    ----------
    x_axis : array_like
        Array containing values from one particular direction.
    gauss : array_like
        Array containing the PDF at the repective values
        from one particular direction.
    peak : float
        Peak of the PDF, locally.

    Returns
    -------
    sigma : float
        Guess of the Gaussian dispersion.

    """

    # Gets the array's index where the peak is located
    i, j = np.argmin(np.abs(x_axis - peak)), np.argmin(np.abs(x_axis - peak))

    # Gets the histogram max value times e^(-1/2): (histogram value at 1 sigma)
    threshold = gauss[i] * np.exp(-1 / 2)

    # Search where the histogram is less than the threshold
    while gauss[i] >= threshold and i > 0:
        i -= 1
    index_left = i
    while gauss[j] >= threshold and j < (len(gauss) - 1):
        j += 1
    index_right = j

    # Assigns sigma as the minimum between the distance took to depass the
    # threshold in the left and in the right of the peak
    sigma = min(x_axis[index_right] - peak, peak - x_axis[index_left])

    return sigma


def clump_fraction(mean, sigma, x_axis, y_axis):
    """
    This function esitmates the fraction of stars belonging to each PM clump.

    Indexes: 0 --> clump 0 / 1 --> clump 1

    Parameters
    ----------
    mean : 2D-array
        Proper motion means (galactic object + Milky Way stars) in one
        particular direction.
    sigma : 2D-array
        Gaussian dispersion values (galactic object + Milky Way stars) in one
        particular direction.
    x_axis : array_like
        Array containing values from one particular direction.
    y_axis : array_like
        Array containing the PDF at the repective values
        from one particular direction.

    Returns
    -------
    fluxes : 2D-array
        Flux fraction from clump 0 and clump 1.

    """

    index_left = np.argmin(np.abs(-x_axis + mean[0] - 3 * sigma[0]))
    index_right = np.argmin(np.abs(-x_axis + mean[0] + 3 * sigma[0]))

    # Flux of stars belonging to the clump 0
    flux1 = integrate.simps(
        y_axis[index_left:index_right], x_axis[index_left:index_right]
    )

    index_left = np.argmin(np.abs(-x_axis + mean[1] - 3 * sigma[1]))
    index_right = np.argmin(np.abs(-x_axis + mean[1] + 3 * sigma[1]))

    # Flux of stars belonging to the clump 1
    flux2 = integrate.simps(
        y_axis[index_left:index_right], x_axis[index_left:index_right]
    )

    return np.asarray([flux1 / (flux1 + flux2), flux2 / (flux1 + flux2)])


def initial_guess(x_data, y_data):
    """
    This function estimates the initial parameters of the global PDF.

    Parameters
    ----------
    x_data : array_like
        Data in x-direction.
    y_data : array_like
        Data in y-direction.

    Returns
    -------
    8D-array
        Initial guesses for some of the proper motion model parameters.

    """

    # Takes off NaN values
    x_nan = np.logical_not(np.isnan(x_data))
    y_nan = np.logical_not(np.isnan(y_data))
    idx_nan = x_nan * y_nan
    x_data = x_data[idx_nan]
    y_data = y_data[idx_nan]

    q_16, q_50, q_84 = quantile(x_data, [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50
    bins_x = int((np.amax(x_data) - np.amin(x_data)) / (min(q_m, q_p) / 4))

    q_16, q_50, q_84 = quantile(y_data, [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50
    bins_y = int((np.amax(y_data) - np.amin(y_data)) / (min(q_m, q_p) / 4))

    # Gets the histogram in PMRA
    x_hist, x_axis = np.histogram(
        x_data, bins=bins_x, range=(np.amin(x_data), np.amax(x_data))
    )
    x_axis = 0.5 * (x_axis[1:] + x_axis[:-1])

    # Gets the histogram in PMDec
    y_hist, y_axis = np.histogram(
        y_data, bins=bins_y, range=(np.amin(y_data), np.amax(y_data))
    )
    y_axis = 0.5 * (y_axis[1:] + y_axis[:-1])

    # Gets the histogram of the 2d (PMRA,PMDec) data
    hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=[bins_x, bins_y])
    hist = hist.T

    # Estimates the PMRA and PMDec means from the two clumps by taking the
    # two main local maxima of the 2d histogram. With that, it estimates the
    # other clump parameters
    peaks = peak_local_max(hist, num_peaks=2)
    y_peak, x_peak = peaks.T[0], peaks.T[1]
    mean_x, mean_y = xedges[x_peak], yedges[y_peak]
    sigma_x = np.asarray(
        [gauss_sig(x_axis, x_hist, mean_x[0]), gauss_sig(x_axis, x_hist, mean_x[1])]
    )
    sigma_y = np.asarray(
        [gauss_sig(y_axis, y_hist, mean_y[0]), gauss_sig(y_axis, y_hist, mean_y[1])]
    )
    frac_x = clump_fraction(mean_x, sigma_x, x_axis, x_hist)
    frac_y = clump_fraction(mean_y, sigma_y, y_axis, y_hist)

    # Tries to find the GC assuming it is the clump with the smaller dispersion
    if min(sigma_x[0], sigma_y[0]) < min(sigma_x[1], sigma_y[1]):
        mu_pmx_go = mean_x[0]
        mu_pmy_go = mean_y[0]
        sig_pm_go = min(sigma_x[0], sigma_y[0])

        mu_pmx_mw = mean_x[1]
        mu_pmy_mw = mean_y[1]
        sr_pm_mw = max(sigma_x[1], sigma_y[1])
    else:
        mu_pmx_go = mean_x[1]
        mu_pmy_go = mean_y[1]
        sig_pm_go = min(sigma_x[1], sigma_y[1])

        mu_pmx_mw = mean_x[0]
        mu_pmy_mw = mean_y[0]
        sr_pm_mw = max(sigma_x[0], sigma_y[0])

    # Assigns the GC fraction of stars as the minimum one between al the clumps
    frc_go_mw = min(min(frac_x[0], frac_y[0]), min(frac_x[1], frac_y[1]))

    # Returns the initial values for the MLE, assuming the field stars PM slope
    # to begin as -6
    return np.asarray(
        [
            mu_pmx_go,
            mu_pmy_go,
            sig_pm_go,
            mu_pmx_mw,
            mu_pmy_mw,
            2 * sr_pm_mw,
            -6,
            frc_go_mw,
        ]
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Fitting procedure"
# ---------------------------------------------------------------------------


def maximum_likelihood(
    X,
    Y,
    eX=None,
    eY=None,
    eXY=None,
    min_method="dif",
    conv=True,
    hybrid=True,
    values=None,
    bounds=None,
):
    """
    Calls a maximum likelihood fit of the proper motion paramters of
    the joint distribution of galactic object plus Milky Way stars.

    Parameters
    ----------
    X : array_like
        Data in x-direction.
    Y : array_like
        Data in y-direction.
    eX : array_like, optional
        Data uncertainty in x-direction.
        The default is None
    eY : array_like, optional
        Data uncertainty in y-direction.
        The default is None
    eXY : array_like, optional
        Correlation of data uncertainty in x and y-direction.
        The default is None
    min_method : string, optional
        Minimization method to be used by the maximum likelihood fit.
        The default is 'dif'.
    conv : boolean, optional
        True, if the user wants to convolve the galactic object PDF with
        Gaussian errors. The defualt is True.
    hybrid : boolean, optional
        True, if the data contain interlopers.
        The default is True.
    values : array_like, optional
        Array containing some of the parameters already fitted. If not fitted,
        they are filled with np.nan.
        The default is None.
    bounds : array_like, optional
        Bounds used in the MLE fit.
        The default is None.


    Returns
    -------
    results : 10D-array
        Best fit parameters of the proper motion model.
    var : 10D-array
        Uncertainty of the fits.

    """

    if hybrid is True:
        if values is None:
            values = np.zeros(10)
            values[:] = np.nan

        # Gets the initial guess of the parameters
        ini = initial_guess(X, Y)

        if bounds is None:
            ranges = [
                [
                    min(ini[0], ini[3]) - 3 * max(ini[2], ini[5]),
                    max(ini[0], ini[3]) + 3 * max(ini[2], ini[5]),
                ],
                [
                    min(ini[1], ini[4]) - 3 * max(ini[2], ini[5]),
                    max(ini[1], ini[4]) + 3 * max(ini[2], ini[5]),
                ],
            ]
        else:
            ranges = [
                [
                    min(bounds[0][0], bounds[3][0])
                    - 3 * max(bounds[2][1], bounds[5][1]),
                    max(bounds[0][1], bounds[3][1])
                    + 3 * max(bounds[2][1], bounds[5][1]),
                ],
                [
                    min(bounds[1][0], bounds[4][0])
                    - 3 * max(bounds[2][1], bounds[5][1]),
                    max(bounds[1][1], bounds[4][1])
                    + 3 * max(bounds[2][1], bounds[5][1]),
                ],
            ]

        if bounds is None:
            bounds = [
                (ini[0] - 3 * ini[2], ini[0] + 3 * ini[2]),
                (ini[1] - 3 * ini[2], ini[1] + 3 * ini[2]),
                (0.1 * ini[2], 10 * ini[2]),
                (ini[3] - 5 * ini[5], ini[3] + 5 * ini[5]),
                (ini[4] - 5 * ini[5], ini[4] + 5 * ini[5]),
                (0.1 * ini[5], 10 * ini[5]),
                (0.1 * ini[5], 10 * ini[5]),
                (-np.pi / 2, np.pi / 2),
                (-20, -3),
                (0.01, 1),
            ]

        for i in range(len(values)):
            if np.logical_not(np.isnan(values[i])):
                bounds[i] = (
                    values[i] - 1e-6 * np.abs(values[i]),
                    values[i] + 1e-6 * np.abs(values[i]),
                )
                ini[i] = values[i]

    else:
        if values is None:
            values = np.zeros(3)
            values[:] = np.nan

        # Gets the initial guess of the parameters
        ini = np.asarray([np.median(X), np.median(Y), 0.5 * (np.std(X) + np.std(Y))])

        if bounds is None:
            ranges = [
                [
                    ini[0] - 3 * ini[2],
                    ini[0] + 3 * ini[2],
                ],
                [
                    ini[1] - 3 * ini[2],
                    ini[1] + 3 * ini[2],
                ],
            ]
        else:
            ranges = [
                [
                    bounds[0][0] - 3 * bounds[2][1],
                    bounds[0][1] + 3 * bounds[2][1],
                ],
                [
                    bounds[1][0] - 3 * bounds[2][1],
                    bounds[1][1] + 3 * bounds[2][1],
                ],
            ]

        if bounds is None:
            bounds = [
                (ini[0] - 3 * ini[2], ini[0] + 3 * ini[2]),
                (ini[1] - 3 * ini[2], ini[1] + 3 * ini[2]),
                (0.1 * ini[2], 10 * ini[2]),
            ]

        for i in range(len(values)):
            if np.logical_not(np.isnan(values[i])):
                bounds[i] = (
                    values[i] - 1e-6 * np.abs(values[i]),
                    values[i] + 1e-6 * np.abs(values[i]),
                )
                ini[i] = values[i]

    idx_x = np.intersect1d(np.where(X < ranges[0][1]), np.where(X > ranges[0][0]))
    idx_y = np.intersect1d(np.where(Y < ranges[1][1]), np.where(Y > ranges[1][0]))

    idxpm = np.intersect1d(idx_x, idx_y)

    X = X[idxpm]
    Y = Y[idxpm]

    if conv is False:
        eX = np.zeros(len(X))
        eY = np.zeros(len(X))
        eXY = np.zeros(len(X))
    else:
        if eX is None:
            eX = np.zeros(len(X))
        else:
            eX = eX[idxpm]

        if eY is None:
            eY = np.zeros(len(X))
        else:
            eY = eY[idxpm]

        if eXY is None:
            eXY = np.zeros(len(X))
        else:
            eXY = eXY[idxpm]

    if hybrid is True:
        if min_method == "dif":
            mle_model = differential_evolution(
                lambda c: likelihood_function(c, X, Y, eX, eY, eXY, values=values),
                bounds,
            )
            results = mle_model.x

        else:

            ini = np.asarray(
                [
                    ini[0],
                    ini[1],
                    ini[2],
                    ini[3],
                    ini[4],
                    ini[5],
                    ini[5],
                    0,
                    ini[6],
                    ini[7],
                ]
            )
            mle_model = minimize(
                lambda c: likelihood_function(c, X, Y, eX, eY, eXY, values=values),
                ini,
                method=min_method,
                bounds=bounds,
            )
            results = mle_model["x"]

        hfun = ndt.Hessian(
            lambda c: likelihood_function(c, X, Y, eX, eY, eXY, values=values),
            full_output=True,
        )

        hessian_ndt, info = hfun(results)
        for i in range(len(values) - 1, -1, -1):
            if np.logical_not(np.isnan(values[i])):
                hessian_ndt = np.delete(hessian_ndt, i, axis=1)
                hessian_ndt = np.delete(hessian_ndt, i, axis=0)
                results[i] = values[i]
        var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    else:
        mle_model = differential_evolution(
            lambda c: likelihood_gauss2d(c, X, Y, eX, eY, eXY, values=values), bounds
        )
        results = mle_model.x

        hfun = ndt.Hessian(
            lambda c: likelihood_gauss2d(c, X, Y, eX, eY, eXY, values=values),
            full_output=True,
        )
        hessian_ndt, info = hfun(results)
        for i in range(len(values) - 1, -1, -1):
            if np.logical_not(np.isnan(values[i])):
                hessian_ndt = np.delete(hessian_ndt, i, axis=1)
                hessian_ndt = np.delete(hessian_ndt, i, axis=0)
                results[i] = values[i]
        var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    return results, var


def gauss_likelihood(X, eX=None, conv=True, hybrid=False):
    """
    Calls a maximum likelihood fit of two (or one) Gaussian 1D fields.

    Parameters
    ----------
    X : array_like
        Data in x-direction.
    eX : array_like, optional
        Data uncertainty in x-direction.
        The default is None
    conv : boolean, optional
        True, if the user wants to convolve the galactic object PDF with
        Gaussian errors. The defualt is True.
    hybrid : boolean, optional
        True, if the data contain interlopers.
        The default is True.


    Returns
    -------
    results : 10D-array
        Best fit parameters of the proper motion model.
    var : 10D-array
        Uncertainty of the fits.

    """

    if hybrid is True:
        # Gets the initial guess of the parameters

        x_nan = np.logical_not(np.isnan(X))
        x_data = X[x_nan]

        q_16, q_50, q_84 = quantile(X, [0.16, 0.5, 0.84])
        q_m, q_p = q_50 - q_16, q_84 - q_50
        bins_x = int((np.amax(x_data) - np.amin(x_data)) / (min(q_m, q_p) / 4))

        x_hist, x_axis = np.histogram(
            x_data, bins=bins_x, range=(np.amin(x_data), np.amax(x_data))
        )
        x_axis = 0.5 * (x_axis[1:] + x_axis[:-1])

        peaks, _ = find_peaks(x_hist)

        maxx = x_hist[peaks]
        argx = x_axis[peaks]

        # Sorts the values according to R_proj
        L = sorted(zip(maxx, argx), key=operator.itemgetter(0), reverse=True)
        maxx, argx = zip(*L)

        peaks_arg = np.asarray([argx[0], argx[1]])
        peaks_val = np.asarray([maxx[0], maxx[1]])

        sigs = np.asarray(
            [
                gauss_sig(x_axis, x_hist, peaks_arg[0]),
                gauss_sig(x_axis, x_hist, peaks_arg[1]),
            ]
        )

        flux = np.asarray(
            [
                peaks_val[0] * sigs[0] * np.sqrt(2 * np.pi),
                peaks_val[1] * sigs[1] * np.sqrt(2 * np.pi),
            ]
        )

        index_go = np.argmin(sigs)
        index_mw = np.argmax(sigs)

        ini = np.asarray(
            [
                peaks_arg[index_go],
                sigs[index_go],
                peaks_arg[index_mw],
                sigs[index_mw],
                flux[index_go] / (flux[index_go] + flux[index_mw]),
            ]
        )

        bounds = [
            (ini[0] - 3 * ini[1], ini[0] + 3 * ini[1]),
            (0.1 * ini[1], 10 * ini[1]),
            (ini[2] - 5 * ini[3], ini[2] + 5 * ini[3]),
            (0.1 * ini[3], 10 * ini[3]),
            (0.01, 1),
        ]

        ranges = [
            min(ini[0], ini[2]) - 3 * max(ini[1], ini[3]),
            max(ini[0], ini[2]) + 3 * max(ini[1], ini[3]),
        ]
    else:
        # Gets the initial guess of the parameters
        ini = np.asarray([np.median(X), np.std(X)])

        bounds = [
            (ini[0] - 3 * ini[1], ini[0] + 3 * ini[1]),
            (0.1 * ini[1], 10 * ini[1]),
        ]

        ranges = [ini[0] - 3 * ini[1], ini[0] + 3 * ini[1]]

    idx_x = np.intersect1d(np.where(X < ranges[1]), np.where(X > ranges[0]))

    X = X[idx_x]

    if conv is False:
        eX = np.zeros(len(X))
    else:
        if eX is None:
            eX = np.zeros(len(X))
        else:
            eX = eX[idx_x]

    if hybrid is True:

        mle_model = differential_evolution(
            lambda c: likelihood_2gauss1d(c, X, eX), bounds
        )
        results = mle_model.x

        hfun = ndt.Hessian(lambda c: likelihood_2gauss1d(c, X, eX), full_output=True)
        hessian_ndt, info = hfun(results)
        var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))
    else:
        mle_model = differential_evolution(
            lambda c: likelihood_1gauss1d(c, X, eX), bounds
        )
        results = mle_model.x

        hfun = ndt.Hessian(lambda c: likelihood_1gauss1d(c, X, eX), full_output=True)
        hessian_ndt, info = hfun(results)
        var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    return results, var


def mcmc(
    X,
    Y,
    eX=None,
    eY=None,
    eXY=None,
    conv=True,
    nwalkers=None,
    steps=1000,
    ini=None,
    bounds=None,
    use_pool=False,
):
    """
    MCMC routine based on the emcee package (Foreman-Mackey et al, 2013).

    The user is strongly encouraged to provide initial guesses,
    which can be derived with the "maximum_likelihood" method,
    previously checked to provide reasonable fits.

    Parameters
    ----------
    X : array_like
        Data in x-direction.
    Y : array_like
        Data in y-direction.
    eX : array_like, optional
        Data uncertainty in x-direction.
        The default is None
    eY : array_like, optional
        Data uncertainty in y-direction.
        The default is None
    eXY : array_like, optional
        Correlation of data uncertainty in x and y-direction.
        The default is None
    conv : boolean, optional
        True, if the user wants to convolve the galactic object PDF with
        Gaussian errors. The defualt is True.
    nwalkers : int, optional
        Number of Markov chains. The default is None.
    steps : int, optional
        Number of steps for each chain. The default is 1000.
    ini : array_like, optional
        Array containing the initial guess of the parameters.
        The order of parameters should be the same returned by the method
        "maximum_likelihood".
        The default is None.
    bounds : array_like, optional
        Array containing the allowed variation of the parameters, with
        respect to the initial guesses.
        The order of parameters should be the same returned by the method
        "maximum_likelihood".
        The default is None.
    use_pool : boolean, optional
        "True", if the user whises to use full CPU power of the machine.
        The default is False.

    Returns
    -------
    chain : array_like
        Set of chains from the MCMC.

    """

    if ini is None:
        ini = initial_guess(X, Y)
        ini = np.asarray(
            [ini[0], ini[1], ini[2], ini[3], ini[4], ini[5], ini[5], 0, ini[6], ini[7]]
        )

    if bounds is None:

        bounds = np.asarray(
            [
                ini[2],
                ini[2],
                ini[2] * 0.5,
                ini[5],
                ini[6],
                ini[5] * 0.5,
                ini[6] * 0.5,
                np.pi / 2,
                1,
                ini[9] * 0.1,
            ]
        )

    ndim = len(ini)  # number of dimensions.
    if nwalkers is None or nwalkers < 2 * ndim:
        nwalkers = int(2 * ndim + 1)

    pos = [ini + 1e-3 * bounds * np.random.randn(ndim) for i in range(nwalkers)]

    if conv is False:
        eX = np.zeros(len(X))
        eY = np.zeros(len(X))
        eXY = np.zeros(len(X))
    else:
        if eX is None:
            eX = np.zeros(len(X))

        if eY is None:
            eY = np.zeros(len(X))

        if eXY is None:
            eXY = np.zeros(len(X))

    if use_pool:

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                likelihood_prob,
                args=(X, Y, eX, eY, eXY, ini, bounds),
                pool=pool,
            )
            sampler.run_mcmc(pos, steps)
    else:

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, likelihood_prob, args=(X, Y, eX, eY, eXY, ini, bounds)
        )
        sampler.run_mcmc(pos, steps)

    chain = sampler.chain

    return chain
