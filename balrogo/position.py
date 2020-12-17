"""
Created on 2020

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2020, Paris
#
# This file contains the main functions concerning positional information.
# It provides MCMC and maximum likelihood fits of surface density,
# as well as robust initial guesses for the (RA,Dec) center of the source.
#
# Documentation is provided on Vitral & Macedo, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

from . import angle
import numpy as np
from skimage.feature import peak_local_max
from scipy.optimize import differential_evolution
from scipy.special import gamma, gammainc, kn
import numdifftools as ndt
import emcee
from multiprocessing import Pool
from multiprocessing import cpu_count

ncpu = cpu_count()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Peak position"
# ---------------------------------------------------------------------------


def gauss_sig(x_axis, gauss, peak):
    """
    This function estimates the dispersion of a Gaussian clump.

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
    while gauss[i] >= threshold:
        i -= 1
    index_left = i
    while gauss[j] >= threshold:
        j += 1
    index_right = j

    # Assigns sigma as the minimum between the distance took to depass the
    # threshold in the left and in the right of the peak
    sigma = min(x_axis[index_right] - peak, peak - x_axis[index_left])

    return sigma


def find_center(x, y, method="mle"):
    """
    Fit a center (peak) of the [x,y] data.

    Parameters
    ----------
    x : array_like
        Data in x-direction
    y : array_like
        Data in y-direction
    method : string, optional
        Method to find the peak of RA, Dec. Available options are:
            - 'iterative'.
            - 'mle'.
        Default is 'mle'.

    Raises
    ------
    ValueError
        Method argument is not one of the following:
            - 'kde'
            - 'iterative'

    Returns
    -------
    center : 2D-array of floats
        Position of the peak: [x_coordinate,y_coordinate]
    unc : float
        Uncertainty in the position.

    """

    if method not in ["iterative", "mle"]:
        raise ValueError("Does not recognize method argument.")

    # Takes off NaN values
    x = x[np.logical_not(np.isnan(x))]
    y = y[np.logical_not(np.isnan(y))]

    if method == "iterative":
        center, unc = center_iterative(x, y)
    elif method == "mle":
        center, unc = center_mle(x, y)

    return center, unc


def center_iterative(x, y):
    """
    Fit a center (peak) of the [x,y] data through an iterative approach.

    Parameters
    ----------
    x : array_like
        Data in x-direction
    y : array_like
        Data in y-direction

    Returns
    -------
    center : 2D-array of floats
        Position of the peak: [x_coordinate,y_coordinate]
    unc : float
        Uncertainty in the position.

    """

    bins_x = good_bin(x)
    bins_y = good_bin(y)

    # Gets the histogram in RA
    x_hist, x_axis = np.histogram(x, bins=bins_x, range=(np.amin(x), np.amax(x)))
    x_axis = 0.5 * (x_axis[1:] + x_axis[:-1])

    # Gets the histogram in Dec
    y_hist, y_axis = np.histogram(y, bins=bins_y, range=(np.amin(y), np.amax(y)))
    y_axis = 0.5 * (y_axis[1:] + y_axis[:-1])

    # Gets the histogram of the 2d (RA,Dec) data
    hist, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y])
    hist = hist.T

    # Estimates the RA and Dec means from the galactic object by taking the
    # main local maxima of the 2d histogram.
    peaks = peak_local_max(hist, num_peaks=1)

    y_peak, x_peak = peaks.T[0], peaks.T[1]
    cmx, cmy = xedges[x_peak], yedges[y_peak]

    sigma_x = np.asarray([gauss_sig(x_axis, x_hist, cmx)])
    sigma_y = np.asarray([gauss_sig(y_axis, y_hist, cmy)])

    sigma = np.nanmean(np.asarray([sigma_x, sigma_y]))

    center = np.asarray([cmx, cmy])
    unc = np.nan

    many_tracers = True
    shift = list()
    count = 0
    while many_tracers is True:

        idx = np.where(
            angle.sky_distance_deg(x, y, center[0], center[1]) < (0.9 ** count) * sigma
        )

        bins_x = good_bin(x[idx])
        bins_y = good_bin(y[idx])

        if len(idx[0]) < bins_x * bins_y and count > 0:

            many_tracers = False
            return center, unc

        hist, xedges, yedges = np.histogram2d(x[idx], y[idx], bins=[bins_x, bins_y])

        xedges = 0.5 * (xedges[1:] + xedges[:-1])
        yedges = 0.5 * (yedges[1:] + yedges[:-1])

        cmx = np.nansum(xedges * np.sum(hist, axis=1)) / np.nansum(np.sum(hist, axis=1))
        cmy = np.nansum(yedges * np.sum(hist, axis=0)) / np.nansum(np.sum(hist, axis=0))

        shift.append(angle.sky_distance_deg(cmx, cmy, center[0], center[1]))

        center = np.asarray([cmx, cmy])
        unc = np.median(shift)

        count += 1

    return center, unc


def center_mle(x, y):
    """
    Fit a center (peak) of the [x,y] data through an iterative approach.

    Parameters
    ----------
    x : array_like
        Data in x-direction
    y : array_like
        Data in y-direction

    Returns
    -------
    center : 2D-array of floats
        Position of the peak: [x_coordinate,y_coordinate]
    unc : float
        Uncertainty in the position.

    """

    cmx, cmy = quantile(x, 0.5), quantile(y, 0.5)
    hmr, norm = initial_guess_sd(x=x, y=y, x0=cmx, y0=cmy)

    bounds = [
        (hmr - 2, hmr + 2),
        (norm - 2, norm + 2),
        (quantile(x, 0.16), quantile(x, 0.84)),
        (quantile(y, 0.16), quantile(y, 0.84)),
    ]

    mle_model = differential_evolution(
        lambda c: likelihood_plummer_freec(c, x, y), bounds
    )
    results = mle_model.x
    hfun = ndt.Hessian(lambda c: likelihood_plummer_freec(c, x, y), full_output=True)

    hessian_ndt, info = hfun(results)
    var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    center = np.asarray([results[2], results[3]])
    unc = np.asarray([var[2], var[3]])

    return center, unc


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Surface density"
# ---------------------------------------------------------------------------


def surface_density(x=None, y=None, x0=None, y0=None):
    """
    Binned surface density derivation.

    Parameters
    ----------
    x : array_like, optional
        Data in x-direction. The default is None.
    y : array_like, optional
        Data in x-direction. The default is None.
    x0 : float, optional
        Peak of data in x-direction. The default is None.
    y0 : TYPE, optional
        Peak of data in y-direction. The default is None.

    Raises
    ------
    ValueError
        No data is provided.

    Returns
    -------
    density : array_like
        Binned array containing, at positions:
            0 : Projected radius.
            1 : Surface density.
            2 : Poisson error on surface density.
            3 : Size inbetween projected radii.

    """

    if (x is None and y is None) or (x is None):
        raise ValueError("Please provide the data to be fitted.")

    if y is None:
        r = x
    else:

        if x0 is None or y0 is None:

            center, unc = find_center(x, y)
            if x0 is None:
                x0 = center[0]
            if y0 is None:
                y0 = center[1]

        r = angle.sky_distance_deg(x, y, x0, y0)

    q_16, q_50, q_84 = quantile(r, [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50
    nbins = int((np.amax(r) - np.amin(r)) / (min(q_m, q_p) / 5))

    counts, binlim = np.histogram(
        r,
        range=(np.amin(r), np.amax(r)),
        bins=np.logspace(np.log10(np.amin(r)), np.log10(np.amax(r)), nbins + 1),
    )

    bincent = 0.5 * (binlim[1:] + binlim[:-1])
    barsize = binlim[1:] - binlim[:-1]
    surface = np.pi * (binlim[1:] ** 2 - binlim[:-1] ** 2)
    s_dens = counts / surface
    errs = np.zeros(len(counts))
    for i in range(len(counts)):
        if counts[i] == 0:
            errs[i] = np.nan
        else:
            errs[i] = s_dens[i] / np.sqrt(counts[i])

    s_dens[np.where(counts == 0)] = np.nan

    density = np.zeros((4, nbins))
    density[0] = bincent
    density[1] = s_dens
    density[2] = errs
    density[3] = barsize

    return density


def initial_guess_sd(x=None, y=None, x0=None, y0=None):
    """
    Initial guess of projected half mass radius.

    Parameters
    ----------
    x : array_like, optional
        Data in x-direction. The default is None.
    y : array_like, optional
        Data in x-direction. The default is None.
    x0 : float, optional
        Peak of data in x-direction. The default is None.
    y0 : float, optional
        Peak of data in y-direction. The default is None.

    Raises
    ------
    ValueError
        No data is provided.

    Returns
    -------
    half_radius : float
        Guess on projected half mass radius.

    """
    if (x is None and y is None) or (x is None):
        raise ValueError("Please provide the data to be fitted.")

    if y is None:
        density = surface_density(x=x)
        size = len(x[0])
    else:

        if x0 is None or y0 is None:

            center, unc = find_center(x, y)
            if x0 is None:
                x0 = center[0]
            if y0 is None:
                y0 = center[1]

        density = surface_density(x=x, y=y, x0=x0, y0=y0)
        size = len(x)

    mw_dens = np.nanmin(density[1])
    density[1] = density[1] - mw_dens
    nilop = (
        mw_dens * np.pi * (density[0][len(density[0]) - 1] ** 2 - density[0][0] ** 2)
    )
    n_local = np.zeros(len(density[0]))
    for i in range(len(density[0])):
        n_local[i] = (
            np.nansum(n_local)
            + 2 * np.pi * density[1][i] * density[0][i] * density[3][i]
        )

    idx = np.nanargmin(np.abs(n_local - n_local[len(n_local) - 1] * 0.5))

    half_radius = density[0][idx]
    norm = size / (size - nilop) - 1
    if norm <= 0:
        norm = 1

    return half_radius, norm


def prob(r, params, model="plummer"):
    """
    Computes the membership probability of the stars based
    on the surface density fits alone.

    Parameters
    ----------
    r : array_like, optional
        Projected radii (in same units as the scale radius).
    params : array_like
        Parameters to be fitted: Output of the maximum_likelihood method.
    model : string, optional
        Surface density model to be considered. Available options are:
             - 'sersic'
             - 'kazantzidis'
             - 'plummer'
        The default is 'plummer'.

    Raises
    ------
    ValueError
        Surface density model is not one of the following:
            - 'sersic'
            - 'kazantzidis'
            - 'plummer'

    Returns
    -------
    probability : array_like
        Probability of a each star to belong to the respective
        a galactic object (considering only proper motions).

    """

    if model not in ["sersic", "plummer", "kazantzidis"]:
        raise ValueError("Does not recognize surface density model.")

    if model == "plummer" or model == "kazantzidis":
        n = 0
        a = 10 ** params[0]
        if params[1] < -10:
            norm = 0
        else:
            norm = 10 ** params[1]
    elif model == "sersic":
        n = params[0]
        a = 10 ** params[1]
        if params[2] < -10:
            norm = 0
        else:
            norm = 10 ** params[2]

    nsys = len(r) / (1 + norm)
    nilop = len(r) - nsys

    Xmax = np.amax(r) / a
    Xmin = np.amin(r) / a
    X = r / a

    if model == "plummer":
        sd = sd_plummer(X) * nsys / (np.pi * a ** 2)
    elif model == "kazantzidis":
        sd = sd_kazantzidis(X) * nsys / (np.pi * a ** 2)
    elif model == "sersic":
        sd = sd_sersic(n, X) * nsys / (np.pi * a ** 2)

    sd_fs = nilop / (np.pi * (Xmax ** 2 - Xmin ** 2))

    probability = sd / (sd + sd_fs)

    return probability


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Sersic profile functions"
# ---------------------------------------------------------------------------

###############################################################################
#
# Functions concerning the Sersic profile (Sersic 1963, 1968).
#
###############################################################################


def b(n):
    """
    Gets the Sersic's b_n term, using the approximation
    from Ciotti & Bertin (1999).

    Parameters
    ----------
    n : array_like, float
        Sersic index

    Returns
    -------
    b : array_like, float
        Sersic's b_n term.

    """

    b = (
        2 * n
        - 1 / 3
        + 4 / (405 * n)
        + 46 / (25515 * n ** 2)
        + 131 / (1148175 * n ** 3)
        - 2194697 / (30690717750 * n ** 4)
    )
    return b


def sd_sersic(n, X):
    """
    Sersic surface density, normalized according to the convention:

    SD(X = R/R_e) = SD_real(R) * pi * R_e^2 / N_infinty

    Parameters
    ----------
    n : array_like, float
        Sersic index.
    X : array_like (same shape as n), float
        Projected radius X = R/R_e.

    Returns
    -------
    SD : array_like (same shape as n), float
        Normalized surface density profile.

    """

    bn = b(n)
    sd = np.exp(-bn * X ** (1 / n)) * (bn ** (2 * n) / (2 * n * gamma(2 * n)))
    return sd


def n_sersic(n, X):
    """
    Sersic projected number, normalized according to the convention:

    N(X = R/R_e) = N(R) / N_infinty

    Parameters
    ----------
    n : array_like, float
        Sersic index.
    X : array_like (same shape as n), float
        Projected radius X = R/R_e.

    Returns
    -------
    N : array_like (same shape as n), float
        Sersic projected number.

    """

    bn = b(n)
    N = gammainc(2 * n, bn * X ** (1 / n))
    return N


def likelihood_sersic(params, Ri):
    """
    Likelihood function of the Sersic profile plus a constant contribution
    from fore/background tracers.

    Parameters
    ----------
    Parameters to be fitted: Sersic index, Sersic characteristic radius R_e and
                             log-ratio of galactic objects and Milky
                             Way stars.
    Ri : array_like
        Array containing the ensemble of projected radii.

    Returns
    -------
    L : float
       Likelihood.

    """

    n = params[0]
    Re = 10 ** params[1]
    if params[2] < -10:
        norm = 0
    else:
        norm = 10 ** params[2]

    Xmax = np.amax(Ri) / Re
    Xmin = np.amin(Ri) / Re
    X = Ri / Re

    N_sys_tot = n_sersic(n, Xmax) - n_sersic(n, Xmin)

    SD = sd_sersic(n, X) + norm * N_sys_tot / (Xmax ** 2 - Xmin ** 2)

    Ntot = N_sys_tot * (1 + norm)

    fi = 2 * (X / Re) * SD / Ntot

    idx_valid = np.logical_not(np.isnan(np.log(fi)))

    L = -np.sum(np.log(fi[idx_valid]))

    return L


def lnprior_s(params, guess, bounds):
    """
    Prior assumptions on the parameters.

    Parameters
    ----------
    Parameters to be fitted: Sersic index, Sersic characteristic radius R_e and
                             log-ratio of galactic objects and Milky
                             Way stars.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.


    Returns
    -------
    log-prior probability : float
        0, if the parameters are within the prior range,
        - Infinity otherwise.

    """

    if (
        (guess[0] - bounds[0] <= params[0] <= guess[0] + bounds[0])
        and (guess[1] - bounds[1] <= params[1] <= guess[1] + bounds[1])
        and (guess[2] - bounds[2] <= params[2] <= guess[2] + bounds[2])
    ):
        return 0.0
    return -np.inf


def lnprob_s(params, Ri, guess, bounds):
    """
    log-probability of fit parameters.

    Parameters
    ----------
    Parameters to be fitted: Sersic index, Sersic characteristic radius R_e and
                             log-ratio of galactic objects and Milky
                             Way stars.
    Ri : array_like
        Array containing the ensemble of projected radii.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.


    Returns
    -------
    log-prior probability : float
        log-probability of fit parameters.

    """

    lp = lnprior_s(params, guess, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp - likelihood_sersic(params, Ri)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Kazantzidis profile functions"
# ---------------------------------------------------------------------------

###############################################################################
#
# Functions concerning the Kazantzidis profile (Kazantzidis et al. 2004).
#
###############################################################################


def sd_kazantzidis(X):
    """
    Kazantzidis surface density, normalized according to the convention:

    SD(X = R/a) = SD_real(R) * pi * a^2 / N_infinty

    Parameters
    ----------
    X : array_like (same shape as n), float
        Projected radius X = R/a

    Returns
    -------
    SD : array_like (same shape as X), float
        Normalized surface density profile.

    """

    sd = kn(0, X) / 2
    return sd


def n_kazantizidis(X):
    """
    Sersic projected number, normalized according to the convention:

    N(X = R/a) = N(R) / N_infinty

    Parameters
    ----------
    X : array_like (same shape as n), float
        Projected radius X = R/a.

    Returns
    -------
    N : array_like (same shape as n), float
        Sersic projected number.

    """

    N = 1 - X * kn(1, X)
    return N


def likelihood_kazantzidis(params, Ri):
    """
    Likelihood function of the Sersic profile plus a constant contribution
    from fore/background tracers.

    Parameters
    ----------
    Parameters to be fitted: Kazantzidis characteristic radius a and
                             log-ratio of galactic objects and Milky
                             Way stars.
    Ri : array_like
        Array containing the ensemble of projected radii.

    Returns
    -------
    L : float
       Likelihood.

    """

    a = 10 ** params[0]
    if params[1] < -10:
        norm = 0
    else:
        norm = 10 ** params[1]

    Xmax = np.amax(Ri) / a
    Xmin = np.amin(Ri) / a
    X = Ri / a

    N_sys_tot = n_kazantizidis(Xmax) - n_kazantizidis(Xmin)

    SD = sd_kazantzidis(X) + norm * N_sys_tot / (Xmax ** 2 - Xmin ** 2)

    Ntot = N_sys_tot * (1 + norm)

    fi = 2 * (X / a) * SD / Ntot

    idx_valid = np.logical_not(np.isnan(np.log(fi)))

    L = -np.sum(np.log(fi[idx_valid]))

    return L


def lnprior_k(params, guess, bounds):
    """
    Prior assumptions on the parameters.

    Parameters
    ----------
    Parameters to be fitted: Kazantzidis characteristic radius a and
                             log-ratio of galactic objects and Milky
                             Way stars.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.

    Returns
    -------
    log-prior probability : float
        0, if the parameters are within the prior range,
        - Infinity otherwise.

    """

    if (guess[0] - bounds[0] <= params[0] <= guess[0] + bounds[0]) and (
        guess[1] - bounds[1] <= params[1] <= guess[1] + bounds[1]
    ):
        return 0.0
    return -np.inf


def lnprob_k(params, Ri, guess, bounds):
    """
    log-probability of fit parameters.

    Parameters
    ----------
        Parameters to be fitted: Kazantzidis characteristic radius a and
                                 log-ratio of galactic objects and Milky
                                 Way stars.
    Ri : array_like
        Array containing the ensemble of projected radii.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.

    Returns
    -------
    log-prior probability : float
        log-probability of fit parameters.

    """

    lp = lnprior_k(params, guess, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp - likelihood_kazantzidis(params, Ri)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Plummer profile functions"
# ---------------------------------------------------------------------------

###############################################################################
#
# Functions concerning the Plummer profile (Plummer 1911).
#
###############################################################################


def sd_plummer(X):
    """
    Plummer surface density, normalized according to the convention:

    SD(X = R/a) = SD_real(R) * pi * a^2 / N_infinty

    Parameters
    ----------
    X : array_like (same shape as n), float
        Projected radius X = R/a

    Returns
    -------
    SD : array_like (same shape as X), float
        Normalized surface density profile.

    """

    sd = 1 / ((1 + X * X) * (1 + X * X))
    return sd


def n_plummer(X):
    """
    Plummer projected number, normalized according to the convention:

    N(X = R/a) = N(R) / N_infinty

    Parameters
    ----------
    X : array_like (same shape as n), float
        Projected radius X = R/a.

    Returns
    -------
    N : array_like (same shape as n), float
        Sersic projected number.

    """

    N = X * X / (1 + X * X)
    return N


def likelihood_plummer(params, Ri):
    """
    Likelihood function of the Plummer profile plus a constant contribution
    from fore/background tracers.

    Parameters
    ----------
    params : array_like
        Parameters to be fitted: Plummer characteristic radius a and
                                 log-ratio of galactic objects and Milky
                                 Way stars.
    Ri : array_like
        Array containing the ensemble of projected radii.

    Returns
    -------
    L : float
       Likelihood.

    """

    a = 10 ** params[0]
    if params[1] < -10:
        norm = 0
    else:
        norm = 10 ** params[1]

    Xmax = np.amax(Ri) / a
    Xmin = np.amin(Ri) / a
    X = Ri / a

    N_sys_tot = n_plummer(Xmax) - n_plummer(Xmin)

    SD = sd_plummer(X) + norm * N_sys_tot / (Xmax ** 2 - Xmin ** 2)

    Ntot = N_sys_tot * (1 + norm)

    fi = 2 * (X / a) * SD / Ntot

    idx_valid = np.logical_not(np.isnan(np.log(fi)))

    L = -np.sum(np.log(fi[idx_valid]))

    return L


def likelihood_plummer_freec(params, x, y):
    """
    Likelihood function of the Plummer profile plus a constant contribution
    from fore/background tracers. Differently from the previous method,
    it also fits the center of the distribution.

    Parameters
    ----------
    params : array_like
        Parameters to be fitted: Plummer characteristic radius a and
                                 log-ratio of galactic objects and Milky
                                 Way stars.
    x : array_like
        Array containing the ensemble of ra data.
    y : array_like
        Array containing the ensemble of dec data.

    Returns
    -------
    L : float
       Likelihood.

    """

    a = 10 ** params[0]
    if params[1] < -10:
        norm = 0
    else:
        norm = 10 ** params[1]

    cmx = params[2]
    cmy = params[3]

    Ri = angle.sky_distance_deg(cmx, cmy, x, y)

    Xmax = np.amax(Ri) / a
    Xmin = np.amin(Ri) / a
    X = Ri / a

    N_sys_tot = n_plummer(Xmax) - n_plummer(Xmin)

    SD = sd_plummer(X) + norm * N_sys_tot / (Xmax ** 2 - Xmin ** 2)

    Ntot = N_sys_tot * (1 + norm)

    fi = 2 * (X / a) * SD / Ntot

    idx_valid = np.logical_not(np.isnan(np.log(fi)))

    L = -np.sum(np.log(fi[idx_valid]))

    return L


def lnprior_p(params, guess, bounds):
    """
    Prior assumptions on the parameters.

    Parameters
    ----------
    params : array_like
        Parameters to be fitted: Plummer characteristic radius a and
                                 log-ratio of galactic objects and Milky
                                 Way stars.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.

    Returns
    -------
    log-prior probability : float
        0, if the parameters are within the prior range,
        - Infinity otherwise.

    """

    if (guess[0] - bounds[0] <= params[0] <= guess[0] + bounds[0]) and (
        guess[1] - bounds[1] <= params[1] <= guess[1] + bounds[1]
    ):
        return 0.0
    return -np.inf


def lnprob_p(params, Ri, guess, bounds):
    """
    log-probability of fit parameters.

    Parameters
    ----------
    params : array_like
    Parameters to be fitted: Plummer characteristic radius a and
                             log-ratio of galactic objects and Milky
                             Way stars.
    Ri : array_like
        Array containing the ensemble of projected radii.
    guess : array_like
        Array containing the initial guess of the parameters.
    bounds : array_like
        Array containing the interval of variation of the parameters.


    Returns
    -------
    log-prior probability : float
        log-probability of fit parameters.

    """

    lp = lnprior_p(params, guess, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp - likelihood_plummer(params, Ri)


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


def good_bin(x):
    """
    Computes an adequate number of bins.

    Parameters
    ----------
    x : array_like
        Array to be binned.

    Returns
    -------
    bins : float
        Number of bins.

    """

    q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84])
    q_m, q_p = q_50 - q_16, q_84 - q_50
    bins = int((np.amax(x) - np.amin(x)) / (min(q_m, q_p) / 4))

    return bins


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Fitting procedure"
# ---------------------------------------------------------------------------


def maximum_likelihood(x=None, y=None, model="plummer", x0=None, y0=None, hybrid=True):
    """
    Calls a maximum likelihood fit of the surface density paramters of
    the joint distribution of galactic object plus Milky Way stars.

    Parameters
    ----------
    x : array_like, optional
        Data in x-direction. The default is None.
    y : array_like, optional
        Data in x-direction. The default is None.
    model : string, optional
        Surface density model to be considered. Available options are:
             - 'sersic'
             - 'kazantzidis'
             - 'plummer'
        The default is 'plummer'.
    x0 : float, optional
        Peak of data in x-direction. The default is None.
    y0 : TYPE, optional
        Peak of data in y-direction. The default is None.
    hydrid :  boolean, optional
        "True", if the user whises to consider field stars in the fit.
        The default is True.

    Raises
    ------
    ValueError
        Surface density model is not one of the following:
            - 'sersic'
            - 'kazantzidis'
            - 'plummer'
        No data is provided.

    Returns
    -------
    results : array
        Best fit parameters of the surface density model.
    var : array
        Uncertainty of the fits.

    """

    if model not in ["sersic", "plummer", "kazantzidis"]:
        raise ValueError("Does not recognize surface density model.")

    if (x is None and y is None) or (x is None):
        raise ValueError("Please provide the data to be fitted.")

    if y is None:
        ri = x
        hmr, norm = initial_guess_sd(x=x)
    else:

        if x0 is None or y0 is None:

            center, unc = find_center(x, y)
            if x0 is None:
                x0 = center[0]
            if y0 is None:
                y0 = center[1]
        ri = np.asarray([angle.sky_distance_deg(x, y, x0, y0)])
        hmr, norm = initial_guess_sd(x=x, y=y, x0=x0, y0=y0)

    hmr = np.log10(hmr)
    norm = np.log10(norm)
    if hybrid is False:
        norm = -50
    if model == "sersic":
        bounds = [(0.5, 10), (hmr - 2, hmr + 2), (norm - 2, norm + 2)]
        mle_model = differential_evolution(lambda c: likelihood_sersic(c, ri), bounds)
        results = mle_model.x
        hfun = ndt.Hessian(lambda c: likelihood_sersic(c, ri), full_output=True)

    elif model == "kazantzidis":
        bounds = [(hmr - 2, hmr + 2), (norm - 2, norm + 2)]
        mle_model = differential_evolution(
            lambda c: likelihood_kazantzidis(c, ri), bounds
        )
        results = mle_model.x
        hfun = ndt.Hessian(lambda c: likelihood_kazantzidis(c, ri), full_output=True)

    elif model == "plummer":
        bounds = [(hmr - 2, hmr + 2), (norm - 2, norm + 2)]
        mle_model = differential_evolution(lambda c: likelihood_plummer(c, ri), bounds)
        results = mle_model.x
        hfun = ndt.Hessian(lambda c: likelihood_plummer(c, ri), full_output=True)

    hessian_ndt, info = hfun(results)
    var = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

    return results, var


def mcmc(
    x=None,
    y=None,
    model="plummer",
    nwalkers=None,
    steps=1000,
    ini=None,
    bounds=None,
    use_pool=False,
    x0=None,
    y0=None,
    hybrid=True,
):
    """
    MCMC routine based on the emcee package (Foreman-Mackey et al, 2013).

    The user is strongly encouraged to provide initial guesses,
    which can be derived with the "maximum_likelihood" method,
    previously checked to provide reasonable fits.

    Parameters
    ----------
    x : array_like
        Data in x-direction.
    y : array_like
        Data in y-direction.
    model : string, optional
        Surface density model to be considered. Available options are:
             - 'sersic'
             - 'kazantzidis'
             - 'plummer'
        The default is 'plummer'.
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
    x0 : float, optional
        Peak of data in x-direction. The default is None.
    y0 : float, optional
        Peak of data in y-direction. The default is None.
    hydrid :  boolean, optional
        "True", if the user whises to consider field stars in the fit.
        The default is True.

    Raises
    ------
    ValueError
        Surface density model is not one of the following:
            - 'sersic'
            - 'kazantzidis'
            - 'plummer'
        No data is provided.

    Returns
    -------
    chain : array_like
        Set of chains from the MCMC.

    """

    if model not in ["sersic", "plummer", "kazantzidis"]:
        raise ValueError("Does not recognize surface density model.")

    if (x is None and y is None) or (x is None):
        raise ValueError("Please provide the data to be fitted.")

    if y is None:
        ri = x
        if ini is None:
            ini, var = maximum_likelihood(x=x, model=model)
    else:

        if x0 is None or y0 is None:

            center, unc = find_center(x, y)
            if x0 is None:
                x0 = center[0]
            if y0 is None:
                y0 = center[1]
        ri = np.asarray([angle.sky_distance_deg(x, y, x0, y0)])

        if ini is None:
            ini, var = maximum_likelihood(x=x, y=y, x0=x0, y0=y0, model=model)
            if model == "sersic":
                ini = np.asarray([2, ini[0], ini[1]])

    ndim = len(ini)  # number of dimensions.
    if nwalkers is None or nwalkers < 2 * ndim:
        nwalkers = int(2 * ndim + 1)

    if bounds is None:
        bounds = 3 * var

    if hybrid is False:
        if model == "sersic":
            ini[2] = -50
        else:
            ini[1] = -50

    pos = [ini + bounds * np.random.randn(ndim) for i in range(nwalkers)]

    if model == "sersic":
        func = lnprob_s

    elif model == "kazantzidis":
        func = lnprob_k

    elif model == "plummer":
        func = lnprob_p

    if use_pool:

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, func, args=(ri, ini, bounds), pool=pool
            )
            sampler.run_mcmc(pos, steps)
    else:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=(ri, ini, bounds))
        sampler.run_mcmc(pos, steps)

    chain = sampler.chain

    return chain
