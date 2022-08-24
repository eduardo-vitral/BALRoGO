"""
Created on 2020

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2020, Paris
#
# This file contains the main functions concerning the handling of the
# Gaia mission data.
#
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from astropy.io import fits
from . import pm
from . import angle
from . import position
from . import hrd

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Gaia handling"
# ---------------------------------------------------------------------------


def polyRiello20(x):
    """
    Polynomial from Riello et al. 2020. With coefficients from their Table 2.

    Parameters
    ----------
    x : array_like
        x = G_BP − G_RP.

    Returns
    -------
    f : array_like
        BP and RP flux excess factor C dependence on the x = G_BP − G_RP.

    """

    f = np.zeros(len(x))

    for i in range(len(x)):

        if np.isnan(x[i]):
            f[i] = np.nan

        elif x[i] < 0.5:
            a0 = 1.154360
            a1 = 0.033772
            a2 = 0.032277

            f[i] = a0 + a1 * x[i] + a2 * x[i] * x[i]

        elif x[i] >= 0.5 and x[i] < 4.0:
            a0 = 1.162004
            a1 = 0.011464
            a2 = 0.049255
            a3 = -0.005879

            f[i] = a0 + a1 * x[i] + a2 * x[i] * x[i] + a3 * x[i] * x[i] * x[i]

        elif x[i] >= 4.0:
            a0 = 1.057572
            a1 = 0.140537

            f[i] = a0 + a1 * x[i]

    return f


def clean_gaia(
    pmra,
    pmdec,
    epmra,
    epmdec,
    corrpm,
    chi2,
    nu,
    g_mag,
    br_mag,
    br_excess,
    ra,
    dec,
    err_lim=None,
    err_handle="relative",
    object_type="gc",
    r_cut=None,
    c=None,
    cleaning="v21",
):
    """
    Cleans the Gaia data based on astrometric flags.

    Parameters
    ----------
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
    chi2 : array_like
        Gaia designation: astrometric_chi2_al.
    nu : array_like
        Gaia designation: astrometric_n_good_obs_al.
    g_mag : array_like
        Gaia designation: phot_g_mean_mag.
    br_mag : array_like
        Gaia designation: bp_rp.
    br_excess : array_like
        Gaia designation: phot_bp_rp_excess_factor.
    ra : array_like
        Gaia designation: ra.
    dec : array_like
        Gaia designation: dec.
    err_lim : float, optional
        Error threshold for proper motions. The default is None.
    err_handle : string, optional
        Way in which the error limit is handled:
            - 'relative', if the err_lim argument is the number of times
            the initial guess on the galactic object proper motion dispersion.
            - 'absolute', if the err_lim argument is the absolute error
            limit in mas/yr.
        The default is 'relative'.
    object_type : string, optional
        Galactic object type, to be considered when defining the default
        value of err_lim:
            - 'gc', for globular cluster.
            - 'dsph', for dwarf spheroidal galaxy.
        The default is 'gc'.
    r_cut : float, optional
        Projected radius limit. The default is None.
    c : 2D array, optional
        (ra,dec) of the center. The default is None.
    cleaning : string, optional
        Cleaning procedure to be adopted:
            - 'v21', to use the same cleaning from Vitral 2021
            - 'vm21', to use the same cleaning from Vitral & Mamon 21
        The default is 'v21'.

    Raises
    ------
    ValueError
        err_handle is not one of the following:
            - 'relative'
            - 'absolute'
        object_type is not one of the following:
            - 'gc'
            - 'dsph'
        cleaning is not one of the following:
            - 'v21'
            - 'vm21'

    Returns
    -------
    idx : array_like
        Array containing the indexes of the original provided arrays
        which correspond to stars with clean astrometry.

    """

    if err_handle not in ["relative", "absolute"]:
        raise ValueError("Does not recognize method to handle errors.")

    if object_type not in ["gc", "dsph"]:
        raise ValueError("Does not recognize object type.")

    if cleaning not in ["v21", "vm21"]:
        raise ValueError("Does not recognize cleaning procedure.")

    if c is None:
        c, unc = position.find_center(ra, dec)

    if r_cut is None:
        ri = angle.sky_distance_deg(ra, dec, c[0], c[1])
        results, var = position.maximum_likelihood(x=np.asarray([ri]), model="plummer")
        r_cut = 10 * 10 ** results[0]

    idx = np.where(angle.sky_distance_deg(ra, dec, c[0], c[1]) < r_cut)

    if err_lim is None:
        ini = pm.initial_guess(pmra[idx], pmdec[idx])
        if object_type == "gc":
            err_lim = 0.5 * ini[2]
        elif object_type == "dsph":
            err_lim = 5 * ini[2]
    else:
        if err_handle == "relative":
            ini = pm.initial_guess(pmra[idx], pmdec[idx])
            err_lim = err_lim * ini[2]

    if cleaning == "v21":

        u = np.sqrt(chi2 / (nu - 5))

        c33 = epmra * epmra
        c34 = epmra * epmdec * corrpm
        c44 = epmdec * epmdec
        err = np.sqrt(
            0.5 * (c33 + c44) + 0.5 * np.sqrt((c44 - c33) ** 2 + 4 * c34 ** 2)
        )

        idx_err = np.where(err < err_lim)
        idx = np.intersect1d(idx, idx_err)

        f = polyRiello20(br_mag)

        C_new = br_excess - f

        sig_c = 0.0059898 + 8.817481 * 1e-12 * g_mag ** (7.618399)

        idx_noise1 = np.where(np.abs(C_new) < 3 * sig_c)
        idx_noise2 = np.where(u < 1.2 * np.maximum(1, np.exp(-0.2 * (g_mag - 19.5))))

        idx_noise = np.intersect1d(idx_noise1, idx_noise2)

        idx = np.intersect1d(idx, idx_noise)

    elif cleaning == "vm21":

        u = np.sqrt(chi2 / (nu - 5))

        c33 = epmra * epmra
        c34 = epmra * epmdec * corrpm
        c44 = epmdec * epmdec
        err = np.sqrt(
            0.5 * (c33 + c44) + 0.5 * np.sqrt((c44 - c33) ** 2 + 4 * c34 ** 2)
        )

        idx_err = np.where(err < err_lim)
        idx = np.intersect1d(idx, idx_err)

        idx_noiseE1 = np.where(1.0 + 0.015 * br_mag ** 2 < br_excess)
        idx_noiseE2 = np.where(br_excess < 1.3 + 0.06 * br_mag ** 2)
        idx_noise1 = np.intersect1d(idx_noiseE1, idx_noiseE2)
        idx_noise2 = np.where(u < 1.2 * np.maximum(1, np.exp(-0.2 * (g_mag - 19.5))))

        idx_noise = np.intersect1d(idx_noise1, idx_noise2)

        idx = np.intersect1d(idx, idx_noise)

    return idx


def find_object(
    ra,
    dec,
    pmra,
    pmdec,
    epmra,
    epmdec,
    corrpm,
    chi2,
    nu,
    g_mag,
    br_mag,
    br_excess,
    sd_model="plummer",
    min_method="dif",
    prob_method="complete",
    prob_limit=0.9,
    use_hrd=True,
    nsig=3,
    bw_hrd=None,
    r_max=None,
    err_lim=None,
    err_handle="relative",
    object_type="gc",
    return_center=False,
    check_fit=False,
    cleaning="v21",
    conv=True,
):

    """
    Finds galactic object by employing the analysis proposed in
    Vitral 2021.

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
    chi2 : array_like
        Gaia designation: astrometric_chi2_al.
    nu : array_like
        Gaia designation: astrometric_n_good_obs_al.
    g_mag : array_like
        Gaia designation: phot_g_mean_mag.
    br_mag : array_like
        Gaia designation: bp_rp.
    br_excess : array_like
        Gaia designation: phot_bp_rp_excess_factor.
    sd_model : string, optional
        Surface density model to be considered. Available options are:
             - 'sersic'
             - 'kazantzidis'
             - 'plummer'
             - 'test', for testing which among Plummer and Sersic should be
             used, based on AICc.
        The default is 'plummer'.
    min_method : string, optional
        Minimization method to be used by the pm maximum likelihood fit.
        The default is 'dif'.
    prob_method : string, optional
        Method of probability cut chosen. Available options are:
            - complete: Considers proper motions and projected radii.
            - pm: Considers only proper motions.
            - position: Considers only projected radii.
        The default is 'complete'.
    prob_limit : float, optional
        Probability threshold (between 0 and 1) considered to cut
        the data. The default is 0.9.
    use_hrd : boolean, optional
        True if the user wants to select stars inside a n-sigma contour
        region inside the color-magnitude diagram.
        The default is True.
    nsig : float, optional
        Number of sigma up to when consider stars inside the KDE
        color-magnitude diagram.
        The default is 3.3.
    bw_hrd : float or string, optional
        Bandwidth method of KDE's scipy method. The default is None.
    r_max : float, optional
        Maximum projected radius up to where consider the data.
        The default is None
    err_lim : float, optional
        Error threshold for proper motions. The default is None.
    err_handle : string, optional
        Way in which the error limit is handled:
            - 'relative', if the err_lim argument is the number of times
            the initial guess on the galactic object proper motion dispersion.
            - 'absolute', if the err_lim argument is the absolute error
            limit in mas/yr.
        The default is 'relative'.
    object_type : string, optional
        Galactic object type, to be considered when defining the default
        value of err_lim:
            - 'gc', for globular cluster.
            - 'dsph', for dwarf spheroidal galaxy.
        The default is 'gc'.
    return_center : boolean, optional
        True if the user wants an estimate of the center as output.
        The default is False.
    check_fit : boolean, optional
        True is the user wants to plot validity checks throughout the fitting
        procedure.
        The default is False.
    cleaning : string, optional
        Cleaning procedure to be adopted:
            - 'v21', to use the same cleaning from Vitral 2021
            - 'vm21', to use the same cleaning from Vitral & Mamon 21
        The default is 'v21'.
    conv : boolean, optional
        True, if the user wants to convolve the galactic object PDF with
        Gaussian errors. The defualt is True.

    Raises
    ------
    ValueError
        Surface density model is not one of the following:
            - 'sersic'
            - 'kazantzidis'
            - 'plummer'
            - 'test'
        Probability method is not one of the following:
            - 'complete'
            - 'pm'
            - 'position'
        Probability threshold is not inbetween 0 and 1.
        nsig is not positive.
        err_handle is not one of the following:
            - 'relative'
            - 'absolute'
        object_type is not one of the following:
            - 'gc'
            - 'dsph'
        cleaning is not one of the following:
            - 'v21'
            - 'vm21'

    Returns
    -------
    idx_final : array_like
        Array containing the indexes of the stars in original file who have
        passed the probability cuts.
    results_sd : array_like
        Results of the surface density fit.
    var_sd : array_like
        Uncertainties of the surface density fit
    results_pm : array_like
        Results of the proper motion fit.
    var_pm : array_like
        Uncertainties of the proper motion fit.

    """

    if sd_model not in ["sersic", "plummer", "kazantzidis", "test"]:
        raise ValueError("Does not recognize surface density model.")

    if prob_method not in ["complete", "pm", "position"]:
        raise ValueError("Does not recognize probability model.")

    if prob_limit > 1 or prob_limit < 0:
        raise ValueError("Probability threshold must be inbetween 0 and 1.")

    if nsig <= 0:
        raise ValueError("nsig must be positive.")

    if err_handle not in ["relative", "absolute"]:
        raise ValueError("Does not recognize method to handle errors.")

    if object_type not in ["gc", "dsph"]:
        raise ValueError("Does not recognize object type.")

    if cleaning not in ["v21", "vm21"]:
        raise ValueError("Does not recognize cleaning procedure.")

    c, unc = position.find_center(ra, dec)

    ri = angle.sky_distance_deg(ra, dec, c[0], c[1])
    if r_max is None:
        ri_sd = ri
    else:
        ri_sd = ri[np.where(ri <= r_max)]

    if sd_model == "sersic":
        results_sd, var_sd = position.maximum_likelihood(
            x=np.asarray([ri_sd]), model="sersic"
        )
        r_cut = 10 * 10 ** results_sd[1]
        nsys = len(ri_sd) / (1 + 10 ** results_sd[2])
        nilop = len(ri_sd) - nsys
        sd = (
            position.sd_sersic(results_sd[0], sorted(ri_sd) / 10 ** results_sd[1])
            * nsys
            / (np.pi * (10 ** results_sd[1]) ** 2)
        )
        prob_sd = position.prob(ri, results_sd, model="sersic")
    elif sd_model == "plummer":
        results_sd, var_sd = position.maximum_likelihood(
            x=np.asarray([ri_sd]), model="plummer"
        )
        r_cut = 10 * 10 ** results_sd[0]
        nsys = len(ri_sd) / (1 + 10 ** results_sd[1])
        nilop = len(ri_sd) - nsys
        sd = (
            position.sd_plummer(sorted(ri_sd) / 10 ** results_sd[0])
            * nsys
            / (np.pi * (10 ** results_sd[0]) ** 2)
        )
        prob_sd = position.prob(ri, results_sd, model="plummer")
    elif sd_model == "kazantzidis":
        results_sd, var_sd = position.maximum_likelihood(
            x=np.asarray([ri_sd]), model="kazantzidis"
        )
        r_cut = 10 * 10 ** results_sd[0]
        nsys = len(ri_sd) / (1 + 10 ** results_sd[1])
        nilop = len(ri_sd) - nsys
        sd = (
            position.sd_kazantzidis(sorted(ri_sd) / 10 ** results_sd[0])
            * nsys
            / (np.pi * (10 ** results_sd[0]) ** 2)
        )
        prob_sd = position.prob(ri, results_sd, model="kazantzidis")
    elif sd_model == "test":
        results_sd_s, var_sd_s = position.maximum_likelihood(
            x=np.asarray([ri_sd]), model="sersic"
        )
        results_sd_p, var_sd_p = position.maximum_likelihood(
            x=np.asarray([ri_sd]), model="plummer"
        )
        results_sd_k, var_sd_k = position.maximum_likelihood(
            x=np.asarray([ri_sd]), model="kazantzidis"
        )

        aicc_s = get_aicc(
            position.likelihood_sersic(results_sd_s, ri_sd), 3, len(ri_sd)
        )
        aicc_p = get_aicc(
            position.likelihood_plummer(results_sd_p, ri_sd), 2, len(ri_sd)
        )
        aicc_k = get_aicc(
            position.likelihood_kazantzidis(results_sd_k, ri_sd), 2, len(ri_sd)
        )

        delta_aicc = aicc_p - aicc_s

        if delta_aicc < 2:
            delta_aicc = aicc_p - aicc_k
            if delta_aicc < 2:
                sd_model = "plummer"
                results_sd, var_sd = results_sd_p, var_sd_p
                r_cut = 10 * 10 ** results_sd[0]
                nsys = len(ri_sd) / (1 + 10 ** results_sd[1])
                nilop = len(ri_sd) - nsys
                sd = (
                    position.sd_plummer(sorted(ri_sd) / 10 ** results_sd[0])
                    * nsys
                    / (np.pi * (10 ** results_sd[0]) ** 2)
                )
                prob_sd = position.prob(ri, results_sd, model="plummer")
            else:
                sd_model = "kazantzidis"
                results_sd, var_sd = results_sd_k, var_sd_k
                r_cut = 10 * 10 ** results_sd[0]
                nsys = len(ri_sd) / (1 + 10 ** results_sd[1])
                nilop = len(ri_sd) - nsys
                sd = (
                    position.sd_kazantzidis(sorted(ri_sd) / 10 ** results_sd[0])
                    * nsys
                    / (np.pi * (10 ** results_sd[0]) ** 2)
                )
                prob_sd = position.prob(ri, results_sd, model="kazantzidis")
        else:
            delta_aicc = aicc_k - aicc_s
            if delta_aicc < 2:
                sd_model = "kazantzidis"
                results_sd, var_sd = results_sd_k, var_sd_k
                r_cut = 10 * 10 ** results_sd[0]
                nsys = len(ri_sd) / (1 + 10 ** results_sd[1])
                nilop = len(ri_sd) - nsys
                sd = (
                    position.sd_kazantzidis(sorted(ri_sd) / 10 ** results_sd[0])
                    * nsys
                    / (np.pi * (10 ** results_sd[0]) ** 2)
                )
                prob_sd = position.prob(ri, results_sd, model="kazantzidis")
            else:
                sd_model = "sersic"
                results_sd, var_sd = results_sd_s, var_sd_s
                r_cut = 10 * 10 ** results_sd[1]
                nsys = len(ri_sd) / (1 + 10 ** results_sd[2])
                nilop = len(ri_sd) - nsys
                sd = (
                    position.sd_sersic(
                        results_sd[0], sorted(ri_sd) / 10 ** results_sd[1]
                    )
                    * nsys
                    / (np.pi * (10 ** results_sd[1]) ** 2)
                )
                prob_sd = position.prob(ri, results_sd, model="sersic")

    if check_fit is True:
        surf_dens = position.surface_density(x=ri_sd)

        fs = nilop / (np.pi * np.amax(ri_sd) ** 2)

        plt.title(sd_model)
        plt.bar(
            surf_dens[0],
            surf_dens[1],
            surf_dens[3],
            ec="k",
            color=(102 / 255, 178 / 255, 1),
            alpha=0.3,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Surface density")
        plt.xlabel("Projected radii")
        plt.errorbar(
            surf_dens[0],
            surf_dens[1],
            color="black",
            yerr=surf_dens[2],
            capsize=3,
            ls="none",
            barsabove=True,
            zorder=11,
        )
        plt.plot(sorted(ri_sd), sd + fs, lw=3, color="red")
        plt.show()

    if r_max is None:
        r_max = r_cut

    idx = clean_gaia(
        pmra,
        pmdec,
        epmra,
        epmdec,
        corrpm,
        chi2,
        nu,
        g_mag,
        br_mag,
        br_excess,
        ra,
        dec,
        c=c,
        r_cut=r_max,
        err_lim=err_lim,
        err_handle=err_handle,
        object_type=object_type,
        cleaning=cleaning,
    )

    prob_sd = prob_sd[idx]
    pmra = pmra[idx]
    pmdec = pmdec[idx]
    epmra = epmra[idx]
    epmdec = epmdec[idx]
    corrpm = corrpm[idx]

    results_pm, var_pm = pm.maximum_likelihood(
        pmra, pmdec, eX=epmra, eY=epmdec, eXY=corrpm, min_method=min_method, conv=conv
    )

    prob_pm = pm.prob(pmra, pmdec, epmra, epmdec, corrpm, results_pm, conv=conv)

    if check_fit is True:
        ellipse_rot = angle.get_ellipse(
            results_pm[5], results_pm[6], results_pm[7], 200
        )
        circle = angle.get_ellipse(results_pm[2], results_pm[2], 0, 200)

        ranges = [
            [pm.quantile(pmra, 0.025)[0], pm.quantile(pmra, 0.975)[0]],
            [pm.quantile(pmdec, 0.025)[0], pm.quantile(pmdec, 0.975)[0]],
        ]

        plt.xlabel("pmra")
        plt.ylabel("pmdec")
        plt.xlim(ranges[0][0], ranges[0][1])
        plt.ylim(ranges[1][0], ranges[1][1])
        plt.hist2d(
            pmra,
            pmdec,
            range=ranges,
            bins=(position.good_bin(pmra), position.good_bin(pmdec)),
            cmap="hot",
            norm=mc.LogNorm(),
        )
        plt.plot(
            results_pm[3] + ellipse_rot[0, :],
            results_pm[4] + ellipse_rot[1, :],
            "green",
            lw=3,
        )
        plt.plot(
            results_pm[0] + circle[0, :], results_pm[1] + circle[1, :], "blue", lw=3
        )
        plt.show()

        x, y = angle.rotate_axis(
            pmra, pmdec, results_pm[7], results_pm[3], results_pm[4]
        )
        mu_x, mu_y = angle.rotate_axis(
            results_pm[0], results_pm[1], results_pm[7], results_pm[3], results_pm[4]
        )

        ranges = [
            [pm.quantile(x, 0.025)[0], pm.quantile(x, 0.975)[0]],
            [pm.quantile(y, 0.025)[0], pm.quantile(y, 0.975)[0]],
        ]

        Ux = np.linspace(ranges[0][0], ranges[0][1], 200)
        Uy = np.linspace(ranges[1][0], ranges[1][1], 200)

        projx = pm.proj_global_pdf(
            Ux, mu_x, results_pm[2], results_pm[5], results_pm[8], results_pm[9]
        )
        projy = pm.proj_global_pdf(
            Uy, mu_y, results_pm[2], results_pm[6], results_pm[8], results_pm[9]
        )

        plt.title("PMs projected on semi-major axis")
        plt.hist(
            x,
            100,
            range=ranges[0],
            alpha=1,
            density=True,
            histtype="stepfilled",
            color=(102 / 255, 178 / 255, 1),
        )
        plt.plot(Ux, projx, color="darkgreen", lw=3)
        plt.xlim(ranges[0][0], ranges[0][1])
        plt.show()

        plt.title("PMs projected on semi-minor axis")
        plt.hist(
            y,
            100,
            range=ranges[1],
            alpha=1,
            density=True,
            histtype="stepfilled",
            color=(102 / 255, 178 / 255, 1),
        )
        plt.plot(Uy, projy, color="darkgreen", lw=3)
        plt.xlim(ranges[1][0], ranges[1][1])
        plt.show()

    if prob_method == "complete":
        probs = prob_pm * prob_sd
    elif prob_method == "pm":
        probs = prob_pm
    elif prob_method == "position":
        probs = prob_sd

    idx_p = np.where(probs > prob_limit)

    if use_hrd is True:
        probs = probs[idx_p]
        fg_mag = g_mag[idx][idx_p]
        fbr_mag = br_mag[idx][idx_p]

        if bw_hrd is None:
            bw_hrd = hrd.bw_silver(fbr_mag, fg_mag) * 0.5

        z = hrd.kde(fbr_mag, fg_mag, ww=probs, bw=bw_hrd)

        max_z = np.amax(z)

        levels = max_z * np.exp(-0.5 * np.array([nsig]) ** 2)

        qx_i, qx_f = np.nanmin(fbr_mag), np.nanmax(fbr_mag)
        qy_i, qy_f = np.nanmin(fg_mag), np.nanmax(fg_mag)
        ranges = [[qx_i, qx_f], [qy_i, qy_f]]

        if check_fit is True:

            fig, ax = plt.subplots(figsize=(7, 6))
            plt.title(r"Isochrone", fontsize=16)
            plt.xlabel(r"color", fontsize=15)
            plt.ylabel(r"main magnitude", fontsize=15)

            cax = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
                cmap="hot_r",
                norm=mc.LogNorm(vmin=0.001),
            )

            cbar = fig.colorbar(cax)
            cbar.ax.set_title("KDE's PDF", fontsize=15)

            contours = plt.contour(
                z,
                origin="lower",
                levels=levels,
                extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
            )
            plt.gca().invert_yaxis()
            plt.show()

        else:

            contours = plt.contour(
                z,
                origin="lower",
                levels=levels,
                extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
            )
            plt.clf()

        idx_final = hrd.inside_contour(br_mag, g_mag, contours, idx[idx_p])

    else:

        idx_final = idx_p

    if return_center is False:
        return idx_final, results_sd, var_sd, results_pm, var_pm
    else:
        return idx_final, results_sd, var_sd, results_pm, var_pm, c


def extract_object(
    path,
    sd_model="plummer",
    min_method="dif",
    prob_method="complete",
    prob_limit=0.9,
    use_hrd=True,
    nsig=3,
    bw_hrd=None,
    r_max=None,
    err_lim=None,
    err_handle="relative",
    object_type="gc",
    return_center=False,
    check_fit=False,
    cleaning="v21",
    conv=True,
):
    """


    Parameters
    ----------
    path : string
        Path to Gaia fits file.
    sd_model : string, optional
        Surface density model to be considered. Available options are:
             - 'sersic'
             - 'kazantzidis'
             - 'plummer'
             - 'test', for testing which among Plummer and Sersic should be
             used, based on AICc.
        The default is 'plummer'.
    min_method : string, optional
        Minimization method to be used by the pm maximum likelihood fit.
        The default is 'dif'.
    prob_method : string, optional
        Method of probability cut chosen. Available options are:
            - complete: Considers proper motions and projected radii.
            - pm: Considers only proper motions.
            - position: Considers only projected radii.
        The default is 'complete'.
    prob_limit : float, optional
        Probability threshold (between 0 and 1) considered to cut
        the data. The default is 0.9.
    use_hrd : boolean, optional
        True if the user wants to select stars inside a n-sigma contour
        region inside the color-magnitude diagram.
        The default is True.
    nsig : float, optional
        Number of sigma up to when consider stars inside the KDE
        color-magnitude diagram.
        The default is 3.3.
    bw_hrd : float or string, optional
        Bandwidth method of KDE's scipy method. The default is None.
    r_max : float, optional
        Maximum projected radius up to where consider the data.
        The default is None
    err_lim : float, optional
        Error threshold for proper motions. The default is None.
    err_handle : string, optional
        Way in which the error limit is handled:
            - 'relative', if the err_lim argument is the number of times
            the initial guess on the galactic object proper motion dispersion.
            - 'absolute', if the err_lim argument is the absolute error
            limit in mas/yr.
        The default is 'relative'.
    object_type : string, optional
        Galactic object type, to be considered when defining the default
        value of err_lim:
            - 'gc', for globular cluster.
            - 'dsph', for dwarf spheroidal galaxy.
        The default is 'gc'.
    return_center : boolean, optional
        True if the user wants an estimate of the center as output.
        The default is False.
    check_fit : boolean, optional
        True is the user wants to plot validity checks throughout the fitting
        procedure.
        The default is False.
    cleaning : string, optional
        Cleaning procedure to be adopted:
            - 'v21', to use the same cleaning from Vitral 2021
            - 'vm21', to use the same cleaning from Vitral & Mamon 21
        The default is 'v21'.
    conv : boolean, optional
        True, if the user wants to convolve the galactic object PDF with
        Gaussian errors. The defualt is True.


    Raises
    ------
    ValueError
        Surface density model is not one of the following:
            - 'sersic'
            - 'kazantzidis'
            - 'plummer'
            - 'test'
        Probability method is not one of the following:
            - 'complete'
            - 'pm'
            - 'position'
        Probability threshold is not inbetween 0 and 1.
        nsig is not positive.
        err_handle is not one of the following:
            - 'relative'
            - 'absolute'
        object_type is not one of the following:
            - 'gc'
            - 'dsph'
        cleaning is not one of the following:
            - 'v21'
            - 'vm21'

    Returns
    -------
    full_data : array_like
        File containing the information of the stars that passed the
        probability cuts.
    results_sd : array_like
        Results of the surface density fit.
    var_sd : array_like
        Uncertainties of the surface density fit
    results_pm : array_like
        Results of the proper motion fit.
    var_pm : array_like
        Uncertainties of the proper motion fit.

    """

    if sd_model not in ["sersic", "plummer", "kazantzidis", "test"]:
        raise ValueError("Does not recognize surface density model.")

    if prob_method not in ["complete", "pm", "position"]:
        raise ValueError("Does not recognize probability model.")

    if prob_limit > 1 or prob_limit < 0:
        raise ValueError("Probability threshold must be inbetween 0 and 1.")

    if nsig <= 0:
        raise ValueError("nsig must be positive.")

    if err_handle not in ["relative", "absolute"]:
        raise ValueError("Does not recognize method to handle errors.")

    if object_type not in ["gc", "dsph"]:
        raise ValueError("Does not recognize object type.")

    if cleaning not in ["v21", "vm21"]:
        raise ValueError("Does not recognize cleaning procedure.")

    hdu = fits.open(path)

    ra = np.asarray(hdu[1].data[:]["ra      "])  # degrees
    dec = np.asarray(hdu[1].data[:]["DEC     "])  # degrees
    pmra = np.asarray(hdu[1].data[:]["pmra    "])  # mas/yr
    epmra = np.asarray(hdu[1].data[:]["pmra_error"])  # mas/yr
    pmdec = np.asarray(hdu[1].data[:]["pmdec   "])  # mas/yr
    epmdec = np.asarray(hdu[1].data[:]["pmdec_error"])  # mas/yr
    corrpm = np.asarray(hdu[1].data[:]["pmra_pmdec_corr"])  # mas/yr
    g_mag = np.asarray(hdu[1].data[:]["phot_g_mean_mag"])  # in mag
    br_mag = np.asarray(hdu[1].data[:]["bp_rp   "])  # in mag
    chi2 = np.asarray(hdu[1].data[:]["astrometric_chi2_al"])  # no units
    nu = np.asarray(hdu[1].data[:]["astrometric_n_good_obs_al"])  # no units
    br_excess = np.asarray(hdu[1].data[:]["phot_bp_rp_excess_factor"])  # no units

    full_data = hdu[1].data[:]

    hdu.close()

    if return_center is False:
        idx_final, results_sd, var_sd, results_pm, var_pm = find_object(
            ra,
            dec,
            pmra,
            pmdec,
            epmra,
            epmdec,
            corrpm,
            chi2,
            nu,
            g_mag,
            br_mag,
            br_excess,
            sd_model=sd_model,
            min_method=min_method,
            prob_method=prob_method,
            prob_limit=prob_limit,
            use_hrd=use_hrd,
            nsig=nsig,
            bw_hrd=bw_hrd,
            r_max=r_max,
            err_lim=err_lim,
            err_handle=err_handle,
            object_type=object_type,
            check_fit=check_fit,
            cleaning=cleaning,
            conv=conv,
        )
    else:
        idx_final, results_sd, var_sd, results_pm, var_pm, center = find_object(
            ra,
            dec,
            pmra,
            pmdec,
            epmra,
            epmdec,
            corrpm,
            chi2,
            nu,
            g_mag,
            br_mag,
            br_excess,
            sd_model=sd_model,
            min_method=min_method,
            prob_method=prob_method,
            prob_limit=prob_limit,
            use_hrd=use_hrd,
            nsig=nsig,
            bw_hrd=bw_hrd,
            r_max=r_max,
            err_lim=err_lim,
            err_handle=err_handle,
            object_type=object_type,
            return_center=return_center,
            check_fit=check_fit,
            cleaning=cleaning,
            conv=conv,
        )

    full_data = full_data[idx_final]

    if return_center is False:
        return full_data, results_sd, var_sd, results_pm, var_pm
    else:
        return full_data, results_sd, var_sd, results_pm, var_pm, center


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Statistical handling"
# ---------------------------------------------------------------------------


def get_aic(lnL, Nf):
    """
    Provides the Akaike Information Criterion (Akaike, 1973), i.e. AIC.

    Parameters
    ----------
    lnL : float
        Minus the log-likelihood.
    Nf : int
        Number of free parameters.

    Returns
    -------
    AIC : float
        AIC indicator.

    """

    AIC = 2 * (lnL + Nf)
    return AIC


def get_bic(lnL, Nf, Nd):
    """
    Provides the Bayes Information Criterion (Schwarz, 1978), i.e. BIC.

    Parameters
    ----------
    lnL : float
        Minus the log-likelihood.
    Nf : int
        Number of free parameters.
    Nd : int
        Size of data sample.

    Returns
    -------
    BIC : float
        BIC indicator.

    """

    BIC = 2 * lnL + Nf * np.log(Nd)
    return BIC


def get_aicc(lnL, Nf, Nd):
    """
    Provides the corrected Akaike Information Criterion (Sugiyara, 1978),
    i.e. AICc.

    Parameters
    ----------
    lnL : float
        Minus the log-likelihood.
    Nf : int
        Number of free parameters.
    Nd : int
        Size of data sample.

    Returns
    -------
    AICc : float
        AICc indicator.

    """

    AICc = get_aic(lnL, Nf) + 2 * Nf * (1 + Nf) / (Nd - Nf - 1)
    return AICc
