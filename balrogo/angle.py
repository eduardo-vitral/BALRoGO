"""
Created on 2020

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2020, Paris
#
# This file contains the main functions concerning the angular tansformations,
# sky projections and spherical trigonomtry.
#
# Documentation is provided on Vitral & Macedo, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Angles handling"
# ------------------------------------------------------------------------------


def sky_distance_deg(RA, Dec, RA0, Dec0):
    """
    Computes the sky distance (in degrees) between two sets of
    sky coordinates, given also in degrees.

    Parameters
    ----------
    RA : array_like, float
        Right ascension (in degrees) of object 1.
    Dec : array_like (same shape as RA), float
        Declination (in degrees) of object 1.
    RA0 : array_like (same shape as RA), float
        Right ascension (in degrees) of object 2.
    Dec0 : array_like (same shape as RA), float
        Declination (in degrees) of object 2.

    Returns
    -------
    R : array_like, float
        Sky distance (in degrees) between object 1 and object 2.

    """

    RA = RA * np.pi / 180
    Dec = Dec * np.pi / 180

    RA0 = RA0 * np.pi / 180
    Dec0 = Dec0 * np.pi / 180

    R = (180 / np.pi) * np.arccos(
        np.sin(Dec) * np.sin(Dec0) + np.cos(Dec) * np.cos(Dec0) * np.cos((RA - RA0))
    )

    return np.asarray(R)


def get_circle_sph_trig(r, a0, d0, nbins=500):
    """
    Generates a circle in spherical coordinates.

    Parameters
    ----------
    r : float
        Distance from the center, in degrees.
    a0 : float
        Right ascention from origin, in degrees.
    d0 : float
        Declination from origin in, in degrees.
    nbins : int, optional
        Number of circle points. The default is 500.

    Returns
    -------
    ra : array_like
        Right ascention in degrees.
    dec : array_like
        Declination in degrees.

    """

    # Converts angles to radians
    r = r * np.pi / 180
    a0 = a0 * np.pi / 180
    d0 = d0 * np.pi / 180

    phi = np.linspace(0, 2 * np.pi, nbins)

    a = np.zeros(nbins)
    d = np.zeros(nbins)
    for i in range(0, nbins):
        a[i], d[i] = polar_to_sky(r, phi[i], a0, d0)

    return a, d


def polar_to_sky(r, phi, a0, d0):
    """
    Transforms spherical polar coordinates (r,phi) into sky coordinates,
    in degrees (RA,Dec).

    Parameters
    ----------
    r : array_like
        Radial distance from center.
    phi : array_like
        Angle between increasing declination and the projected radius
        (pointing towards the source).
    a0 : float
        Right ascention from origin in radians.
    d0 : float
        Declination from origin in radians.

    Returns
    -------
    ra : array_like
        Right ascention in degrees.
    dec : array_like
        Declination in degrees.

    """

    d = np.arcsin(np.cos(r) * np.sin(d0) + np.cos(d0) * np.cos(phi) * np.sin(r))

    if phi < np.pi:
        if (np.cos(r) - np.sin(d) * np.sin(d0)) / (np.cos(d) * np.cos(d0)) > 0:
            a = a0 + np.arccos(np.sqrt(1 - (np.sin(phi) * np.sin(r) / np.cos(d)) ** 2))
        else:
            a = a0 + np.arccos(-np.sqrt(1 - (np.sin(phi) * np.sin(r) / np.cos(d)) ** 2))

    if phi >= np.pi:
        if (np.cos(r) - np.sin(d) * np.sin(d0)) / (np.cos(d) * np.cos(d0)) > 0:
            a = a0 - np.arccos(np.sqrt(1 - (np.sin(phi) * np.sin(r) / np.cos(d)) ** 2))
        else:
            a = a0 - np.arccos(-np.sqrt(1 - (np.sin(phi) * np.sin(r) / np.cos(d)) ** 2))

    ra = a * 180 / np.pi
    dec = d * 180 / np.pi

    return ra, dec


def sky_to_polar(a, d, a0, d0):
    """
    Transforms sky coordinates, in degrees (RA,Dec), into
    spherical polar coordinates (r,phi).

    Parameters
    ----------
    a : array_like
        Right ascention in degrees.
    d : array_like
        Declination in degrees.
    a0 : float
        Right ascention from origin.
    d0 : float
        Declination from origin.

    Returns
    -------
    r : array_like
        Radial distance from center.
    p : array_like
        Angle between increasing declination and the projected radius
        (pointing towards the source), in radians.

    """

    r = sky_distance_deg(a, d, a0, d0) * np.pi / 180

    a = a * np.pi / 180
    d = d * np.pi / 180

    sp = np.cos(d) * np.sin(a - (a0 * np.pi / 180)) / np.sin(r)

    p = np.zeros(len(sp))

    spp = np.where(sp > 0)
    spm = np.where(sp <= 0)
    dp = np.where(d > (d0 * np.pi / 180))
    dm = np.where(d <= (d0 * np.pi / 180))

    p[np.intersect1d(spp, dp)] = np.arcsin(sp[np.intersect1d(spp, dp)])
    p[np.intersect1d(spp, dm)] = np.pi - np.arcsin(sp[np.intersect1d(spp, dm)])
    p[np.intersect1d(spm, dp)] = 2 * np.pi + np.arcsin(sp[np.intersect1d(spm, dp)])
    p[np.intersect1d(spm, dm)] = np.pi - np.arcsin(sp[np.intersect1d(spm, dm)])

    return r, p


def angular_sep_vector(v0, v):
    """
    Returns separation angle in radians between two 3D arrays.

    Parameters
    ----------
    v0 : 3D array
        Vector 1.
    v : 3D array
        Vector 2.

    Returns
    -------
    R : array_like
        Separation between vector 1 and vector 2 in radians.

    """

    try:
        v0.shape
    except NameError:
        print("You did not give a valid input.")
        return
    v = v / np.linalg.norm(v)
    B = np.arcsin(v[2])
    cosA = v[0] / np.cos(B)
    sinA = v[1] / np.cos(B)

    if cosA > 0 and sinA >= 0:
        A = np.arcsin(sinA)
    if cosA <= 0 and sinA > 0:
        A = np.arccos(cosA)
    if cosA < 0 and sinA <= 0:
        A = 2 * np.pi - np.arccos(cosA)
    if cosA >= 0 and sinA < 0:
        A = np.arcsin(sinA)

    v0 = v0 / np.linalg.norm(v0)
    B0 = np.arcsin(v0[2])
    cosA0 = v0[0] / np.cos(B0)
    sinA0 = v0[1] / np.cos(B0)

    if cosA0 > 0 and sinA0 >= 0:
        A0 = np.arcsin(sinA0)
    if cosA0 <= 0 and sinA0 > 0:
        A0 = np.arccos(cosA0)
    if cosA0 < 0 and sinA0 <= 0:
        A0 = 2 * np.pi - np.arccos(cosA)
    if cosA0 >= 0 and sinA0 < 0:
        A0 = np.arcsin(sinA0)

    cosR = np.sin(B0) * np.sin(B) + np.cos(B0) * np.cos(B) * np.cos(A - A0)
    R = np.arccos(cosR)

    return R


def rodrigues_formula(k, v, theta, debug=False):
    """
    Returns the rotation of v of an angle theta with respect to the vector k
    Applies the Rodrigues formula from:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Parameters
    ----------
    k : array_like
        Vector with respect to which v will be rotated.
    v : array_like
        Vector to be rotated.
    theta : float
        Angle to rotate v.
    debug : boolean, optional
       True if the reader wants to print debug diagnistics.
       The default is False.

    Returns
    -------
    v_rot : array_like
        Rotated vector.

    """

    try:
        v.shape
    except NameError:
        print("You did not give a valid input for the Rodrigues formula.")
        return

    if len(np.shape(v)) == 1 and len(v) == 3:
        v_rot = (
            v * np.cos(theta)
            + np.cross(k, v) * np.sin(theta)
            + k * np.dot(k, v) * (1 - np.cos(theta))
        )

    elif len(np.shape(v)) == 2 and np.shape(v)[0] == 3:
        v_rot = np.zeros((np.shape(v)[1], 3))
        for i in range(0, len(v_rot)):

            v0 = np.asarray([v[0][i], v[1][i], v[2][i]])
            v_rot[i] = (
                v0 * np.cos(theta)
                + np.cross(k, v0) * np.sin(theta)
                + k * np.dot(v0, k) * (1 - np.cos(theta))
            )

            if debug is True and i < 10:
                print("v0   :", v0)
                print("v_rot:", v_rot[i])
    else:
        print("You did not give a valid input for the Rodrigues formula.")
        return

    return v_rot


def sky_coord_rotate(v_i, v0_i, v0_f, debug=False):
    """
    Gets new angles (RA,Dec) in degrees of a rotated vector.

    Parameters
    ----------
    v_i : array_like
        Vectors to be rotated.
    v0_i : array_like
        Vector pointing to the initial centroid position.
    v0_f : array_like
        Vector pointing to the final centroid position.
    debug : boolean, optional
       True if the reader wants to print debug diagnistics.
       The default is False.

    Returns
    -------
    array_like, array_like
        Arrays containing the new rotated (RA,Dec) positions.

    """

    if (v0_i == v0_f).all():
        if debug is True:
            print("Pure rotation")
        k = v0_i / np.linalg.norm(v0_i)
        try:
            theta = args[0]
        except NameError:
            ("You did not provide an angle to rotate. Default is zero.")
            theta = 0
    else:
        if debug is True:
            print("Translation in spherical geometry")
        k = np.cross(v0_i, v0_f) / np.linalg.norm(np.cross(v0_i, v0_f))
        theta = angular_sep_vector(v0_i, v0_f)

    if debug is True:
        print("Vector k:", k)
        print("Angle of separation [degrees]:", theta * 180 / np.pi)

    v_f = rodrigues_formula(k, v_i, theta)

    try:
        v_f.shape
    except NameError:
        print("You did not give a valid input for the Rodrigues formula.")
        return

    if len(np.shape(v_f)) == 1 and len(v_f) == 3:
        v_f = v_f / np.linalg.norm(v_f)
        B = np.arcsin(v_f[2])
        cosA = v_f[0] / np.cos(B)
        sinA = v_f[1] / np.cos(B)

        if cosA > 0 and sinA >= 0:
            A = np.arcsin(sinA)
        if cosA <= 0 and sinA > 0:
            A = np.arccos(cosA)
        if cosA < 0 and sinA <= 0:
            A = 2 * np.pi - np.arccos(cosA)
        if cosA >= 0 and sinA < 0:
            A = np.arcsin(sinA)

    elif len(np.shape(v_f)) == 2:
        A = np.zeros(len(v_f))
        B = np.zeros(len(v_f))
        for i in range(0, len(v_f)):
            v_f[i] = v_f[i] / np.linalg.norm(v_f[i])
            B[i] = np.arcsin(v_f[i][2])
            cosA = v_f[i][0] / np.cos(B[i])
            sinA = v_f[i][1] / np.cos(B[i])

            if cosA > 0 and sinA >= 0:
                A[i] = np.arcsin(sinA)
            if cosA <= 0 and sinA > 0:
                A[i] = np.arccos(cosA)
            if cosA < 0 and sinA <= 0:
                A[i] = 2 * np.pi - np.arccos(cosA)
            if cosA >= 0 and sinA < 0:
                A[i] = np.arcsin(sinA)

            if debug is True and i < 10:
                print("A, B [degrees]:", A[i] * (180 / np.pi), B[i] * (180 / np.pi))
    else:
        print("You did not give a valid input for the Rodrigues formula.")
        return

    return A * (180 / np.pi), B * (180 / np.pi)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Axis rotation"
# ------------------------------------------------------------------------------


def rotate_axis(x, y, theta, mu_x=0, mu_y=0):
    """
    Rotates and translates the two main cartesian axis.

    Parameters
    ----------
    x : float or array_like
        Data in x-direction.
    y : float or array_like
        Data in y-direction.
    theta : float
        Rotation angle in radians.
    mu_x : float, optional
        Center of new x axis in the old frame. The default is 0.
    mu_y : float, optional
        Center of new y axis in the old frame. The default is 0.

    Returns
    -------
    x_new : float or array_like
        Data in new x-direction.
    y_new : float or array_like
        Data in new y-direction.

    """

    x_new = (x - mu_x) * np.cos(theta) + (y - mu_y) * np.sin(theta)
    y_new = -(x - mu_x) * np.sin(theta) + (y - mu_y) * np.cos(theta)

    return x_new, y_new


def get_ellipse(a, b, theta, nbins):
    """
    Provides an array describing a rotated ellipse.

    Parameters
    ----------
    a : float
        Ellipse semi-major axis.
    b : float
        Ellipse semi-minor axis.
    theta : float
        Rotation angle in radians.
    nbins : int
        Number of points in the final array.

    Returns
    -------
    ellipse_rot : array_like
        Rotated ellipse array.

    """

    t = np.linspace(0, 2 * np.pi, nbins)
    ellipse = np.array([a * np.cos(t), b * np.sin(t)])
    m_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ellipse_rot = np.zeros((2, ellipse.shape[1]))
    for i in range(ellipse.shape[1]):
        ellipse_rot[:, i] = np.dot(m_rot, ellipse[:, i])

    return ellipse_rot
