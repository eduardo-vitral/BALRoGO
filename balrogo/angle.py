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
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Global variables"
# ------------------------------------------------------------------------------

# Right ascention of the north galactic pole, in radians
a_NGP = 192.85947789 * (np.pi / 180)
# Declination of the north galactic pole, in radians
d_NGP = 27.12825241 * (np.pi / 180)
# Longitude of the north celestial pole, in radians
l_NCP = 122.93192526 * (np.pi / 180)

# Vertical waves in the solar neighbourhood in Gaia DR2
# Bennett & Bovy, 2019, MNRAS
# --> Sun Z position (kpc), in galactocentric coordinates
z_sun = 0.0208

# Sun Y position (kpc), in galactocentric coordinates, by definition.
y_sun = 0

# A geometric distance measurement to the Galactic center black hole
# with 0.3% uncertainty
# Gracity Collaboration, 2019, A&A
# --> Sun distance from the Galactic center, in kpc.
d_sun = 8.178

# Sun X position (kpc), in galactocentric coordinates
x_sun = np.sqrt(d_sun * d_sun - z_sun * z_sun)

# On the Solar Velocity
# Ronald Drimmel and Eloisa Poggio, 2018, RNAAS
# --> Sun X velocity (km/s), in galactocentric coordinates
vx_sun = -12.9
# --> Sun Y velocity (km/s), in galactocentric coordinates
vy_sun = 245.6
# --> Sun Z velocity (km/s), in galactocentric coordinates
vz_sun = 7.78

# Multuplying factor to pass from kpc to km
kpc_to_km = 3.086 * 10 ** 16

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

    A = np.arctan2(sinA, cosA)

    v0 = v0 / np.linalg.norm(v0)
    B0 = np.arcsin(v0[2])
    cosA0 = v0[0] / np.cos(B0)
    sinA0 = v0[1] / np.cos(B0)

    A0 = np.arctan2(sinA0, cosA0)

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


def sky_coord_rotate(v_i, v0_i, v0_f, theta=0, debug=False):
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
    theta : float, optional
        Angle to rotate (for no translation), in radians. The default is 0.
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

        A = np.arctan2(sinA, cosA)

    elif len(np.shape(v_f)) == 2:
        A = np.zeros(len(v_f))
        B = np.zeros(len(v_f))
        for i in range(0, len(v_f)):
            v_f[i] = v_f[i] / np.linalg.norm(v_f[i])
            B[i] = np.arcsin(v_f[i][2])
            cosA = v_f[i][0] / np.cos(B[i])
            sinA = v_f[i][1] / np.cos(B[i])

            A[i] = np.arctan2(sinA, cosA)

            if debug is True and i < 10:
                print("A, B [degrees]:", A[i] * (180 / np.pi), B[i] * (180 / np.pi))
    else:
        print("You did not give a valid input for the Rodrigues formula.")
        return

    return A * (180 / np.pi), B * (180 / np.pi)


def sky_vector(a, d, a0, d0, af, df):
    """
    Transforms sky coordinates in vectors to be used by sky_coord_rotate.

    Parameters
    ----------
    a : float, array_like
        Original set of RA in degrees.
    d : float, array_like
        Original set of Dec in degrees.
    a0 : float
        Original centroid RA in degrees.
    d0 : float
        Original centroid Dec in degrees.
    af : float
        Final centroid RA in degrees.
    df : float
        Final centroid Dec in degrees.

    Returns
    -------
    v_i : array_like
        Vectors to be rotated.
    v0_i : array_like
        Vector pointing to the initial centroid position.
    v0_f : array_like
        Vector pointing to the final centroid position.

    """

    a = a * (np.pi / 180)
    d = d * (np.pi / 180)
    a0 = a0 * (np.pi / 180)
    d0 = d0 * (np.pi / 180)
    af = af * (np.pi / 180)
    df = df * (np.pi / 180)

    v_i = np.asarray([np.cos(a) * np.cos(d), np.sin(a) * np.cos(d), np.sin(d)])

    v0_i = np.asarray([np.cos(a0) * np.cos(d0), np.sin(a0) * np.cos(d0), np.sin(d0)])

    v0_f = np.asarray([np.cos(af) * np.cos(df), np.sin(af) * np.cos(df), np.sin(df)])

    return v_i, v0_i, v0_f


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Axis rotation"
# ------------------------------------------------------------------------------


def transrot_source(a, d, a0, d0, af, df):
    """
    Translades and then rotates a source in spherical coordinates, so
    the original directions remain the same.

    Parameters
    ----------
    a : float, array_like
        Original set of RA in degrees.
    d : float, array_like
        Original set of Dec in degrees.
    a0 : float
        Original centroid RA in degrees.
    d0 : float
        Original centroid Dec in degrees.
    af : float
        Final centroid RA in degrees.
    df : float
        Final centroid Dec in degrees.

    Returns
    -------
    a : array_like
        Right ascention in degrees.
    d : array_like
        Declination in degrees.

    """

    v_i, v0_i, v0_f = sky_vector(a, d, a0, d0, af, df)

    a, d = sky_coord_rotate(v_i, v0_i, v0_f)

    rmax = np.nanmax(sky_distance_deg(a, d, af, df)) * (np.pi / 180)

    narr = np.asarray([0, 0.1, 0.25, 0.5]) * rmax
    vt = np.asarray(
        [
            np.cos(a0 + narr) * np.cos([d0, d0, d0, d0]),
            np.sin(a0 + narr) * np.cos([d0, d0, d0, d0]),
            np.sin([d0, d0, d0, d0]),
        ]
    )

    at, dt = sky_coord_rotate(vt, v0_i, v0_f)

    r0, p0 = sky_to_polar(a0 + narr, d0 * np.ones(len(narr)), a0, d0)
    rf, pf = sky_to_polar(at, dt, af, df)

    phi = np.nanmean(pf - p0)

    v_i, v0_i, v0_f = sky_vector(a, d, af, df, af, df)

    a, d = sky_coord_rotate(v_i, v0_i, v0_f, theta=phi)

    return a, d


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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------------------------------
"Transform coordinates"
# ------------------------------------------------------------------------------


def cart_to_sph(x, y, z, vx, vy, vz):
    """
    Transforms 6D cartesian coordinates to spherical coordinates.

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

    Returns
    -------
    r : array_like, float
        r-axis.
    phi : array_like, float
        phi-angle.
    theta : array_like, float
        theta-angle.
    vr : array_like, float
        r-axis velocity.
    vphi : array_like, float
        phi-angle velocity.
    vtheta : array_like, float
        theta-angle velocity.

    """

    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)

    vr = (vx * x + vy * y + vz * z) / r
    vphi = (vy * x - vx * y) / np.sqrt(x * x + y * y)
    vtheta = -(vz * (x * x + y * y) - z * (vx * x + vy * y)) / (
        np.sqrt(x * x + y * y) * r
    )

    return r, phi, theta, vr, vphi, vtheta


def sph_to_cart(r, phi, theta, vr, vphi, vtheta):
    """
    Transforms 6D spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    r : array_like, float
        r-axis.
    phi : array_like, float
        phi-angle.
    theta : array_like, float
        theta-angle.
    vr : array_like, float
        r-axis velocity.
    vphi : array_like, float
        phi-angle velocity.
    vtheta : array_like, float
        theta-angle velocity.

    Returns
    -------
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

    """

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    vx = (
        vr * np.sin(theta) * np.cos(phi)
        + vtheta * np.cos(theta) * np.cos(phi)
        - vphi * np.sin(phi)
    )
    vy = (
        vr * np.sin(theta) * np.sin(phi)
        + vtheta * np.cos(theta) * np.sin(phi)
        + vphi * np.cos(phi)
    )
    vz = vr * np.cos(theta) - vtheta * np.sin(theta)

    return x, y, z, vx, vy, vz


def radec_to_lb(a, d, dadt=None, dddt=None):
    """
    Transforms celestial coordinates into galactic coordinates.

    Parameters
    ----------
    a : array_like, float
        Right ascention in degrees.
    d : array_like, float
        Declination in degrees.
    dadt : array_like, float, optional
        Right ascention velocity (PMRA), in mas/yr.
        The default is None
    dddt : array_like, float, optional
        Declination velocity (PMDec), in mas/yr.
        The default is None

    Returns
    -------
    lon : array_like, float
        Galactic longitude, in degrees.
    b : array_like, float
        Galactic latitude, in degrees.
    dldt : array_like, float, optional
        Galactic longitude velocity, in mas/yr
    dbdt : array_like, float, optional
        Galactic latitude velocity, in mas/yr. The default is None

    """

    a = a * (np.pi / 180)
    d = d * (np.pi / 180)

    sind = np.sin(d)
    cosd = np.cos(d)
    sind_NGP = np.sin(d_NGP)
    cosd_NGP = np.cos(d_NGP)
    cosda = np.cos(a - a_NGP)
    sinda = np.sin(a - a_NGP)

    sinb = sind_NGP * sind + cosd_NGP * cosd * cosda
    cosb_sindl = cosd * sinda
    cosb_cosdl = cosd_NGP * sind - sind_NGP * cosd * cosda

    b = np.arcsin(sinb)

    cosb = np.cos(b)

    sindl = cosb_sindl / cosb
    cosdl = cosb_cosdl / cosb

    if np.isscalar(sindl):
        lon = l_NCP - np.arctan2(sindl, cosdl)
    else:
        lon = np.zeros(len(sindl))
        for i in range(len(sindl)):
            lon[i] = l_NCP - np.arctan2(sindl[i], cosdl[i])

    if np.isscalar(lon):
        lon = lon % (2 * np.pi)
    else:
        for i in range(len(lon)):
            lon[i] = lon[i] % (2 * np.pi)

    if dadt is not None and dddt is not None:
        sinb = np.sin(b)
        cosdl = np.cos(l_NCP - lon)

        dbdt = (
            sind_NGP * cosd * dddt
            - cosd_NGP * cosda * sind * dddt
            - cosd_NGP * sinda * dadt
        ) / cosb

        dldt = (
            cosb * (sind * dddt * sinda - cosda * dadt) - cosd * sinda * sinb * dbdt
        ) / (cosb * cosb * cosdl)

        lon = lon * (180 / np.pi)
        b = b * (180 / np.pi)

        return lon, b, dldt, dbdt

    else:

        lon = lon * (180 / np.pi)
        b = b * (180 / np.pi)

        return lon, b


def lb_to_radec(lon, b, dldt=None, dbdt=None):
    """
    Transforms galactic coordinates into celestial coordinates.

    Parameters
    ----------
    lon : array_like, float
        Galactic longitude, in degrees.
    b : array_like, float
        Galactic latitude, in degrees.
    dldt : array_like, float, optional
        Galactic longitude velocity, in mas/yr. The default is None
    dbdt : array_like, float, optional
        Galactic latitude velocity, in mas/yr. The default is None

    Returns
    -------
    a : array_like, float
        Right ascention in degrees.
    d : array_like, float
        Declination in degrees.
    dadt : array_like, float, optional
        Right ascention velocity (PMRA), in mas/yr.
    dddt : array_like, float, optional
        Declination velocity (PMDec), in mas/yr.

    """

    lon = lon * (np.pi / 180)
    b = b * (np.pi / 180)

    sind_NGP = np.sin(d_NGP)
    cosd_NGP = np.cos(d_NGP)
    sinb = np.sin(b)
    cosb = np.cos(b)
    cosdl = np.cos(l_NCP - lon)
    sindl = np.sin(l_NCP - lon)

    sind = sind_NGP * sinb + cosd_NGP * cosb * cosdl
    cosd_sinda = cosb * sindl
    cosd_cosda = cosd_NGP * sinb - sind_NGP * cosb * cosdl

    d = np.arcsin(sind)

    cosd = np.cos(d)

    sinda = cosd_sinda / cosd
    cosda = cosd_cosda / cosd

    if np.isscalar(sinda):
        a = np.arctan2(sinda, cosda) + a_NGP
    else:
        a = np.zeros(len(sinda))
        for i in range(len(sinda)):
            a[i] = np.arctan2(sinda[i], cosda[i]) + a_NGP

    if np.isscalar(a):
        a = a % (2 * np.pi)
    else:
        for i in range(len(a)):
            a[i] = a[i] % (2 * np.pi)

    if dbdt is not None and dldt is not None:
        sind = np.sin(d)
        cosda = np.cos(a - a_NGP)

        dddt = (
            sind_NGP * cosb * dbdt
            - cosd_NGP * cosdl * sinb * dbdt
            + cosd_NGP * cosb * sindl * dldt
        ) / cosd

        dadt = -(
            cosd * (sinb * dbdt * sindl + cosb * cosdl * dldt)
            - cosb * sindl * sind * dddt
        ) / (cosd * cosd * cosda)

        dadt = dadt * cosd

        a = a * (180 / np.pi)
        d = d * (180 / np.pi)

        return a, d, dadt, dddt

    else:

        a = a * (180 / np.pi)
        d = d * (180 / np.pi)

        return a, d


def cart_to_radec(x, y, z, vx=None, vy=None, vz=None):
    """
    Transforms galactocentric coordinates into celestial coordinates.

    Parameters
    ----------
    x : array_like, float
        x-axis.
    y : array_like, float
        y-axis.
    z : array_like, float
        z-axis.
    vx : array_like, float, optional
        x-axis velocity. The default is None.
    vy : array_like, float, optional
        y-axis velocity. The default is None.
    vz : array_like, float, optional
        z-axis velocity. The default is None.

    Raises
    ------
    ValueError
        Velocity components are incomplete
        (number of velocity dimensions equal to 1 or 2).

    Returns
    -------
    a : array_like, float
        Right ascention in degrees.
    d : array_like, float
        Declination in degrees.
    r : array_float
        Distance of the source, in kpc.
    dadt : array_like, float, optional
        Right ascention velocity (PMRA), in mas/yr.
    dddt : array_like, float, optional
        Declination velocity (PMDec), in mas/yr.
    vr : array_like, float
        Line of sight velocity, in km/s.

    """
    if vx is None or vy is None or vz is None:

        onlypos = True

        if vx is not None:
            raise ValueError("Please provide other velocity components.")
        if vy is not None:
            raise ValueError("Please provide other velocity components.")
        if vz is not None:
            raise ValueError("Please provide other velocity components.")

        vx = 0
        vy = 0
        vz = 0

    else:

        onlypos = False

    lon, b, r, dldt, dbdt, vr = cart_to_lb(x, y, z, vx=vx, vy=vy, vz=vz)

    a, d, dadt, dddt = lb_to_radec(lon, b, dldt=dldt, dbdt=dbdt)

    if onlypos is True:
        return a, d, r
    else:
        return a, d, r, dadt, dddt, vr


def radec_to_cart(a, d, r, mua=None, mud=None, vr=None):
    """
    Transforms celestial coordinates into galactocentric coordinates.

    Parameters
    ----------

    a : array_like, float
        Right ascention in degrees.
    d : array_like, float
        Declination in degrees.
    r : array_float
        Distance of the source, in kpc.
    mua : array_like, float, optional
        Right ascention velocity (PMRA), in mas/yr.
        The default is None.
    mud : array_like, float, optional
        Declination velocity (PMDec), in mas/yr.
        The default is None.
    vr : array_like, float, optional
        Line of sight velocity, in km/s.
        The default is None.

    Raises
    ------
    ValueError
        Velocity components are incomplete
        (number of velocity dimensions equal to 1 or 2).

    Returns
    -------

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

    """

    masyr_to_kms = 4.7405 * r

    if mua is None or mud is None or vr is None:

        onlypos = True

        if mua is not None:
            raise ValueError("Please provide other velocity components.")
        if mud is not None:
            raise ValueError("Please provide other velocity components.")
        if vr is not None:
            raise ValueError("Please provide other velocity components.")

        mua = 0
        mud = 0
        vr = 0

    else:

        onlypos = False

    lon, b, dldt, dbdt = radec_to_lb(a, d, dadt=mua, dddt=mud)

    dldt = dldt * masyr_to_kms
    dbdt = dbdt * masyr_to_kms
    r = r * kpc_to_km

    phi = lon * (np.pi / 180)
    theta = np.pi * 0.5 - b * (np.pi / 180)

    x, y, z, vx, vy, vz = sph_to_cart(r, phi, theta, vr, dldt, -dbdt)

    x = -x / kpc_to_km + x_sun
    y = y / kpc_to_km - y_sun
    z = z / kpc_to_km - z_sun

    vx = -vx + vx_sun
    vy = vy + vy_sun
    vz = vz + vz_sun

    if onlypos is True:
        return x, y, z
    else:
        return x, y, z, vx, vy, vz


def cart_to_lb(x, y, z, vx=None, vy=None, vz=None):
    """
    Transforms galactocentric coordinates into galactic coordinates.

    Parameters
    ----------
    x : array_like, float
        x-axis.
    y : array_like, float
        y-axis.
    z : array_like, float
        z-axis.
    vx : array_like, float, optional
        x-axis velocity. The default is None.
    vy : array_like, float, optional
        y-axis velocity. The default is None.
    vz : array_like, float, optional
        z-axis velocity. The default is None.

    Raises
    ------
    ValueError
        Velocity components are incomplete
        (number of velocity dimensions equal to 1 or 2).

    Returns
    -------
    lon : array_like, float
        Galactic longitude, in degrees.
    b : array_like, float
        Galactic latitude, in degrees.
    r : array_float
        Distance of the source, in kpc.
    dldt : array_like, float, optional
        Galactic longitude velocity, in mas/yr.
    dbdt : array_like, float, optional
        Galactic latitude velocity, in mas/yr.
    vr : array_like, float
        Line of sight velocity, in km/s.

    """

    x = (-x + x_sun) * kpc_to_km
    y = (y + y_sun) * kpc_to_km
    z = (z + z_sun) * kpc_to_km

    if vx is None or vy is None or vz is None:

        onlypos = True

        if vx is not None:
            raise ValueError("Please provide other velocity components.")
        if vy is not None:
            raise ValueError("Please provide other velocity components.")
        if vz is not None:
            raise ValueError("Please provide other velocity components.")

        vx = 0
        vy = 0
        vz = 0

    else:

        onlypos = False

    vx = -vx + vx_sun
    vy = vy - vy_sun
    vz = vz - vz_sun

    r, phi, theta, vr, vphi, vtheta = cart_to_sph(x, y, z, vx, vy, vz)

    lon = phi * (180 / np.pi)
    b = (np.pi * 0.5 - theta) * (180 / np.pi)

    dldt = (vphi / r) * (3600 * 1000 * 180 / np.pi) * (3.154 * 10 ** 7)
    dbdt = (-vtheta / r) * (3600 * 1000 * 180 / np.pi) * (3.154 * 10 ** 7)
    r = r / kpc_to_km

    if onlypos is True:
        return lon, b, r
    else:
        return lon, b, r, dldt, dbdt, vr


def lb_to_cart(lon, b, r, dldt=None, dbdt=None, vr=None):
    """
    Transforms galactic coordinates into galactocentric coordinates.

    Parameters
    ----------
    lon : array_like, float
        Galactic longitude, in degrees.
    b : array_like, float
        Galactic latitude, in degrees.
    r : array_float
            Distance of the source, in kpc.
    dldt : array_like, float, optional
        Galactic longitude velocity, in mas/yr. The default is None
    dbdt : array_like, float, optional
        Galactic latitude velocity, in mas/yr. The default is None
    vr : array_like, float, optional
        Line of sight velocity, in km/s. The default is None.

    Raises
    ------
    ValueError
        Velocity components are incomplete
        (number of velocity dimensions equal to 1 or 2).

    Returns
    -------

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
    """

    masyr_to_kms = 4.7405 * r

    if dldt is None or dbdt is None or vr is None:

        onlypos = True

        if dldt is not None:
            raise ValueError("Please provide other velocity components.")
        if dbdt is not None:
            raise ValueError("Please provide other velocity components.")
        if vr is not None:
            raise ValueError("Please provide other velocity components.")

        dldt = 0
        dbdt = 0
        vr = 0

    else:

        onlypos = False

    r = r * kpc_to_km
    phi = lon * (np.pi / 180)
    theta = np.pi * 0.5 - b * (np.pi / 180)

    vphi = dldt * masyr_to_kms
    vtheta = -dbdt * masyr_to_kms

    x, y, z, vx, vy, vz = sph_to_cart(r, phi, theta, vr, vphi, vtheta)

    x = -x / kpc_to_km + x_sun
    y = y / kpc_to_km - y_sun
    z = z / kpc_to_km - z_sun

    vx = -vx + vx_sun
    vy = vy + vy_sun
    vz = vz + vz_sun

    if onlypos is True:
        return x, y, z
    else:
        return x, y, z, vx, vy, vz
