"""
Created on 2021

@author: Eduardo Vitral
"""

###############################################################################
#
# November 2021, Paris
#
# This file contains the main functions concerning the creation and handling
# of mock data sets.
#
# Documentation is provided on Vitral, 2021.
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

from . import angle
import numpy as np
import os
import glob

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Writing files"
# ---------------------------------------------------------------------------


def read_columns(file, transp=False):
    """
    Read txt file.

    Parameters
    ----------
    file : string
        Path to file.
    transp : boolean, optional
        True if you want the transposed matrix of the file.
        Default is False

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """

    data = np.loadtxt(file)

    if transp is False:
        return data
    else:
        return data.T


def write_cart(data, path):
    """
    Writes a file of cartesian coordinates.

    Parameters
    ----------
    data : array_like
        Data to be written.
    path : string
        location to store the data.

    Raises
    ------
    ValueError
        Whenever the path already exists or the data shape is not good.

    Returns
    -------
    None.

    """

    os.chdir(path)
    for file in glob.glob("*.dat"):
        if file == path:
            raise ValueError("File with the same name already exists.")

    # Creates file
    astrometric = open(path, "w")

    if np.shape(data[0]) == 6:
        astrometric.write("# x y z v_x v_y v_z \n")
        astrometric.write("# kpc kpc kpc km/s km/s km/s \n")
        for i in range(0, np.shape(data)[1]):
            # write line to output file
            string = "{: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} "
            line = string.format(
                data[0, i], data[1, i], data[2, i], data[3, i], data[4, i], data[5, i]
            )
            astrometric.write(line)
            astrometric.write("\n")
            astrometric.close()

        astrometric.close()
    elif np.shape(data[0]) == 9:
        astrometric.write("# x y z v_x ev_x v_y ev_y v_z ev_z \n")
        astrometric.write("# kpc kpc kpc km/s km/s km/s km/s km/s km/s \n")
        for i in range(0, np.shape(data)[1]):
            # write line to output file
            string = (
                "{: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} "
                + " {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} "
            )
            line = string.format(
                data[0, i],
                data[1, i],
                data[2, i],
                data[3, i],
                data[4, i],
                data[5, i],
                data[6, i],
                data[7, i],
                data[8, i],
            )
            astrometric.write(line)
            astrometric.write("\n")
            astrometric.close()
    else:
        astrometric.close()
        raise ValueError("Does not recognize data shape.")


def write_gaia(data, path):
    """
    Writes a file of sky coordinates.

    Parameters
    ----------
    data : array_like
        Data to be written.
    path : string
        location to store the data.

    Raises
    ------
    ValueError
        Whenever the path already exists or the data shape is not good.

    Returns
    -------
    None.

    """

    # Creates file
    astrometric = open(path, "w")

    if np.shape(data)[0] == 6:
        astrometric.write("# RA Dec D PMRA PMDec vLOS \n")
        for i in range(0, np.shape(data)[1]):
            # write line to output file
            string = "{: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} "
            line = string.format(
                data[0, i], data[1, i], data[2, i], data[3, i], data[4, i], data[5, i]
            )
            astrometric.write(line)
            astrometric.write("\n")

        astrometric.close()
    elif np.shape(data)[0] == 9:
        astrometric.write("# RA Dec D PMRA ePMRA PMDec ePMDec vLOS evLOS \n")
        for i in range(0, np.shape(data)[1]):
            # write line to output file
            string = (
                "{: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} "
                + " {: 9.4f} {: 9.4f} {: 9.4f} {: 9.4f} "
            )
            line = string.format(
                data[0, i],
                data[1, i],
                data[2, i],
                data[3, i],
                data[4, i],
                data[5, i],
                data[6, i],
                data[7, i],
                data[8, i],
            )
            astrometric.write(line)
            astrometric.write("\n")
        astrometric.close()
    else:
        astrometric.close()
        raise ValueError("Does not recognize data shape.")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Random generators"
# ---------------------------------------------------------------------------


def sphcap_generator(a0, d0, rmax, n):
    """
    Generates an spherical cap uniformally distributed.

    Parameters
    ----------
    a0 : float
        Right ascention from origin.
    d0 : float
        Declination from origin.
    rmax : float
        Maximum radius from center in degrees.
    n : int
        length of data set.

    Returns
    -------
    a : array_like
        Right ascention in degrees.
    d : array_like
        Declination in degrees.

    """

    u_1 = np.random.uniform(0, 1, n)
    u_2 = np.random.uniform(0, 1, n)
    r = np.arccos((1 - u_1) + u_1 * np.cos(rmax * np.pi / 180))
    phi = 2 * np.pi * u_2

    a = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):

        a[i], d[i] = angle.polar_to_sky(
            r[i], phi[i], a0 * np.pi / 180, d0 * np.pi / 180
        )

    return a, d


def pm_generator(a, d, n, pmra0=0, pmdec0=0):
    """
    Random field stars proper motions following a Pearson VII distribution.

    Parameters
    ----------
    a : float
        Proper motion scale radius.
    d : float
        Proper motion slope.
    n : int
        length of data set.
    pmra0 : float, optional
        Mean pmra of interlopers.
    pmdec0 : float, optional
        Mean pmra of interlopers.

    Returns
    -------
    pmra : array_like
        Random pmra variables.
    pmdec : array_like
        Random pmdec variables.

    """

    u_1 = np.random.uniform(0, 1, n)
    u_2 = np.random.uniform(0, 1, n)
    r = a * np.sqrt(u_1 ** (1 / (1 + 0.5 * d)) - 1.0)
    phi = 2 * np.pi * u_2

    pmra = r * np.cos(phi) + pmra0
    pmdec = r * np.sin(phi) + pmdec0

    return pmra, pmdec


def gaussian_generator(mean, dev):
    """
    Random Gaussian variables of mean "mean" and standard deviation "dev".

    Parameters
    ----------
    mean : array_like
        Gaussian mean.
    dev : array_like
        Gaussian dispersion.

    Returns
    -------
    x : array_like
        Random Gaussian variables.

    """

    x = np.random.normal(mean, dev, size=len(mean))

    return x


def magnitude_generator(n, a=0.25, b=21):
    """
    Random G magnitude generator. Supposes a Gaia like distribution.

    Parameters
    ----------
    n : int
        number of points to be generated.
    a : float
        slope of log10 magnitude cumulative distribution.
        The default is 0.25
    b : float
        Cutting G magnitude.
        The default is 21.

    Returns
    -------
    x : array_like
        Random magnitudes.

    """

    u = np.random.uniform(low=0.0, high=1.0, size=n)

    x = np.log10(u) / a + b

    return x


def get_error_pm(n):
    """
    Random proper motion error generator. Supposes a Gaia like distribution.

    Parameters
    ----------
    n : int
        number of points to be generated.

    Returns
    -------
    x : array_like
        Random error in pmra.
    y : array_like
        Random error in pmdec.

    """

    m = magnitude_generator(n)

    x = 10 ** (0.26 * (m - 21.5))
    y = 10 ** (0.26 * (m - 21.7))

    return x, y


def get_error_rv(n):
    """
    Random radial velocity generator. Supposes a Gaia like distribution.

    Parameters
    ----------
    n : int
        number of points to be generated.

    Returns
    -------
    x : array_like
        Random error in radial velocity.

    """

    m = magnitude_generator(n)

    x = 10 ** (0.0026 * m * m * m - 0.061 * m * m + 0.56 * m - 2.7)

    return x


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------------------------------------------------------------------------
"Data conversion"
# ---------------------------------------------------------------------------


def cart6d_to_gaia(data, error=False):
    """
    Transforms a cartesian 6D file into Gaia-like coordinates.

    Parameters
    ----------
    data : array_like
        Cartesian data.
    error : boolean, optional
        True if the user wants to provide Gaia like velocity errors.
        The default is False.

    Returns
    -------
    data_gaia : array_like
        Original file in Gaia-like coordinates.

    """

    coord = angle.cart_to_radec(data[0], data[1], data[2], data[3], data[4], data[5])

    RA = coord[0]
    Dec = coord[1]
    DD = coord[2]
    PMRA = coord[3]
    PMDec = coord[4]
    v_LOS = coord[5]

    if error is False:
        data_gaia = np.asarray([RA, Dec, DD, PMRA, PMDec, v_LOS])
    else:
        ePMRA, ePMDec = get_error_pm(len(PMRA))
        ev_LOS = get_error_rv(len(v_LOS))

        PMRA = gaussian_generator(PMRA, ePMRA)
        PMDec = gaussian_generator(PMDec, ePMDec)
        v_LOS = gaussian_generator(v_LOS, ev_LOS)

        data_gaia = np.asarray([RA, Dec, DD, PMRA, ePMRA, PMDec, ePMDec, v_LOS, ev_LOS])

    return data_gaia.astype(float)
