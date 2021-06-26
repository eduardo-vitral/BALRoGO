"""
Created on 2020

@author: Eduardo Vitral
"""
###############################################################################
#
# 2020, Paris
#
# This file is based on the Python corner package (Copyright 2013-2016
# Dan Foreman-Mackey & contributors, The Journal of Open Source Software):
# https://joss.theoj.org/papers/10.21105/joss.00024
# I have done some modifications on it so it allows some new features and
# so it takes into account some choices as default. I thank Gary Mamon
# for his good suggestions concerning the plot visualization.
#
###############################################################################

from __future__ import print_function, absolute_import
import logging
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

__all__ = ["corner", "hist2d", "quantile"]


def dgauss(x, mu, sig, A):
    f = A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return f


def corner(
    xs,
    range=None,
    weights=None,
    colormain="tab:blue",
    colorbackgd=(250 / 255, 245 / 255, 228 / 255),
    colorhist=(25 / 255, 70 / 255, 104 / 255),
    colorgauss="tab:green",
    labels=None,
    smooth=True,
    smooth1d=False,
    gauss_prior=None,
    gauss_mu=None,
    gauss_sig=None,
    truths=None,
    reals=None,
    truth_color="red",
    reals_color="orange",
    marker="X",
    own_dist=True,
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=4,
    use_math_text=True,
    label_size=None,
    tick_size=None,
    dist_title=None,
    dist_ylab=None,
    tick_rotate=0,
    title_above=True,
    tick_above=True,
    prior_display=0,
):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like[nsamples, ndim]
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    range : float(ndim,2)
        Array containing the ranges to be used in the plots.

    weights : array_like[nsamples,]
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    color : str
        A ``matplotlib`` style color for all histograms.

    labels : iterable (ndim,)
        A list of names for the dimensions. If a ``xs`` is a
        ``pandas.DataFrame``, labels will default to column names.

    smooth, smooth1d : float
       The standard deviation for Gaussian kernel passed to
       `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
       respectively. If `None` (default), no smoothing is applied.

    gauss_prior : boolean array
        Boolean array with size equals the number of displayed parameters.
        It is True if the variable had Gaussian priors and False if not.
        It will display the relative box with green contours.

    gauss_mu : array
        Array containing the mean of the Gaussian priors.

    gauss_sig : array
        Array containing the dispersion of the Gaussian priors.

    truths : iterable (ndim,)
        A list of reference values to indicate on the plots.  Individual
        values can be omitted by using ``None``.

    reals : iterable (ndim,)
        A list of second reference values to indicate on the plots.  Individual
        values can be omitted by using ``None``.

    truth_color : str
        A ``matplotlib`` style color for the ``truths`` makers.

    reals_color : str
        A ``matplotlib`` style color for the ``reals`` makers.

    marker : str
        A ``matplotlib`` marker style. The default is "X".

    own_dist : Boolean
        True if the user wants to use specific implemantations for ticks/
        tiles/labels sizes. False if the user wants to use it as in the
        Python corner package.

    scale_hist : bool
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool
        If true, print the values of the computed quantiles.

    max_n_ticks: int
        Maximum number of ticks to try to use

    fig : matplotlib.Figure
        Overplot onto the provided figure object.

    use_math_text : bool
        If true, then axis tick labels for very large or small exponents will
        be displayed as powers of 10 rather than using `e`.

    label_size : float
        Size of plot labels

    tick_size : float
        Size of plot ticks

    dist_title : float
        Distance between the title and the box

    dist_ylab : float
        Distance between the y-axis title and the box

    tick_rotate : float
        Rotation angle of x-ticks, in degrees.

    title_above : Boolean
        Boolean indicating it the user wants the label above the box.

    tick_above : Boolean
        Boolean indicating it the user wants the ticks above the box.

    prior_display : int
        Integer indicating the kind of display of the gaussian prior:
            0: No display: Default.
            1: Green box: Must provide "gauss_prior".
            2: Green Gaussian: Must provide valid "gauss_prior" and "range".

    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if quantiles is None:
        quantiles = []

    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = np.shape(xs)[1] * [""]
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more " "dimensions than samples!"
    )

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    range_provided = False
    # Parse the parameter ranges.
    if range is None:

        range = [[x.min(), x.max()] for x in xs]
        # Check for parameters that never change.
        m = np.array([e[0] == e[1] for e in range], dtype=bool)
        if np.any(m):
            raise ValueError(
                (
                    "It looks like the parameter(s) in "
                    "column(s) {0} have no dynamic range. "
                    "Please provide a `range` argument."
                ).format(", ".join(map("{0}".format, np.arange(len(m))[m])))
            )

    else:
        range_provided = True
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range = list(range)
        for i, _ in enumerate(range):
            try:
                emin, emax = range[i]
            except TypeError:
                q = [0.5 - 0.5 * range[i], 0.5 + 0.5 * range[i]]
                range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    bins = np.zeros(len(range)).astype(int)

    # Define labels size
    n_params = len(range)
    if n_params > 10:
        var = 10
    else:
        var = n_params

    if n_params > 6:
        max_n_ticks -= 1
    # if (n_params > 4) :
    #     max_n_ticks -= 1

    # Change here the sizes of labels/titles and ticks
    if label_size is None:
        label_size = var * 0.625 + 14
    if tick_size is None:
        tick_size = var * 0.5 + 13
    if dist_title is None:
        if tick_rotate == 0:
            dist_title = var * 0.008 + 1.15
        else:
            dist_title = (var * 0.008 + 1.15) * 1.1
    if dist_ylab is None:
        dist_ylab = var * 0.025 + 0.25

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0  # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except ValueError:
            raise ValueError(
                "Provided figure has {0} axes, but data has "
                "dimensions K={1}".format(len(fig.axes), K)
            )

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    for i, x in enumerate(xs):

        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]
        # Plot the histograms.

        # Define the bin quantity for each histogram
        q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84], weights=weights)
        q_m, q_p = q_50 - q_16, q_84 - q_50
        bins[i] = int((np.amax(range[i]) - np.amin(range[i])) / (min(q_m, q_p) / 4))

        if smooth1d:
            smooth1d = bins[i]

        if smooth1d is False:
            n, b, p = ax.hist(
                x,
                bins=bins[i],
                weights=weights,
                range=np.sort(range[i]),
                ec="k",
                histtype="stepfilled",
                color=colormain,
            )
            if gauss_prior is not None:
                if gauss_prior[i]:
                    if prior_display == 2:
                        if range_provided:
                            xx = np.linspace(min(range[i]), max(range[i]), 50)
                            disp0 = gauss_sig[i]
                            mu0 = gauss_mu[i]
                            amp = np.amax(n)
                            ax.plot(
                                xx,
                                dgauss(xx, mu0, disp0, amp),
                                color=colorgauss,
                                lw=4,
                                alpha=0.8,
                            )

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(2.2)
                if gauss_prior is not None:
                    if prior_display == 0:
                        prior_display = 1
                    if gauss_prior[i]:
                        if prior_display == 1:
                            ax.spines[axis].set_color(colorgauss)

        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")

            n, b, p = ax.hist(
                x,
                bins=bins[i],
                weights=weights,
                range=np.sort(range[i]),
                ec="k",
                histtype="stepfilled",
                color=colormain,
            )
            if gauss_prior is not None:
                if gauss_prior[i]:
                    if prior_display == 2:
                        if range_provided:
                            xx = np.linspace(min(range[i]), max(range[i]), 50)
                            disp0 = gauss_sig[i]
                            mu0 = gauss_mu[i]
                            amp = np.amax(n)
                            ax.plot(
                                xx,
                                dgauss(xx, mu0, disp0, amp),
                                color=colorgauss,
                                lw=4,
                                alpha=0.8,
                            )

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(2.2)
                if gauss_prior is not None:
                    if prior_display == 0:
                        prior_display = 1
                    if gauss_prior[i]:
                        if prior_display == 1:
                            ax.spines[axis].set_color(colorgauss)

        if truths is not None and truths[i] is not None:

            arrow_width = (np.amax(b) - np.amin(b)) / 20
            arrow_length = (np.amax(n) - np.amin(n)) / 12

            ax.arrow(
                truths[i],
                0,
                0,
                np.amax(n) * 0.6,
                head_width=arrow_width,
                head_length=arrow_length,
                color=truth_color,
                width=arrow_width / 4,
                zorder=10,
            )

        if reals is not None and reals[i] is not None:

            arrow_width = (np.amax(b) - np.amin(b)) / 20
            arrow_length = (np.amax(n) - np.amin(n)) / 12

            ax.arrow(
                reals[i],
                0,
                0,
                np.amax(n) * 0.4,
                head_width=arrow_width,
                head_length=arrow_length,
                color=reals_color,
                width=arrow_width / 4,
                zorder=10,
            )

        # Add in the column name if it's given.
        if labels is not None:
            title = "{0}".format(labels[i])
        else:
            raise ValueError("You did not give labels for your parameters.")

        if title is not None:
            if title_above:
                if own_dist:
                    ax.set_title(title, y=dist_title, fontsize=label_size)
                else:
                    ax.set_title(title)
        # Set up the axes.
        ax.set_xlim(range[i])
        ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
        ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))

        ax.tick_params(
            labeltop=tick_above,
            labelright=False,
            top=True,
            right=True,
            axis="both",
            which="major",
            direction="in",
            length=8,
            width=1.3,
            labelsize=tick_size,
        )
        ax.tick_params(
            labeltop=tick_above,
            labelright=False,
            top=True,
            right=True,
            axis="both",
            which="minor",
            direction="in",
            length=4,
            width=1.3,
            labelsize=tick_size,
        )
        if tick_rotate != 0:
            ax.tick_params(axis="x", labelrotation=tick_rotate)
        ax.minorticks_on()

        # Take off ticks that overlap left neighbour
        locations = ax.xaxis.get_ticklocs()
        size_x = max(range[i]) - min(range[i])
        if np.amin(locations) - min(range[i]) < size_x / 9:
            ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
        if np.amin(locations) - min(range[i]) < size_x / 10:
            ax.xaxis.get_major_ticks()[0].label2.set_visible(False)

        if i < K - 1:
            pass

        else:
            [labs.set_rotation(tick_rotate) for labs in ax.get_xticklabels()]
            if labels is not None:
                if own_dist:
                    ax.set_xlabel(labels[i], fontsize=label_size)
                else:
                    ax.set_xlabel(labels[i])
                if tick_rotate != 0:
                    ax.xaxis.set_label_coords(0.5, -dist_ylab)
                else:
                    ax.xaxis.set_label_coords(0.5, -0.2)

        # use MathText for axes ticks
        ax.xaxis.set_major_formatter(
            ScalarFormatter(useMathText=use_math_text, useOffset=False)
        )

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(
                y,
                x,
                ax=ax,
                range=[range[j], range[i]],
                weights=weights,
                ccolormain=colormain,
                ccolorbackgd=colorbackgd,
                ccolorhist=colorhist,
                bins=[bins[j], bins[i]],
                smooth=smooth,
                ticksize=tick_size,
            )

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    # "What's at the X? Pirate treasure?"
                    ax.plot(
                        truths[j], truths[i], marker, color=truth_color, markersize=10
                    )

            if reals is not None:
                if reals[i] is not None and reals[j] is not None:
                    # "What's at the X? Pirate treasure?"
                    ax.plot(reals[j], reals[i], marker, color=reals_color, markersize=7)

            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [labs.set_rotation(tick_rotate) for labs in ax.get_xticklabels()]
                if labels is not None:

                    # This concerns the titles on the bottom of the corner plot
                    if own_dist:
                        ax.set_xlabel(labels[j], fontsize=label_size)
                    if tick_rotate != 0:
                        ax.xaxis.set_label_coords(0.5, -dist_ylab)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.2)

                # Take off ticks that overlap left neighbour
                locations = ax.xaxis.get_ticklocs()
                size_x = max(range[j]) - min(range[j])
                if np.amin(locations) - min(range[j]) < size_x / 10:
                    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
                if max(range[j] - np.amax(locations)) < size_x / 10:
                    arg = len(locations) - 1
                    if max(range[j]) - locations[arg] < size_x / 10:
                        ax.xaxis.get_major_ticks()[arg].label1.set_visible(False)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [labs.set_rotation(tick_rotate) for labs in ax.get_yticklabels()]
                if labels is not None:
                    # This concerns the titles in the left side of the corner
                    # plot
                    if own_dist:
                        ax.set_ylabel(labels[i], fontsize=label_size)
                        ax.yaxis.set_label_coords(-dist_ylab, 0.5)
                    if tick_rotate != 0:
                        ax.xaxis.set_label_coords(0.5, -dist_ylab)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.2)

                # Take off ticks that overlap bottom neighbour
                locations = ax.yaxis.get_ticklocs()
                size_x = max(range[i]) - min(range[i])
                if np.amin(locations) - min(range[i]) < size_x / 10:
                    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    return fig


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def hist2d(
    x,
    y,
    bins=None,
    ticksize=5,
    range=None,
    weights=None,
    levels=None,
    smooth=True,
    ax=None,
    ccolormain=None,
    ccolorbackgd=None,
    ccolorhist=None,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    no_fill_contours=False,
    fill_contours=True,
):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """
    if ax is None:
        ax = pl.gca()

    ax.tick_params(
        labeltop=False,
        labelright=False,
        top=True,
        right=True,
        axis="both",
        which="major",
        direction="in",
        length=8,
        width=1.3,
        labelsize=ticksize,
    )
    ax.tick_params(
        labeltop=False,
        labelright=False,
        top=True,
        right=True,
        axis="both",
        which="minor",
        direction="in",
        length=4,
        width=1.3,
        labelsize=ticksize,
    )
    ax.minorticks_on()

    ax.set_facecolor(ccolorbackgd)

    # Set the default range based on the data range if not provided.
    if range is None:
        range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if ccolormain is None:
        ccolormain = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    color_hist = ccolorhist
    rgba_color = colorConverter.to_rgba(color_hist)
    contour_cmap = [list(rgba_color) for levs in levels] + [rgba_color]
    for i, levs in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins,
            range=list(map(np.sort, range)),
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )

    if smooth:

        factor = 5
        sig_smooth = [factor * np.std(x.flatten()), factor * np.std(y.flatten())]

        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, sig_smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except ImportError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(
            X2, Y2, H2.T, [V.min(), H.max()], cmap=white_cmap, antialiased=False
        )

    if plot_contours and fill_contours:
        contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
        ax.contourf(
            X2,
            Y2,
            H2.T,
            np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
            **contourf_kwargs
        )

    # Plot the contour edge colors.
    if plot_contours:
        contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color_hist)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.2)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
