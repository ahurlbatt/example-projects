"""
Functions used within the beamdiags module for various image analyses.

Included are wrappers around commonly used routines, mainly for brevity's sake, fitting routines using curve_fit and
least_squares as well as their prototype functions, and some more complex routines for extended functionality.
"""


import numpy as np
import skimage.feature as feature
import skimage.transform as transform
from scipy.special import erf
from scipy.optimize import curve_fit, least_squares
from scipy.spatial.distance import cdist


def homogeneous_transform(coords, matrix):
    """ A wrapper for a transformation of coordinates, via conversion to homogeneous coordinates. """
    # return (matrix @ np.pad(coords, ((0, 1), (0, 0)), constant_values=1.0))[0:2, :]
    return (dst := (matrix @ np.pad(coords, ((0, 1), (0, 0)), constant_values=1.0)))[0:-1, :] / dst[-1, :]


def weighted_projective_estimate(src, dst, weights=None):
    """Estimate the projective transformation matrix between 'src' and 'dst' with weights for each pair of points.

    This function is a recreation of the 'estimate' method of the class skimage.transform.ProjectiveTransform, but
    with an additional inclusion of uncertainties via linear algebra. It is essentially a least squares fitting, but a
    more in depth description of the algorithm is given there.


    Parameters
    ----------
    src : array-like, size (n, 2)
        Source coordinates.
    dst : array-like, size (n, 2)
        Destination coordinates.
    weights : array-like, size (n,), optional
        Vector of weight values for each pair of points. The default is None.

    Returns
    -------
    np.ndarray, size (3, 3)
        Estimated transformation matrix.

    """

    # The number of provided points to use.
    n = src.shape[0]

    # Normalise the provided coordinates, for stability of the matrix inversions.
    src_matrix, src = transform._geometric._center_and_normalize_points(src)
    dst_matrix, dst = transform._geometric._center_and_normalize_points(dst)

    # Construct the coefficient matrix for the corresponding set of homogeneous linear equations.
    A = np.block(
        [
            [src, np.ones((n, 1)), np.zeros((n, 3)), -dst[:, 0:1] * src, dst[:, 0:1]],
            [np.zeros((n, 3)), src, np.ones((n, 1)), -dst[:, 1:2] * src, dst[:, 1:2]],
        ]
    )

    if weights is None:
        # We can avoid some matrix calcuations if no weights have been provided
        _, _, V = np.linalg.svd(A)
    else:
        # Otherwise normalise, then duplicate the weights into a (2n, 2n) matrix.
        W = np.diag(np.tile(np.sqrt(weights.flatten() / np.max(weights)), 2))

        # After the weights are applied, use SVD to solve Ax=0
        _, _, V = np.linalg.svd(W @ A)
    # Check against poor fitting
    if np.isclose(V[-1, -1], 0):
        raise RuntimeError("Projective transform estimate returned degenerate transformation matrix.")
    # The transformation coordinates will go in here.
    H = np.zeros((3, 3))

    # The solution is in the last column of V
    H.flat[list(range(8)) + [-1]] = -V[-1, :-1] / V[-1, -1]
    H[2, 2] = 1

    # Invert and de-normalise
    H = np.linalg.inv(dst_matrix) @ H @ src_matrix

    # Make sure we recover having the last element as 1. Some small errors creep in through inversions.
    return H / H[-1, -1]


def average_around_pixel(pixel, image, bin_size=np.array([2, 2])):
    """Return the average value around a given pixel, using the given bin size."""

    # Generate the pixel meshes
    ii_pixels, jj_pixels = np.meshgrid(
        np.ceil(pixel[0] + bin_size[0] * np.array([-0.5, 0.5])).astype(int),
        np.ceil(pixel[1] + bin_size[1] * np.array([-0.5, 0.5])).astype(int),
    )

    # Extract the average, and create a vector if we're given a list of images.
    if type(image) == list:
        return np.array([np.mean(frame[ii_pixels, jj_pixels]) for frame in image])
    else:
        return np.mean(image[ii_pixels, jj_pixels])


def find_template(image, template, min_distance=None, threshold_abs=0.9):
    """Wrapper around match_template and peak_local_max from skimage.feature for matching templates in an image."""

    # If no minimum distance is given, use a little smaller than the smallest of the template dimensions
    if min_distance is None:
        min_distance = np.round(np.min(template.shape) * 0.9).astype(int)
    # Find the correlation of the image with the template
    corr_image = feature.match_template(image, template)

    # Find the peaks in the correlation image to find the matches.
    return feature.peak_local_max(corr_image, min_distance=min_distance, threshold_abs=threshold_abs)


def coords_to_integer_grid_2D(matrix_args, coords):
    """Prototype function for transforming a collection of points onto a regular, integer valued grid.

    Given a set of coordinates and a set of matrix elements, work out how close the transformation matrix brings
    those coordinates to aligning on an integer grid i.e. (0, 0), (0, 1), (1, 0), (1, 1) etc.

    The 8 matrix_args should be given as indicated below. The value X is always 1.
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, X]]


    Parameters
    ----------
    matrix_args : array-like, size (8,)
        Elements of the transformation matrix, as detailed above.
    coords : array-like, size (2, n)
        Coordinates of the points as column vectors.

    Returns
    -------
    grid_errors : np.ndarray, size (n,)
        Vector of distances of the transformed points from their nearest integer location.

    """

    # Create the transformation matrix from matrix_args
    H = np.concatenate((matrix_args, np.ones(1))).reshape((3, 3))

    # Transform the coordinates
    new_coords = homogeneous_transform(coords, H)

    # The grid errors are the difference between the transformed coordinates and these coordinates rounded to the
    # nearest integer.
    return np.sum((new_coords - np.round(new_coords)) ** 2, axis=0)


def fit_coords_to_integer_grid(coords, matrix_0=None):
    """Find a transformation matrix that can transform the given points onto a regular integer-valued 2D grid.

    From a given list of points, find a transformation matrix that minimises their distances from a regular and integer
    valued 2D grid. If no initial estimate is given, then one is calculated assuming that the provided points are close
    to regular. and do not require rotation. WARNING: this problem is quite unstable, and has many local minima that the
    solver likes to collapse into. If no initial estimate is known, and the number of points is large (> 20), it is
    recommended to first use a small number of points to find an initial estimate, otherwise there is a high probability
    of the returned matrix being a transformation to a diagonal line. It has also not been tested with large rotations.

    The transformation is a projective transformation via homogeneous coordinates, so the last element of the
    transformation matrix is always 1.


    Parameters
    ----------
    coords : array-like, size (2, n)
        The list of points to fit to, given as column vectors.
    matrix_0 : array-like, size (3, 3), optional
        Initial estimate of the transformation matrix. The default is None.

    Returns
    -------
    np.ndarray
        Estimate of the transformation matrix.

    """

    # If we've not been given an initial condition, estimate one
    if matrix_0 is None:

        # For an initial estimate, make a transform matrix that scales uniformly so that the minimum (non-zero)
        # distance between points is unity.
        all_distances = cdist(coords.T, coords.T)

        # Find the scale factor as 1/(minimum value that isn't zero)
        scale_factor = 1.0 / np.min(all_distances[all_distances != 0.0])

        # Build the initial matrix estimate as a simple scale transform.
        matrix_0 = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    # Get the relevant values from the transform matrix (i.e. all but the last)
    p0 = matrix_0.flatten()[0:8]

    # Perform least squares optimisation on the matrix
    ls_result = least_squares(fun=coords_to_integer_grid_2D, x0=p0, method="lm", args=(coords,),)

    # Raise an error if the fit hasn't converged for some reason
    if not ls_result.success:
        RuntimeError("Optimal parameters not found: " + ls_result.message)
    # Otherwise, return the estimated transformation matrix
    return np.concatenate((ls_result.x, np.ones(1))).reshape((3, 3))


def careful_coords_to_integer_grid(coords):
    """Run integer grid fitting on progressively larger numbers of points, to avoid local minima

    Parameters
    ----------
    coords : array-like, size (2, n)
        The list of points to fit to, given as column vectors.

    Returns
    -------
    np.ndarray
        Estimate of the transformation matrix.

    """
    # How many coordinates there are
    N = coords.shape[1]

    # Just run the fitting once if the number is low
    if N <= 15:
        return fit_coords_to_integer_grid(coords)
    # Get a first estimate with a low number of points
    matrix_estimate = fit_coords_to_integer_grid(coords[:, 0:10])

    # Improve the estimate with increasing numbers of points. Increase quadratically, roughly adding a layer each time
    for ii in np.arange(3, int(np.sqrt(N)) + 2) ** 2:
        matrix_estimate = fit_coords_to_integer_grid(coords[:, 0 : min(N, (10 + ii))], matrix_0=matrix_estimate)
    # Give back the estimate calculated with all points
    return matrix_estimate


def plate_over_background(x, intensity, width, middle, blur, background):
    """Prototype function for a blurry image of a plate over a background.

    This function models a 1D element with uniform value 'intensity', width 'width', and a central position 'middle'
    being imaged above a background with uniform value 'background' via a Gaussian blurring defined by the 'blur' value.
    This is done using the convolution of a Boxcar function and a Guassian function, which results in two back-to-back
    error-functions.


    Parameters
    ----------
    x : array-like
        The positional coordinate axes.
    intensity : float
        The absolute value of the object.
    width : float
        Width of the object, in the units of the x axis.
    middle : float
        Location of the centre of the object, in the units of the x axis.
    blur : float
        The standard deviation (i.e. sigma) of a Guassian blur from which the image suffers, in the units of the x axis.
    background : float
        Absolute value of the background.

    Returns
    -------
    np.ndarray
        Evaluation of the function at the points given in 'x'.

    """

    # The function - two scaled error functions plus a background.
    return background + (intensity - background) * (
        0.5 * erf((x - (middle - width)) / blur) + 0.5 * erf(((middle + width) - x) / blur)
    )


def fit_plate_over_background(x, y, width_estimate=5.0, blur_estimate=1.0, lsq_kwargs=None):
    """Fit a 1D section taken from a blurry image of a plate over a background.

    Uses scipy.optimize.curve_fit() to estimate the parameters required to fit plate_over_background() to the data
    provided in the x and y vectors.


    Parameters
    ----------
    x : array-like, size (n,)
        Positional axis of the data.
    y : array-like, size (n,)
        Values taken from the image.
    width_estimate : float, optional
        Initial estimate of the width of the object, in the units of 'x'. The default is 5.
    blur_estimate : float, optional
        Initial estimate of image blurring, in the units of 'x'.. The default is 1.

    Returns
    -------
    (dict, dict)
        A pair of dicts containing the fitted parameters and their uncertainties, respectively.

    """

    # Build the initial estimates
    p0 = (np.max(y), width_estimate, np.mean(x), blur_estimate, np.min(y))

    # Fitting options to pass through to least_squares, default to empty.
    if lsq_kwargs is None:
        lsq_kwargs = {}
    # Run the fitting, which returns the optimised parameters and the covariance matrix
    p_fit, p_cov = curve_fit(plate_over_background, x, y, p0=p0, bounds=(0, np.inf), **lsq_kwargs)

    # A list of the names of the parameters being fitted
    param_names = ["intensity", "width", "middle", "blur", "background"]

    # Extract the uncertainty in the fit parameters as sigma = sqrt(variance) (assuming no correlation).
    p_sigma = np.sqrt(np.diagonal(p_cov))

    # Return both the fitted values and the uncertainties as dicts of parameters
    return dict(zip(param_names, p_fit)), dict(zip(param_names, p_sigma))


def xy_line_cross(x_line, y_line):
    """Find the crossing point of two straight lines in 2D space given their coefficients.

    The two lines are defined orthoginally from each other:
        x_line = [a, b] --> y = a*x + b
        y_line = [p, q] --> x = p*y + q

    From rearranging and equating:
        x = (p*b + q)/(1 - a*p)
        y = (a*q + b)/(1 - a*p)


    Parameters
    ----------
    x_line : array-like, size (2,)
        [a, b] of a line defined by y = a*x + b.
    y_line : array-like, size (2,)
        [p, q] of a line defined by x = p*y + q.

    Returns
    -------
    np.ndarray, size (2,)
        (x, y) coordinates of the crossing point of the two lines.

    """

    x = (y_line[0] * x_line[1] + y_line[1]) / (1 - x_line[0] * y_line[0])
    y = (x_line[0] * y_line[1] + x_line[1]) / (1 - x_line[0] * y_line[0])

    return np.array((x, y,))
