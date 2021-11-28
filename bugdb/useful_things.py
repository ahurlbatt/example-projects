import numpy as np
import sklearn.cluster as skl_c
from scipy.signal import get_window


def group_df_by(df, group_param: str, n_groups: int, group_num_name: str = "group_num"):
    """
    Detect groups within the given parameter of a given DataFrame using KMeans Clustering.
    Group IDs are append to the DF as parameter 'group_num', unless a group_num_name is provided.
    Groups are sorted small --> big. Group centres are also returned separately.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be grouped.
    group_param : str
        DF parameter/column to group by.
    n_groups : int
        Number of groups in the DF.
    group_num_name : str, optional
        Name of the new column with group IDs. The default is 'group_num'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with group IDs appended.
    group_centres : np.ndarray
        The mean values of each group.

    """

    # Do automatic clustering on the group parameter
    kmeans = skl_c.KMeans(n_clusters=n_groups).fit(df[group_param].to_numpy().reshape(-1, 1))

    # The labels from the clustering are in arbitrary order. We want to sort them.
    # Get the indices of the cluster centres if they were to be sorted
    sort_idx = np.argsort(kmeans.cluster_centers_.reshape(-1))

    # Initialise a lookup table to translate between old and new labels
    lut = np.zeros_like(sort_idx)

    # The lookup table is the inverse permutation of sort_idx:
    lut[sort_idx] = np.arange(n_groups)

    # Use the lookup table to make a column that groups the scan
    df["group_num"] = lut[kmeans.labels_]

    # Also extract the group centres
    group_centres = kmeans.cluster_centers_.reshape(-1)[sort_idx]

    # Return the dataframe and optionally the group centres
    return df, group_centres


def roundto(value, target):
    """Round values to the closest multiple of the target."""
    return np.round(value / target) * target


def match_pattern_1d(values, pattern):
    """Return a list of indices for the starting postions of EXACT pattern matches in a 1D array-like of values"""
    idx = [np.s_[ii:len(values)-len(pattern)+1+ii] for ii in range(len(pattern))]
    return np.flatnonzero(np.all([values[idx[ii]] == pattern[ii] for ii in range(len(pattern))], axis=0))


def smooth_1D(y, x=None, width=None, window="boxcar"):
    """Smooth a 1D signal based on a given window width, optionally in the units of a specified x-axis.

    The signal (y) is smoothed by convolution with a normalised window of a certain length. The size of the window is
    determined by the 'width' parameter, which may be given in the same units as the provided x-axis. If no x-axis is
    given, then 'width' gives the numer of elements in the window. The window defaults to a 'boxcar' type, i.e. uniform.
    This can be changed by providing another name to the 'window' parameter. See scipy.signal.get_window() for types.


    Parameters
    ----------
    y : ndarray
        1D signal to be smoothed.
    x : ndarray, optional
        The x-axis of the signal. The step size of this is used for finding the window width. The default is None.
    width : float, optional
        Width of the window in array elements or units of 'x'. The default is None.
    window : str, optional
        The name of the type of window to use. The default is "boxcar".

    Returns
    -------
    ndarray
        The smoothed signal, the same size and shape as 'y'.

    """
    # If we're not given a window width, default to 5 elements. Otherwise calculate.
    if width is None:
        window_size = 5
    else:

        # Find the size of the window using 'width' either as a number of elements or in units of 'x'.
        if x is None:
            window_size = np.round(width).astype(int)
        else:
            window_size = np.round(width / np.mean(np.diff(x))).astype(int)

    # Convolve the input signal with the window, making sure the window is normalised (i.e. sum = unity)
    return np.convolve(y, (w := get_window(window, window_size, fftbins=False)) / np.sum(w), mode="same")
