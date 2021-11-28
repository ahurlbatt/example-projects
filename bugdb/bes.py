"""
Operations to obtain and process data from BES Diagnostic at BUG

A Hurlbatt, May 2021
"""


import datetime
from dateutil import tz
import pandas as pd
import numpy as np

from bugdb import shotinfo
from bugdb import errors

SHOTFILES = "/PATH/TO/FOLDER/"
BES_DB_FOLDER = "/PATH/TO/OTHER/FOLDER/"


def _parse_besfile_date(datfile_str):
    """Turn the strings found in the second column of BES DB .txt files into datetime objects."""

    datetime_out = datetime.datetime.strptime(datfile_str, "%d-%b-%Y %H:%M:%S")

    return datetime_out.replace(tzinfo=tz.gettz("Europe/Berlin"))


def dat_file_to_dataframe(datfile_name):
    """Read a BUG-DB BES .txt file into a pandas DataFrame."""

    # Read into a pandas DataFrame, including parsing the DateTime column
    datfile_contents = pd.read_csv(
        datfile_name, sep="\t", header=0, index_col=0, parse_dates=[1], date_parser=_parse_besfile_date
    )

    # Column headers have whitespace all over the place, so trim that
    return datfile_contents.rename(columns=str.strip)


def get_bes_dat_df(day):
    """
    Return the contents of the BES .txt file for the given day as a pandas.DataFrame.
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Build the file name. Subfolder is sorted by year.
    dat_file_name = BES_DB_FOLDER + day.strftime("%Y") + "/" + day.strftime("%Y%m%d") + "_BES.txt"

    # Read it in
    return dat_file_to_dataframe(dat_file_name)


def get_day_bes_info(day):
    """
    Extract the bes .dat file for a given day, and return as a pandas.DataFrame
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Get the contents of the day's .dat file, with possibility of no .dat file◘
    try:
        return get_bes_dat_df(day)
    except FileNotFoundError as exc:
        raise errors.NoDayFileFoundError(day) from exc


def get_bes_shots(shot_nums, ignoremissing: bool = False, longerthan: float = None):
    """
    Return a DataFrame containing all the information for list of shots and the BES results.

    Parameters
    ----------
    shot_nums : array-like
        List of shot numbers to fetch.
    ignoremissing : bool, optional
        If true, do not throw an error if shot info not found. The default is False.
    longerthan : float, optional
        Discard shots that have a beam shorter than this value. The default is None.

    Returns
    -------
    bes_df: pd.DataFrame
        pandas DataFrame of BES/shot info.

    """

    # Get a list of dates of the shots requested, with fallback for single shot
    try:
        all_datetimes = [shotinfo.shotnum_2_date(s) for s in shot_nums]
    except TypeError:
        all_datetimes = [shotinfo.shotnum_2_date(shot_nums)]

    # Get the uniqe dates
    all_dates = np.unique([d.date() for d in all_datetimes])

    # Get all of the possible BES information for these shots
    bes_df = pd.concat([get_day_bes_info(d) for d in all_dates])

    # Get all of the possible BUG information for these shots
    all_bug_df = pd.concat([shotinfo.get_day_db_info(d) for d in all_dates])

    # Merge the DFs, labelling duplicate columns
    bes_df = all_bug_df.join(bes_df, how="right", rsuffix="_frombes")

    # Drop the duplicate columns
    bes_df.drop(columns=bes_df.filter(regex="_frombes$"), inplace=True)

    # Trim down to that just from the requested shots, raising an error unless we're ignoring it
    # Double [[]] are to ensure we return a DataFrame, not a Series
    try:
        if hasattr(shot_nums, "__iter__"):
            bes_df = bes_df.loc[shot_nums]
        else:
            bes_df = bes_df.loc[[shot_nums]]
    except KeyError as exc:
        if ignoremissing:
            bes_df = bes_df.loc[bes_df.index.intersection(shot_nums)]
        else:
            # find which shots aren't found
            not_found = np.setdiff1d(shot_nums, bes_df.index)
            raise errors.ShotsNotFoundError(not_found) from exc

    # If we're asked to, get rid of any with a beam below longerthan
    if longerthan is not None:
        return bes_df[bes_df["t-on"] > longerthan]
    else:
        return bes_df
