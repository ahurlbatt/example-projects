"""Operations to obtain and process data from CFC Diagnostic at BUG"""


import datetime
from dateutil import tz
import pandas as pd
import numpy as np
import pathlib
import math

from bugdb import shotinfo
from bugdb import errors


SHOTFILES = "/PATH/TO/FOLDER/"
CFC_DB_FOLDER = "/PATH/TO/OTHER/FOLDER/"
SMB_SHOTFILES = "/PATH/TO/ANOTHER/FOLDER/"


def _parse_cfcfile_date(datfile_str):
    """Turn the strings found in the second column of CFC DB .txt files into datetime objects."""

    datetime_out = datetime.datetime.strptime(datfile_str, "%d-%b-%Y %H:%M:%S")

    return datetime_out.replace(tzinfo=tz.gettz("Europe/Berlin"))


def dat_file_to_dataframe(datfile_name):
    """Read a BUG-DB CFC .txt file into a pandas DataFrame."""

    # Read into a pandas DataFrame, including parsing the DateTime column
    datfile_contents = pd.read_csv(
        datfile_name, sep="\t", header=0, index_col=0, parse_dates=[1], date_parser=_parse_cfcfile_date
    )

    # Column headers have whitespace all over the place, so trim that
    return datfile_contents.rename(columns=str.strip)


def get_cfc_dat_df(day):
    """
    Return the contents of the CFC .txt file for the given day as a pandas.DataFrame.
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Build the file name. Subfolder is sorted by year.
    dat_file_name = CFC_DB_FOLDER + day.strftime("%Y") + "/" + day.strftime("%Y%m%d") + ".txt"

    # Read it in
    return dat_file_to_dataframe(dat_file_name)


def get_day_cfc_info(day):
    """
    Extract the CFC .dat file for a given day, and return as a pandas.DataFrame
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Get the contents of the day's .dat file, with possibility of no .dat file
    try:
        return get_cfc_dat_df(day)
    except FileNotFoundError as exc:
        raise errors.NoDayFileFoundError(day) from exc


def get_cfc_shots(shot_nums, ignoremissing=False):
    """
    Return a DataFrame containing all the information for list of shots and the CFC results.

    Parameters
    ----------
    shot_nums : array-like
        List of shot numbers to fetch.
    ignoremissing : bool, optional
        If true, do not throw an error if shot info not found. The default is False.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame of CFC/shot info.

    """

    # Get a list of dates of the shots requested, with fallback for single shot
    try:
        all_datetimes = list()
        for s in shot_nums:
            try:
                all_datetimes.append(shotinfo.shotnum_2_date(s))
            except errors.ShotsNotFoundError as err:
                if ignoremissing:
                    continue
                else:
                    raise err
    except TypeError:
        all_datetimes = [shotinfo.shotnum_2_date(shot_nums)]

    # Get the uniqe dates
    all_dates = np.unique([d.date() for d in all_datetimes])

    # Get all of the possible CFC information for these shots
    cfc_day_dfs = list()
    for date in all_dates:
        try:
            cfc_day_dfs.append(get_day_cfc_info(date))
        except Exception as err:
            if ignoremissing:
                continue
            else:
                raise err

    # Get all of the possible BUG information for these shots
    bug_day_dfs = list()
    for date in all_dates:
        try:
            bug_day_dfs.append(shotinfo.get_day_db_info(date))
        except Exception as err:
            if ignoremissing:
                continue
            else:
                raise err

    # Merge the DFs, labelling duplicate columns
    cfc_df = pd.concat(bug_day_dfs).join(pd.concat(cfc_day_dfs), how="right", rsuffix="_fromcfc")

    # Drop the duplicate columns
    cfc_df.drop(columns=cfc_df.filter(regex="_fromcfc$"), inplace=True)

    # Trim down to that just from the requested shots, raising an error unless we're ignoring it
    # Double [[]] are to ensure we return a DataFrame, not a Series
    try:
        if hasattr(shot_nums, "__iter__"):
            cfc_df = cfc_df.loc[shot_nums]
        else:
            cfc_df = cfc_df.loc[[shot_nums]]
    except KeyError as exc:
        if ignoremissing:
            return cfc_df.loc[cfc_df.index.intersection(shot_nums)]
        else:
            # find which shots aren't found
            not_found = np.setdiff1d(shot_nums, cfc_df.index)
            raise errors.ShotsNotFoundError(not_found) from exc


def shotnum_to_cfc_seq(shot_num):

    my_file = pathlib.Path(SMB_SHOTFILES + f"/{math.floor(shot_num/1000):03d}/BUG_CFC_{shot_num}.seq")

    if not my_file.exists():
        FileNotFoundError(str(my_file))

    return my_file
