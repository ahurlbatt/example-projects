"""
A collection of functions for querying information from the BUG database, including electrical measurements and
other basic shot information.
"""


import datetime
from dateutil import tz
import math
import os
import pandas as pd
import numpy as np

from bugdb import errors

SHOTFILES = "/PATH/TO/FOLDER/"
DB_FOLDER = "/PATH/TO/OTHER/FOLDER/"
SIMATIC_FILE = "/PATH/TO/FILE"


def _parse_datfile_date(datfile_str):
    """Turn the strings found in the second column of DB .dat files into datetime objects."""

    datetime_out = datetime.datetime.strptime(datfile_str, "%d%b%y %H:%M:%S")

    return datetime_out.replace(tzinfo=tz.gettz("Europe/Berlin"))


def dat_file_to_dataframe(datfile_name):
    """Read a BUG-DB .dat (or .set) file into a pandas DataFrame."""

    # Read into a pandas DataFrame, including parsing the DateTime column
    try:
        datfile_contents = pd.read_csv(
            datfile_name,
            sep="|",
            header=0,
            index_col=0,
            parse_dates=[1],
            date_parser=_parse_datfile_date,
            encoding="ansi",
        )
    except UnicodeDecodeError as exc:
        print(datfile_name)
        raise exc

    # Column headers have whitespace all over the place, so trim that
    return datfile_contents.rename(columns=str.strip)


def get_dat_df(day):
    """
    Return the contents of the .dat file for the given day as a pandas.DataFrame.
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Build the file name. Subfolder is sorted by year.
    dat_file_name = DB_FOLDER + day.strftime("%Y") + "/" + day.strftime("%Y_%m_%d") + ".dat"

    # Read it in
    return dat_file_to_dataframe(dat_file_name)


def get_set_df(day):
    """
    Return the contents of the .set file for the given day as a pandas.DataFrame.
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Build the file name. Subfolder is sorted by year.
    set_file_name = DB_FOLDER + day.strftime("%Y") + "/" + day.strftime("%Y_%m_%d") + ".set"

    # Read it in
    return dat_file_to_dataframe(set_file_name)


def get_current_shot_num():
    """Return the current shot number, as held by SIMATIC."""

    # !! If SIMATIC is refreshing the file, it will be empty when we try to read it. !!

    # Initialise the file contents lit, so that we can check a successful read
    sim_file_lines = []

    while len(sim_file_lines) == 0:
        # In case it's being refreshed, repeat the read until we get a non-zero number of lines
        with open(SIMATIC_FILE, "r") as sim_file:
            sim_file_lines = sim_file.read().splitlines()

    # Shot number is stored as the first item on the first line.
    return int(sim_file_lines[0].split()[0])


def shotnum_2_date(shot_num):
    """
    Find the timestamp of the provided shot number by reading the SIMATIC comments for the shot.

    There are two systems of files, which changed on shot number 113001. Before this, data was recorded but not
    timestamped. The best option here is to use the creation time of one of the data files. After this, the
    timestamp is recorded in a 'comments.txt', which is the 'Rechnerdaten_schreiben.txt' file for the shot with
    an extra line at the end, on which can be found the timestamp.

    """

    my_gantner_folder = SHOTFILES + f"GANTNER/{math.floor(shot_num/1000)}/"

    if shot_num < 113000:
        # Old system -> look for a data file to find the timestamp of

        my_shot_files = [f for f in os.listdir(my_gantner_folder) if f[0:6] == f"{shot_num}"]

        if my_shot_files:
            # Get modification time of one of these files
            shot_datetime = datetime.datetime.fromtimestamp(os.path.getctime(my_gantner_folder + my_shot_files[0]))
        else:
            raise errors.ShotInfoNotFoundError(shot_num, "Gantner file for timestamp")

    else:
        # New system -> Last (non-empty) line of comments.txt

        # If shot_num is ABCDEF then Gantner sub-folder is /ABC/DEF/, and we want comments.txt
        my_shot_file = my_gantner_folder + f"{(shot_num % 1000):03d}/comments.txt"

        try:
            with open(my_shot_file, "r") as comments_file:
                comments_file_lines = comments_file.read().splitlines()
        except FileNotFoundError as exc:
            raise errors.ShotsNotFoundError(shot_num) from exc

        # Timestamp should be in last line of file
        try:
            shot_datetime = datetime.datetime.strptime(comments_file_lines[-1], "%d %b %Y; %H:%M:%S")
        except Exception as exc:
            raise errors.ShotsNotFoundError(shot_num) from exc

    # Add the timezone, assuming it was always created using local time
    return shot_datetime.replace(tzinfo=tz.gettz("Europe/Berlin"))


def datetime_2_shotnum(datetime_input):
    """
    Find the shot number that corresponds to the provided datetime object. This is done by using the BUG database,
    which is stored on a day-by-day basis to narrow down to a specific day, then finding the shot on that day that
    matched the time closest. As the timing at BUG is a bit messed up, there can be small differences in the timing
    with errors on the order of seconds. For this reason the time is matched to the nearest minute.
    """

    # First check if a timezone is given. If not, assume the local timezone
    if datetime_input.tzinfo is None:
        datetime_input = datetime_input.replace(tzinfo=tz.gettz("Europe/Berlin"))

    # Get the contents of the day file
    day_file_df = get_day_db_info(datetime_input)

    # Get the shot that has the closest time to requested one
    closest_shot_df = day_file_df.sort_values(by="Date, Time", key=lambda dt: abs(dt - datetime_input)).head(1)

    # Check that this shot matches to within a given threshold (currently hardcoded to 60 seconds)
    SHOT_TIME_THRESHOLD = datetime.timedelta(seconds=60)

    if abs(closest_shot_df["Date, Time"].iat[0] - datetime_input) > SHOT_TIME_THRESHOLD:
        raise errors.NoMatchingShotFoundError(datetime_input, "DateTime")

    # The dataframe is indexed by shot number, so just return the first (only) index
    return closest_shot_df.index[0]


def get_day_db_info(day):
    """
    Extract and combine the .dat and .set files for a given date, returning as a pandas.DataFrame
    'day' is a datetime.datetime object, and it doesn't matter if there are hours/minutes/seconds included
    """

    # Get the contents of the day's .dat file, with possibility of no .dat file
    try:
        dat_file_df = get_dat_df(day)
    except FileNotFoundError as exc:
        raise errors.NoDayFileFoundError(day) from exc

    # As it's used by many things, but not in the DB by default, calculate the HV Ratio
    dat_file_df["HV-ratio"] = dat_file_df["U-acc"] / dat_file_df["U-extr"]

    # If the .dat file exists, then the .set file has to, too. Get it.
    set_file_df = get_set_df(day)

    # Merge the two DFs, labelling duplicate columns from the .set file
    return dat_file_df.join(set_file_df, how="left", rsuffix="_set")


def fetch_shot_params(shot_nums, ignoremissing: bool = False):
    """
    Return all of the parameters found for a given list of shots from both the .dat and .set files in the BUG DB.
    Data is returned as a pandas.DataFrame with the shot number as the row index and column names matching the
    .dat file. Where the .set file has column names matching those in the .dat file, '_set' is appended.
    """

    # Get a list of dates of the shots requested, with fallback for single shot
    try:
        all_datetimes = [[] for _ in shot_nums]
        for ii, s in enumerate(shot_nums):
            try:
                all_datetimes[ii] = shotnum_2_date(s)
            except errors.ShotsNotFoundError as exc:
                if ignoremissing:
                    continue
                else:
                    raise exc
    except TypeError:
        all_datetimes = [shotnum_2_date(shot_nums)]

    # Get the uniqe dates, with truthy-ness testing for empty values.
    all_dates = np.unique([d.date() for d in all_datetimes if d])

    # Get all of the possible BUG information for these shots
    day_info = [[] for _ in all_dates]
    for ii, d in enumerate(all_dates):
        try:
            day_info[ii] = get_day_db_info(d)
        except errors.NoDayFileFoundError as exc:
            if ignoremissing:
                continue
            else:
                raise exc

    df = pd.concat([d for d in day_info if type(d) == pd.DataFrame])

    # Return the information
    try:
        if hasattr(shot_nums, "__iter__"):
            return df.loc[shot_nums]
        else:
            return df.loc[[shot_nums]]
    except KeyError as exc:
        if ignoremissing:
            return df.loc[df.index.intersection(shot_nums)]
        else:
            # find which shots aren't found
            not_found = np.setdiff1d(shot_nums, df.index)
            raise errors.ShotsNotFoundError(not_found) from exc
