"""
Errors that may be raised by functions and scripts within the bugdb package.
"""

import datetime


class ShotsNotFoundError(Exception):
    """Raised if a number of shots does not seem to exist."""

    def __init__(self, shot_nums):
        self.shot_nums = shot_nums

    def __str__(self):
        if hasattr(self.shot_nums, "__iter__"):
            return "Unable to find following shots: #" + ", #".join(f"{s}" for s in self.shot_nums) + "."
        else:
            return f"Unable to find shot #{self.shot_nums}."


class ShotInfoNotFoundError(Exception):
    """
    Raised if information about a shot is not found. info_type should be a string.
    """

    def __init__(self, shot_num, info_type):
        self.shot_num = shot_num
        self.info_type = info_type

    def __str__(self):
        return f"{self.info_type} for shot #{self.shot_num} not found."


class NoMatchingShotFoundError(Exception):
    """
    Raised if a shot is not found matching certain criteria.

    info_type should already be a string, and refers to the value being looked for instead of type(info_value)
    """

    def __init__(self, info_value, info_type):
        self.info_value = info_value
        self.info_type = info_type

    def __str__(self):

        # Format the info_value into a string first, depending on type
        if type(self.info_value) is datetime.datetime:
            info_string = self.info_value.strftime("%d.%m.%Y %H:%M:%S")
        else:
            # Fallback to letting python handle it
            info_string = f"{self.info_value}"

        return "No shot found with " + self.info_type + " of " + info_string


class NoDayFileFoundError(Exception):
    """Raised if not .dat file found for a given date."""

    def __init__(self, day):
        self.day = day

    def __str__(self):
        return "No .dat file found for " + self.day.strftime("%d.%m.%Y")
