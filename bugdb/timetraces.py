"""Provides the TimeTrace class that creates a useable object from Gantner binary files of a given shot number

Create a TimeTrace object with TimeTrace(shot_number[, module name, sample rate, force cache refresh]).

The .dat files created by the Gantner DAQ system are decoded using their provided tool. This tool is not especially
friendly for automisation, as it creates a dialogue box for each file ... We're stuck with it until someone reverse
engineers the binary data type. Upon creation, a cached text file will be created locally in the directory
decode_cache. They can be overwritten by setting the "force_cache_refresh" flag.

Data is held in TimeTrace.DataFrame indexed by time (seconds from start of acquisition). Individual time traces can
be accessed as attributes of the TimeTrace object.

Information on variable types is held in TimeTrace.VariableDefinitions indexed by variable name, but can also by
accessed as attributes like TimeTrace.Unit.{variable name}.

A list of variable names is held as TimeTrace.Names

"""
import time
import math
import shutil
import pathlib
import subprocess
import pandas as pd
import numpy as np

from bugdb import useful_things


MY_FOLDER = pathlib.Path(__file__).parent.resolve()
DECODE_CMD = str(MY_FOLDER.joinpath("DecodeBin2Ascii.exe"))
LOCAL_CACHE = MY_FOLDER.joinpath("decode_cache")
SHOTFILES = pathlib.Path("//afs/ipp-garching.mpg.de/home/n/niti/shotfiles")
KILL_ARGS = ["powershell", "Stop-Process", "-Id"]


class TimeTrace(object):
    def __init__(self, shot_num: int, module: str = "A", sample_rate: int = None, force_cache_refresh: bool = False):
        """Create an object that can be queried for time traces held for a specific BUG shot"""

        # The default sample rate depends on the module being asked for. Currently only A and B modules are known about.
        if sample_rate is None:
            if module == "A":
                sample_rate = 20
            elif module == "B":
                sample_rate = 100
            else:
                raise NotImplementedError(f"No default sample rate known for odule with label '{module}'")

        # Get the name of the file in the cache, and make it if we need to.
        txt_file = _create_cached_txt(shot_num, module, sample_rate, force_cache_refresh)

        # Parse the header to get information on the file and the variables/columns
        self.FileInfo, self.VariableDefinitions, header_lines = self._parse_header(txt_file)

        # Include the input information in the file info
        self.FileInfo["ShotNumber"] = shot_num
        self.FileInfo["Module"] = module

        # Extract the data into a DF, using what we learned from the header.
        self.DataFrame = pd.read_csv(
            txt_file,
            sep="\t",
            header=header_lines,
            index_col=False,
            names=["Seconds"] + list(self.VariableDefinitions.index),
            encoding="ansi",
        )

        # The files have delimiters at line ends, so we had to make "Seconds" a column, but now we can make it the index
        # We make it relative first, then round to the sample rate as we've changed the precision of the data
        self.DataFrame.Seconds = useful_things.roundto(
            self.DataFrame.Seconds - self.DataFrame.Seconds.iloc[0], 1.0 / sample_rate
        )
        self.DataFrame.set_index("Seconds", inplace=True)

        # That's all we need. The rest is the getattr overloading.
        return None

    def __getattr__(self, attribute):
        """Attributes are overloaded to return pd.Series of either a time trace or a type of variable property"""
        if attribute in self.DataFrame.columns:
            return self.DataFrame[attribute]
        elif attribute in self.VariableDefinitions.columns:
            return self.VariableDefinitions[attribute]
        else:
            return object.__getattribute__(self, attribute)

    @property
    def Names(self):
        """Quick access to the variable names"""
        return list(self.DataFrame.columns)

    @staticmethod
    def _parse_header(file):
        """Detect the text file header and extract information about the file and its contents"""

        def split_defs(line):
            return [item for item in line.split(" / ") if len(item) > 0]

        def defs_to_dict(defs):
            return dict([[element.strip() for element in item.split(": ", maxsplit=1)] for item in defs])

        # Read lines until we find one that's all stars to get the file header
        with file.open() as f:
            header = [f.readline().strip("\n")]
            while not header[-1] == "*" * len(header[-1]):
                header += [f.readline().strip("\n")]
        # Variable definition lines can be recognised easily
        is_variable_definition = [line[0:8] == "NameLen:" for line in header]

        # Flatten out all of the definitions that aren't variable defs and parse
        FileInfo = defs_to_dict(
            [
                item
                for line in header[0: np.flatnonzero(is_variable_definition)[0]]
                for item in split_defs(line)
                if len(item) > 0
            ]
        )

        # Parse the lines defining variables into a DataFrame indexed by the variable name
        VariableDefinitions = pd.DataFrame(
            [defs_to_dict(split_defs(line)) for line, is_def in zip(header, is_variable_definition) if is_def]
        ).set_index("Name")

        # Return the information
        return FileInfo, VariableDefinitions, len(header)


def _create_cached_txt(shot_num: int, module: str = "A", sample_rate: int = 20, force_cache_refresh: bool = False):
    """From the shot number and other info, find the .dat time trace file and create a locally cached txt file of it"""

    # Get name of remote .dat file
    remote_dat = _shot_timetrace_dat(shot_num, module, sample_rate)

    # Build the name of the local copy of the .dat, including the shot nummber!
    local_dat = LOCAL_CACHE.joinpath(f"{shot_num:06d}_" + remote_dat.name)

    # The name of the cached text file. This Decoder program is daft.
    local_txt = pathlib.Path(str(local_dat) + ".txt")

    # If it already exists, return the cached copy, unless we're forced to remake it.
    if local_txt.exists() and not force_cache_refresh:
        return local_txt

    # At this point, we will try to use the local cache folder. Check it's there
    LOCAL_CACHE.mkdir(exist_ok=True)

    # Otherwise we need to copy the .dat, decode it, then delete it again. Making links doesn't seem to work :(
    shutil.copy2(remote_dat, local_dat)
    _decode_file(local_dat)
    local_dat.unlink()

    # Done
    return local_txt


def _shot_timetrace_dat(shot_num: int, module: str = "A", sample_rate: int = 20):
    """Create the Path for a Gantner time trace .dat file for a given shot"""
    return (
        SHOTFILES
        / "GANTNER"
        / f"{math.floor(shot_num/1000):03d}"
        / f"{(shot_num % 1000):03d}"
        / f"BATMAN_{module}_{sample_rate:d}Hz.dat"
    )


def _decode_file(file: pathlib.Path):
    """Run the Decoder on a given file, (almost) silently, via powershell, file must be full absolute path"""

    # This is the txt file that will be created. No way to change this, unfortunately.
    file_out = pathlib.Path(str(file) + ".txt")

    # The arguments to pass to powershell
    run_args = [
        "powershell",
        "Start-Process",
        DECODE_CMD,
        str(file),
        r"-PassThru",
        r"| % Id",
    ]

    # Run the decoder and grab the process ID
    pid = int((subprocess.run(run_args, capture_output=True, text=True)).stdout)

    # Give it some time to complete
    time.sleep(0.25)

    # Kill the process to get rid of the dialogue box
    subprocess.run(KILL_ARGS + [str(pid)], capture_output=False)

    return file_out
