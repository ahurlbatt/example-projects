## Python BUG Database Fetcher
A functional-based package for retrieving diagnostic information from the BUG database.

The current BUG database, which has grown organically over the last 20 years, is currently based on text files. As that's probably not going to change any time soon, this package provides a way for fetching the data and turning it into pandas DataFrames (DFs), so that the data can be used in a sensible way. It is still a work in progress, but currently includes fetchers for the basic shot information, the BES results, the CFC results, and time traces from the Ganter DAQ systems. As well as creating DFs from the files, one can request information for a specific shot or list of shots. Accessing time traces requires the use of the provided decoder .exe, and will create a local cached version of the text-based output.

**High level functions**

* `fetch_shot_params(shot_nums)`

 Return a DF containing basic information for a given shot number of list of shot numbers.

* `get_bes_shots(shot_nums, ignoremissing: bool = False, longerthan: float = None)`

 Return a DF containing basic information and BES diagnostic results for a given shot number of list of shot numbers.

* `get_cfc_shots(shot_nums, ignoremissing: bool = False, longerthan: float = None)`

 Return a DF containing basic information and CFC diagnostic results for a given shot number of list of shot numbers.
 
* `group_df_by(df, group_param: str, n_groups: int, group_num_name: str = 'group_num')`

 Group a DF by the provided parameter, for example to find similar voltages with a 2D scan.
 
* `get_current_shot_num()`

 Return the current experiment shot number.

* `TimeTrace(shot_num: int, module: str = "A", sample_rate: int = None, force_cache_refresh: bool = False)`

 Create a queryable TimeTrace object for the provided shot number and Gantner module.
 
**Example Use**
```
# There is a parameter scan in the shots 133000 to 133100
scan_shots = np.arange(133000, 133100 + 1)

# This is a 2D scan. Want one scan param, and one group param, and the number of groups.
scan_param = 'P/P0'    # Normalised perveance
group_param = 'pSfill' # Source filling pressure
n_groups = 5

# A separate parameter to sort the shots by
sort_param = 'PHF-f'   # Forward RF power

# Get the data for all these shots
bes_df = bugdb.get_bes_shots(scan_shots, longerthan = 2, ignoremissing = False)

# Sort by the scan parameter for nice graphs
bes_df.sort_values(by=sort_param, inplace=True)

# Find the groups of the scan for each DF
bes_df, bes_group_centres = bugdb.group_df_by(bes_df, group_param, n_groups)

# Investigate the time trace of a shot that has returned some unusual results.
suspect_time_trace = TimeTrace(scan_shots[24], module = "B")
```
