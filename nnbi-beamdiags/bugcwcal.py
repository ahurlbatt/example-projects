"""
Everything needed to turn an infrared video of the BUG CW calorimeter into an estimate of the incident power density.

The main objects in here are the BUGFoilCalorimeter and BUGFoilPair classes, which work together to recognise foil
pairs in the infrared images, generate the transform from pixels to absolute position in mm, and extract the temperature
delta for conversion into power density.

Quantities computed from fits, including the power density, are wrapped into ufloats or arrays thereof from the
uncertainties package, for inclusion of their estimated uncertainty.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, transform
from scipy.optimize import curve_fit, least_squares
from scipy.linalg import svd
from uncertainties import unumpy as unp
from uncertainties import umath, ufloat
from beamdiags import imageanalysis
import flirseq
import bugdb


# Our current folder, so we can build absolute paths for template files.
MY_FOLDER = str(pathlib.Path(__file__).parent.resolve()) + "\\"

# Static file names for the image templates for foils and thermocouples
FOIL_TEMPLATE_FILE_NAME = MY_FOLDER + "image_templates\\bug_cw_cal_foil_template.txt"
TC_TEMPLATE_FILE_NAME = MY_FOLDER + "image_templates\\bug_cw_cal_tc_template.txt"

# Estimated/assumed parameters relative for the optics that influence the IR measurement
BACKGROUND_TEMPERATURE = 20 + 273.15
RELATIVE_HUMIDITY = 0.5
DISTANCE_CAMERA_TO_WINDOW = 0.1
WINDOW_TRANSMITTANCE = 0.95
WINDOW_REFLECTANCE = 0.03
DISTANCE_WINDOW_TO_CAL_SURFACE = 0.3


class CalorimeterDetectionError(Exception):
    """This is raised if a known problem is encountered"""

    pass


class BUGFoilCalorimeter(object):
    """Container for extracting and containing information from an IR movie of the CW calorimeter at BUG

    Call with a movie (list() of images) taken using the IR camera looking at the back side of the CW Calorimeter
    at BUG and the class instance will attempt to recognise the foils heated by the beam, using the hottest overall
    image. If there are enough to work with, it will build a transform from pixel space to position (in mm) relative
    to the hottest foil on the calorimeter. If it can recognise the thermocouples in the image, the positioning will
    be made relative to the centre of the calorimeter.

    Foil pairs (BUGFoilPair objects) are stored in a 2D array representing their actual positions on the calorimeter.
    Information about the foils can either be extracted from BUGFoilCalorimeter.foil_array[ii, jj] directly, or by
    using the helper functions.

    Selected Methods
    -------
    list_of_foil_values(attribute: str, valid_only: bool = False)
        Returns a flat list of the provided attribute extracted from each of the foil pairs.
    array_of_foil_values(attribute: str)
        Returns an array of the provided attribute extracted from each of the foil pairs. Depending on the type of
        value contained in the attribute, the returned array might be a 2D array of scalars, a 2D array of objects,
        or an ndarray where the first two indices represent the 2D arrangement of foil pairs.

    Selected Attributes
    -------------------
    valid_foils: (N, M) array of BUGFoilPair
        A flattened array of just the foils which have been found on the calorimeter. This is not 2D, as such an array
        might be sparse.
    foil_positions_mm: (N, M, 2) array
        The coordinates of the measurement location of each foil pair, relative to the estimated centre of the
        calorimeter. The origin may be off by some multiple of the distance between foil pairs, depending if an
        absolute position has been found via the thermocouples.
    foil_positions_pix: (N, M, 2) array
        The foil measurement locations translated into pixel coordinates
    foil_array_axes: tuple((M,) array, (N,) array)
        The x and y axes of the foil array. Be aware these correspond to the image orientation, not the calorimeter
        orientation!
    power_density: tuple((M, N) array, (N,) array, (M,) array)
        The calculated steady state power density for this calorimeter movie. The arrays, in order, are power density
        in MW per m², and the x and y axes in mm. These arrays are oriented as if one were observing the calorimeter
        from the FRONT. A motion of the beam toward positive x would be a deflection to the right.
    thermocouple_time_traces: pd.DataFrame
        A DataFrame containing time traces of the temperatures at the locations given in THERMOCOUPLE_POSITIONS_MM.
    image_blur: ufloat
        The estimate blurryness of the sample image, corresponding to a general uncertainty (in units of pixels)
    """

    # Estimated emittance of the foil surface
    SURFACE_EMITTANCE = 0.9

    # Information about the foils, their sizes, and their locations
    NUMBER_OF_FOILPAIRS = np.array([30, 40])
    FOILPAIR_GRID_SPACING_MM = np.array([20.0, 20.0])
    FIRST_FOIL_OFFSET = NUMBER_OF_FOILPAIRS % 2 + 0.5 * FOILPAIR_GRID_SPACING_MM

    # The size of the array made to contain the foils, as the "first" detected foil could be anywhere
    FOIL_ARRAY_OVERSIZE = np.max(NUMBER_OF_FOILPAIRS) * 2

    # How far a foil finger needs to be from the edge of the image
    IMAGE_EDGE_BUFFER = 2

    # Sometimes fitting returns an unrealistically small blur value with a high confidence, which then messes up
    # the weighted averaging. A simple way to deal with this a with a hard limit.
    MINIMUM_BLUR = 0.4

    # Thresholds for fitting continuation - both in number of foils and how much their temperature changes
    MIN_REQUIRED_FOIL_MATCHES = 40
    FOIL_TEMPERATURE_DELTA_FITTING_THRESHOLD = 0.5

    # Information about the locations of thermocouples - these are also used for absolute positioning
    THERMOCOUPLE_POSITIONS_MM = np.array(
        [
            [-60.0, -158.0],
            [-60.0, -78.0],
            [-60.0, 2.0],
            [-60.0, 82.0],
            [-60.0, 162.0],
            [60.0, -158.0],
            [60.0, -78.0],
            [60.0, 2.0],
            [60.0, 82.0],
            [60.0, 162.0],
        ]
    )

    # Approximate pixel locations of the thermocouples
    TC_TEMPLATE_POSSIBLE_MATCHES = np.array([[181, 190], [181, 255], [181, 320], [181, 385], [181, 450]])

    # Absolute position of the foil closest to the thermocouple centre (in the direction of the origin)
    TC_TEMPLATE_FOIL_ABSOLUTE_MM = np.array([[-70, -170], [-70, -90], [-70, -10], [-70, 70], [-70, 150]])

    # Size of the thermocouple in image pixels, for extracting temperatures
    TC_PIXEL_SIZE = np.array([2.0, 2.0])
    TC_MATCH_TOLERANCE = 10

    # Work out what the foil_array axes should be, based on the foil pair arrangement
    X_AXIS_SHOULDBE = np.linspace(
        *((NUMBER_OF_FOILPAIRS - 1) * FOILPAIR_GRID_SPACING_MM)[1] * np.array([-0.5, 0.5]), NUMBER_OF_FOILPAIRS[1],
    )
    Y_AXIS_SHOULDBE = np.linspace(
        *((NUMBER_OF_FOILPAIRS - 1) * FOILPAIR_GRID_SPACING_MM)[0] * np.array([-0.5, 0.5]), NUMBER_OF_FOILPAIRS[0],
    )

    def __init__(self, kelvin_frames, frame_times=None, fit=True):
        """Create an object from the provided movie in kelvin_frames, including running the fitting process

        Parameters
        ----------
        kelvin_frames : list
            A list of images (2D arrays) representing a movie of the BUG CW calorimeter
        frame_times : (N,) array, optional
            The times of each frame in kelvin_frames, in seconds.
        fit : bool, optional
            Whether to run the image recognition process or not. The default is True.

        Returns
        -------
        None.

        """

        # The video of the calorimeter, in Kelvin
        self.kelvin_frames = kelvin_frames

        # Grab the frame times, if we've been given them
        if frame_times is None:
            self.frame_times = np.arange(len(kelvin_frames))
        else:
            self.frame_times = frame_times
        # Run the fitting, unless we've been told not to
        if fit:
            self.run_fitting()

    def run_fitting(self):
        """Run the sequence of methods needed to calculate power density"""

        # Create a normalised sample image and an interpolater for it
        self.get_sample_image()

        # Find locations matching the foil template
        self.match_foil_templates()

        # Find the trasform that turns template coordinates into their positions in an array
        self.foil_templates_to_grid()

        # Create the Foil objects in the array, and try to fit their positions
        self.fit_foil_positions()

        # Fit the horizontal and vertical lines to the rows and columns of fitted foil pairs
        self.fit_foil_lines()

        # From the reported line crossings, estimate the projective transformation matrix
        self.fit_transform_matrix()

        # Find the pixel coordinates of the foil pairs, and discard those that fall outside the image
        self.transform_foil_coords()

        # Get the temperature time traces for each foil pair
        self.extract_foil_temperatures()

        # Get each foil to fit its temperature delta
        self.fit_foil_temperature_deltas()

        # Attempt to align the mm coordinates with the absolute position of thermocouples.
        self.absolute_align_via_thermocouples()

    def get_sample_image(self):
        """Normalise the hottest frame to use as a sample image, on which recognition can be run"""

        # Find the hottest overall frame.
        self.hottest_frame_idx = np.argmax([frame.sum() for frame in self.kelvin_frames])

        # Noramlise this frame to run recognition on
        self.sample_image = (
            (sample_image := self.kelvin_frames[self.hottest_frame_idx]) - np.min(sample_image)
        ) / np.ptp(sample_image)

    def match_foil_templates(self):
        """Find all of the parts of the sample image that match the foil pair template"""

        # Find all the 'correct orientation' template matches, with a reasonable threshold.
        template_noflip_corners = imageanalysis.find_template(
            self.sample_image, BUGFoilPair.TEMPLATE, threshold_abs=0.9
        )

        # Find all the 'flipped orientation' template matches.
        template_flip_corners = imageanalysis.find_template(
            self.sample_image, np.flip(BUGFoilPair.TEMPLATE, axis=0), threshold_abs=0.9
        )

        # An array of all template corners
        self.foil_match_corners = np.concatenate((template_noflip_corners, template_flip_corners))

        # An array of all template centres
        self.foil_match_centres = self.foil_match_corners + 0.5 * np.array(BUGFoilPair.TEMPLATE.shape)

        # A list of whether or not each template is flipped
        self.is_foilmatch_flipped = [False] * template_noflip_corners.shape[0]
        self.is_foilmatch_flipped += [True] * template_flip_corners.shape[0]

        # Check we have enough template matches to sensibly fit the calorimeter. If there are too few, it likely means
        # that the beam failed, or the calorimeter was not fully exposed.
        if not self.foil_match_centres.shape[0] >= self.MIN_REQUIRED_FOIL_MATCHES:
            raise CalorimeterDetectionError("Not enough foil pairs found for accurate detection.")

    def foil_templates_to_grid(self):
        """Find 2D grid indices for each of the matched foil pairs"""

        # We want to centre things in the hottest part of the image, so we want to find out which of the templates
        # is closest to that. First we find the hottest pixel (in a median filtered image).
        hottest_pixel = np.unravel_index(
            np.argmax(filters.median(self.sample_image, np.ones((5, 5)))), self.sample_image.shape
        )

        # Then we find out which of the template centres is closest to it
        hottest_template = np.argmin(np.sum((self.foil_match_centres - hottest_pixel) ** 2, axis=1))

        # Extract the coordinates of the templates as a collection of column vectors, relative to the hottest
        template_vecs = (self.foil_match_centres - self.foil_match_centres[hottest_template, :]).T

        # Estimate a transformation matrix to convert the template coordinates into 2D array indices.
        template_grid_transform = imageanalysis.careful_coords_to_integer_grid(template_vecs)

        # Use this to make a list of array indices for each template, and transpose back!
        self.template_grid_indices = (
            np.round(imageanalysis.homogeneous_transform(template_vecs, template_grid_transform)).astype(int).T
        )

    def fit_foil_positions(self):
        """Create a BUGFoilPair object that fits finger positions for each matched template

        For each of the template matches, a BUGFoilPair object is initialised with the part of the image corresponding
        to the template match. This then fits approximate positions (in pixel space) for the centres of the hot and
        cold fingers. Positions in the 2D array foil_array that do not have a coressponding template match are
        initialised with an unfitted BUGFoilPair object. In this way information can be extracted hereon from the
        whole array without necessarily worrying about where the template matches are.
        """

        # Initialise an array for holding the foil pairs, much bigger than it needs to be
        self.foil_array = np.empty((self.FOIL_ARRAY_OVERSIZE,) * 2, dtype=type(BUGFoilPair()))

        # Fill the array with the foils from the template, making sure to translate [0,0] to the array centre
        for tt in range(self.foil_match_centres.shape[0]):

            # Calculate the indices (for brevity)
            ii = self.template_grid_indices[tt, 0] + self.FOIL_ARRAY_OVERSIZE // 2
            jj = self.template_grid_indices[tt, 1] + self.FOIL_ARRAY_OVERSIZE // 2

            # Assign the foil only if this element is already empty, otherwise we have a problem!
            if self.foil_array[ii, jj] is None:

                # Get the chunk of image that we want to use for this foil pair
                image_chunk = self.sample_image[
                    slice(
                        self.foil_match_corners[tt, 0],
                        self.foil_match_corners[tt, 0] + BUGFoilPair.TEMPLATE.shape[0],
                        1,
                    ),
                    slice(
                        self.foil_match_corners[tt, 1],
                        self.foil_match_corners[tt, 1] + BUGFoilPair.TEMPLATE.shape[1],
                        1,
                    ),
                ]

                # If there is little variation in the image chunk, then fitting it isn't worth the time.
                if np.ptp(image_chunk) < 0.1:
                    continue
                # Create a foil pair object within the array, by providing the image chunk and letting it fit itself
                self.foil_array[ii, jj] = BUGFoilPair(
                    self.foil_match_corners[tt, :], image_chunk, is_flipped=self.is_foilmatch_flipped[tt],
                )
            else:

                raise CalorimeterDetectionError("Foil detection returned overlapping foils.")
        # Work out if odd or even rows are flipped. This could be done with just the hottest foil, but this is
        # also a good opportunity to check for odd behaviour and misalignments.

        # Get a list of which rows contain flipped templates (np.nonzero() returns a tuple)
        rows_with_flipped = np.nonzero(
            np.any(np.vectorize(lambda foil: foil is not None and foil.is_flipped)(self.foil_array), axis=1)
        )[0]

        # Row numbers modulo 2 gives an array or 0/1 -> False/True. If we have flipped templates only on even or only
        # on odd rows, then any() and all() should be equal.
        if np.any(flipped_rows_are_odd := rows_with_flipped % 2) == np.all(flipped_rows_are_odd):
            odd_rows_are_flipped = np.any(flipped_rows_are_odd)
        else:
            raise CalorimeterDetectionError("Flipped templates detected on neighbouring rows.")
        # Fill the rest of the array with unfitted foils (flipped if need be), and tell all the foils where they are
        # relative to the hottest one.
        for ii in range(self.foil_array.shape[0]):
            relative_ii = ii - self.FOIL_ARRAY_OVERSIZE // 2
            is_flipped_row = (ii % 2) * odd_rows_are_flipped
            for jj in range(self.foil_array.shape[1]):
                relative_jj = jj - self.FOIL_ARRAY_OVERSIZE // 2
                if self.foil_array[ii, jj] is None:
                    self.foil_array[ii, jj] = BUGFoilPair(is_flipped=is_flipped_row)
                self.foil_array[ii, jj].relative_offset_mm = self.FOILPAIR_GRID_SPACING_MM * np.array(
                    [relative_ii, relative_jj], dtype=float
                )
        # Keep a record of whcih foils have been fitted
        self.fitted_foils = np.vectorize(lambda foil: foil.is_fitted)(self.foil_array)

    def fit_foil_lines(self):
        """Fit vertical and horizontal lines to the fitted positions of the hot and cold fingers

        Due to the arrangement of the foil fingers over the background, each finger has only been fitted in one
        dimension - the other dimension is estimated from the template. To find full 2D positions for the centre of
        each finger, fit vertical lines to those that have their vertical positions estimated (i.e. horizontal position
        fitted), and vice versa. By combining the two lines, a more accurate position can be found as their crossing
        point. Note that this assumes that the transformation between pixels and millimetres is a homography - straight
        lines are preserved, but parallelism may not be. This excludes some effects from lenses, and is designed to
        account only for changes in perspective.

        TODO : These lines can stay in this class, they don't really need to be given to the foils to get the crossings
        """

        # Loop through the rows of the foil array
        for ii in range(self.foil_array.shape[0]):

            # Get the list of which foils in this row are fitted and check we have at least two points to fit
            if not (use_columns := np.nonzero(self.fitted_foils[ii, :])[0]).size >= 2:
                continue
            # Extract the coordinates found for the cold finger for each foil
            temp_line = np.stack([foil.pix_for_horizontal_line for foil in self.foil_array[ii, use_columns]])

            # Fit the line with linear least squares (uncertainties are implicit :D )
            temp_line_params = _fit_line_via_ulinalg(temp_line[:, 1], temp_line[:, 0])

            # Tell each of the foils used for this line what the result is
            for jj in use_columns:
                self.foil_array[ii, jj].horizontal_line = temp_line_params
        # Do the same again but iterate over columns, and fit a vertical line
        for jj in range(self.foil_array.shape[1]):

            # Get the list of which foils in this column are fitted and check we have at least two points to fit
            if not (use_rows := np.nonzero(self.fitted_foils[:, jj])[0]).size >= 2:
                continue
            # Extract the coordinates for the line
            temp_line = np.stack([foil.pix_for_vertical_line for foil in self.foil_array[use_rows, jj]])

            # Fit the line with linear least squares (uncertainties are implicit :D )
            temp_line_params = _fit_line_via_ulinalg(temp_line[:, 0], temp_line[:, 1])

            # Give it back to the foils that provided data - note this is in the format x = a*y + b!
            for ii in use_rows:
                self.foil_array[ii, jj].vertical_line = temp_line_params

    def fit_transform_matrix(self):
        """Fit a projective transform to the points where fitted lines cross, as these are known in pixels and mm"""

        # Extract the coordinates in mm at which the horizontal and vertical lines cross for each fitted foil pair
        line_cross_mm = np.array([foil.line_cross_mm for foil in self.foil_array[self.fitted_foils]])

        # Extract the estimated pixel coordinates of this crossing point
        line_cross_pix = np.array([foil.line_cross_pix for foil in self.foil_array[self.fitted_foils]])

        # It is possible that a foil has been fitted, but not assigned a line, if it alone on a row or column
        # Check for that by finding nans in line_cross_pix, and remove these entries
        foil_has_line_cross = ~(np.any(np.vectorize(umath.isnan)(np.array(line_cross_pix)), axis=1))
        line_cross_mm = line_cross_mm[foil_has_line_cross, :]
        line_cross_pix = line_cross_pix[foil_has_line_cross, :]

        # For estimating the transform, we need two lists of coordinates; a source and a destination.
        source = unp.nominal_values(line_cross_mm)
        destination = unp.nominal_values(line_cross_pix)

        # In addition, to get the best fit possible, we weight the coordinates against the uncertainties from the
        # image fitting
        weights = 1 / np.mean(unp.std_devs(line_cross_pix) ** 2, axis=1)

        # Fit the transformation matrix to the points, including the weighting
        self.transformation_matrix = imageanalysis.weighted_projective_estimate(source, destination, weights)

        # Construct the transformation object using the fitted matrix
        self.transform_mm_to_pix = transform.ProjectiveTransform(matrix=self.transformation_matrix)

    def transform_foil_coords(self):
        """Batch transform mm positions of hot/cold fingers into pixel space, then trim to the image size"""

        # Extract all the hot/cold foil positions in mm
        cold_finger_mm = np.array([foil.cold_finger_centre_mm for foil in self.foil_array.flatten()])
        hot_finger_mm = np.array([foil.hot_finger_centre_mm for foil in self.foil_array.flatten()])

        # Transform them into pixel coordinates
        cold_finger_pix = self.transform_mm_to_pix(cold_finger_mm)
        hot_finger_pix = self.transform_mm_to_pix(hot_finger_mm)

        # For each foil, both the hot and cold finger centres need to be within the image.
        # Check the coordinates against the lower image bound
        foils_inside_lower_bound = np.logical_and(
            np.all(hot_finger_pix > self.IMAGE_EDGE_BUFFER, axis=1),
            np.all(cold_finger_pix > self.IMAGE_EDGE_BUFFER, axis=1),
        )

        # And against the upper bound
        foils_inside_upper_bound = np.logical_and(
            np.all(hot_finger_pix < (np.array(self.sample_image.shape) - self.IMAGE_EDGE_BUFFER), axis=1),
            np.all(cold_finger_pix < (np.array(self.sample_image.shape) - self.IMAGE_EDGE_BUFFER), axis=1),
        )

        # logical_and the two results to get a list of valid foils
        foils_inside_bound = np.logical_and(foils_inside_lower_bound, foils_inside_upper_bound)

        # Foils that can be used for measurement are told what the pixel coordinates are of their foils
        for ii in np.nonzero(foils_inside_bound)[0]:
            self.foil_array.flat[ii].cold_finger_centre_pix = cold_finger_pix[ii, :]
            self.foil_array.flat[ii].hot_finger_centre_pix = hot_finger_pix[ii, :]
        # To avoid running checks every time, store which foils are valid for measurement
        self._valid_foils = foils_inside_bound.reshape(self.foil_array.shape)

        # Trim down the foil array so that we only have the interesting area
        slice_0 = slice(*(np.flatnonzero(np.any(self._valid_foils, axis=1))[[0, -1]]) + [0, 1])
        slice_1 = slice(*(np.flatnonzero(np.any(self._valid_foils, axis=0))[[0, -1]]) + [0, 1])
        self.foil_array = self.foil_array[slice_0, slice_1]
        self._valid_foils = self._valid_foils[slice_0, slice_1]
        self.fitted_foils = self.fitted_foils[slice_0, slice_1]

        # Reposition the origin of the system assuming that we have captured all the foils, by coercing the mm values
        # for the x- and y-axes of foil_array into the range that they should be.

        # Get the axes as reported by the current foil_array
        x, y = self.foil_array_axes

        # Work out the smallest shifts in x and y that move these axes into the range that they should be
        x_shift = np.min(x[[0, -1]] - self.X_AXIS_SHOULDBE[[0, -1]])
        y_shift = np.min(y[[0, -1]] - self.Y_AXIS_SHOULDBE[[0, -1]])

        # This shift points towards the new origin, as well as ensuring the axes are aligned with any offset needed.
        self._move_mm_origin_to([y_shift, x_shift])

    def extract_foil_temperatures(self):
        """Give each hot/cold finger its temperature time trace"""

        # Get the temperature time trace for each finger
        for foil in self.valid_foils:

            # Frame by frame, get the temperature as the average of the pixel mask
            foil.cold_finger_temperature_vs_time = imageanalysis.average_around_pixel(
                foil.cold_finger_centre_pix, self.kelvin_frames, BUGFoilPair.FINGER_PIXEL_SIZE
            )
            foil.hot_finger_temperature_vs_time = imageanalysis.average_around_pixel(
                foil.hot_finger_centre_pix, self.kelvin_frames, BUGFoilPair.FINGER_PIXEL_SIZE
            )

    def fit_foil_temperature_deltas(self):
        """Fit the temperature deltas with an exponential heat up/cool down, in bulk to ensure timings are the same"""

        # Get the time trace of temperature from each foil
        temperature_deltas = self.list_of_foil_values("temperature_delta_vs_time")

        # Work out which foils will be fitted, based on thresholding the temperature deltas (nans are excluded here)
        fit_foils = np.array(
            [
                np.all(np.greater((np.max(td), np.ptp(td)), self.FOIL_TEMPERATURE_DELTA_FITTING_THRESHOLD))
                if td is not None
                else False
                for td in temperature_deltas
            ]
        )

        # Build the temperature deltas into an array for faster fitting
        delta_array = np.stack([temperature_deltas[idx] for idx in np.flatnonzero(fit_foils)])

        # Get the number of foils to fit, as it's used a lot
        n_fits = delta_array.shape[0]

        # Build the upper and lower bounds for the fitting of this large array. Variables are, in order:
        # temperature_delta * N, temperature_offset * N, tau_on, tau_off, t_on, on_time
        ls_lb = np.concatenate((np.zeros(n_fits), np.full(n_fits, -1.0), [0.4, 0.4, -3600.0, 0.0]))
        ls_ub = np.concatenate((np.full(n_fits, 100.0), np.full(n_fits, 1.0), [2.0, 2.0, self.frame_times[-1], 3600.0]))

        # Find the foil with the greatest delta, to fit to this one first
        hottest_foil = np.argmax(np.max(delta_array, axis=1))

        # Specify some bounds for this hottest foil
        hottest_lb = [0.0, 0.4, 0.4, -3600.0, 0.0, -5.0]
        hottest_ub = [100.0, 3.0, 3.0, 3600.0, 3600.0, 5.0]

        # Make initial conditions for this fit, and bracket it within the bounds
        hottest_x0 = np.clip(
            [np.max(delta_array[hottest_foil, :]), 1.0, 1.0, 1.0, np.mean(self.frame_times), 0.0],
            hottest_lb,
            hottest_ub,
        )

        # Fit this foil to get an estimate of the timing parameters
        hottest_fit = curve_fit(
            f=_heatup_cooldown,
            xdata=self.frame_times,
            ydata=delta_array[hottest_foil, :],
            p0=hottest_x0,
            bounds=(hottest_lb, hottest_ub),
        )

        # This fit result should have good estimates for tau_on, tau_off, t_on and on_time, in that order
        estimated_timings = hottest_fit[0][1:5]

        # Build initial conditions for the least squares fitting of delta_array, and bracket within the bounds.
        ls_x0 = np.clip(
            np.concatenate((np.max(delta_array, axis=1), np.zeros(n_fits), estimated_timings)), ls_lb, ls_ub
        )

        # TODO This ends up with slightly incorrect values for the timings, leading to underestimations of the higher
        # power densities. Look at getting the timings from just the hottest few foils, and see where that ends up.

        # TODO Some of the heating curves are not well described by an exponential, possibly due to water being heated?

        # Run the least squares fitting
        ls_result = least_squares(
            fun=_heatup_cooldown_array_loss_func,
            x0=ls_x0,
            bounds=(ls_lb, ls_ub),
            args=(self.frame_times, delta_array),
            method="trf",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )

        # Collect the results as an uncertainties array
        fitting_results = unp.uarray(ls_result.x, np.sqrt(np.diagonal(_pcov_from_jac(ls_result.jac))))

        # Give the foils their temperature deltas
        for ff, foil in enumerate(self.foil_array.flat[fit_foils]):
            foil.fitted_temperature_delta = fitting_results[ff]
            foil.fitted_temperature_offset = fitting_results[n_fits + ff]
        # Also store the timing parameters for myself
        self.tau_on = fitting_results[-4]
        self.tau_off = fitting_results[-3]
        self.t_on = fitting_results[-2]
        self.on_time = fitting_results[-1]

    def absolute_align_via_thermocouples(self):
        """Try to find some thermocouples using template matching, to get positions relative to the true centre"""

        # Grab the template for matching thermocouples in the image
        self.tc_template = np.loadtxt(TC_TEMPLATE_FILE_NAME)

        # Find the centre pixels of matches in the image
        tc_matches = imageanalysis.find_template(
            self.sample_image, self.tc_template, min_distance=40, threshold_abs=0.9
        ) + (self.tc_template.shape * np.array([0.5, 0.5]))

        if tc_matches.size == 0:
            raise CalorimeterDetectionError("No thermocouples found for absolute position.")
        # See if any of these matches coincide with the expected locations, to with a tolerance
        tc_match_indices = [
            np.nonzero(np.sqrt(np.sum((t - self.TC_TEMPLATE_POSSIBLE_MATCHES) ** 2, axis=1)) < self.TC_MATCH_TOLERANCE)[
                0
            ]
            for t in tc_matches
        ]

        # For each of the thermocouple matches that are close enough to be considered, use its known absolute position
        # to work out the position of the foil closest to it (towards the image origin) and from this the offset that
        # needs to be applied to the current mm origin

        # Some of these might stay empty
        new_origins = np.empty_like(tc_matches)
        for tt, match in enumerate(tc_matches):

            if tc_match_indices[tt].size > 0:

                # Work out the foils that are towards the origin from this match
                is_towards_origin = np.all(
                    (relative_vector := self.foil_positions_pix.reshape(-1, 2) - match) < 0, axis=1
                )

                # Sort them by distance to the match
                foil_order_to_match = np.argsort(np.sqrt(np.sum(relative_vector ** 2, axis=1)))

                # Take the foil closest that's in the direction of the origin
                key_foil = foil_order_to_match[is_towards_origin[foil_order_to_match]][0]

                # From the known absolute positions, work out what the origin shift must be for this match
                new_origins[tt] = (
                    self.foil_array.flat[key_foil].measurement_position_mm
                    - self.TC_TEMPLATE_FOIL_ABSOLUTE_MM[tc_match_indices[tt], :]
                )
        # Check that we have some calculated origins that can be used
        if (useful_origins := new_origins[[bool(t.size) for t in tc_match_indices], :]).size == 0:
            raise CalorimeterDetectionError("No thermocouples found for absolute position.")
        # If all the origin shifts agree, the shift the origin.
        if np.allclose(useful_origins, np.mean(useful_origins, axis=0)):
            self._move_mm_origin_to(np.mean(useful_origins, axis=0))

    def _move_mm_origin_to(self, new_origin):
        """Helper function for moving the centre of the calorimeter to a new location"""

        # Turn the new origin into a transformation matrix for a translation
        (translation := np.eye(3))[0:2, 2] = new_origin

        # This translation wants to happen firsst (i.e. in mm space), so update the transform matrix appropriately
        self.transformation_matrix = self.transformation_matrix @ translation

        # Update the ProjectiveTransform object
        self.transform_mm_to_pix = transform.ProjectiveTransform(matrix=self.transformation_matrix)

        # Move the relative locations of the foils
        for foil in self.foil_array.flatten():
            foil.relative_offset_mm -= new_origin

    def list_of_foil_values(self, attribute):
        """Extract a given attribute from each foil and return as a list"""
        return [getattr(foil, attribute) for foil in self.foil_array.flatten()]

    def array_of_foil_values(self, attribute):
        """Extract a given attribute from each foil and return as a 2-or-more D array"""

        # Get the items as a flat list
        item_list = self.list_of_foil_values(attribute)
        # Determine if we need to build the array as object-type, because there are multiple data types
        use_obj = len(set([type(item) for item in item_list])) > 1

        if (
            (not use_obj)
            and isinstance(item_list[0], np.ndarray)
            and (len(set([item.shape for item in item_list])) == 1)
        ):
            # If every item is an N-D array with the same shape, then construct a (2+N)-D array
            return np.array(item_list).reshape(self.foil_array.shape + item_list[0].shape)
        else:
            # Otherwise return a 2D array, with either object or assumed data type
            return np.array(item_list, dtype=[None, object][use_obj]).reshape(self.foil_array.shape)

    @property
    def valid_foils(self):
        """A flat list of all foils within the image"""
        return self.foil_array[self._valid_foils]

    @property
    def foil_positions_mm(self):
        """The mm positions of all foil measurement points, relative to the centre"""
        return self.array_of_foil_values("measurement_position_mm")

    @property
    def foil_positions_pix(self):
        """The pixel positions of all foil measurement points"""
        return self.transform_mm_to_pix(self.foil_positions_mm.reshape(-1, 2)).reshape(self.foil_array.shape + (2,))

    @property
    def foil_array_axes(self):
        """The foil measurement positions converted to x/y axes suitable for use with the foil_array"""

        # Extract the x and y coordinates from every foil to make sure everything is aligned.
        foil_positions = self.foil_positions_mm

        # Get the unique non-nan values of x and y in each column or row respectively.
        x_columns = [np.unique(column[~(np.isnan(column))]) for column in foil_positions[:, :, 1].T]
        y_rows = [np.unique(row[~(np.isnan(row))]) for row in foil_positions[:, :, 0]]

        # Get the sizes of each of these unique values
        x_column_sizes = np.array([column.size for column in x_columns])
        y_row_sizes = np.array([row.size for row in y_rows])

        # Throw an error if something's misaligned.
        if np.any(x_column_sizes > 1) or np.any(y_row_sizes > 1):
            raise CalorimeterDetectionError("Ambiguity in foil position axes.")
        # Otherwise, build the axes first with nans
        x_axis = np.full(self.foil_array.shape[1], np.nan)
        y_axis = np.full(self.foil_array.shape[0], np.nan)

        # Then extract the values
        x_axis[np.flatnonzero(x_column_sizes)] = [x_val[0] for x_val in x_columns if x_val.size == 1]
        y_axis[np.flatnonzero(y_row_sizes)] = [y_val[0] for y_val in y_rows if y_val.size == 1]

        return x_axis, y_axis

    @property
    def power_density(self):
        """The power density from all the foils, with any non-recognised foil pairs set to zero"""

        # Grab the power density and zero out anything that's NaN.
        power_density = self.array_of_foil_values("power_density")
        power_density[unp.isnan(power_density)] = ufloat(0.0, 0.0)

        # Get the axes of this array
        x, y = self.foil_array_axes

        # Work out how much we need to pad the output array to get the dimensions it should be.
        needs_padding_x = np.flatnonzero(~np.isin(self.X_AXIS_SHOULDBE, x))
        needs_padding_y = np.flatnonzero(~np.isin(self.Y_AXIS_SHOULDBE, y))

        # Pad the array, not forgetting that y is in the zeroth axis.
        power_density = np.pad(
            power_density,
            (
                (
                    np.sum(needs_padding_y < self.NUMBER_OF_FOILPAIRS[0] / 2),
                    np.sum(needs_padding_y > self.NUMBER_OF_FOILPAIRS[0] / 2),
                ),
                (
                    np.sum(needs_padding_x < self.NUMBER_OF_FOILPAIRS[1] / 2),
                    np.sum(needs_padding_x > self.NUMBER_OF_FOILPAIRS[1] / 2),
                ),
            ),
            mode="constant",
            constant_values=ufloat(0.0, 0.0),
        )

        # Return with the necessary rotation and swapping of x/y for plotting as if one is looking downstream
        return np.rot90(power_density, k=3), self.Y_AXIS_SHOULDBE, self.X_AXIS_SHOULDBE

    @property
    def thermocouple_time_traces(self):
        """A DataFrame of the timetraces from where the thermocouples should be"""
        df_out = pd.DataFrame(
            {
                "mm": list(self.THERMOCOUPLE_POSITIONS_MM),
                "pix": list(self.transform_mm_to_pix(self.THERMOCOUPLE_POSITIONS_MM)),
            }
        )
        df_out["temperature"] = df_out["pix"].apply(
            imageanalysis.average_around_pixel, args=(self.kelvin_frames, self.TC_PIXEL_SIZE)
        )
        return df_out

    @property
    def image_blur(self):
        """The weighted average of the blur values from each foil, if they are over the threshold"""
        all_blurs = np.concatenate([x for x in self.list_of_foil_values("blur_values") if x is not None])
        return _mean_of_uarray(all_blurs[all_blurs > self.MINIMUM_BLUR])


class BUGFoilPair:
    """A single pair of hot/cold foil fingers from which a power density can be calculated

    This object serves a number of intertwined purposes that are isolated to each foil pair:
        - Estimation/fitting of finger locations from a sample image
        - Providing spatial positioning of features relative to a given point
        - Finding a power density from time traces of temperature of requested locations

    Instances may be initialised with or without a sample image corresponding to the area of the total frame that was
    found to match the template. If one is provided, then an attempt is made to locate the centres of the hot and cold
    fingers. If the 'corner' is provided, then it is able to return these as positions relative to the origin of the
    whole image - 'corner' should be the corner of the sample image that is closest to the origin.

    Attribute handling is changed so that mixtures of instances with and without sample images can be manipulated in
    arrays without throwing errors. Attributes for which setters and getters are used (via the @property decorator), and
    would normally return a scalar or a vector of known size, will return nans instead if these values are not set.
    Other attributes will return None.

    The local origin for positions given in mm is a point chosen such that the foils lie on a regular 2D grid,
    regardless of whether their template was flipped or not. Instances of this class should be provided with a value for
    'relative_offset_mm', which is the location of their local origin relative to the origin of the whole calorimeter.
    Other positions are returned with this offset included, so that they can be correctly placed on the calorimeter.

    Selected Attributes
    -------------------
    relative_offset_mm: (2, ) array
        The position of this instance relative to the origin of the calorimeter.
    blur_values: (2, ) array
        The estimates of the image blurryness from the fits to the hot and cold finger positions
    temperature_delta_vs_time: (N, ) array
        The time trace of the difference in temperature between the hot and cold fingers
    fitted_temperature_delta: float
        The estimated steady-state difference in temperature found by fitting the time trace
    power_density: float
        The power density in MW per m² found from the fitted temperature delta

    Positions of elements
    ---------------------
    - measurement_position_mm
    - hot_finger_centre_mm
    - cold_finger_centre_mm
    - hot_finger_centre_pix   <-- This must be provided
    - cold_finger_centre_pix  <-- This must be provided
    """

    # Conversion factor from Kelvin Difference to Power Density
    FOIL_DELTA_T_TO_MW_PER_M2 = 26.5

    # If the template for this foil is flipped, multiply offsets by this vector
    FLIP_VECTOR = np.array([-1.0, 1.0])

    # The centres of the hot and cold fingers are offset from a regular grid
    HOT_FINGER_OFFSET_MM = np.array([-6.0, 0.0])  # [y, x]
    COLD_FINGER_OFFSET_MM = np.array([3.0, 7.5])  # [y, x]

    # The distance from the the reference point to the measurement hole
    MEASUREMENT_OFFSET_MM = np.array([0.0, 0.0])  # [y, x]

    # Distance from foil reference point at which the fitted horizontal and vertical lines cross
    LINECROSS_OFFSET_MM = np.array([3.0, 0.0])  # [y, x]

    # The template describing this type of foil pair
    TEMPLATE = np.loadtxt(FOIL_TEMPLATE_FILE_NAME)

    # How large, in pixels, each finger is in the image
    FINGER_PIXEL_SIZE = np.array([2.0, 2.0])

    def __init__(self, corner=None, sample_image=None, is_flipped=False):
        """Create a foil pair object, including fitting positions in an image if provided

        Parameters
        ----------
        corner : (2,) array, optional
            The absolute pixel coordinates of the corner of the recognised template that is closest to (0, 0).
        sample_image : (N, M) array, optional
            The section of the main image that matches to the template.
        is_flipped : bool, optional
            Whether the template was flipped to match this foil pair.

        Returns
        -------
        None.

        """

        # Grab the input values
        self.corner = corner
        self.sample_image = sample_image
        self.is_flipped = is_flipped

        # Calculate the specific flip vector for this foil (basically, turn -1 into 1 if we're not flipped)
        self._my_flip_vector = self.FLIP_VECTOR ** self.is_flipped

        # Record whether this foil has been fit successfully
        self.is_fitted = False

        # If we've been given an image, try to fit the positions, but don't worry if it doesn't work.
        try:
            if sample_image is not None:
                self.fit_finger_middles()
                self.is_fitted = True
        except RuntimeError:
            pass

    def __getattr__(self, attribute):
        """Ensure non-existing attributes return None instead of causing exceptions.

        Some attributes are not created on un-fitted foils, and we want to indicate this without creating exceptions,
        as foils are often accessed within loops where it's awkward to build try/except cases.
        """

        # In theory, _getattr__() is only called if __getattribute__() has already raised AttributeError, but let's be
        # explicit about it:
        try:
            return object.__getattribute__(self, attribute)
        except AttributeError:
            return None

    def _have_or_nanvec(attribute=None, length=2):
        """Create a decorator for properties that should return an array, but may have unfulfilled dependencies

        Parameters
        ----------
        attribute : str, optional
            The name of the attribute on which the property depends. The default is the name of the decorated function,
            prepended with an underscore.
        length : int, optional
            The length of the vector that should be returned. The default is 2. A length of 1 still returns a vector!

        """

        def attribute_check_decorator(fun):

            # Determine the attribute to be checked for
            if attribute is None:
                check_for = "_" + fun.__name__
            else:
                check_for = attribute

            # Replace the function with one that returns a vector of nans if the dependencies are not met.
            def wrapper(self, *args, **kwargs):
                if getattr(self, check_for, None) is not None:
                    return fun(self, *args, **kwargs)
                else:
                    return np.full(length, np.nan)

            return wrapper

        return attribute_check_decorator

    def fit_finger_middles(self):
        """Fit the position of the hot and cold fingers from the sample image"""

        # For the hot finger, we're fitting a horizontal slice. These are the coordinates of that slice.
        xx_hot = np.arange(0, 13)
        yy_hot = 3

        # For the cold finger we're fitting a vertical slice.
        xx_cold = 11
        yy_cold = np.arange(6, 16)

        # If we're flipped, flip the y axis
        if self.is_flipped:
            yy_hot = self.sample_image.shape[0] - yy_hot
            yy_cold = self.sample_image.shape[0] - yy_cold
        # Get the fit results and uncertainties for the two sample regions
        hot_finger_fit_values, hot_finger_fit_sigmas = imageanalysis.fit_plate_over_background(
            xx_hot, self.sample_image[yy_hot, xx_hot]
        )
        cold_finger_fit_values, cold_finger_fit_sigmas = imageanalysis.fit_plate_over_background(
            yy_cold, self.sample_image[yy_cold, xx_cold]
        )

        # Check for unusual fit behaviour by ensuring we have uncertainties for all the fit parameters
        if any([v == 0 for v in cold_finger_fit_sigmas.values()]) or any(
            [v == 0 for v in hot_finger_fit_sigmas.values()]
        ):
            raise RuntimeError
        # Wrap up the results into single objects (with uncertainties) for the fit results
        self._cold_finger_fit = dict(
            zip(
                cold_finger_fit_values.keys(),
                unp.uarray(list(cold_finger_fit_values.values()), list(cold_finger_fit_sigmas.values())),
            )
        )
        self._hot_finger_fit = dict(
            zip(
                hot_finger_fit_values.keys(),
                unp.uarray(list(hot_finger_fit_values.values()), list(hot_finger_fit_sigmas.values())),
            )
        )
        # Package the fit results into pixel coordinates with the uncertainties included.
        self._hot_middle_pix = unp.uarray(
            (yy_hot, hot_finger_fit_values["middle"]), (0, hot_finger_fit_sigmas["middle"])
        )
        self._cold_middle_pix = unp.uarray(
            (cold_finger_fit_values["middle"], xx_cold), (cold_finger_fit_sigmas["middle"], 0)
        )

    @property
    def line_cross_pix(self):
        return imageanalysis.xy_line_cross(self.vertical_line, self.horizontal_line)

    @property
    @_have_or_nanvec("_hot_middle_pix")
    def pix_for_vertical_line(self):
        return self.corner + self._hot_middle_pix

    @property
    @_have_or_nanvec("_cold_middle_pix")
    def pix_for_horizontal_line(self):
        return self.corner + self._cold_middle_pix

    @property
    def line_cross_mm(self):
        return self.relative_offset_mm + self.LINECROSS_OFFSET_MM * self._my_flip_vector

    @property
    def measurement_position_mm(self):
        return self.relative_offset_mm + self.MEASUREMENT_OFFSET_MM * self._my_flip_vector

    @property
    def hot_finger_centre_mm(self):
        return self.relative_offset_mm + self.HOT_FINGER_OFFSET_MM * self._my_flip_vector

    @property
    def cold_finger_centre_mm(self):
        return self.relative_offset_mm + self.COLD_FINGER_OFFSET_MM * self._my_flip_vector

    @property
    def blur_values(self):
        try:
            return np.array([self._cold_finger_fit["blur"], self._hot_finger_fit["blur"]])
        except TypeError:
            return None

    @property
    def temperature_delta_vs_time(self):
        try:
            return self._hot_finger_temperature_vs_time - self._cold_finger_temperature_vs_time
        except TypeError:
            return None

    @property
    def power_density(self):
        return self.fitted_temperature_delta / self.FOIL_DELTA_T_TO_MW_PER_M2

    @property
    @_have_or_nanvec()
    def relative_offset_mm(self):
        return self._relative_offset_mm

    @relative_offset_mm.setter
    def relative_offset_mm(self, offset):
        self._relative_offset_mm = offset

    @property
    @_have_or_nanvec()
    def vertical_line(self):
        return self._vertical_line

    @vertical_line.setter
    def vertical_line(self, line_params):
        self._vertical_line = line_params  # Should be in format x = a * y + b

    @property
    @_have_or_nanvec()
    def horizontal_line(self):
        return self._horizontal_line

    @horizontal_line.setter
    def horizontal_line(self, line_params):
        self._horizontal_line = line_params  # Should be in format y = a * x + b

    @property
    @_have_or_nanvec()
    def hot_finger_centre_pix(self):
        return self._hot_finger_centre_pix

    @hot_finger_centre_pix.setter
    def hot_finger_centre_pix(self, pixel_coords):
        self._hot_finger_centre_pix = pixel_coords

    @property
    @_have_or_nanvec()
    def cold_finger_centre_pix(self):
        return self._cold_finger_centre_pix

    @cold_finger_centre_pix.setter
    def cold_finger_centre_pix(self, pixel_coords):
        self._cold_finger_centre_pix = pixel_coords

    @property
    def cold_finger_temperature_vs_time(self):
        return self._cold_finger_temperature_vs_time

    @cold_finger_temperature_vs_time.setter
    def cold_finger_temperature_vs_time(self, temperature):
        self._cold_finger_temperature_vs_time = temperature

    @property
    def hot_finger_temperature_vs_time(self):
        return self._hot_finger_temperature_vs_time

    @hot_finger_temperature_vs_time.setter
    def hot_finger_temperature_vs_time(self, temperature):
        self._hot_finger_temperature_vs_time = temperature

    @property
    @_have_or_nanvec(length=1)
    def fitted_temperature_delta(self):
        return self._fitted_temperature_delta

    @fitted_temperature_delta.setter
    def fitted_temperature_delta(self, temperature):
        self._fitted_temperature_delta = temperature

    @property
    def fitted_temperature_offset(self):
        return self._fitted_temperature_offset

    @fitted_temperature_offset.setter
    def fitted_temperature_offset(self, temperature):
        self._fitted_temperature_offset = temperature


def cw_cal_camera_optics():
    """Create a list of the optical components between the camera and the calorimeter using the default values"""

    return [
        flirseq.infroptics.Atmosphere(BACKGROUND_TEMPERATURE, DISTANCE_CAMERA_TO_WINDOW, RELATIVE_HUMIDITY),
        flirseq.infroptics.Window(BACKGROUND_TEMPERATURE, WINDOW_TRANSMITTANCE, WINDOW_REFLECTANCE),
        flirseq.infroptics.Vacuum(DISTANCE_WINDOW_TO_CAL_SURFACE),
    ]


def make_new_foil_template(
    shot_num=130096, template_x=208, template_y=222, template_width=16, template_height=16,
):
    """Using the defined shot number and pixel coordinates, create a template of a well defined foil pair"""

    # Go through the process of grabbing the SEQ file for the given shot number
    seq_file = bugdb.cwcal.shotnum_to_cw_seq(shot_num)

    # Get the calibrated optics for the CW Cal
    my_optics = cw_cal_camera_optics()

    # Convert the file to Kelvin
    K, *_ = flirseq.seq_to_kelvin(
        seq_file, my_optics, BACKGROUND_TEMPERATURE, BUGFoilCalorimeter.SURFACE_EMITTANCE, frames="all"
    )

    # Get the hottest frame
    sample_image = K[np.argmax([frame.mean() for frame in K])]

    # Normalise this frame to [0, 1]
    sample_image = (sample_image - np.min(sample_image)) / np.ptp(sample_image)

    # Build the slices based on the size and location given. First index is y!
    template_slice = (
        slice(template_y, template_y + template_height, 1),
        slice(template_x, template_x + template_width, 1),
    )

    # Extract the chuck of image coressponding to the template
    template = sample_image[template_slice].copy()

    template_header = (
        "Template of a hot/cold foil pair from the BUG CW Calorimeter, "
        + f"taken from a normalised image of shot #{shot_num}. "
        + f"Corners are {(template_x, template_y)} + {(template_width, template_height)}."
    )

    # Save to txt
    np.savetxt(FOIL_TEMPLATE_FILE_NAME, template, fmt="%.15f", header=template_header)


def make_new_thermocouple_template(
    shot_num=130096, template_x=185, template_y=173, template_width=10, template_height=23,
):
    """Using the defined shot number and pixel coordinates, create a template of a well defined thermocouple"""

    # Go through the process of grabbing the SEQ file for the given shot number
    seq_file = bugdb.cwcal.shotnum_to_cw_seq(shot_num)

    # Get the calibrated optics for the CW Cal
    my_optics = cw_cal_camera_optics()

    # Convert the file to Kelvin
    K, *_ = flirseq.seq_to_kelvin(
        seq_file, my_optics, BACKGROUND_TEMPERATURE, BUGFoilCalorimeter.CAL_SURFACE_EMITTANCE, frames="all"
    )

    # Get the hottest frame
    sample_image = K[np.argmax([frame.mean() for frame in K])]

    # Normalise this frame to [0, 1]
    sample_image = (sample_image - np.min(sample_image)) / np.ptp(sample_image)

    # Build the slices based on the size and location given. First index is y!
    template_slice = (
        slice(template_y, template_y + template_height, 1),
        slice(template_x, template_x + template_width, 1),
    )

    # Extract the chuck of image coressponding to the template
    template = sample_image[template_slice].copy()

    template_header = (
        "Template of a thermocouple attached to the BUG CW Calorimeter, "
        + f"taken from a normalised image of shot #{shot_num}. "
        + f"Corners are {(template_x, template_y)} + {(template_width, template_height)}."
    )

    # Save to txt
    np.savetxt(TC_TEMPLATE_FILE_NAME, template, fmt="%.15f", header=template_header)


def _heatup_cooldown(time_axis, target_temperature, tau_on, tau_off, t_on, on_time, offset):
    """Calculate the time trace of an object heating up then cooling down exponentially.

    This function returns a time trace of an object subject to a heating effect limited in time. The heat up and cool
    down phases are both assumed to happen exponentially, though with individual time constants. Before t_on, the object
    has a constant temperature 'offset'. At t = t_on, it starts heating toward a 'target_temperature' with time constant
    'tau_on'. After another 'on_time' of time-units, the heating stops and the object cools back down towards 'offset'
    with a time constant 'tau_off'.

    The equation during the on-time is:
        T(t) = offset + target_temperature*(1 - exp( - tau_on * (t - t_on)))
    The equation during the off-time is:
        T(t) = offset + target_temperature * (1 - exp( - tau_on * (on_time))) * exp( - tau_off * (t - (t_on + on_time)))


    Parameters
    ----------
    time_axis : (N, ) array
        Time axis of the process.
    target_temperature : float
        The temperature up to which the object would eventually heat.
    tau_on : float
        Time constant of the heating process.
    tau_off : float
        Time constant of the cooling process.
    t_on : float
        Time at which the heating starts.
    on_time : float
        Length of time for which the heating lasts.
    offset : float
        Uniform offset in the temperature.

    Returns
    -------
    time_trace : (N, ) array
        Calculated time trace of the object's temperature.

    """

    # Get the indices of the on/off stages. It doesn't matter if these are empty.
    heat_idx = np.flatnonzero((time_axis > t_on) & (time_axis <= t_on + on_time))
    cool_idx = np.flatnonzero(time_axis > t_on + on_time)

    # Initialise the time_trace as the same shape as the time_axis, and already set the offset.
    time_trace = np.full_like(time_axis, offset)

    # Add the heating process as per the equation.
    time_trace[heat_idx] += target_temperature * (1 - np.exp(-tau_on * (time_axis[heat_idx] - t_on)))

    # Add the cooling process
    time_trace[cool_idx] += (
        target_temperature
        * (1 - np.exp(-tau_on * (on_time)))
        * np.exp(-tau_off * (time_axis[cool_idx] - t_on - on_time))
    )

    return time_trace


def _heatup_cooldown_array(time_axis, target_temperatures, tau_on, tau_off, t_on, on_time, offsets):
    """Array form of _heatup_cooldown to calculate multiple objects with different temperatures, but the same timing

    Parameters
    ----------
    time_axis : (M, ) array
        Time axis of the process.
    target_temperature : (N, ) array
        The temperature up to which the N objects would eventually heat.
    tau_on : float
        Time constant of the heating processes.
    tau_off : float
        Time constant of the cooling processes.
    t_on : float
        Time at which the heating starts.
    on_time : float
        Length of time for which the heating lasts.
    offset : (N, ) array
        Uniform offset in the temperature of the N objects.

    Returns
    -------
    time_trace : (N, M) array
        The time traces of temperature for each of the N objects.

    """

    # Get the size of the array to be created
    n = target_temperatures.size
    m = time_axis.size

    # Initialise the time_trace as the same shape as the time_axis, and already set the offset.
    time_trace = np.zeros((n, m)) + offsets.reshape((-1, 1))

    # Get the indices of the on/off stages. It doesn't matter if these are empty.
    heat_idx = np.flatnonzero((time_axis > t_on) & (time_axis <= t_on + on_time))
    cool_idx = np.flatnonzero(time_axis > t_on + on_time)

    # Add the heating process as per the equation, but we can do matrix multiplication of (N, 1) and (1, M) vectors
    time_trace[:, heat_idx] += target_temperatures.reshape((-1, 1)) @ (
        1 - np.exp(-tau_on * (time_axis[heat_idx] - t_on))
    ).reshape((1, -1))

    # Add the cooling process
    time_trace[:, cool_idx] += (
        (1 - np.exp(-tau_on * (on_time)))
        * target_temperatures.reshape((-1, 1))
        @ np.exp(-tau_off * (time_axis[cool_idx] - t_on - on_time)).reshape((1, -1))
    )

    return time_trace


def _heatup_cooldown_array_loss_func(p, times, temperature_array):
    """Calculate a loss function for fitting _heatup_cooldown_array

    The loss function is found as the sum of squared differences for each time trace, to give N outputs for N objects,
    the weighted by the fitted target temperatures, as the behaviour is better for being biased to the hotter, and
    therefore less noisy, signals.
    """
    n_fit = temperature_array.shape[0]
    return np.sqrt(
        np.sum(
            (
                temperature_array
                - _heatup_cooldown_array(times, p[:n_fit], p[-4], p[-3], p[-2], p[-1], p[n_fit : (n_fit * 2)])
            )
            ** 2,
            axis=1,
        )
        * p[:n_fit]
    )


def _fit_line_via_ulinalg(x, y):
    """Fit a line y = m * x + c via matrix linear least squares and the uncertainties package."""
    return (unp.ulinalg.inv((C := np.stack((x, unp.uarray(np.ones_like(x), 0),))) @ C.T) @ C @ y).flatten()


def _mean_of_uarray(array):
    """Weighted mean of an array, assuming values are all estimates of the same parameter"""
    v = 1.0 / np.sum(1.0 / unp.std_devs(array) ** 2)
    return ufloat(np.sum(unp.nominal_values(array) / unp.std_devs(array) ** 2) * v, np.sqrt(v))


def _pcov_from_jac(jac):
    """Calculate a covarience matrix from the (estimated) Jacobian, stolen from scipy.optimize.curve_fit"""
    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    return np.dot(VT.T / s ** 2, VT)

