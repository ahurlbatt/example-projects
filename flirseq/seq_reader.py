"""
This module contains reader objects and methods for obtaining data from FLIR SEQ and FFF infrared imaging files.
As well as the ability to create interrogation objects and fine tune conversions, there are high level functions 
for bulk reading and conversion of files.
"""

import os
import numpy as np
import math

from flirseq import fff_reader_objs as fff_read
from flirseq import fff_module_constants as MOD
from flirseq import infroptics


class FrameNotFoundError(Exception):
    """Raised if a frame is requested that doesn't exist."""

    def __init__(self, frame_number: int, n_frames: int):
        self.frame_number = frame_number
        self.n_frames = n_frames

    def __str__(self):
        if self.n_frames is None:
            return f"Invalid from number requested: {self.frame_number}"
        else:
            return f"Invalid from number requested: {self.frame_number} (of possible {self.n_frames})"


class MissingInfoError(Exception):
    """ Raised if information for a requested action has not yet been provided. """

    def __init__(self, missing_info: str, requested_action: str = None):
        self.missing_info = missing_info
        self.requested_action = requested_action

    def __str__(self):
        if self.requested_action is None:
            return f"Information/Data missing: {self.missing_info}."
        else:
            return f"Information/Data missing for {self.requested_action}: {self.missing_info}."


def seq_to_kelvin(file: str, optics, background_temperature: float, object_emittance: float, frames="all"):
    """
    Extract the data from a given SEQ file and use the provided environmental parameters to convert the images into 
    the temperature of the object in Kelvin.

    Parameters
    ----------
    file : str/path
        Path to the file that contains the frame data.
    optics : list[optics]
        A list of optical components (including air) between the camera and the object
        optics[0] is closest to the camera.
    background_temperature : float
        Temperature of the background environment in Kelvin.
    object_emittance : float
        Emittance of the object of interest in the interval [0, 1].
    frames : list/int, optional
        A list of which frames to read. The default is 'all'.

    Returns
    -------
    Kelvin: list
        List of np.ndarray, each of which is an image in Kelvin
    s: SeqData
        The containing object, if more things need to be done with it

    """

    # Create the object and read the header info, including the number of frames
    s = SeqData(file)

    # Read the requested frames
    s.read(frames)

    # Apply the optical set up
    s.optics = optics

    # Set the background temperature and object emittance
    s.set_background_temperature(background_temperature)
    s.set_object_emittance(object_emittance)

    # Calculate values in Kelvin for each of the frames read
    return s.kelvin, s


class SeqData:
    """
    This is a container for a FLIR .seq ('sequence') file, which itself is just a bunch of FFF (FLIR File Format)
    frames concatenated into the same file. On initialisation with a given file name, the file is parsed to determine
    the total number of frames and to read header information for each.
    
    After initialisation, frames can be read from disk individually, in any order, using 'read(readme)', where 'readme'
    is either of the keywords 'all' or 'last', or a list of integer indices, or a single index. Negative indexing is
    supported, and the default is 'all'.

    A list of the optics between the camera and the object must be provided before any values of object radiance or
    temperature can be obtained. These are defined in the 'infroptics' module.'

    Calculated radiances and temperatures are accessed as attributes, and only calculated on invocation. These are
    purposefully not cached, so that values can be returned with different optics if needed for advanced use.

    Selected Methods
    ----------------
    read(frames="all")
        Invoke the read method of the specified frames to extract their data from the SEQ file. 'frames' can be a
        single integer, an iterable of ints, or the keywords "last" or "all".
    set_background_temperature(temperature: float)
        Provide the temperature of the background, in Kelvin, to use in calculating object radiances and temperatures.
    def set_object_emittance(object_emittance: float)
        Provide the surface emittance of the object being imaged, for calculating the estimated surface radiance and
        temperature.
    
    Selected Attributes
    ----------
    frames : list of <FFF>
        All of the frames contained in this file. See class FFF for details.
    optics : list of <infroptics.Optics>
        The optical objects appearing between the camera and the target.
    measured_radiance : list of (N, M) array
        The measured radiance from each of the frames that have been read from the file.
    object_radiance : list of (N, M) array
        The estimated radiance of the target object from each of the frames that have been read from the file.
    kelvin : list of (N, M) array
        The estimated temperature of the target object from each of the frames that have been read from the file.
    frame_times : (N, ) array
        The times (in seconds) of each frame relative to the first.
    frame_times_absolute : (N, ) array of DateTime.DateTime
        The absolute times of each frame as recorded by the camera.
    
    """

    def __repr__(self):

        str_out = "   <SEQ File>"
        str_out += f"\n {self.n_frames} Frames ({len([1 for f in self.frames if f.is_read])} read)"

        if hasattr(self, "optics"):
            str_out += "\n Has optics"
        else:
            str_out += "\n NO optics"

        if hasattr(self, "background"):
            str_out += "\n Has background"
        else:
            str_out += "\n NO background"

        if hasattr(self, "object_emittance"):
            str_out += "\n Has object emittance"
        else:
            str_out += "\n NO object emittance"

        return str_out

    def __init__(self, file: str):
        """
        Opens a given file (probably with .seq extension) and parses binary data for header information of the frames 
        contained within, finding all of the frames in the file.

        Parameters
        ----------
        file : str/path
            The name of the file to be read.

        Returns
        -------
        None.

        """

        # Definitely want to record what file we're using, as reading can happen multiple times.
        self.file = file

        # There's no explicit information as to how many frames are contained in this file,
        # so we need to compare to the file size
        self.filesize = os.path.getsize(file)

        # Initialise frames as empty
        self.frames = []

        # Open file, grab frames and their headers until we reach the end
        with open(file, "rb") as f:
            self.frames = [FFF(f)]
            while self.frames[-1].endbyte < self.filesize:
                self.frames.append(FFF(f, self.frames[-1].endbyte))

        # Record the number of frames, as it's useful
        self.n_frames = len(self.frames)

    def read(self, frames="all"):
        """
        Invoke the read() method of the frames contained in this file. Selection can be made on the frames by passing
        a list of frame numbers or 'all' of 'last'.

        Parameters
        ----------
        frames : list/int, optional
            A list of which frames to read. The default is 'all'.

        Returns
        -------
        None.

        """

        # Turn keyword arguments into indices
        if type(frames) == str:
            if frames == "all":
                frames = range(self.n_frames)
            elif frames == "last":
                frames = [-1]

        # Check for a single digit index, and convert to list
        if type(frames) == int:
            frames = [frames]

        # Catch any out of range indices before starting the read, as reading is expensive
        if any(x >= self.n_frames or x < -self.n_frames for x in frames):
            bad_frames = [x for x in frames if x >= self.n_frames or x < -self.n_frames]
            raise FrameNotFoundError(bad_frames[0], self.n_frames)

        # Tell each frame to be read
        for ff in frames:
            self.frames[ff].read()

        # Grab the camera calibration from the first frame that's been read
        self.CameraCalibration = self.frames[frames[0]].CameraCalibration

    @property
    def optics(self):
        return self._optics

    @optics.setter
    def optics(self, optics_in):
        """ Create the list of optical components between the camera and the object. First in the list are those
		closest to the camera. """

        # Overwrite the current optics with the provided one
        self._optics = optics_in

        # Assume they're not calibrated, so try to calibrate. If it's not iterable, we've only got one.
        try:
            [o.calibrate(self.CameraCalibration) for o in self._optics]
        except TypeError:
            self._optics = [optics_in]
            [o.calibrate(self.CameraCalibration) for o in self._optics]
        except AttributeError:
            raise TypeError("Adding optics requires Optics objects or a list thereof.")

        # Once we're sure all the optics are correct, add them to each frame
        for f in self.frames:
            f.optics = self._optics

    def set_background_temperature(self, temperature: float):
        """ Create and calibrate a background temperature for radiance reflections, then copy it to the frames. """

        self.background = infroptics.Background(temperature)
        self.background.calibrate(self.CameraCalibration)

        for f in self.frames:
            f.set_background(self.background)

    def set_object_emittance(self, object_emittance: float):
        """ Record the estimated emittance of the object of interest. """

        self.object_emittance = object_emittance

        for f in self.frames:
            f.set_object_emittance(self.object_emittance)

    @property
    def measured_radiance(self):
        return [f.RadianceApparent for f in self.frames if f.is_read]

    @property
    def object_radiance_coefficients(self):
        return infroptics.find_radiance_coefficients(self.object_emittance, self.optics, self.background)

    @property
    def object_radiance(self):
        """ Get the estimated object radiance for all the frames that have been read. """
        coeffs = self.object_radiance_coefficients
        return [coeffs[0] * f.RadianceApparent + coeffs[1] for f in self.frames if f.is_read]

    @property
    def kelvin(self):
        """ Get the esimated object temperature for all the frames that have been read. """
        return [
            infroptics.radiance_to_temp(
                M, self.CameraCalibration["R"], self.CameraCalibration["B"], self.CameraCalibration["F"]
            )
            for M in self.object_radiance
        ]

    @property
    def frame_times_absolute(self):
        """ Get the esimated object temperature for all the frames that have been read. """
        return np.array([f.BasicData.ImageInfo.datetime for f in self.frames if f.is_read])

    @property
    def frame_times(self):
        """ Get the esimated object temperature for all the frames that have been read. """
        fta = self.frame_times_absolute
        return np.array([t.total_seconds() for t in (fta - fta[0])])


class FFF:
    """
    The FLIR File Format (FFF) is a binary storage format for images and additional information from FLIR infrared
    cameras, as created by the ResearchIR program. Every frame has a Header and a list of Indexes detailing the 
    contents of the frame. Each Index corresponds to a data block and describes the type of data stored there.
    Possible data types are information on the camera status, calibration factors, image data, correction factors, 
    data from other sensors, and more. Each block is typically a nested structure of some of the data types.
    
    The details of the possible data types and their translation from binary have been put together through a 
    combination of spotty documentation and reverse engineering. As such, although names are known for most of the 
    parameters and data types available, their actual use and meaning is not necessarily clear. No guarantee is made on
    the accuracy or interpretation of the data, and testing has only been performed on data from a particular system.
    
    Attributes
    ----------
    file : str/path
        Path to the file that contains the frame data.
    startbyte : int
        The starting position (in bytes) of this frame within the file
    header : FFFHeader
        Information about the file version and how many types of data to expect.
    indexes : [FFFFileIndex]
        Details the types, sizes, and locations of data blocks associated with this frame.
    endbyte : int
        The last byte of this frame. Used for SEQ files to know where the next frame starts.
    is_read : Bool
        True if the data from the data blocks has been read.
    
    Other attributes are created when read() is called. Each descriptor (may) create a new attribute of a type found
    in fff_reader_objs.
    
    """

    def __repr__(self):
        """Describe the frame using the types of descriptors found, and whether it's been read yet or not."""

        return "<FFF Frame> [" + ", ".join([d.tagtype for d in self.indexes]) + "]" + " (Unread)" * (not self.is_read)

    @fff_read.file_open_and_seek
    def __init__(self, file, startbyte=0):
        """
        Create a FFF object by reading header and desscriptor information from a given location in a given file.
        Decorator from fff_reader_objs ensures that the file is open before any reading happens.

        Parameters
        ----------
        file : str/path/IOStream
            The file from which information is to be read.
        startbyte : int
            The first byte of this frame.

        Returns
        -------
        None.

        """

        # We want to be able to read more stuff later, so record the file name
        self.file = file

        # Record the starting byte of this frame
        self.startbyte = startbyte

        # Get the header for this frame from the initial position
        self.header = fff_read.FFFHeader(file, self.startbyte)

        # Work out where the first descriptor starts
        indexes_startbyte = self.startbyte + self.header.dwIndexOff

        # Get the Descriptors as a list
        self.indexes = [
            fff_read.FFFFileIndex(file, indexes_startbyte + offset)
            for offset in np.arange(0, self.header.dwNumUsedIndex, 1) * MOD.FFF_INDEX_BYTES
        ]

        # Work out where the frame ends
        self.endbyte = self.startbyte + max([d.dwDataPtr + d.dwDataSize for d in self.indexes])

        # Frames always end on a LONGWORD boundary (i.e. 4 bytes)
        self.endbyte = 4 * math.ceil(self.endbyte / 4)

        # We want to track if this frame has been read or not
        self.is_read = False

    @fff_read.file_open_and_seek
    def read(self, file, startbyte):
        """
        Look at each descriptor in turn and read the data block described by it. Each descriptor gives the location and
        type of data as an index. The data type is converted into a reader class using the FFF_TAG_READER_OBJ dict, 
        which is then told where to start reading from. The string name of the data type is used to name a new attribute
        of this frame to contain the data that has been read.

        Parameters
        ----------
        file : str/path/IOStream
            The file from which information is to be read.
        startbyte : int
            The first byte of this frame.

        Returns
        -------
        None.

        """

        # Cycle through the Descriptors, calling the appropriate reader and making the appropriate attribute
        for dd, index in enumerate(self.indexes):

            # Get the reader class name
            my_reader = MOD.FFF_TAG_READER_OBJ.get(index.wMainType, "Unknown")

            # Convert into the actual class to use
            try:
                my_reader_class = getattr(fff_read, my_reader)
            except AttributeError:
                continue
                raise fff_read.UnsupportedTypeError("Descriptor", index.tagtype)

            # Find the actual starting byte, as the Descriptor only knows about the local frame
            my_start_byte = self.startbyte + index.dwDataPtr

            # Read into the appropriate attribute, with a workaround for pixel data, which needs to know the subtype
            if my_reader == "FFFPixelData":
                setattr(self, index.tagtype, my_reader_class(file, my_start_byte, index.dwDataSize, index.wSubType))
            else:
                setattr(self, index.tagtype, my_reader_class(file, my_start_byte, index.dwDataSize))

        # Extract the camera calibration parameters
        self.CameraCalibration = {REDACTED}

        # Extract the Radiance values from the pixels, if they are raw counts
        # Other types of pixel values are not supported
        if self.REDACTED == 1:

            # Pixels also need trimming based on the REDACTED provided
            x1 = self.REDACTED
            x2 = self.REDACTED
            y1 = self.REDACTED
            y2 = self.REDACTED

            # Conversion from raw signal into Radiance using camera's calibrated offset and gain.
            self.RadianceApparent = (
                self.Pixels.pixels[y1:y2, x1:x2] + self.CameraCalibration["offset"]
            ) * self.CameraCalibration["gain"]

        else:
            raise NotImplementedError(f"Pixels of type {self.REDACTED} not yet supported.")

        # Mark as read
        self.is_read = True

    @property
    def optics(self):
        return self._optics

    @optics.setter
    def optics(self, optics_in):
        """ Create or add to the list of optical components between the camera and the object. First in the list are
        those closest to the camera. """

        # Overwrite the current optics with the provided one
        self._optics = optics_in

        # Try to calibrate if anything isn't yet. If it's not iterable, we've only got one.
        try:
            [o.calibrate(self.CameraCalibration) for o in self._optics if not o.is_calibrated]
        except TypeError:
            self._optics = [optics_in]
            [o.calibrate(self.CameraCalibration) for o in self._optics if not o.is_calibrated]
        except AttributeError:
            raise TypeError("Adding optics requires Optics objects or a list thereof.")

    def set_background(self, background):
        """ Apply a calibrated radiance background for this frame """

        self.background = background

    def set_object_emittance(self, object_emittance):
        """ Record the estimated emittance of the object of interest. """

        self.object_emittance = object_emittance

    @property
    def object_radiance(self):
        """
        Pass the optical system and observed radiance to the optics module for calculating an estimated object radiance.

        """

        if not hasattr(self, "_optics"):
            raise MissingInfoError("Optical Components", "Object Radiance Calculation")

        if not hasattr(self, "background"):
            raise MissingInfoError("Background Temperature", "Object Radiance Calculation")

        # Calculate and use a conversion from Apparent to Object radiance according to the optical system
        return infroptics.find_object_radiance(
            self.RadianceApparent, self.object_emittance, self._optics, self.background
        )

    @property
    def kelvin(self):
        """ Use the camera calibrations to convert the estimated object radiance into degrees Kelvin """

        return infroptics.radiance_to_temp(
            self.object_radiance, self.CameraCalibration["R"], self.CameraCalibration["B"], self.CameraCalibration["F"]
        )
