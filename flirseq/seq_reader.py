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
from flirseq import ir_imaging


def seq_to_kelvin(
    file: str,
    target_object: ir_imaging.TargetObject,
    object_system: list,
    detector_temperature: float = None,
    frames="all",
):
    """Extract the data from a given SEQ file and use the provided environmental parameters to convert the images into
        the temperature of the object in Kelvin.

    Parameters
    ----------
    file : str/path
        Path to the file that contains the frame data.
    target_object : ir_imaging.TargetObject
        Description of the object being imaged.
    object_system : list
        A list of object between the camera and the target.
    detector_temperature : float, optional
        A value of the detector temperature to override built in sensors.
    frames : list/int, optional
        A list of which frames to read. The default is 'all'.

    Returns
    -------
    kelvin: list
        List of np.ndarray, each of which is an image in Kelvin
    s: SeqData
        The containing object, if more things need to be done with it

    """

    # Create the object and read the header info, including the number of frames
    s = SeqData(file)

    # Read the requested frames
    s.read(frames)

    # Apply the system of objects
    s.target_object = target_object
    s.object_system = object_system

    # Override the detector temperature if requested
    if detector_temperature is not None:
        s.detector_temperature = detector_temperature

    # Calculate values in Kelvin for each of the frames read
    return s.kelvin, s


class SeqData:
    """Container for a FLIR .seq ('sequence') file, which itself is just a bunch of FFF (FLIR File Format) frames

    On initialisation with a given file name, the file is parsed to determine the total number of frames and to read
    header information for each.

    After initialisation, frames can be read from disk individually, in any order, using 'read(readme)', where 'readme'
    is either of the keywords 'all' or 'last', or a list of integer indices, or a single index. Negative indexing is
    supported, and the default is 'all'.

    A list of the objects between the camera and the target object must be provided before any values of object
    radiance or temperature can be obtained. These are defined in the 'ir_imaging' module.

    Calculated radiances and temperatures are accessed as attributes, and only calculated on invocation. These are
    purposefully not cached, so that values can be returned with different conditions if needed for advanced use.

    Selected Attributes
    ----------
    frames : list of <FFF>
        All of the frames contained in this file. See class FFF for details.
    object_system : list of <ir_imaging.InfraredObject>
        The objects appearing between the camera and the target.
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
            raise RuntimeError(f"Invalid from number requested: {bad_frames[0]} (of possible {self.n_frames})")

        # Tell each frame to be read
        for ff in frames:
            self.frames[ff].read()

        # Take one frame as a reference frame through which further calibrations and calculations can be made.
        # It is assumed that all the frames were taken with the same camera under the same conditions!
        self.reference_frame = self.frames[frames[0]]

    @property
    def detector_temperature(self):
        return self.reference_frame.detector_temperature

    @detector_temperature.setter
    def detector_temperature(self, temperature: float):
        self.reference_frame.detector_temperature = temperature

    @property
    def detector(self):
        return self.reference_frame.detector

    @property
    def object_system(self):
        return self.reference_frame.object_system

    @object_system.setter
    def object_system(self, object_system: list):
        self.reference_frame.object_system = object_system

    @property
    def target_object(self):
        return self.reference_frame.target_object

    @target_object.setter
    def target_object(self, target_object: ir_imaging.TargetObject):
        self.reference_frame.target_object = target_object

    @property
    def measured_radiance(self):
        return [f.measured_radiance for f in self.frames if f.is_read]

    @property
    def object_radiance_coefficients(self):
        return ir_imaging.find_radiance_coefficients(
            self.reference_frame.target_object, self.reference_frame.detector, self.reference_frame.object_system
        )

    @property
    def object_radiance(self):
        """ Get the estimated object radiance for all the frames that have been read. """
        coeffs = self.object_radiance_coefficients
        return [coeffs[0] * radiance + coeffs[1] for radiance in self.measured_radiance]

    @property
    def kelvin(self):
        """ Get the esimated object temperature for all the frames that have been read. """
        return [radiance_o.temperature for radiance_o in self.object_radiance]

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

        # Extract the calibration parameters between radiance/temperature
        self.CameraCalibration = ir_imaging.DetectorCalibration(
            REDACTED
        )

        # Get the calibration parameters from raw signal to radiance
        self.SignalCalibration = {
            REDACTED
        }

        # Extract the Radiance values from the pixels, if they are raw counts
        # Other types of pixel values are not supported
        if self.REDACTED == 1:

            # Pixels also need trimming based on the GeomInfo provided
            x1 = self.REDACTED
            x2 = self.REDACTED
            y1 = self.REDACTED
            y2 = self.REDACTED

            # Conversion from raw signal into Radiance using camera's calibrated offset and gain.
            self.measured_radiance = ir_imaging.CalibratedRadiance(
                radiance=(self.Pixels.pixels[y1:y2, x1:x2] + self.SignalCalibration["offset"])
                * self.SignalCalibration["gain"],
                calibration=self.CameraCalibration,
            )

        else:
            raise NotImplementedError(f"Pixels of type {self.REDACTED} not yet supported.")

        # Mark as read
        self.is_read = True

    @property
    def detector_temperature(self):
        try:
            return self._detector_temperature
        except AttributeError:
            if self.is_read:
                return self.REDACTED
            else:
                raise RuntimeError("Detector properties not available: Frame not read.")

    @detector_temperature.setter
    def detector_temperature(self, temperature: float):
        self._detector_temperature = temperature

    @property
    def detector(self):
        if self.is_read:
            return ir_imaging.Detector(self.detector_temperature, self.CameraCalibration)
        else:
            raise RuntimeError("Detector properties not available: Frame not read.")

    @property
    def object_system(self):
        try:
            return self._object_system
        except AttributeError:
            raise RuntimeError("No infrared object system set.")

    @object_system.setter
    def object_system(self, object_system: list):
        self._object_system = object_system

    @property
    def target_object(self):
        try:
            return self._target_object
        except AttributeError:
            raise RuntimeError("No target object set.")

    @target_object.setter
    def target_object(self, target_object: ir_imaging.TargetObject):
        self._target_object = target_object

    @property
    def object_radiance(self):
        return ir_imaging.find_object_radiance(
            self.measured_radiance, self.target_object, self.detector, self.object_system
        )

    @property
    def kelvin(self):
        return self.object_radiance.temperature
