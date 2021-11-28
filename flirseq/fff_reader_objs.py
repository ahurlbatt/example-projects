"""
This module contains the functions and objects required to convert binary data found in an SEQ or FFF file into
the data types defined by FLIR, so that they can be quereied and read, to be converted into useful images.

"""

import os
import io
import numpy as np
import datetime
import struct

from flirseq import fff_module_constants as MOD


class CorruptFileError(Exception):
    """Raised if possible corruption found in file."""

    def __init__(self, filepart: str):
        self.filepart = filepart

    def __str__(self):
        return "Error reading " + self.filepart + ". Possible file corruption."

    pass


class UnsupportedTypeError(Exception):
    """Raised if requested type/value of object [label] is unsupported."""

    def __init__(self, label: str, value: str):
        self.label = label
        self.value = value

    def __str__(self):
        return 'Unsupported Type/Value "' + self.value + '" of item "' + self.label + '".'


def unpack_array(data_type: str, buffer: bytes, byte_order: str = None):
    """
    Unpack data from a bytes object or byte array into a scalar object or tuple of objects. Object type is specified by
    data_type. See below table for valid types, or 'struct' documentation for more information. If 's' is provided, the
    unpacked chars are converted to a string and trailing whitespace is trimmed.
    
      Format    C Type               Python type         Bytes
        c     , char               , bytes of length 1 , 1    
        b     , signed char        , integer           , 1    
        B     , unsigned char      , integer           , 1    
        ?     , _Bool              , bool              , 1    
        h     , short              , integer           , 2    
        H     , unsigned short     , integer           , 2    
        i     , int                , integer           , 4    
        I     , unsigned int       , integer           , 4    
        l     , long               , integer           , 4    
        L     , unsigned long      , integer           , 4    
        q     , long long          , integer           , 8    
        Q     , unsigned long long , integer           , 8    
        e     , half               , float             , 2    
        f     , float              , float             , 4    
        d     , double             , float             , 8    
        s     , char[]             , string            , X

    Parameters
    ----------
    data_type : str
        Single char of type specification from struct module.
    buffer : bytes
        bytes object to be unpacked.
    buffer : str
        Specify big endian ('>') or little endian ('<'). Default is taken from module constants

    Returns
    -------
    [data_type] or tuple
        Single item of specified type, or tuple of specified types if buffer contains multiple.

    """

    # The size of our data type, in bytes
    type_size = struct.calcsize(data_type)

    # Find out how many items we're unpacking
    # If buffer is a single int, then check we really want a single byte, then convert back to bytes.
    # (in other cases, struct will throw the right error for us later if buffer is the wrong size)
    try:
        n_items = int(len(buffer) / type_size)
    except TypeError as exc:
        if type(buffer) == int and type_size == 1:
            buffer = bytes([buffer])
            n_items = 1
        else:
            raise exc

    # Get the byte order, if it hasn't already been given
    if byte_order is None:
        byte_order = MOD.FFF_BYTE_ORDER

    # Build the format string
    format_str = byte_order + str(n_items) + data_type

    # Unpack the data - this always returns a tuple!
    unpacked = struct.unpack(format_str, buffer)

    if n_items == 1:

        # If we've only got one item, take it out of the tuple
        return unpacked[0]

    elif data_type == "s":

        # If we've been asked for a string, convert and trim
        return unpacked[0].decode("utf-8").split("\x00", 1)[0]

    else:

        # Otherwise return unchanged
        return unpacked


def file_open_and_seek(fun):
    def wrapper(self, file=None, startbyte: int = 0, *args, **kwargs):
        """
        Ensure a file is open before trying to read from it using a class method. If the file is not open (either a
        filename/path is passed, or the stream has been closed) it is opened within a context block, seek()ed to
        startbyte, then passed through. If a valid and open stream is passed as file, then this is seek()ed and passed
        through without further modification. No context manager is used in this case.
        
        This decorator is used exclusively on class methods, hence the references to self. It can also be used with 
        objects that have attributes 'file' and 'startbyte', instead of passing them as arguments.

        Parameters
        ----------
        file : str/path/stream object
            Either a filename/path to open, or an open file, or a different stream object.
        startbyte : int, optional
            The starting byte of the item that will be read. The default is 0.
        
        Further arguments are passed through.

        Returns
        -------
        Whatever the wrapped function returns.
        
        """

        # If we've not been passed a file, see if we can get it (and startbyte) from the object we're working on.
        if file is None:
            file = self.file
            startbyte = self.startbyte

        # Determine whether we have a filename/path, an open/closed file stream, or a different stream object, then do
        # the appropriate open/seek/read
        if hasattr(file, "read"):

            # Even if it's a valid stream, it might be closed. So check before passing through
            if not file.closed:

                # We're an open file stream or in-memory stream. Seek and pass through.
                file.seek(startbyte, os.SEEK_SET)
                return fun(self, file, startbyte, *args, **kwargs)

            else:

                # We're a closed file stream. Get the file name, open it, seek, then pass through.
                with open(file.name, "rb") as f:
                    f.seek(startbyte, os.SEEK_SET)
                    return fun(self, f, startbyte, *args, **kwargs)

        else:

            # We've been given a filename or path. Open it, seek, and pass.
            with open(file, "rb") as f:
                self.file = file
                f.seek(startbyte, os.SEEK_SET)
                return fun(self, f, startbyte, *args, **kwargs)

    return wrapper


"""
The following class defs are all objects defined by the FFF documentation. Most of the code is specific unpacking from 
bytes into parameters, but some classes have more complicated routines for reading and conversion into a useful format.
This is not an exhaustive list of definitions, and only includes the ones that have been needed to date.

These are classes instead of functions mainly for the different things that want to be shown when they're displayed in 
a terminal. And the different checks and conversions that happen to some of them mean that there isn't really a 
standard 'read' that can be made to accomodate all of them. 

Despite this, the classes are only really treated like functions. They are called with a file (or other IOStream), a 
starting byte number, and any other parameters that the particular objects need. All class __init__s are decorated to
ensure the file is open and that the stream is in the correct place before reading happens. This way the file can be
read in any order, and partial reads can happen, without worrying about making sure the file is already open.
"""


class FFFHeader:
    def __repr__(self):
        return f"<FFF Header ({self.REDACTED} Tags)>"

    def __str__(self):
        return (
            f"FFF Header Version {self.REDACTED}.\n"
            + f"Source:         {self.REDACTED}.\n"
            + f"Number of Tags: {self.REDACTED}."
        )

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_HEADER_BYTES)

        # Convert the raw bytes into the header attributes
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])
        self.REDACTED = unpack_array("L", all_bytes[*****:*****])


class FFFFileIndex:
    def __repr__(self):
        return f"<FFF Descriptor {self.REDACTED}>"

    def __str__(self):
        return (
            f"FFF Descriptor Version {self.REDACTED}.\n"
            + f"Size:           {self.REDACTED} B.\n"
            + f"Data Type:      {self.REDACTED}.\n"
            + f"[Pixel Subtype: {self.REDACTED}.]"
        )

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_INDEX_BYTES)

        # Unpack from the byte array
        self.REDACTED = unpack_array("H", all_bytes[*****:*****])
        self.REDACTED = unpack_array("H", all_bytes[*****:*****])

        # Get the descriptions of the data type
        self.tagtype = MOD.FFF_TAG_MAIN_TYPE.get(self.REDACTED, "Unknown")
        self.tagsubtype = MOD.FFF_PIXELS_SUBTAG_TYPE.get(self.REDACTED, "Unknown")


class FFFBasicInfo:
    def __repr__(self):
        return "<FFF Basic Info>"

    @file_open_and_seek
    def __init__(self, file, startbyte, datasize):

        # Different versions of the file format add more things, so we need to check against the data size before
        # reading certain objects.

        # Read all of these items in order. The file stream will make sure we get the right bytes.
        self.GeomInfo = FFFGeomInfo(file, file.tell())
        self.ObjectParams = FFFObjectParams(file, file.tell())
        self.CalibParams = FFFCalibParams(file, file.tell())
        self.CalibInfo = FFFCalibInfo(file, file.tell())
        self.AdjustParams = FFFAdjustParams(file, file.tell())
        self.PresentParams = FFFPresentationParams(file, file.tell())
        self.DisplayParams = FFFDisplayParams(file, file.tell())
        self.ImageInfo = FFFImageInfo(file, file.tell())
        self.DistributionData = FFFDistributionData(file, file.tell())

        if datasize == *****:
            return None

        self.ExtendedImageInfo = FFFExtendedImageInfo(file, file.tell())

        if datasize == *****:
            return None

        self.wbRBFparams = FFFwbRBFparams(file, file.tell())

        if datasize == *****:
            return None

        self.ExtPresentParams = FFFExtendedPresentationParams(file, file.tell())

        if datasize == *****:
            return None
        else:
            raise UnsupportedTypeError("BasicInfo", f"{datasize} B")


class FFFPixelData:
    def __repr__(self):
        return f"<FFF Pixels ({self.REDACTED}x{self.REDACTED})>"

    @file_open_and_seek
    def __init__(self, file, startbyte, datasize, subtype):

        # Get the image geometry info
        self.GeomInfo = FFFGeomInfo(file, file.tell())

        # Pixel data comes immediately after, but we need to work out how many and their size

        # Total number of pixels:
        n_pix = self.REDACTED * self.REDACTED

        # Pixel type/size - they are always UINT of some size
        try:
            pixel_type = MOD.FFF_PIXEL_SIZES[self.REDACTED]
        except KeyError:
            raise UnsupportedTypeError("PixelBytes", str(self.REDACTED))

        # Pixels can apparently be big or little endian, even if the file itself is something else.
        # They might also be stored as PNG data, but we've not yet written a reader for that.
        if subtype == *****:
            endianness = ">"
        elif subtype == *****:
            endianness = "<"
        elif subtype == *****:
            raise NotImplementedError("Pixel data REDACTED not supported.")
        elif subtype == *****:
            raise NotImplementedError("Pixel data REDACTED not supported.")
        elif subtype == *****:
            raise NotImplementedError("Pixel data REDACTED not supported.")
        else:
            raise UnsupportedTypeError("PixelData", f"Unknown [{subtype}]")

        # Grab the bytes that contains all the pixels
        all_bytes = file.read(n_pix * self.REDACTED)

        # Unpack and reshape into a nxm array.
        self.pixels = np.reshape(
            unpack_array(pixel_type, all_bytes, endianness), (self.REDACTED, self.REDACTED)
        )


class FFFGeomInfo:
    def __repr__(self):
        return "<FFF Image Geometry Info>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_GEOMINFO_BYTES)

        # Unpack, with some shortcuts

        (
            self.REDACTED,
            self.REDACTED,
            self.REDACTED,
            self.REDACTED,
            self.REDACTED,
            self.REDACTED,
        ) = unpack_array("H", all_bytes[*****:*****])
        self.REDACTED = unpack_array("B", all_bytes[*****:*****])
        self.REDACTED = unpack_array("B", all_bytes[*****:*****])


class FFFObjectParams:
    def __repr__(self):
        return "<FFF Object Info>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_OBJECTPARAMS_BYTES)

        # Unpack
        (
            self.REDACTED,
            self.REDACTED,
            self.REDACTED,
        ) = unpack_array("f", all_bytes[*****:*****])
        self.REDACTED = unpack_array("L", all_bytes[*****:*****])


class FFFCalibParams:
    def __repr__(self):
        return f"<FFF Calib Params ({self.REDACTED})>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_CALIBPARAMS_BYTES)

        # Unpack
        self.REDACTED = unpack_array("L", all_bytes[*****:*****])
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])

        self.pixeltypetag = MOD.FFF_CALIB_PIXELS_TYPE.get(self.REDACTED, "Unknown")


class FFFCalibInfo:
    def __repr__(self):
        return f"<FFF Calibration Info ({self.camera_name})>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_CALIBINFO_BYTES)

        # Unpack
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])

        # Optics components are tricked into reading from an in memory byte stream
        self.REDACTED = FFFOpticsInfo(io.BytesIO(all_bytes[*****:*****]))
        self.REDACTED = FFFOpticsInfo(io.BytesIO(all_bytes[*****:*****]))


class FFFOpticsInfo:
    def __repr__(self):
        if self.structUsed:
            return f"<FFF Optics Info ({self.descr})>"
        else:
            return "<FFF Optics Info (ununsed)>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_OPTICSINFO_BYTES)

        # Unpack
        self.REDACTED = unpack_array("l", all_bytes[*****:*****])
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])


class FFFAdjustParams:
    def __repr__(self):
        return "<FFF Adjustment Params>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_ADJUSTPARAMS_BYTES)

        # Unpack
        self.REDACTED = unpack_array("l", all_bytes[*****:*****])
        self.REDACTED = unpack_array("f", all_bytes[*****:*****])


class FFFPresentationParams:
    def __repr__(self):
        return "<FFF Presentation Params>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_PRESENTATIONPARAMS_BYTES)

        # Unpack
        self.REDACTED = unpack_array("l", all_bytes[*****:*****])
        self.REDACTED = unpack_array("l", all_bytes[*****:*****])


class FFFDisplayParams:
    def __repr__(self):
        return "<FFF Display Params>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_DISPLAYPARAMS_BYTES)

        # Unpack
        self.REDACTED = unpack_array("f", all_bytes[*****:*****])
        self.REDACTED = unpack_array("f", all_bytes[*****:*****])


class FFFImageInfo:
    def __repr__(self):
        return "<FFF Image Info>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_IMAGEINFO_BYTES)

        # Unpack
        self.REDACTED = unpack_array("L", all_bytes[*****:*****])
        self.REDACTED = unpack_array("L", all_bytes[*****:*****])

        # Convert the epoch seconds, and milliseconds into DateTime
        # Current experience shows that the timestamp is always in UTC, regardless of what the system time
        # actually is or if there's Daylight Savings time or not
        self.datetime = datetime.datetime.fromtimestamp(
            self.REDACTED + self.REDACTED * 0.001, datetime.timezone.utc
        )


class FFFDistributionData:
    def __repr__(self):
        return "<FFF Distribution Data>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_DISTRIBUTIONDATA_BYTES)

        # Unpack
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])
        self.REDACTED = unpack_array("H", all_bytes[*****:*****])


class FFFExtendedImageInfo:
    def __repr__(self):
        return "<FFF Extended Image Info>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # This type consists solely of sub types. Namely 20 temperature sensor data and 20 other detector parameters
        # It still makes sense to read everything in one go, then pass as memory streams, as it's less than 1 kB total

        n_bytes = 20 * MOD.FFF_TEMPSENSORDATA_BYTES + 20 * MOD.FFF_DETECTORPARAM_BYTES

        # Grab all the bytes
        all_bytes = file.read(n_bytes)

        # Make a reader object
        byte_stream = io.BytesIO(all_bytes)

        self.REDACTED = [FFFTempSensorData(byte_stream, ii * MOD.FFF_TEMPSENSORDATA_BYTES) for ii in range(*****)]
        self.REDACTED = [
            FFFDetectorParam(byte_stream, ***** * MOD.FFF_TEMPSENSORDATA_BYTES + ii * MOD.FFF_DETECTORPARAM_BYTES)
            for ii in range(*****)
        ]


class FFFTempSensorData:
    def __repr__(self):
        return f"<FFF Temperature Sensor ({self.REDACTED}: {self.ftstemp} K)>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_TEMPSENSORDATA_BYTES)

        # Unpack
        self.REDACTED = unpack_array("f", all_bytes[*****:*****])
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])

        # Convert time stamps to datetime object
        self.datetime = datetime.datetime.fromtimestamp(
            self.REDACTED + self.REDACTED * 0.001, datetime.timezone.utc
        )


class FFFDetectorParam:
    def __repr__(self):
        return f"<FFF Detector Param ({self.REDACTED}: {self.fdata})>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_DETECTORPARAM_BYTES)

        # Unpack
        self.REDACTED = unpack_array("f", all_bytes[*****:*****])
        self.REDACTED = unpack_array("s", all_bytes[*****:*****])


class FFFwbRBFparams:
    def __repr__(self):
        return "<FFF Waveband RBF Params>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_WBRBFPARAMS_BYTES)

        # Unpack
        self.REDACTED = unpack_array("l", all_bytes[*****:*****])
        self.REDACTED = unpack_array("h", all_bytes[*****:*****])


class FFFExtendedPresentationParams:
    def __repr__(self):
        return "<FFF Extended Presentation Params>"

    @file_open_and_seek
    def __init__(self, file, startbyte):

        # Grab the right number of bytes
        all_bytes = file.read(MOD.FFF_EXTENDEDPRESENTATIONPARAMS_BYTES)

        # Unpack
        self.REDACTED = unpack_array("H", all_bytes[*****:*****])
        self.REDACTED = unpack_array("H", all_bytes[*****:*****])
