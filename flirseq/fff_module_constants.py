"""
This file contains constants used by the FLIR SEQ reader module, including the size of various objects (in bytes), 
string descriptions of data types (that are only given an integer in the files), as well as the name of the reader 
classes for each of the data types (if known).
"""

# Default encoding and endianness of the seq files
FFF_CHAR_ENCODING = "utf-8"
FFF_BYTE_ORDER = "<"

# The size of each known static data type in bytes
FFF_HEADER_BYTES = *****
FFF_INDEX_BYTES = *****
FFF_GEOMINFO_BYTES = *****
FFF_OBJECTPARAMS_BYTES = *****
FFF_CALIBPARAMS_BYTES = *****
FFF_CALIBINFO_BYTES = *****
FFF_OPTICSINFO_BYTES = *****
FFF_ADJUSTPARAMS_BYTES = *****
FFF_PRESENTATIONPARAMS_BYTES = *****
FFF_DISPLAYPARAMS_BYTES = *****
FFF_IMAGEINFO_BYTES = *****
FFF_DISTRIBUTIONDATA_BYTES = *****
FFF_TEMPSENSORDATA_BYTES = *****
FFF_DETECTORPARAM_BYTES = *****
FFF_WBRBFPARAMS_BYTES = *****
FFF_EXTENDEDPRESENTATIONPARAMS_BYTES = *****

# String decriptions of data types found in the descriptors
FFF_TAG_MAIN_TYPE = {
    0: "REDACTED",
}

# For pixel data, a subtype declares how they're stored
FFF_PIXELS_SUBTAG_TYPE = {0: "REDACTED",}

# The reader objects that should be used for each of the data types
FFF_TAG_READER_OBJ = {
    0: "REDACTED",
}

# For pixel data, this gives the type of data described
FFF_CALIB_PIXELS_TYPE = {
    0: "REDACTED",
}

# Conversion from pixel size (in bytes) to struct data type
FFF_PIXEL_SIZES = {1: "REDACTED"}
