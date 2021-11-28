---

The file layout information contained in the original version of this package is the subject of an NDA between IPP and FLIR and has been redacted or removed.

---

## Python SEQ Reader

A reader and converter for binary SEQ and FFF files from FLIR infrared camera systems, including a general infrared optics module for consideration of window and atmosphere effects.

* Support for partial reading of large files
* Automatic conversion to Apparent Radiance based on embedded camera calibration
* Calculation of object temperatures in Kelvin from given environment parameters
* Consideration of multi-component optical path between camera and object
* High-level functions for direct reading and conversion

**Usage**
```
import flirseq

# Define the environment conditions
background_temperature = 20 + 273.15
relative_humidity = 0.5

# Create a list of optical components between the camera and the object
my_optics = [
    flirseq.infroptics.Atmosphere(background_temperature, 0.1, relative_humidity),
    flirseq.infroptics.Window(background_temperature, 0.97, 0.01),
    flirseq.infroptics.Vacuum(0.3),
    flirseq.infroptics.Mirror(background_temperature,0.99),
    flirseq.infroptics.Vacuum(1)
    ]

# Pass these arguments to the top-level function for direct conversion to Kelvin of the last frame in the file
K, s = flirseq.seq_to_kelvin(file_name, 
              my_optics,
              background_temperature,
              object_emittance,
              frames = 'all')
```

**Top level objects**
* `FFF`: a single frame, containing header and pixel information
* `SeqData`: a collection of frames

**High level functions**
* `seq_to_kelvin(file, optics, background_temperature, object_emittance, frames)` directly returns a list of images in Kelvin.
