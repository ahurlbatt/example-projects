---

The file layout information contained in the original version of this package is the subject of an NDA between IPP and FLIR and has been redacted or removed.

---

## Python SEQ Reader

A reader and converter for binary SEQ and FFF files from FLIR infrared camera systems, including a module for consideration of objects between the detector and target object, for including e.g. window and atmosphere effects.

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
    flirseq.ir_imaging.Atmosphere(temperature=background_temperature, length=0.1, relative_humidity=relative_humidity),
    flirseq.ir_imaging.Window(
            temperature=background_temperature,
            background_temperature=background_temperature,
            transmittance=0.97,
            reflectance=0.02,
            axis_alignment=True,
        ),
    flirseq.ir_imaging.Vacuum(0.3),
    flirseq.ir_imaging.Mirror(
            reflectance=0.99, temperature=background_temperature, background_temperature=background_temperature,
        ),
    flirseq.ir_imaging.Vacuum(1),
    ]

# Define the estimated emittance of the object of interest
target_object = flirseq.ir_imaging.TargetObject(
    emittance=0.9, background_temperature=background_temperature, diffuse_fraction=1.0,
)

# Pass these arguments to the top-level function for direct conversion to Kelvin of the last frame in the file
K, s = flirseq.seq_reader.seq_to_kelvin(file_name, target_object, object_system, frames="last")
```

**Top level objects**
* `FFF`: a single frame, containing header and pixel information
* `SeqData`: a collection of frames

**High level functions**
* `seq_to_kelvin(file, optics, background_temperature, object_emittance, frames)` directly returns a list of images in Kelvin.
