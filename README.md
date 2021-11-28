## Example Projects
This is a small portfolio of recent Python work created for use with diagnostics and databases of negative ion beam experiments. There are examples of working with `pandas`, `skimage` for image recognition and transforms, raw file wrangling, linear algebra, and nonlinear least squares for curve fitting.

### `bugdb`
A collection of functional based modules for conversion of the mostly csv-file based experimental database into `pandas` DataFrames.

### `flirseq`
A reader class for extracting data from binary .seq files created when using FLIR infrared cameras. As well as reading the raw data, a separate `infroptics` module provides functionality for calculating the temperature of observed objects including the effects of everything between the camera and the object via some recursive linear algebra. Unfortunately the file structure information for .seq files is proprietary, and has had to be redacted to comply with an NDA.

### `nnbi-beamdiags`
A collection of modules for analysing data obtained from beam diagnostics of negative ion beams. Currently the collection size is effectively one. The `bugcwcal` and `imageanalysis` modules work together using image recognition, projective transformations, and nonlinear least squares solvers to extract the absolute power density from an infrared video of a beam hitting a specially adapted target.