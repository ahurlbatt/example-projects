## NNBI Beam Diagnostic Analyses
An under-development package of beam diagnostic analysis pipelines for the NNBI devices at IPP-Garching.

The previous suite of beam diagnostic analysis scripts for the NNBI test facilities at IPP Garching is held in miscellaneous source code files, distributed between people and places. The aim of this package is threefold:
- Ensure analyses are traceable and recreatable
- Homogenise analysis methods between the NNBI test facilities
- Give everyone simple access

## Package Contents
This package is still young, so there's not so much here yet.

### `bugcwcal.py`
Contains the `BUGFoilCalorimeter` class for analysing infrared videos of the CW Calorimeter at BUG. Provided with video data, as an iterable of thermal images, it use image recognition to transform the images into real-space. The accuracy of the transformation is high enough to predict the locations of the 2-by-2 pixel measurement targets (foils) across the whole image, even when they are not warm enough to be resolved. By analysing the time dependent behaviour of the measurement targets, an equilibrium temperature for each target can be estimated. This can then be used to calculated an absolute power density, accessible as an attribute of the `BUGFoilCalorimeter` instance. Uncertainties in values are traced through calculations using the `uncertainties` package.

### `imageanalysis.py`
A module of routines and helper functions related to image recognition and analysis. Contains:
- A convenience wrapper for matching templates with the `skimage.feature` package
- Methods for estimating and using image transformations, including an extension of `skimage.transform.ProjectiveTransform` to allow weighting of source-destination pairs
- Fitting and using a transform that makes an integer-valued grid from a set of scattered points, useful for assigning array indices to objects
- Specific fitting routines for improving the accuracy of position information of calorimeter measurement targets

The extension for weighted estimation of projective transformations is now in the next release candidate for `skimage`, 0.19.0rc0, so it might soon be redundant here!
