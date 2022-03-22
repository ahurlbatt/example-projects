import numpy as np
from typing import Union


class DetectorCalibration(object):
    """A holder for camera/detector calibration coefficients that use R/B/F formulae

    Assuming a microbolometer camera, the response curve can be estimated by a function with the same form as
    Planck's Law for blackbody radiation, where instead of universal constants, the calibration values for the
    specific camera are used.

    N. Horny, "FPA Camera Standardisation", https://doi.org/10.1016/S1350-4495(02)00183-4

    via

    H. Budzier, G. Gerlach, "Calibration of Infrared Cameras with Microbolometers",
    https://doi.org/10.5162/irs2015/1.1

    """

    def __init__(self, R: float, B: float, F: float):
        self.R = R
        self.B = B
        self.F = F

    def to_radiance(self, temperature: Union[float, np.ndarray]):
        """Convert Temperature of an assumed blackbody into a Radiance that would be measured by a this detector

        Parameters
        ----------
        temperature : Union[float, np.ndarray]
            Temperature of the blackbody in Kelvin.

        Returns
        -------
        Radiance: Union[float, np.ndarray]
            Measured Radiance in [W m^-2 sr^1 um].

        """

        return self.R / (np.exp(self.B / temperature) - self.F)

    def to_temperature(self, radiance: Union[float, np.ndarray]):
        """
        For recovering the temperature from a radiance measured by a camera, the formula for temp_to_radiance() is
        simply inverted, and the same calibration parameters used.

        Parameters
        ----------
        radiance : Union[float, np.ndarray]
            Measured Radiance in [W m^-2 sr^1 um].

        Returns
        -------
        Temperature: Union[float, np.ndarray]
            Temperature of the blackbody in Kelvin.

        """

        return self.B / np.log(self.R / radiance + self.F)


class CalibratedRadiance(object):
    """A container for values of radiance and temperature, and the calibration values"""

    def __init__(
        self,
        temperature: Union[float, np.ndarray] = None,
        radiance: Union[float, np.ndarray] = None,
        calibration: DetectorCalibration = None,
    ):
        """Create a CalibratedRadiance with either a single float or an array of temperature or radiance values

        Parameters
        ----------
        temperature : Union[float, np.ndarray], optional
            Temperatures in Kelvin.
        radiance : Union[float, np.ndarray], optional
            (Calibrated) radiance values in W/sr/m^2.
        calibration : DetectorCalibration, optional
            The calibration factors for conversion between the two, for the detector in use.

        """

        # Check we have one, and only one, of the input values.
        if temperature is None and radiance is None:
            raise ValueError("One of temperature/value must be specified")
        if temperature is not None and radiance is not None:
            raise ValueError("Only one of temperature/value can be specified")

        # Save both.
        self.temperature = temperature
        self.radiance = radiance

        # If we have already got the calibration factors, then use them.
        if calibration is not None:
            self.calibrate(calibration)

    def calibrate(self, calibration: DetectorCalibration):
        """Apply the given calibration to find the missing temperature or radiance"""

        # Store the calibration values
        self.calibration = calibration

        if self.temperature is None:
            self.temperature = self.calibration.to_temperature(self.radiance)

        if self.radiance is None:
            self.radiance = self.calibration.to_radiance(self.temperature)

    def __add__(self, other):
        """Allow addition of radiance values by other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(self.radiance + other), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(self.radiance + other.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Addition of 'CalibratedRadiance' and '{type(other).__name__}' not supported."
                ) from exc

    def __radd__(self, other):
        """Allow addition of radiance values by other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(other + self.radiance), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(other.radiance + self.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Addition of '{type(other).__name__}' and 'CalibratedRadiance' not supported."
                ) from exc

    def __sub__(self, other):
        """Allow subtraction of other values or scalars/arrays from radiance values"""
        try:
            return CalibratedRadiance(radiance=(self.radiance - other), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(self.radiance - other.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Subtraction of '{type(other).__name__}' from 'CalibratedRadiance' not supported."
                ) from exc

    def __rsub__(self, other):
        """Allow subtraction of radiance values from other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(other - self.radiance), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(other.radiance - self.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Subtraction of '{type(other).__name__}' from 'CalibratedRadiance' not supported."
                ) from exc

    def __mul__(self, other):
        """Allow multiplication of radiance values by other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(self.radiance * other), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(self.radiance * other.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Multiplication of 'CalibratedRadiance' and '{type(other).__name__}' not supported."
                ) from exc

    def __rmul__(self, other):
        """Allow multiplication of radiance values by other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(other * self.radiance), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(other.radiance * self.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Multiplication of '{type(other).__name__}' 'CalibratedRadiance' not supported."
                ) from exc

    def __div__(self, other):
        """Allow division of radiance values by other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(self.radiance / other), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(self.radiance / other.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Division of 'CalibratedRadiance' by '{type(other).__name__}' not supported."
                ) from exc

    def __rdiv__(self, other):
        """Allow division of radiance values into other values or scalars/arrays"""
        try:
            return CalibratedRadiance(radiance=(other / self.radiance), calibration=self.calibration)
        except TypeError:
            try:
                return CalibratedRadiance(radiance=(other.radiance / self.radiance), calibration=self.calibration)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Division of 'CalibratedRadiance' into '{type(other).__name__}' not supported."
                ) from exc


class InfraredObject(object):
    """An object that appears in front of an infrared camera/detector

    Properties:

    Temperatures and radiances:
        The ideal radiance of both the object and its background are needed for calculations of self emission and
        reflected radiation. These should be specified via a temperature in Kelvin. As the relationship between
        temperature and radiance is dependent on the detector in use, the detector's calibration coefficients are needed
        to calculate the radiance values.

    Emittance, Transmittance, and Reflectance:
        These unitless values in [0, 1] characterise how the object emits radiation and interacts with incoming
        radiation. Due to the identity of Emittance + Transmittance + Reflectance = 1, only two of the three
        parameters need to be provided.

    Diffuse Fraction:
        Reflections are modelled as a combination of two extreme possibilities: specular and diffuse. This parameter
        specifies the fraction of reflections that are diffuse in nature.

    Axis Alignment:
        If this value is True, then specular reflections are aligned with the direction of the detector, and so
        radiation is reflected along the path between the detector and the target. If it is False, then specular
        reflections of the background are assumed to be sent toward both the detector and the target object.

    Calibration:
        The calibration factors for the appropriate detector that allow the conversion between temperature and
        radiance. These can be provided either during creation, or later using the calibrate() method.

    """

    def __init__(
        self,
        temperature: float,
        background_temperature: float,
        emittance: float = None,
        transmittance: float = None,
        reflectance: float = None,
        diffuse_fraction: float = 0.0,
        axis_alignment: bool = False,
        calibration: DetectorCalibration = None,
    ):
        """Create an InfraredObject instance using given values

        Parameters
        ----------
        temperature : float
            Temperature of the object in Kelvin.
        background_temperature : float
            Temperature of the object's background in Kelvin.
        emittance : float, optional
            Emittance of the object's surface in [0, 1].
        transmittance : float, optional
            Transmittance of the object in [0, 1].
        reflectance : float, optional
            Reflectance of the object in [0, 1].
        diffuse_fraction : float, optional
            Fraction of reflection from the object that can be considered diffuse, as opposed to specular.
        axis_alignment : bool, optional
            True if specularly reflected light remains along the axis of the camera/detector.
            (Imagine the camera being able to see itself in the reflection)
        calibration : DetectorCalibration, optional
            Calibration coefficients for conversion between temperature and radiance.
            Can be provided after object creation.

        """

        # Create CalibratedRadiance instances for the provided temperatures
        self.radiance = CalibratedRadiance(temperature=temperature)
        self.background_radiance = CalibratedRadiance(temperature=background_temperature)

        # Check we have at least two of emittance, transmittance, and reflectance, then handle them.
        if (number_missing := sum(item is None for item in [emittance, transmittance, reflectance])) > 1:
            raise ValueError("At least two from emittance, transmittance, reflectance must be specified.")
        elif number_missing > 0:
            if emittance is None:
                emittance = 1.0 - transmittance - reflectance
            elif transmittance is None:
                transmittance = 1.0 - emittance - reflectance
            elif reflectance is None:
                reflectance = 1.0 - transmittance - emittance

        # These should all now have values
        self.emittance = emittance
        self.transmittance = transmittance
        self.reflectance = reflectance

        # These properties should all be positive and add to one.
        assert emittance >= 0.0
        assert transmittance >= 0.0
        assert reflectance >= 0.0
        assert np.isclose(emittance + transmittance + reflectance, 1.0)

        # Grab the other properties
        self.diffuse_fraction = diffuse_fraction
        self.axis_alignment = axis_alignment

        # If we're provided a calibration already, then use it
        if calibration is not None:
            self.calibrate(calibration)

    def calibrate(self, calibration: DetectorCalibration):
        """Apply the given calibration factors to the object and background CalibratedRadiance

        If the attributes radiance and background_radiance are not instances of CalibratedRadiance, they are
        assumed to be values of radiance. A CalibratedRadiance instance is then created using the value and the
        calibration.

        Parameters
        ----------
        calibration : DetectorCalibration
            Calibration coefficients of the detector being used.

        """

        try:
            self.radiance.calibrate(calibration)
        except AttributeError:
            self.radiance = CalibratedRadiance(radiance=self.radiance, calibration=calibration)

        try:
            self.background_radiance.calibrate(calibration)
        except AttributeError:
            self.background_radiance = CalibratedRadiance(radiance=self.background_radiance, calibration=calibration)

    @property
    def modification_matrix(self):
        """Matrix defining the impact on infrared radiation travelling through the object

        Returns
        -------
        modification_matrix : (2,2) array

        """
        if self.axis_alignment:
            return np.array(
                [
                    [1.0 / self.transmittance, -(1.0 - self.diffuse_fraction) * self.reflectance / self.transmittance],
                    [
                        (1.0 - self.diffuse_fraction) * self.reflectance,
                        self.transmittance
                        - (1.0 - self.diffuse_fraction) ** 2 * self.reflectance ** 2 / self.transmittance,
                    ],
                ]
            )
        else:
            return np.array([[1 / self.transmittance, 0.0], [0.0, self.transmittance]])

    @property
    def additional_radiation(self):
        """Radiation that must be added to the forward/backward values on the detector side

        Returns
        -------
        additional_radiation : (2,) array

        """

        # Check if our radiance values have been calibrated yet
        if self.radiance.radiance is None or self.background_radiance.radiance is None:
            raise RuntimeError("CalibratedRadiance values not calibrated.")

        if self.axis_alignment:
            return np.array(
                [
                    [-1.0 / self.transmittance],
                    [1.0 - (1.0 - self.diffuse_fraction) * self.reflectance / self.transmittance],
                ]
            ) * (
                self.emittance * self.radiance.radiance
                + self.diffuse_fraction * self.reflectance * self.background_radiance.radiance
            )
        else:
            return np.array([[-1.0 / self.transmittance], [1.0]]) * (
                self.emittance * self.radiance.radiance + self.reflectance * self.background_radiance.radiance
            )


class Window(InfraredObject):
    """Alias for InfraredObject"""

    pass


class Mirror(InfraredObject):
    """An InfraredObject that changes internal behaviour to support reflections as the desired outcome

    The modification_matrix and additional_radiation properties are calculated differently to the parent class, as the
    reflectance defines how much radiation is "transmitted".

    Reflectance, Emittance, and Transmittance:
        These unitless values in [0, 1] characterise how the object emits radiation and interacts with incoming
        radiation. For a Mirror, the transmittance is assumed to be zero, so only the reflectance is required.
        The emittance is then calculated as 1 - reflectance.

    Temperatures and radiances:
        The ideal radiance of both the object and its background are needed for calculations of self emission and
        reflected radiation. These should be specified via a temperature in Kelvin. As the relationship between
        temperature and radiance is dependent on the detector in use, the detector's calibration coefficients are needed
        to calculate the radiance values.

    Diffuse Fraction:
        Reflections are modelled as a combination of two extreme possibilities: specular and diffuse. This parameter
        specifies the fraction of reflections (if any) that are diffuse in nature.

    Calibration:
        The calibration factors for the appropriate detector that allow the conversion between temperature and
        radiance.

    """

    def __init__(
        self,
        reflectance: float,
        temperature: float,
        background_temperature: float,
        diffuse_fraction: float = 0.0,
        calibration: DetectorCalibration = None,
    ):
        """Create a Mirror instance using given values

        Parameters
        ----------
        reflectance : float
            Reflectance of the Mirror in [0, 1].
        temperature : float
            Temperature of the Mirror in Kelvin.
        background_temperature : float
            Temperature of the Mirror's background in Kelvin.
        diffuse_fraction : float, optional
            Fraction of reflection from the object that can be considered diffuse, as opposed to specular.
        calibration : DetectorCalibration, optional
            Calibration coefficients for conversion between temperature and radiance.
            Can be provided after object creation.
        """

        # The Mirror has all the same properties as a normal InfraredObject, but some values are fixed.
        super().__init__(
            temperature=temperature,
            background_temperature=background_temperature,
            transmittance=0.0,
            reflectance=reflectance,
            diffuse_fraction=diffuse_fraction,
            axis_alignment=False,
            calibration=calibration,
        )

    @property
    def modification_matrix(self):
        """Matrix defining the impact on infrared radiation travelling through the object

        Differs from the parent class in that "transmission" of radiation is defined by the (specular) reflectance of
        the object. In addition, there are no specular reflections of radiation back to where it came from, and no
        specular reflections of the background are possible.

        Returns
        -------
        modification_matrix : (2,2) array

        """
        return np.array(
            [
                [1 / ((1.0 - self.diffuse_fraction) * self.reflectance), 0.0],
                [0.0, (1.0 - self.diffuse_fraction) * self.reflectance],
            ]
        )

    @property
    def additional_radiation(self):
        """Radiation that must be added to the forward/backward values on the detector side

        Differs from the parent class in that "transmission" of radiation is defined by the (specular) reflectance of
        the object. In addition, there are no specular reflections of radiation back to where it came from, and no
        specular reflections of the background are possible.

        Returns
        -------
        additional_radiation : (2,) array

        """
        # Check if our radiance values have been calibrated yet
        if self.radiance.radiance is None or self.background_radiance.radiance is None:
            raise RuntimeError("CalibratedRadiance values not calibrated.")

        return np.array([[-1.0 / ((1.0 - self.diffuse_fraction) * self.reflectance)], [1.0]]) * (
            self.emittance * self.radiance.radiance
            + self.diffuse_fraction * self.reflectance * self.background_radiance.radiance
        )


class Atmosphere(InfraredObject):
    """A packet of air that appears in front of an infrared camera/detector

    Uses the measureable properties of temperature, length, and relative humidity to estimate relevant parameters of
    transmittance and emittance of infrared radiation via emperical formulae. The air is assumed to be at atmospheric
    pressure.

    Properties:

    Temperature:
        Temperature of the air in Kelvin. Used to estimate emittance and transmittance values, as well as the ideal
        radiance via the detector calibration.

    Length, Relative Humidity:
        These properties of the air are required for the estimation of the emittance and transmittance.

    Calibration:
        The calibration factors for the appropriate detector that allow the conversion between temperature and
        radiance. These can be provided either during creation, or later using the calibrate() method.

    """

    def __init__(
        self, temperature: float, length: float, relative_humidity: float, calibration: DetectorCalibration = None
    ):
        """Create an Atmosphere object that affects infrared radiation travelling through it

        Parameters
        ----------
        temperature : float
            Temperature of the air in Kelvin.
        length : float
            Length of the radiation path through the air in metres.
        relative_humidity : float
            Relative humidity of the air in [0, 1].
        calibration : DetectorCalibration, optional
            Calibration coefficients for conversion between temperature and radiance.
            Can be provided after object creation.

        """
        # Transmittance of the chunk of atmosphere is found through emperical formulae.
        transmittance = atmospheric_transmittance(temperature, length, relative_humidity)

        # Use the transmittance and temperature values to create the rest of the required values, with the assumption
        # that the air is not reflective ...
        super().__init__(
            temperature=temperature,
            background_temperature=temperature,
            transmittance=transmittance,
            reflectance=0.0,
            diffuse_fraction=0.0,
            axis_alignment=False,
            calibration=calibration,
        )

        # Save the input values for reference
        self.temperature = temperature
        self.length = length
        self.relative_humidity = relative_humidity


class Vacuum(Atmosphere):
    """An empty object that doesn't affect the radiation travelling through it"""

    def __init__(self, length: float = None):

        # Make a packet of air with zero length and humidity, but a temperature of 300 K.
        # If we set a temperature of zero, then we'll have to rewrite lots to deal with that.
        # As the emittance and reflectance are zero, the temperature we set here has no effect.
        super().__init__(300.0, 0.0, 0.0)

        # Override the internal length attribute, for reference.
        self.length = length


class TargetObject(object):
    """An object that is assumed to be the target for the infrared measurement, so the temperature is unknown

    Properties:

    Background Temperatures and radiances:
        The ideal radiance of the background is needed for calculations of reflected radiation. This should be specified
        via a temperature in Kelvin. As the relationship between temperature and radiance is dependent on the detector
        in use, the detector's calibration coefficients are needed to calculate the radiance values.

    Emittance and Reflectance:
        These unitless values in [0, 1] characterise how the object emits radiation and interacts with incoming
        radiation. Due to the identity of Emittance + Transmittance + Reflectance = 1, and the assumption of zero
        transmittance, only one of these parameters needs to be provided.

    Diffuse Fraction:
        Reflections are modelled as a combination of two extreme possibilities: specular and diffuse. This parameter
        specifies the fraction of reflections (if any) that are diffuse in nature.

    Axis Alignment:
        If this value is True, then specular reflections are aligned with the direction of the detector, and so
        radiation is reflected along the path between the detector and the target. If it is False, then specular
        reflections of the background are assumed to be sent toward both the detector and the target object.

    Calibration:
        The calibration factors for the appropriate detector that allow the conversion between temperature and
        radiance. These can be provided either during creation, or later using the calibrate() method.

    """

    def __init__(
        self,
        background_temperature: float = None,
        emittance: float = None,
        reflectance: float = None,
        diffuse_fraction: float = 0.0,
        axis_alignment: bool = False,
        calibration: DetectorCalibration = None,
    ):
        """Create a TargetObject instance using given values to find those not provided

        All inputs are optional, but only certain combinations of inputs are valid.

        Parameters
        ----------
        background_temperature : float, optional
            Temperature of the object's background in Kelvin. May be omitted if background_radiance is provided.
        background_radiance : Union[CalibratedRadiance, float], optional
            (Calibrated) radiance of the object's background, either as a CalibratedRadiance object or value
            in W/m^2. May be omitted if background_temperature is provided.
        emittance : float, optional
            Emittance of the object's surface in [0, 1].
        reflectance : float, optional
            Reflectance of the object in [0, 1].
        diffuse_fraction : float, optional
            Fraction of reflection from the object that can be considered diffuse, as opposed to specular.
        axis_alignment : bool, optional
            True if specularly reflected light remains along the axis of the camera/detector.
            (Imagine the camera being able to see itself in the reflection)
        calibration : DetectorCalibration, optional
            Calibration coefficients for conversion between temperature and radiance.
            Can be provided after object creation.

        """

        # Create a CalibratedRadiance instance using the background_temperature
        self.background_radiance = CalibratedRadiance(temperature=background_temperature)

        # Check we have at least two of emittance, transmittance, and reflectance, then handle them.
        if emittance is None and reflectance is None:
            raise ValueError("At least one of emittance or reflectance must be specified.")
        else:
            if emittance is None:
                emittance = 1.0 - reflectance
            elif reflectance is None:
                reflectance = 1.0 - emittance

        # These should both have values
        self.emittance = emittance
        self.reflectance = reflectance

        # These properties should be positive and add to one.
        assert emittance >= 0.0
        assert reflectance >= 0.0
        assert np.isclose(emittance + reflectance, 1.0)

        # Grab the other properties
        self.diffuse_fraction = diffuse_fraction
        self.axis_alignment = axis_alignment

        # If we're provided a calibration already, then use it
        if calibration is not None:
            self.calibrate(calibration)

    def calibrate(self, calibration: DetectorCalibration):
        """Apply the given calibration factors to the background CalibratedRadiance

        If the attribute background_radiance is not an instance of CalibratedRadiance, it is assumed to be a value of
        radiance. A CalibratedRadiance instance is then created using the value and the calibration.

        Parameters
        ----------
        calibration : DetectorCalibration
            Calibration coefficients of the detector being used.

        """

        try:
            self.background_radiance.calibrate(calibration)
        except AttributeError:
            self.background_radiance = CalibratedRadiance(radiance=self.background_radiance, calibration=calibration)

    @property
    def emission_matrix(self):
        """In combination with the reflections, allows conversion of outgoing radiation into radiance value"""
        if self.axis_alignment:
            return np.array([1.0, -(1.0 - self.diffuse_fraction) * self.reflectance]) / self.emittance
        else:
            return np.array([1.0 / self.emittance, 0.0])

    @property
    def reflection(self):
        """In combination with the emittance, allows conversion of outgoing radiation into radiance value"""
        if self.axis_alignment:
            return np.array([self.diffuse_fraction * self.reflectance / self.emittance])
        else:
            return np.array([self.reflectance / self.emittance])


class Detector(object):
    """Container for physical properties of the infrared detector

    Properties:

    Temperatures and radiances:
        The ideal radiance of the detector is needed for calculations of self emission. This should be specified via a
        temperature in Kelvin. As the relationship between temperature and radiance is dependent on the detector in use,
        the detector's calibration coefficients are needed to calculate the radiance values. Note that reflections from
        the background are assumed negligible.

    Calibration:
        The calibration factors for the detector that allow the conversion between temperature and radiance.
        These can be provided either during creation, or later using the calibrate() method.

    """

    def __init__(
        self, temperature: float, calibration: DetectorCalibration = None,
    ):
        """Create an Detector instance with the given temperature

        Parameters
        ----------
        temperature : float
            Temperature of the object in Kelvin.
        calibration : DetectorCalibration, optional
            Calibration coefficients for conversion between temperature and radiance.
            Can be provided after object creation.

        """
        # Create a CalibratedRadiance instance for the provided temperature
        self.radiance = CalibratedRadiance(temperature=temperature)

        # If we're provided a calibration already, save it then use it
        if calibration is not None:
            self.calibration = calibration
            self.calibrate(calibration)

    def calibrate(self, calibration: DetectorCalibration):
        """Apply the given calibration factors to the object and background CalibratedRadiance

        If radiance is not an instance of CalibratedRadiance, it is assumed to be values of radiance.
        A CalibratedRadiance instance is then created using the value and the 9calibration.

        Parameters
        ----------
        calibration : DetectorCalibration
            Calibration coefficients of the detector being used.

        """

        # Save the calibration to be used by other objects
        self.calibration = calibration

        try:
            self.radiance.calibrate(calibration)
        except AttributeError:
            self.radiance = CalibratedRadiance(radiance=self.radiance, calibration=calibration)

    @property
    def additional_radiation(self):
        """Self emission from the detector"""

        if self.radiance.radiance is None:
            raise RuntimeError("CalibratedRadiance values not calibrated.")

        return self.radiance.radiance


def atmospheric_transmittance(temperature, length, relative_humidity):
    """
    Estimate the transmittance of a chuck of air using its length, relative humidity, and temperature
    using an empirical formula. For more detail see Section 3 of:

    W. Minkina and D. Klecha,
    "Modeling of Atmospheric Transmission Coefficient in Infrared for Thermovision Measurements"
    https://doi.org/10.5162/irs2015/1.4

    A pressure of 1 atm is assumed!

    Parameters
    ----------
    length : float
        Length of the air packet in metres.
    relative_humidity: float
        Relative humidity as a fraction in the interval [0,1]
    temperature: float
        Temperature of the air in Kelvin

    Returns
    -------
    Transmittance: float
        Transmittance of the air as a value [0,1].

    """

    # Check Relative Humidity is a valid value
    assert (
        relative_humidity >= 0.0 and relative_humidity <= 1.0
    ), f"Relative humidity of {relative_humidity} is invalid."

    # From the publication, the following constants are used for the calculation of absolute humidity:
    h1 = 6.8455e-7
    h2 = -2.7816e-4
    h3 = 6.939e-2
    h4 = 1.5587

    # Calculation of the absolute humidity needs the temperature in Celcius, not Kelvin.
    T_C = temperature - 273.15

    # Find the absolute humidty from Equation 2 of the publication.
    absolute_humidity = relative_humidity * np.exp(h1 * T_C ** 3 + h2 * T_C ** 2 + h3 * T_C + h4)

    # From the publication, the following constants are used for the atmospheric transmittance:
    K_atm = 1.9
    a1 = 0.0066
    a2 = 0.0126
    b1 = -0.0023
    b2 = -0.0067

    # Find the transmittance using Equation 3 of the publication
    return K_atm * np.exp(-np.sqrt(length) * (a1 + b1 * np.sqrt(absolute_humidity))) + (1 - K_atm) * np.exp(
        -np.sqrt(length) * (a2 + b2 * np.sqrt(absolute_humidity))
    )


def system_transfer_matrices(object_system: list):
    """Combine the modification matrices and additional radiation sources from a list of InfraredObject instances

    object_system should be ordered with object_system[0] closest to the detector.

    For the vector of forward/backward radiation passing through, there is a multiplication by the modification matrix,
    then the addition of the additional radiation, for each object in the system. The end result can be given by a
    single multiplication matrix and single addition vector, but finding these requires combining the individual
    contributions in a series of iterative matrix multiplications and additions.

    For (2,1) forward/backward radiation vector X, (2,2) modification matrix A, and (2,1) additional radiation vector b:
        X_n = A_n @ X_(n-1) + b_n

    We want to find X_n given X_0, [A_0 ... A_n], and [b_0 ... b_n]. Starting with X_0:
        X_1 = A_1 @ X_0 + b_1
        X_2 = A_2 @ X_1 + b_2
            = A_2 @ (A_1 @ X_0 + b_1) + b_2
            = A_2 @ A_1 @ X_0 + A_2 @ b_1 + b_2
        ...
               /             \             /             \
              |  n-1          |       n-1 | n-i-1         |
              | _____         |        _  | _____         |
        X_n = |  | |  A_(n-i) | X_0 + \   |  | |  A_(n-j) | b_i + b_n
              |  | |          |       /_  |  | |          |
              |               |       i=1 |               |
               \ i=0         /             \ j=0         /

        where big-pi and big-sigma notation is used to define (matrix-) product and summation respectively.
        (N.B. indexing with e.g. (n-i) to preseve correct ordering during matrix multiplication)

    Extracting the two resulting matrices gives:

        X_n = E_n X_0 + C_n

        where:

               /             \                   /             \
              |  n-1          |             n-1 | n-i-1         |
              | _____         |              _  | _____         |
        E_n = |  | |  A_(n-i) |  and  C_n = \   |  | |  A_(n-j) | b_i + b_n
              |  | |          |             /_  |  | |          |
              |               |             i=1 |               |
               \ i=0         /                   \ j=0         /

    Parameters
    ----------
    object_system : list
        List of InfraredObject (or subclass) instances describing objects between a detector and target object, with
        object_system[0] being closest to the detector.

    Returns
    -------
    modification_output : (2,2) array
        Matrix E_N from above.
    additional_output : (2,) array
        Vector b_N from above.

    """
    # Work out how many object are in our list
    n_objects = len(object_system)

    # If we have more than one object, then we should collect all of the matrices/vectors, then perform the
    # multiplications and summations. If we only have one, then there's no collection to be done and we simply return
    # the properties of that one object.

    if n_objects == 1:
        modification_output = object_system[0].modification_matrix
        additional_output = object_system[0].additional_radiation
    else:
        # Fetch all the modifaction matrices and additional radiation values
        modifications = [o.modification_matrix for o in object_system]
        additionals = [o.additional_radiation for o in object_system]

        # Use multi_dot to perform matrix multiplication on the list of matrices.
        modification_output = np.linalg.multi_dot(list(reversed(modifications)))
        additional_output = (
            sum(
                [
                    np.linalg.multi_dot(
                        [modifications[n_objects - jj] for jj in range(1, n_objects - ii)] + [additionals[ii]]
                    )
                    for ii in range(0, (n_objects - 1))
                ]
            )
            + additionals[-1]
        )

    return (modification_output, additional_output)


def find_radiance_coefficients(target_object: TargetObject, detector: Detector, object_system: list):
    """Combine properties of an IR imaging system into a linear relationship between measured/estimated radiance

    Parameters
    ----------
    target_object : TargetObject
        The object that is the target of the thermographic imaging.
    detector : Detector
        Calibrated detector taking infrared measurements.
    object_system : list
        List of InfraredObject (or subclass) instances describing objects between a detector and target object, with
        object_system[0] being closest to the detector.

    Returns
    -------
    radiance_scale : float
        Scale of the linear relationship between measured and estimated radiance of the target.
    radiance_offset : float
        Offset of the linear relationship between measured and estimated radiance of the target.

    """

    # Pass the detector calibration to the objects to ensure radiance values are present
    for o in object_system:
        o.calibrate(detector.calibration)

    # Pass the calibration to the target object too
    target_object.calibrate(detector.calibration)

    # Get the transfer matrices for the list of objects
    system_modification_matrix, system_additional_radiation = system_transfer_matrices(object_system)

    # Combine with the properties of the target object into a single modification matrix
    modification_matrix = target_object.emission_matrix @ system_modification_matrix

    # The additional radiation combines into a single value
    additional_radiation = float(target_object.emission_matrix @ system_additional_radiation - target_object.reflection)

    # Relationship scale is the effective transmittance of the whole system, which includes emittance of the target.
    radiance_scale = modification_matrix[0]

    # Offset of the relationship is related to the self emission of intermediate objects and reflected background
    radiance_offset = modification_matrix[1] * detector.additional_radiation + additional_radiation

    return (radiance_scale, radiance_offset)


def find_object_radiance(
    measured_radiance: CalibratedRadiance, target_object: TargetObject, detector: Detector, object_system: list
):
    """Estimate the radiance (and temperature) of a target object measured by a given detector through a given system

    Parameters
    ----------
    measured_radiance : CalibratedRadiance
        Radiance measured by the detector. Does not need to be calibrated.
    target_object : TargetObject
        The object that is the target of the measurement.
    detector : Detector
        The detector that has performed the measurement. Must have a calibration applied.
    object_system : list
        The objects between the detector and the target object, starting at the detector.

    Returns
    -------
    object_radiance : CalibratedRadiance
        The estimated radiance (and temperature) of the target object after consideration of the imaging system and the
        properties of the detector and the target object itself.

    """

    # Get the linear scaling between measured and object radiance for this system
    radiance_linear_coefficients = find_radiance_coefficients(target_object, detector, object_system)

    # Apply the detector calibration to the measured radiance
    measured_radiance.calibrate(detector.calibration)

    # Return the input radiance after this linear transformation
    return measured_radiance * radiance_linear_coefficients[0] + radiance_linear_coefficients[1]
