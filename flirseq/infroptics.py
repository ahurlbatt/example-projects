"""
This module contains everything needed to convert from radiance measured by a microbolometer infrared camera to a
temperature (in Kelvin) of a target object, including accounting for objects such as air and windows that lie between
the camera and the target.
"""

import numpy as np


class ValueOutOfRangeError(Exception):
    def __init__(self, value, label, validrange: str = None):
        self.value = value
        self.label = label
        self.validrange = validrange

    def __str__(self):
        if self.validrange is None:
            return f"Value {self.value} of " "{self.label}" " is out of range."
        else:
            return (
                f'Value {self.value} of "{self.label}" is out of range '
                + f"[{self.validrange[0]}, {self.validrange[1]}]."
            )


def find_object_radiance(
    RadianceApparent: np.ndarray, object_emittance: float, optics: list, background, object_reflectance: float = 0.0
):
    """Direct application of radiance conversion coefficients on a given measured radiance"""
    coeffs = find_radiance_coefficients(object_emittance, optics, background)
    return RadianceApparent * coeffs[0] + coeffs[1]


def find_radiance_coefficients(object_emittance: float, optics: list, background, object_reflectance: float = 0.0):
    """Find the linear coefficients for estimating object radiance from measured radiance
    
    From the properites of each optical component, a pair of coupled recursive linear equations can be created, 
    describing the radiance that must be moving across the boundary between two elements in either direction. As the 
    equations are linear, they can be turned into a single (inhomogeneous) recursive matrix equation, which can then
    be used to find a single non-recursive matrix equation that describes the whole system. In the following
    definitions, 'n' = 0 corresponds to the camera/detector, and 'n' increases towards the object. F_n is the radiance
    towards the camera at the boundary between the nth and (n+1)th component, and B_n is the radiance away from the 
    camera. F_0 is our measured radiance and we want F_(N), where N is the number of optical components. Each compontent
    has t_n, r_n, e_n, s_n and M_n as their transmittance, reflectance, emittance, diffuse reflectance, and radiance
    respectively. The target object also has these properties, with subscript T, and there is a background radiance of 
    M_B.
    
    In the forward direction:
        F_(n-1) = t_n*F_n + r_n*B_(n-1) + e_n*M_n + s_n*M_B
        --> F_n = (F_(n-1) - r_n*B_(n-1) - e_n*M_n - s_n*M_B)/t_n
    In the backward direction:
        B_n = r_n*F_n + t_n*B_(n-1) + e_n*M_n + s_n*M_B
            = r_n*(F_(n-1) - r_n*B_(n-1) - e_n*M_n - s_n*M_B)/t_n + t_n*B_(n-1) + M_n
            = (r_n/t_n)*F_(n-1) + (t_n - r_n^2/t_n)*B_(n-1) + (1 - r_n/t_n)*(e_n*M_n + s_n*M_B)
    
    Defining matrices:
        X_n = [[F_n]      A_n = [[ 1/t_n         -r_n/t_n ]        b_n = [[ -(e_n*M_n + s_n*M_B)/t_n ]
               [B_n]]            [ r_n/t_n   t_n - r_n^2/t_n ]]           [ (1-r_n/t_n)*(e_n*M_n + s_n*M_B) ]]
    
    The recursive matrix equation becomes:
        X_n = A_n X_(n-1) + b_n
    
    The recursion can be unraveled into a single formula:
    
               /             \             /             \                
              |   n           |       n-2 | n-i-1         |               
              | _____         |        _  | _____         |               
        X_n = |  | |  A_(n-i) | X_0 + \   |  | |  A_(n-j) | b_i + b_n 
              |  | |          |       /_  |  | |          |               
              |               |       i=0 |               |               
               \ i=1         /             \ j=1         /                
                           
        where big-pi and big-sigma notation is used to define (matrix-) product and summation respectively.
        (N.B. indexing with e.g. (n-i) to preseve correct ordering during matrix multiplication)
        
    This is still a linear equation, and the two parameters can be defined as follows:
        
        X_n = D_n X_0 + C_n
        
        where:
            
                   /             \                   /             \                
                  |   n           |             n-2 | n-i-1         |               
                  | _____         |              _  | _____         |               
            D_n = |  | |  A_(n-i) |  and  C_n = \   |  | |  A_(n-j) | b_i + b_n 
                  |  | |          |             /_  |  | |          |               
                  |               |             i=0 |               |               
                   \ i=1         /                   \ j=1         /                
    
    
    We have N and X_0, being the number of optical of optical components and the measured radiance M_A respectively
    (with the assumption that B_0 = 0). The value of X at the object is then:
        
        X_N = D_N X_0 + C_N
    
    The reflections from the target object also need to be considered, similarly to how F_n is defined above:
        
        M_T = (F_N - s_T*M_B - r_T*B_N)/e_T
    
    This can be gathered into another matrix expression:
        
        M_T = K X_0 + L
        
        where:
            
            K = P_T D_N
            l = P_T C_N - (s_T/e_T)*M_B
            P_T = [1/e_T  -r_T/e_T]
            
    Knowing that X_0[0] = M_M and X_0[1] = 0, we only need the first element of K, being k = K[0], and the value of l,
    to define a linear relationship between M_M and M_T.
    
    The output of this function is the coefficients k and l.
    
    Parameters
    ----------
    RadianceApparent : np.ndarray
        The image of apparent radiance as recorded by the camera.
    object_emittance : float [0, 1]
        The estimated emittance of the object being observed.
    optics : list of Optics
        The optical components that lie between the camera and the object. optics[0] is closest to the camera.
    background : Background
        The background that provides radiance reflected from objects.

    Returns
    -------
    (u, v): np.ndarry
        The estimated coefficients for calculating M_T from M_A.

    """

    # Sanity check on the provided object emittance
    if not (0.0 <= object_emittance <= 1.0):
        raise ValueOutOfRangeError(object_emittance, "object_emittance", [0, 1])
    if not (0.0 <= object_reflectance <= 1.0):
        raise ValueOutOfRangeError(object_reflectance, "object_reflectance", [0, 1])
    # We also need the diffuse reflectance of the object. this can be found by assuming e_T + r_T + s_T = 1
    object_reflectance_diffuse = 1.0 - object_emittance - object_reflectance

    # Extract the properties of the optical components into arrays, for speed and easier reading, including
    # any radiation that is reflected from the background.
    transmittances = np.array([o.transmittance for o in optics])
    radiances = np.array([o.radiance for o in optics])
    emittances = np.array([o.emittance for o in optics])
    reflectances_diffuse = np.array([o.reflectance_diffuse for o in optics])

    # Only solid objects ('Window's) have a reflectance that returns radiance back toward the source
    reflectances = np.array([o.reflectance * (isinstance(o, Window) and not o.offaxis) for o in optics])

    # Build the matrices A_n, initially with n indexing the 3rd dimension of the array, but moved to the 1st
    A = np.moveaxis(
        np.array(
            [
                [1 / transmittances, -reflectances / transmittances],
                [reflectances / transmittances, transmittances - reflectances ** 2 / transmittances],
            ]
        ),
        2,
        0,
    )

    # Build the matrices b similarly, and also 3D, even though each b_n is a column vector
    b = np.array(
        [
            [-(emittances * radiances + reflectances_diffuse * background.radiance) / transmittances],
            [
                (1 - reflectances / transmittances)
                * (emittances * radiances + reflectances_diffuse * background.radiance)
            ],
        ]
    )

    # Get the number of optical elements
    n = len(optics)

    # Calculate D from the A_n matrices in reverse order
    D = np.linalg.multi_dot(A[::-1, :, :])

    # Calculate C using nested list comprehensions.
    # multi_dot needs a list of at least two matrices, so b_ii is appended to the respective list of A_n
    C = sum([(np.linalg.multi_dot([*A[:ii:-1, :, :]] + [b[:, :, ii]])) for ii in range(0, n - 1)]) + b[:, :, n - 1]

    # Create the object property matrix P_T
    P = np.array([1.0 / object_emittance, -object_reflectance / object_emittance])
    
    # Calculate K and l
    K = P @ D
    l = P @ C - object_reflectance_diffuse * background.radiance / object_emittance

    # Return the coefficients - l is a single element array, extract to a scalar.
    return (K[0], l[0])


def atmos_transmittance(length, relative_humidity, temperature):
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
    if not (relative_humidity >= 0 and relative_humidity <= 1):
        raise ValueOutOfRangeError(relative_humidity, "relative_humidity", [0, 1])
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


def temp_to_radiance(T, R, B, F):
    """
    Convert Temperature of an assumed blackbody into a Radiance that would be measured by a specific camera.
    
    Assuming a microbolometer camera, the response curve can be estimated by a function with the same form as
    Planck's Law for blackbody radiation, where instead of universal constants, the calibration values for the 
    specific camera are used.
    
    N. Horny, "FPA Camera Standardisation", https://doi.org/10.1016/S1350-4495(02)00183-4
    
    via
    
    H. Budzier, G. Gerlach, "Calibration of Infrared Cameras with Microbolometers",
    https://doi.org/10.5162/irs2015/1.1

    Parameters
    ----------
    T : float
        Temperature of the blackbody in Kelvin.
    R, B, F : float
        Calibration parameters of the camera.
    
    Returns
    -------
    Radiance: float
        Estimated Radiance in [W m^-2 sr^1 um].

    """

    return R / (np.exp(B / T) - F)


def radiance_to_temp(M, R, B, F):
    """
    For recovering the temperature from a radiance measured by a camera, the formula for temp_to_radiance() is
    simply inverted, and the same calibration parameters used.

    Parameters
    ----------
    M : float
        Measured Radiance in [W m^-2 sr^1 um].
    R, B, F : float
        Calibration parameters of the camera.
    
    Returns
    -------
    Temperature: float
        Temperature of the blackbody in Kelvin.

    """

    return B / np.log(R / M + F)


class Optics:
    """ A base class for optical objects that lie between the camera and the item of interest. """

    def __init__(self):
        self.is_calibrated = False
        return None

    def calibrate(self, CameraCalibration):
        return None


class Atmosphere(Optics):
    """
    A packet of air that exists between an object and a detector. It modifies radiation passing through, and emits its
    own radiation. From the given temperature, length, and relative humidity, further properties are calculated. By
    providing a camera calibration, an effective radiance can be found. Atmospheric pressure is
    assumed.
    
    Attributes
    ----------
    temperature: float
        Temperature in Kelvin
    length: float
        Length/size of the the air packet in metres
    relative_humidity: float
        Relative humidity in interval [0,1]
    transmittance: float
        Estimated transmittance of external radiances, given in the interval [0,1]
    radiance:
        Radiance of a blackbody at the same temperature of this air packet.
    emittance:
        Emittance, form which the net self emission can be calculated.
    reflectance:
        How much radiation can be reflected specularly.
    reflectance_diffuse:
        How much diffuse reflection there is (i.e. from the environment)
    
    """

    # Atmosphere does not reflect.
    reflectance = 0.0
    reflectance_diffuse = 0.0

    def __init__(self, temperature, length, relative_humidity):
        """
        Create a packet of air with the given parameters.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        length : float
            Length/size of the the air packet in metres.
        relative_humidity : float
            Relative humidity in interval [0,1].

        Returns
        -------
        None.

        """

        self.is_calibrated = False

        self.temperature = temperature
        self.length = length
        self.relative_humidity = relative_humidity

        # Use the emperical conversion formulae to find the transmittance of this air packet.
        self.transmittance = atmos_transmittance(self.length, self.relative_humidity, self.temperature,)

        # Check transmittance is a valid value
        if not (self.transmittance >= 0 and self.transmittance <= 1):
            raise ValueOutOfRangeError(self.transmittance, "transmittance", [0, 1])
        # The effective emittance can be found by assuming reflectance + emittance + transmittance = 1
        self.emittance = 1 - self.transmittance

    def calibrate(self, CameraCalibration):
        """
        Apply a camera calibration to the air packet, so that the radiance and self output can be calculated.

        Parameters
        ----------
        CameraCalibration : Dict
            Dict containing the calibration parameters of the camera.

        Returns
        -------
        None.

        """

        # Convert the temperature of the air into a radiance value for this camera
        self.radiance = temp_to_radiance(
            self.temperature, CameraCalibration["R"], CameraCalibration["B"], CameraCalibration["F"]
        )

        self.is_calibrated = True


class Vacuum(Atmosphere):
    """ A packet of 'air' that doesn't contain (much) air. Transmittance is 1 and radiance is 0."""

    relative_humidity = 0.0
    absolute_humidity = 0.0
    transmittance = 1.0
    reflectance = 0.0
    reflectance_diffuse = 0.0
    emittance = 0.0
    radiance = 0.0
    temperature = 0.0
    is_calibrated = True

    def __init__(self, length):

        self.length = length

    def calibrate(self, CameraCalibration):
        return None


class Window(Optics):
    """
    A window that exists between an object and a detector. It modifies radiation passing through, and emits its
    own radiation. From the given temperature, transmittance, and reflectance, further properties are calculated.
    By providing a camera calibration, an effective radiance can be found.
    
    If the 'offaxis' parameter is set, then specular reflections are ignored.
    
    Attributes
    ----------
    temperature: float
        Temperature in Kelvin
    transmittance: float
        Transmittance of external radiances, given in the interval [0,1]
    reflectance: float
        Reflectance of external radiances, given in the interval [0,1]
    radiance:
        Radiance of a blackbody at the same temperature of this window.
    contribution: float
        Net radiance contribution of this window toward the camera
    """

    def __init__(
        self, temperature, transmittance, reflectance, reflectance_diffuse: float = 0.0, offaxis: bool = False
    ):

        self.is_calibrated = False

        # Check transmittance and reflectances are valid values
        if not (0.0 <= transmittance <= 1.0):
            raise ValueOutOfRangeError(transmittance, "transmittance", [0, 1])
        if not (0.0 <= reflectance <= 1.0):
            raise ValueOutOfRangeError(reflectance, "reflectance", [0, 1])
        if not (0.0 <= reflectance_diffuse <= 1.0):
            raise ValueOutOfRangeError(reflectance, "reflectance", [0, 1])
        if not (reflectance_diffuse + reflectance + transmittance <= 1):
            raise ValueOutOfRangeError(
                reflectance_diffuse + reflectance + transmittance, "reflectance + transmittance", [0, 1]
            )
        # Calculate/apply missing properties
        self.temperature = temperature
        self.transmittance = transmittance
        self.reflectance = reflectance
        self.reflectance_diffuse = reflectance_diffuse
        self.emittance = 1 - self.transmittance - self.reflectance - self.reflectance_diffuse
        self.offaxis = offaxis

    def calibrate(self, CameraCalibration):
        """
        Apply a camera calibration to the object, so that the self emitted radiance can be calculated.

        Parameters
        ----------
        CameraCalibration : Dict
            Dict containing the calibration parameters of the camera.

        Returns
        -------
        None.
        
        """

        self.radiance = temp_to_radiance(
            self.temperature, CameraCalibration["R"], CameraCalibration["B"], CameraCalibration["F"]
        )

        self.is_calibrated = True


class Mirror(Window):
    """
    Mirrors are modelled as Windows where the transmittance equals the given reflectance, and the emittance *and* 
    reflectance are one minus the given reflectance, for calculating self and background radiances.
    
    The 'offaxis' parameter is set by default, as reflections that return radiation back towards the source are highly
    unlikely. 
    
    'calibrate' method is inherited
    
    """

    def __init__(self, temperature, reflectance, reflectance_diffuse: float = 0.0, offaxis: bool = True):

        self.is_calibrated = False

        # Throw an error if the reflectance value is silly
        if not (0.0 <= reflectance <= 1.0):
            raise ValueOutOfRangeError(reflectance, "reflectance", [0, 1])
        self.temperature = temperature
        self.transmittance = reflectance
        self.reflectance = 0.0
        self.reflectance_diffuse = reflectance_diffuse
        self.emittance = 1 - reflectance - reflectance_diffuse
        self.offaxis = offaxis


class Background:
    """ An object describing the radiance coming from the background environment """

    def __init__(self, temperature):
        self.is_calibrated = False
        self.temperature = temperature

    def calibrate(self, CameraCalibration):

        self.radiance = temp_to_radiance(
            self.temperature, CameraCalibration["R"], CameraCalibration["B"], CameraCalibration["F"]
        )

        self.is_calibrated = True
