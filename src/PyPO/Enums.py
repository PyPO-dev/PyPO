"""!
@file
File containing enum types for PyPO methods.
"""

from enum import Enum

import numpy as np

class CustomEnumScalar(Enum):
    def __str__(self):
        return self.name

class CustomEnumString(Enum):
    def __str__(self):
        return self.name
    
    def __rmul__(self, other):
        return NotImplemented
    
    def __rtruediv__(self, other):
        return NotImplemented

class CustomEnumTuple(Enum):
    def __str__(self):
        return self.name
    
    def __rmul__(self, other):
        return other * self.value[0]
    
    def rdiv(self, other):
        return other / self.value[0]

class FieldComponents(CustomEnumScalar):
    """!
    Enum types for electric and magnetic field components.
   
    These are special options that are used whenever a specific component of a field or current distribution is required.
    This could be, for example, for visualisation or an efficiency calculation.
    
    Options:\n
    Ex      : x-component of E-field\n
    Ey      : y-component of E-field\n
    Ez      : z-component of E-field\n
    Hx      : x-component of H-field\n
    Hy      : y-component of H-field\n
    Hz      : z-component of H-field\n
    NONE    : No component. Use for scalarfield objects.
    
    @ingroup public_api_argopts
    """

    Ex = 0
    Ey = 1
    Ez = 2
    Hx = 3
    Hy = 4
    Hz = 5
    NONE = None

class CurrentComponents(CustomEnumScalar):
    """!
    These are special options that are used whenever a specific component of a field or current distribution is required.
    This could be, for example, for visualisation or an efficiency calculation.
    
    Options:\n
    Jx      : x-component of J-current\n
    Jy      : y-component of J-current\n
    Jz      : z-component of J-current\n
    Mx      : x-component of M-current\n
    My      : y-component of M-current\n
    Mz      : z-component of M-current
    
    @ingroup public_api_argopts
    """
    
    Jx = 0
    Jy = 1
    Jz = 2
    Mx = 3
    My = 4
    Mz = 5

class Projections(CustomEnumScalar):
    """!
    Enum types for projections on two axes for plotting purposes.
    
    When an object such as a frame, or a field distribution, needs to be plotted onto a 2D figure, a projection needs to be specified.
    This projection defaults to the x-axis on the abscissa and the y-axis on the ordinate.

    Options:\n
    xy      : Project on x and y-axes\n
    yz      : Project on y and z-axes\n
    zx      : Project on z and x-axes\n
    yx      : Project on y and x-axes\n
    zy      : Project on z and y-axes\n
    xz      : Project on z and x-axes

    @ingroup public_api_argopts
    """
    
    xy = 0
    yz = 1
    zx = 2
    yx = 3
    zy = 4
    xz = 5

class Units(CustomEnumTuple):
    """!
    Enum types for units for display and conversion.
    
    In `PyPO`, the default spatial unit is millimeters and the default angular unit is degrees.
    For plotting purposes, it is possible to scale the units on the axes. 
    These special options can be used to specify units on plotting axes.

    Options:\n
    M       : Set unit to meters\n
    CM      : Set unit to centimeters\n
    MM      : Set unit to millimeters (default unit)\n
    UM      : Set unit to micrometers\n
    NM      : Set unit to nanometers\n
    IN      : Set unit to inches\n
    MIL/THOU: Set unit to thousandths of an inch\n
    UIN     : Set unit to microinches\n
    FT      : Set unit to feet\n
    DEG     : Set unit to degrees. Use for far-fields only\n
    AM      : Set unit to arcminutes. Use for far-fields only\n
    AS      : Set unit to arcseconds. Use for far-fields only\n
    RAD     : Set unit to radians. Use for far-fields only\n
    MRAD    : Set unit to milliradians. Use for far-fields only\n
    URAD    : Set unit to microradians. Use for far-fields only\n

    @ingroup public_api_argopts
    """
    
    M = (1e3, "spatial")
    CM = (1e-2*M[0], "spatial")
    MM = (1e-3*M[0], "spatial")
    UM = (1e-6*M[0], "spatial")
    NM = (1e-9*M[0], "spatial")
    IN = (25.4*MM[0], "spatial")
    MIL = (IN[0]*1e-3, "spatial")
    THOU = (IN[0]*1e-3, "spatial")
    UIN = (IN[0]*1e-6, "spatial")
    FT = (12*IN[0], "spatial")
    DEG = (1., "angular")
    AM = (60., "angular")
    AS = (3600., "angular")
    RAD = (180/np.pi*DEG[0], "angular")
    MRAD = (RAD[0]*1e-3, "angular")
    URAD = (RAD[0]*1e-6, "angular")

class Scales(CustomEnumScalar):
    """!
    Enum types for units for setting scalings for quantities.
    
    This special option is used for specifying the scaling of the heatmap for 2D plots, but also for the scaling of beam cross-sections.
    In addition, the scaling argument sets the scale for fitting Gaussians.
    For example, using the scale argument, it is possible to fit a Gaussian to a beam pattern in linear, logarithmic and decibel space.

    Options:\n
    LIN       : Set scaling mode to linear\n
    dB        : Set scaling to decibels

    @ingroup public_api_argopts
    """

    LIN = 0
    dB = 2

class Objects(CustomEnumString):
    """!
    Enum types for objects such as elements, groups, frames, etc.

    Options:\n
    ELEMENT   : Choose element\n
    GROUP     : Choose a group\n
    FRAME     : Choose frame

    @ingroup public_api_argopts
    """

    ELEMENT = "elements"
    GROUP = "groups"
    FRAME = "frames"
    FIELD = "fields"
    CURRENT = "currents"

class Modes(CustomEnumScalar):
    """!
    Enum types for transformation modes, absolute or relative.

    Options:\n
    REL       : Relative translation/rotation\n
    ABS       : Absolute translation/rotation\n

    @ingroup public_api_argopts
    """

    REL = 0
    ABS = 1

class AperShapes(CustomEnumScalar):
    """!
    Enum types for aperture object shapes.

    The apertures used for efficiency calculations and plotting can be created in elliptical and rectangular shapes.

    Options:\n
    ELL       : Use elliptical aperture\n
    RECT      : Use rectangular aperture

    @ingroup public_api_argopts
    """

    ELL = 0
    RECT = 1

class OptDoF(CustomEnumScalar):
    """!
    Enum types for selecting degrees-of-freedom (DoF) for optimisation of an element or group.

    Options:\n
    TX      : Include x-translations\n
    TY      : Include y-translations\n
    TZ      : Include z-translations\n
    RX      : Include x-rotations\n
    RY      : Include y-rotations\n
    RZ      : Include z-rotations
    
    @ingroup public_api_argopts
    """

    TX = 0
    TY = 1
    TZ = 2
    RX = 3
    RY = 4
    RZ = 5

class Dielectrics(CustomEnumScalar):
    """!
    Enum types for commonly used dielectrics for lens materials.
    
    """

    BK7 = 1
    FS = 2
    K5 = 3
