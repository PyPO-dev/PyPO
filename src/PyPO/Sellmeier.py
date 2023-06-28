"""!
@file
File containing classes representing dielectric materials commonly used in lenses.
Each class holds the Sellmeier coefficients in order to calculate the wavelength-dependent refractive index for a lens.
"""

from dataclasses import dataclass
import numpy as np

def Sellmeier(A1, A2, A3, B1, B2, B3, lam):
    n = np.sqrt((A1/(lam**2 - B1) + A2/(lam**2 - B2) + A3/(lam**2 - B3)) * lam**2 + 1)
    return n

class BK7:
    """!
    Class for representing borosilicate crown glass (BK7).
    """

    A1 = 1.03961212
    A2 = 0.231792344
    A3 = 1.01046945

    B1 = 0.00600069867
    B2 = 0.0200179144
    B3 = 103.560653

    def __init__(self, lam):
        """!
        Constructor. Sets the wavelength at which refractive index will be calculated.
        Note that the wavelength has to be given in millimeters, otherwise PyPO will not return the correct refractive index.
        """

        self.lam_um = lam * 1e3
        self.n = Sellmeier(self.A1, self.A2, self.A3,
                self.B1, self.B2, self.B3, self.lam_um)
    
class FS:
    """!
    Class for representing fused silica glass (FS).
    """

    A1 = 0.6961663
    A2 = 0.4079426
    A3 = 0.8974794

    B1 = 0.0684043**2
    B2 = 0.1162414**2
    B3 = 9.896161**2

    def __init__(self, lam):
        """!
        Constructor. Sets the wavelength at which refractive index will be calculated.
        Note that the wavelength has to be given in millimeters, otherwise PyPO will not return the correct refractive index.
        """

        self.lam_um = lam * 1e3
        self.n = Sellmeier(self.A1, self.A2, self.A3,
                self.B1, self.B2, self.B3, self.lam_um)


