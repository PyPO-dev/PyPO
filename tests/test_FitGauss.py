"""!
@file
File containing tests for Gaussian fitting in PyPO.
"""

import unittest
from nose2.tools import params

from PyPO.System import System
from PyPO.FitGauss import fitGaussAbs, generateGauss
from PyPO.Enums import FieldComponents, CurrentComponents, Modes

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

class Test_FitGauss(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setOverride(False) 
    
    @params(Modes.dB, Modes.LIN)
    def test_fitGauss(self, mode):
            popt = self.s.fitGaussAbs(TestTemplates.GPOfield["name"], FieldComponents.Ex, thres=-100, mode=mode, full_output=True, ratio=None)
            self.assertTrue(len(popt) == 6)
            self.assertTrue(f"fitGauss_{TestTemplates.GPOfield['name']}" in self.s.scalarfields)

            popt = self.s.fitGaussAbs(TestTemplates.GPOfield["name"], FieldComponents.Ex, thres=-100, mode=mode, full_output=True, ratio=1)
            self.assertTrue(len(popt) == 6)
            self.assertTrue(f"fitGauss_{TestTemplates.GPOfield['name']}" in self.s.scalarfields)
           
    def test_calcHPBW(self):
        E, H = self.s.calcHPBW(TestTemplates.GPOfield["name"], FieldComponents.Ex)
        self.assertTrue(isinstance(E, float))
        self.assertTrue(isinstance(H, float))

if __name__ == "__main__":
    import nose2
    nose2.main()
