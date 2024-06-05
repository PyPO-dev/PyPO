"""!
@file
File containing tests for the PO efficiencies in PyPO.
"""

import unittest
from nose2.tools import params

try:
    from . import TestTemplates
except:
    import TestTemplates

from PyPO.System import System
from PyPO.Enums import FieldComponents, CurrentComponents

class Test_SystemEfficiencies(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setLoggingVerbosity(False)
    
    @params(TestTemplates.aperDictEll, TestTemplates.aperDictRect)
    def test_spillover(self, aper):
        eta_s = self.s.calcSpillover(TestTemplates.GPOfield["name"], FieldComponents.Ex, aper)
        self.assertTrue(isinstance(eta_s, float))
    
    @params(TestTemplates.aperDictEll, TestTemplates.aperDictRect)
    def test_taper(self, aper):
        eta_t = self.s.calcTaper(TestTemplates.GPOfield["name"], FieldComponents.Ex)
        self.assertTrue(isinstance(eta_t, float))
        
        eta_t = self.s.calcTaper(TestTemplates.GPOfield["name"], FieldComponents.Ex, aperDict=aper)
        self.assertTrue(isinstance(eta_t, float))

    def test_Xpol(self):
        eta_x = self.s.calcXpol(TestTemplates.GPOfield["name"], FieldComponents.Ex, FieldComponents.Ex)
        self.assertTrue(isinstance(eta_x, float))
    
    def test_mainBeam(self):
        eta_mb = self.s.calcMainBeam(TestTemplates.GPOfield["name"], FieldComponents.Ex)
        self.assertTrue(isinstance(eta_mb, float))

if __name__ == "__main__":
    import nose2
    nose2.main()

