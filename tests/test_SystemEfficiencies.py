"""!
@file
File containing tests for the PO efficiencies in PyPO.
"""

import unittest

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
    
    def test_spillover(self):
        eta_s = self.s.calcSpillover(TestTemplates.GPOfield["name"], FieldComponents.Ex, TestTemplates.aperDict)
        self.assertTrue(isinstance(eta_s, float))
    
    def test_taper(self):
        eta_t = self.s.calcTaper(TestTemplates.GPOfield["name"], FieldComponents.Ex)
        self.assertTrue(isinstance(eta_t, float))
        
        eta_t = self.s.calcTaper(TestTemplates.GPOfield["name"], FieldComponents.Ex, aperDict=TestTemplates.aperDict)
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

