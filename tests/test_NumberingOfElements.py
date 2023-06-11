import unittest
import numpy as np
from PyPO.System import System
from PyPO.Checks import InputReflError, InputRTError, InputPOError

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setOverride(False) 

    def test_namingReflector(self):
        for plane in TestTemplates.getPlaneList():
            for i in range(2):
                self.s.addPlane(plane)
                self.assertTrue(plane['name'] + f"_{i+1}" in self.s.system)
        
        for parabola in TestTemplates.getParaboloidList():
            for i in range(2):
                self.s.addParabola(parabola)
                self.assertTrue(parabola['name'] + f"_{i+1}" in self.s.system)
        
        for hyperbola in TestTemplates.getHyperboloidList():
            for i in range(2):
                self.s.addHyperbola(hyperbola)
                self.assertTrue(hyperbola['name'] + f"_{i+1}" in self.s.system)
        
        for ellipse in TestTemplates.getEllipsoidList():
            for i in range(2):
                self.s.addEllipse(ellipse)
                self.assertTrue(ellipse['name'] + f"_{i+1}" in self.s.system)


    def test_namingGroup(self):
        self.s.groupElements('g')
        self.s.groupElements('g')
        self.s.groupElements('g')
        self.assertTrue("g_1" in self.s.groups)
        self.assertTrue("g_2" in self.s.groups)

    def test_namingFrames(self):
        for i in range(2):
            self.s.createTubeFrame(TestTemplates.TubeRTframe)
            self.assertTrue(TestTemplates.TubeRTframe["name"] + f"_{i+1}" in self.s.frames)
        
            self.s.createGRTFrame(TestTemplates.GaussRTframe)
            self.assertTrue(TestTemplates.GaussRTframe["name"] + f"_{i+1}" in self.s.frames)

    def test_addPOFields(self):
        for plane in TestTemplates.getPlaneList():
            if plane["gmode"] == "AoE":
                break
        
        for i in range(2):
            self.s.createPointSource(TestTemplates.PS_Ufield, plane["name"])
            self.assertTrue(TestTemplates.PS_Ufield["name"] + f"_{i+1}" in self.s.fields)
            self.assertTrue(TestTemplates.PS_Ufield["name"] + f"_{i+1}" in self.s.currents)
            
            self.s.createGaussian(TestTemplates.GPOfield, plane["name"])
            self.assertTrue(TestTemplates.GPOfield["name"] + f"_{i+1}" in self.s.fields)
            self.assertTrue(TestTemplates.GPOfield["name"] + f"_{i+1}" in self.s.currents)
        
            self.s.createPointSourceScalar(TestTemplates.PS_Ufield, plane["name"])
            self.assertTrue(TestTemplates.PS_Ufield["name"] + f"_{i+1}" in self.s.scalarfields)
            
            self.s.createScalarGaussian(TestTemplates.GPOfield, plane["name"])
            self.assertTrue(TestTemplates.GPOfield["name"] + f"_{i+1}" in self.s.scalarfields)

if __name__ == "__main__":
    unittest.main()
        
