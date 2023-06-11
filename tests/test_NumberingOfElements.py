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
        
        for i in range(2):
            self.s.createGRTFrame(TestTemplates.GaussRTframe)
            self.assertTrue(TestTemplates.GaussRTframe["name"] + f"_{i+1}" in self.s.frames)

    def test_addPOFields(self):
        validPlane = {
            "name"      : "plane",
            "gmode"     : "xy",
            "lims_x"    : np.array([-1,1]),
            "lims_y"    : np.array([-1,1]),
            "gridsize"  : np.array([3, 3])
            }
        PSBeam = {
            "name"          : 'PSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            }
        ScalarPSBeam = {
            "name"          : 'SPSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        UPSBeam = {
            "name"          : 'UPSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        ScalarUPSBeam = {
            "name"          : 'SUPSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        GaussBeam = {
            "name"          : 'GaussBeam',
            "lam"           : 1,
            "w0x"           : 1,
            "w0y"           : 1,
            "n"             : 1,
            "E0"            : 1,
            "phase"         : 1,
            "pol"           : np.array([0,0,1]),
            }
        ScalarGaussBeam = {
            "name"          : 'SGaussBeam',
            "lam"           : 1,
            "w0x"           : 1,
            "w0y"           : 1,
            "n"             : 1,
            "E0"            : 1,
            "phase"         : 1,
            "pol"           : np.array([0,0,1]),
            }
        self.s.addPlane(validPlane)
        for func, validElem in [
            (self.s.createPointSource, PSBeam), 
            (self.s.createUniformSource, UPSBeam), 
            (self.s.createGaussian, GaussBeam),
            ]:
            names = [validElem['name'] + '_1', validElem['name'] + '_2']
            func(validElem, "plane")
            func(validElem, "plane")
            func(validElem, "plane")

            for n in names:
                self.assertTrue(n in self.s.fields)
                self.assertTrue(n in self.s.currents)
            
            
        for func, validElem in [
            (self.s.createPointSourceScalar, ScalarPSBeam),
            (self.s.createUniformSourceScalar, ScalarUPSBeam),
            (self.s.createScalarGaussian, ScalarGaussBeam),
            ]:
            names = [validElem['name'] + '_1', validElem['name'] + '_2']
            func(validElem, "plane")
            func(validElem, "plane")
            func(validElem, "plane")

            for n in names:
                self.assertTrue(n in self.s.scalarfields)
            
            

if __name__ == "__main__":
    unittest.main()
        
