import unittest
import numpy as np
from src.PyPO.System import System as pypoSystem
from src.PyPO.Checks import InputReflError, InputRTError, InputPOError


class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self) -> None:
        self.s = pypoSystem(override=False, verbose=False)

    def test_namingReflector(self):
        validParabola = {
            "name"      : "parabola",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
            }
        validHyperbola = {           
            "name"      : "hyperbola",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "flip"      : True,
            "focus_1"   : np.array([0,0,3.5e3]),
            "focus_2"   : np.array([0,0,3.5e3 - 5606.286]),
            "ecc"       : 1.08208248,
            "lims_u"    : np.array([0,310]),
            "lims_v"    : np.array([0,360]),
            "gridsize"  : np.array([501,1501])
            }
        validEllipse = {
            "name"      : "ellipse",
            "pmode"     : "manual",
            "gmode"     : "xy",
            "coeffs"    : np.array([3199.769638 / 2, 3199.769638 / 2, 3689.3421 / 2]),
            "flip"      : False,
            "lims_x"    : np.array([1435, 1545]),
            "lims_y"    : np.array([-200, 200]),
            "gridsize"  : np.array([401, 401])
            }
        validPlane = {
            "name"      : "plane",
            "gmode"     : "xy",
            "lims_x"    : np.array([-1,1]),
            "lims_y"    : np.array([-1,1]),
            "gridsize"  : np.array([3, 3])
            }
        name = validParabola['name']
        for func, validElem in [
            (self.s.addPlane, validPlane),
            (self.s.addParabola, validParabola),
            (self.s.addHyperbola, validHyperbola),
            (self.s.addEllipse, validEllipse)
            ]:
            
            func(validElem)
            func(validElem)
            func(validElem)
            func(validElem)
            func(validElem)
            
            name = validElem['name']+"_1"
            self.assertTrue(name in self.s.system)
            
            name = validElem['name']+"_2"
            self.assertTrue(name in self.s.system)
            
            name = validElem['name']+"_3"
            self.assertTrue(name in self.s.system)
            
            name = validElem['name']+"_4"
            self.assertTrue(name in self.s.system)

    def test_namingGroup(self):
        self.s.groupElements('g')
        self.s.groupElements('g')
        self.assertTrue("g_1" in self.s.groups)
        self.s.groupElements('g')
        self.assertTrue("g_2" in self.s.groups)

    def test_namingFrames(self):
        tubeFrameDict = {
            "name"          : "tf",
            "nRays"         : 0,
            "nRing"         : 0,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }
        gaussFrame = {
            "name"          : 'gf',
            "nRays"         : 1,
            "n"             : 1,
            "lam"           : 1,
            "x0"            : 5,
            "y0"            : 5,
            "setseed"       : 'set',
            "seed"          : 1,
            }
        self.s.createTubeFrame(tubeFrameDict)
        self.s.createTubeFrame(tubeFrameDict)
        self.assertTrue("tf_1" in self.s.frames)
        self.s.createTubeFrame(tubeFrameDict)
        self.assertTrue("tf_2" in self.s.frames)
        self.s.createTubeFrame(tubeFrameDict)
        self.assertTrue("tf_3" in self.s.frames)

        self.s.createGRTFrame(gaussFrame)
        self.s.createGRTFrame(gaussFrame)
        self.assertTrue("gf_1" in self.s.frames)
        self.s.createGRTFrame(gaussFrame)
        self.assertTrue("gf_2" in self.s.frames)
        self.s.createGRTFrame(gaussFrame)
        self.assertTrue("gf_3" in self.s.frames)


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
            name = validElem['name'] + '_1'
            func(validElem, "plane")
            func(validElem, "plane")
            self.assertTrue(name in self.s.fields)
            self.assertTrue(name in self.s.currents)
            
            func(validElem, "plane")
            name = validElem['name'] + '_2'
            self.assertTrue(name in self.s.fields)
            self.assertTrue(name in self.s.currents)
            
            func(validElem, "plane")
            name = validElem['name'] + '_3'
            self.assertTrue(name in self.s.fields)
            self.assertTrue(name in self.s.currents)
            
        for func, validElem in [
            (self.s.createPointSourceScalar, ScalarPSBeam),
            (self.s.createUniformSourceScalar, ScalarUPSBeam),
            (self.s.createScalarGaussian, ScalarGaussBeam),
            ]:
            name = validElem['name'] + '_1'
            func(validElem, "plane")
            func(validElem, "plane")

            self.assertTrue(name in self.s.scalarfields)
            
            func(validElem, "plane")
            name = validElem['name'] + '_2'
            self.assertTrue(name in self.s.scalarfields)
            
            func(validElem, "plane")
            name = validElem['name'] + '_3'
            self.assertTrue(name in self.s.scalarfields)

if __name__ == "__main__":
    unittest.main()
        
