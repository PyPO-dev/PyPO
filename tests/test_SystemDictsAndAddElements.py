import unittest
from parameterized import parameterized
import numpy as np
from src.PyPO.System import System as pypoSystem
from src.PyPO.Checks import InputReflError, InputRTError


class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self) -> None:
        self.s = pypoSystem()
        self.dictNames = ["system", "groups", "frames", "currents", "fields", "scalarfields"]
    
        self.invalidDict = {'name' : 'someElement'}
        self.validParabola = self.s.addParabola, {
            "name"      : "p1",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
            }
        self.validHyperbola = self.s.addHyperbola, {           
            "name"      : "sec",
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
        self.validEllipse = self.s.addEllipse, {
            "name"      : "e_wo",
            "pmode"     : "manual",
            "gmode"     : "xy",
            "coeffs"    : np.array([3199.769638 / 2, 3199.769638 / 2, 3689.3421 / 2]),
            "flip"      : False,
            "lims_x"    : np.array([1435, 1545]),
            "lims_y"    : np.array([-200, 200]),
            "gridsize"  : np.array([401, 401])
            }
        self.validPlane = self.s.addPlane, {
            "name"      : "plane1",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }
        self.validRT_TubeFrame = self.s.createTubeFrame, {
            "name"          : "tubeFrame",
            "nRays"         : 8,
            "nRing"         : 1,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }
        self.validRT_GaussFrame = self.s.createGRTFrame, {
            "name"          : 'gaussFrame',
            "nRays"         : 100,
            "n"             : 1,
            "lam"           : 1,
            "x0"            : 5,
            "y0"            : 5,
            "setseed"       : 'set',
            "seed"          : 1,
            }
        
    def test_dictsExist(self):
        for i in self.dictNames:
            self.assertIsNotNone(self.s.__getattribute__(i))


    def test_addReflector(self):
        sysLen = 0
        for func, validElem in [self.validPlane, self.validParabola, self.validHyperbola, self.validEllipse]:
            with self.assertRaises(InputReflError):
                func(self.invalidDict)
            self.assertEquals(len(self.s.system), sysLen)
            func(validElem)
            self.assertEquals(len(self.s.system), sysLen+1)
            sysLen += 1

    def test_addRTFrame(self):
        frameLen = 0
        for func, validElem in [self.validRT_TubeFrame, self.validRT_GaussFrame]:
            print("Gonna test",func, validElem)
            with self.assertRaises(InputRTError):
                func(self.invalidDict)
            self.assertEquals(len(self.s.frames), frameLen)
            func(validElem)
            self.assertEquals(len(self.s.frames), frameLen+1)
            frameLen += 1







if __name__ == "__main__":
    unittest.main()
        