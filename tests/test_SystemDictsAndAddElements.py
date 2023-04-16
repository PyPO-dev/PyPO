import unittest
import numpy as np
from src.PyPO.System import System as pypoSystem
from src.PyPO.Checks import InputReflError, InputRTError, InputPOError


class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self) -> None:
        self.s = pypoSystem()
        self.dictNames = ["system", "groups", "frames", "currents", "fields", "scalarfields"]

        ### Define tuples of system functions and their input dictionary
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
        self.validRT_TubeFrameTuple = self.s.createTubeFrame, {
            "name"          : "tubeFrame",
            "nRays"         : 0,
            "nRing"         : 0,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }
        self.validRT_GaussFrameTuple = self.s.createGRTFrame, {
            "name"          : 'gaussFrame',
            "nRays"         : 1,
            "n"             : 1,
            "lam"           : 1,
            "x0"            : 5,
            "y0"            : 5,
            "setseed"       : 'set',
            "seed"          : 1,
            }
        self.validPO_PSBeamTuple = self.s.createPointSource, {
            "name"          : 'PSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        self.validPO_GaussBeamTuple = self.s.createGaussian, {
            "name"          : 'GaussBeam',
            "lam"           : 1,
            "w0x"           : 1,
            "w0y"           : 1,
            "n"             : 1,
            "E0"            : 1,
            "phase"         : 1,
            "pol"           : np.array([0,0,1]),
            }
        
        self.validPO_ScalarPSBeamTuple = self.s.createPointSourceScalar, {
            "name"          : 'SPSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        self.validPO_ScalarGaussBeamTuple = self.s.createScalarGaussian, {
            "name"          : 'SGaussBeam',
            "lam"           : 1,
            "w0x"           : 1,
            "w0y"           : 1,
            "n"             : 1,
            "E0"            : 1,
            "phase"         : 1,
            "pol"           : np.array([0,0,1]),
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
        for func, validElem in [self.validRT_TubeFrameTuple, self.validRT_GaussFrameTuple]:
            with self.assertRaises(InputRTError):
                func(self.invalidDict)
            self.assertEquals(len(self.s.frames), frameLen)
            func(validElem)
            self.assertEquals(len(self.s.frames), frameLen+1)
            frameLen += 1

    def test_addPOFields(self):
        self.s.addPlane(self.validPlane[1])
        fieldsLen = 0
        for func, validElem in [self.validPO_PSBeamTuple, self.validPO_GaussBeamTuple]:
            # print("Gonna test",func, validElem)
            with self.assertRaises(InputPOError):
                func(self.invalidDict, 'plane1')
            self.assertEquals(len(self.s.fields), fieldsLen)
            func(validElem, 'plane1')
            self.assertEquals(len(self.s.fields), fieldsLen+1)
            fieldsLen += 1


    def test_addPOScalarFields(self):
        self.s.addPlane(self.validPlane[1])
        fieldsLen = 0
        for func, validElem in [ self.validPO_ScalarGaussBeamTuple]:
            print("Gonna test",func, validElem)
            # with self.assertRaises(InputPOError):
            #     func(self.invalidDict, 'plane1')
            self.assertEquals(len(self.s.scalarfields), fieldsLen)
            func(validElem, 'plane1')
            self.assertEquals(len(self.s.scalarfields), fieldsLen+1)
            fieldsLen += 1

    







if __name__ == "__main__":
    unittest.main()
        
