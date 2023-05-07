import unittest
import numpy as np
from src.PyPO.System import System as pypoSystem
from src.PyPO.Checks import InputReflError, InputRTError, InputPOError


class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self) -> None:
        self.s = pypoSystem(override=False)
        self.s.setLoggingVerbosity(False)

        self.invalidDict = {'name' : 'someElement'}
        
    def test_dictsExist(self):
        for i in ["system", "groups", "frames", "currents", "fields", "scalarfields"]:
            self.assertIsNotNone(self.s.__getattribute__(i))

    def test_addReflector(self):
        validParabola = {
            "name"      : "p1",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
            }
        validHyperbola = {           
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
        validEllipse = {
            "name"      : "e_wo",
            "pmode"     : "manual",
            "gmode"     : "xy",
            "coeffs"    : np.array([3199.769638 / 2, 3199.769638 / 2, 3689.3421 / 2]),
            "flip"      : False,
            "lims_x"    : np.array([1435, 1545]),
            "lims_y"    : np.array([-200, 200]),
            "gridsize"  : np.array([401, 401])
            }
        validPlane = {
            "name"      : "plane1",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }
        sysLen = 0
        for func, validElem in [
            (self.s.addPlane, validPlane),
            (self.s.addParabola, validParabola),
            (self.s.addHyperbola, validHyperbola),
            (self.s.addEllipse, validEllipse)
            ]:
            ##Fails
            with self.assertRaises(InputReflError):
                func(self.invalidDict)
            self.assertEqual(len(self.s.system), sysLen)
            ##Succeeds
            func(validElem)
            self.assertEqual(len(self.s.system), sysLen+1)
            sysLen += 1

    def test_addGroup(self):
        validPlane = {
            "name"      : "r1",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }
        validParabola = {
            "name"      : "r2",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
            }
        validHyperbola = {           
            "name"      : "r3",
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
            "name"      : "r4",
            "pmode"     : "manual",
            "gmode"     : "xy",
            "coeffs"    : np.array([3199.769638 / 2, 3199.769638 / 2, 3689.3421 / 2]),
            "flip"      : False,
            "lims_x"    : np.array([1435, 1545]),
            "lims_y"    : np.array([-200, 200]),
            "gridsize"  : np.array([401, 401])
            }
        
        self.s.addPlane(validPlane)
        self.s.addParabola(validParabola)
        self.s.addHyperbola(validHyperbola)
        self.s.addEllipse(validEllipse)

        sysLen = 0
        ##Fails
        with self.assertRaises(TypeError):
            self.s.groupElements()
        self.assertEqual(len(self.s.groups), sysLen)
        ##Succeeds
        self.s.groupElements('g', "r1", "r2", "r3", "r4")
        self.assertEqual(len(self.s.groups), sysLen+1)
        sysLen += 1

    def test_addRTFrame(self):
        validRT_TubeFrameTuple = {
            "name"          : "tubeFrame",
            "nRays"         : 0,
            "nRing"         : 0,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }
        validRT_GaussFrameTuple = {
            "name"          : 'gaussFrame',
            "nRays"         : 1,
            "n"             : 1,
            "lam"           : 1,
            "x0"            : 5,
            "y0"            : 5,
            "setseed"       : 'set',
            "seed"          : 1,
            }
        frameLen = 0
        for func, validElem in [
            (self.s.createTubeFrame, validRT_TubeFrameTuple), 
            (self.s.createGRTFrame, validRT_GaussFrameTuple)
            ]:
            ##Fails
            with self.assertRaises(InputRTError):
                func(self.invalidDict)
            self.assertEqual(len(self.s.frames), frameLen)
            ##Succeeds
            func(validElem)
            self.assertEqual(len(self.s.frames), frameLen+1)
            frameLen += 1

    def test_addPOFields(self):
        validPlane = {
            "name"      : "plane1",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }
        validPO_PSBeamTuple = {
            "name"          : 'PSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        validPO_USBeamTuple = {
            "name"          : 'USBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        validPO_GaussBeamTuple = {
            "name"          : 'GaussBeam',
            "lam"           : 1,
            "w0x"           : 1,
            "w0y"           : 1,
            "n"             : 1,
            "E0"            : 1,
            "phase"         : 1,
            "pol"           : np.array([0,0,1]),
            }
        self.s.addPlane(validPlane)
        fieldsLen = 0
        for func, validElem in [
            (self.s.createPointSource, validPO_PSBeamTuple), 
            (self.s.createUniformSource, validPO_USBeamTuple), 
            (self.s.createGaussian, validPO_GaussBeamTuple)
            ]:
            with self.assertRaises(InputPOError):
                func(self.invalidDict, 'plane1')
            self.assertEqual(len(self.s.fields), fieldsLen)
            self.assertEqual(len(self.s.currents), fieldsLen)
            func(validElem, 'plane1')
            self.assertEqual(len(self.s.fields), fieldsLen+1)
            self.assertEqual(len(self.s.currents), fieldsLen+1)
            fieldsLen += 1

    def test_addPOScalarFields(self):
        validPlane = {
            "name"      : "plane1",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }
        validPO_ScalarPSBeamTuple = {
            "name"          : 'SPSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        validPO_ScalarUSBeamTuple = {
            "name"          : 'SUSBeam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([0,0,1]),
            }
        validPO_ScalarGaussBeamTuple = {
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
        fieldsLen = 0
        for func, validElem in [
            (self.s.createPointSourceScalar, validPO_ScalarPSBeamTuple),
            (self.s.createUniformSourceScalar, validPO_ScalarUSBeamTuple),
            (self.s.createScalarGaussian, validPO_ScalarGaussBeamTuple)
            ]: ##TODO: uncomment once segmentation fault is resolved
            ##Fails
            with self.assertRaises(InputPOError):
                func(self.invalidDict, 'plane1')
            self.assertEqual(len(self.s.scalarfields), fieldsLen)
            ##Succeeds
            func(validElem, 'plane1')
            self.assertEqual(len(self.s.scalarfields), fieldsLen+1)
            fieldsLen += 1

    
if __name__ == "__main__":
    unittest.main()
        
