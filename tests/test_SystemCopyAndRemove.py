import unittest
import numpy as np
from src.PyPO.System import System as PyPOSystem
from src.PyPO.Checks import InputReflError, InputRTError, InputPOError

##
# @file
# File containing tests for copy and remove methods for reflectors, fields, currents, frames and scalar fields.
class Test_SystemCopyAndRemove(unittest.TestCase):
    def setUp(self) -> None:
        self.s = PyPOSystem(override=False, verbose=False)
        
        Parabola = {
            "name"      : "par",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
            }

        Hyperbola = {           
            "name"      : "hyp",
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

        Ellipse = {
            "name"      : "ell",
            "pmode"     : "manual",
            "gmode"     : "xy",
            "coeffs"    : np.array([3199.769638 / 2, 3199.769638 / 2, 3689.3421 / 2]),
            "flip"      : False,
            "lims_x"    : np.array([1435, 1545]),
            "lims_y"    : np.array([-200, 200]),
            "gridsize"  : np.array([401, 401])
            }

        Plane = {
            "name"      : "plane",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }

        self.s.addParabola(Parabola)
        self.s.addHyperbola(Hyperbola)
        self.s.addEllipse(Ellipse)
        self.s.addPlane(Plane)

        RTDict = {
            "name"          : "frame",
            "nRays"         : 0,
            "nRing"         : 0,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }

        self.s.createTubeFrame(RTDict)
        
        PODict = {
            "name"          : 'beam',
            "lam"           : 1,
            "E0"            : 1,
            "phase"         : 0,
            "pol"           : np.array([1,0,0]),
            }

        self.s.createPointSource(PODict, "plane")
        self.s.createPointSourceScalar(PODict, "plane")

    def test_copyElement(self):
        self.s.copyElement("par", "par_copy")
        self.s.copyElement("hyp", "hyp_copy")
        self.s.copyElement("ell", "ell_copy")
        self.s.copyElement("plane", "plane_copy")

        self.assertTrue("par_copy" in self.s.system)

        for (key1, item1), (key2, item2) in zip(self.s.system["par"].items(), self.s.system["par_copy"].items()):
            self.assertEqual(key1, key2)

            if isinstance(item1, int) or isinstance(item1, float) or isinstance(item1, bool) or isinstance(item1, str):
                self.assertEqual(id(item1), id(item2))
            
            else:
                self.assertNotEqual(id(item1), id(item2))

        self.assertTrue("hyp_copy" in self.s.system)

        for (key1, item1), (key2, item2) in zip(self.s.system["hyp"].items(), self.s.system["hyp_copy"].items()):
            self.assertEqual(key1, key2)

            if isinstance(item1, int) or isinstance(item1, float) or isinstance(item1, bool) or isinstance(item1, str):
                self.assertEqual(id(item1), id(item2))
            
            else:
                self.assertNotEqual(id(item1), id(item2))

        self.assertTrue("ell_copy" in self.s.system)

        for (key1, item1), (key2, item2) in zip(self.s.system["ell"].items(), self.s.system["ell_copy"].items()):
            self.assertEqual(key1, key2)

            if isinstance(item1, int) or isinstance(item1, float) or isinstance(item1, bool) or isinstance(item1, str):
                self.assertEqual(id(item1), id(item2))
            
            else:
                self.assertNotEqual(id(item1), id(item2))

        self.assertTrue("plane_copy" in self.s.system)

        for (key1, item1), (key2, item2) in zip(self.s.system["plane"].items(), self.s.system["plane_copy"].items()):
            self.assertEqual(key1, key2)

            if isinstance(item1, int) or isinstance(item1, float) or isinstance(item1, bool) or isinstance(item1, str):
                self.assertEqual(id(item1), id(item2))
            
            else:
                self.assertNotEqual(id(item1), id(item2))


    def test_removeElement(self):
        self.s.groupElements("testgroup", "par", "ell")
        self.s.removeElement("par")
        
        self.assertFalse("par" in self.s.groups["testgroup"]["members"])

        self.s.removeElement("hyp", "ell", "plane")

        self.assertFalse("par" in self.s.system)
        self.assertFalse("hyp" in self.s.system)
        self.assertFalse("ell" in self.s.system)
        self.assertFalse("plane" in self.s.system)

    def test_copyGroup(self):
        self.s.groupElements("testgroup", "par", "ell")
        self.s.copyGroup("testgroup", "testgroup_copy")
        
        self.assertTrue("testgroup_copy" in self.s.groups)

        for (key1, item1), (key2, item2) in zip(self.s.groups["testgroup"].items(), self.s.groups["testgroup_copy"].items()):
            self.assertEqual(key1, key2)

            if isinstance(item1, int) or isinstance(item1, float) or isinstance(item1, bool) or isinstance(item1, str):
                self.assertEqual(id(item1), id(item2))
            
            else:
                self.assertNotEqual(id(item1), id(item2))

    def test_removeFrame(self):
        self.s.removeFrame("frame")
        self.assertFalse("frame" in self.s.frames)
    
    def test_removeField(self):
        self.s.removeField("beam")
        self.assertFalse("beam" in self.s.fields)
    
    def test_removeCurrent(self):
        self.s.removeCurrent("beam")
        self.assertFalse("beam" in self.s.currents)

    def test_removeScalarField(self):
        self.s.removeScalarField("beam")
        self.assertFalse("beam" in self.s.scalarfields)
    
if __name__ == "__main__":
    unittest.main()
        


