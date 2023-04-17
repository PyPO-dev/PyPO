import sys
import os
import random
import shutil

import unittest
import numpy as np
from pathlib import Path

from src.PyPO.System import System

class Test_SystemSnapAndHome(unittest.TestCase):
    def setUp(self):
        self.s = System(verbose=False)
        
        self.validParabola = {
            "name"      : "par",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
            }
        
        self.validHyperbola = {           
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
        
        self.validEllipse = {
            "name"      : "ell",
            "pmode"     : "manual",
            "gmode"     : "xy",
            "coeffs"    : np.array([3199.769638 / 2, 3199.769638 / 2, 3689.3421 / 2]),
            "flip"      : False,
            "lims_x"    : np.array([1435, 1545]),
            "lims_y"    : np.array([-200, 200]),
            "gridsize"  : np.array([401, 401])
            }
        
        self.validPlane = {
            "name"      : "plane",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }

        self.validRT_TubeFrame = {
            "name"          : "tubeFrame",
            "nRays"         : 8,
            "nRing"         : 1,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }
        
        self.validRT_GaussFrame = {
            "name"          : 'gaussFrame',
            "nRays"         : 100,
            "n"             : 1,
            "lam"           : 1,
            "x0"            : 5,
            "y0"            : 5,
            "setseed"       : 'set',
            "seed"          : 1,
            }
        self.names = ["par", "hyp", "ell", "plane"]

        self.fr_names = ["tubeFrame", "gaussFrame"]
        
        self.s.addParabola(self.validParabola)
        self.s.addHyperbola(self.validHyperbola)
        self.s.addEllipse(self.validEllipse)
        self.s.addPlane(self.validPlane)

        self.s.createTubeFrame(self.validRT_TubeFrame)
        self.s.createGRTFrame(self.validRT_GaussFrame)

        self.s.groupElements("testgroup", "par", "hyp", "ell", "plane")

    def test_snapObj(self):
        trans = np.random.rand(3) * 100
        rot = np.random.rand(3) * 300

        for name in self.names:
            self.s.translateGrids(name, trans)
            self.s.rotateGrids(name, rot)
            self.s.snapObj(name, "test")

            self.assertFalse(id(self.s.system[name]["snapshots"]["test"]) == id(self.s.system[name]["transf"]))
            for tr, trs in zip(self.s.system[name]["snapshots"]["test"].ravel(), 
                                self.s.system[name]["transf"].ravel()):
                
                self.assertEqual(tr, trs)

        self.s.translateGrids("testgroup", trans, obj="group")
        self.s.rotateGrids("testgroup", rot, obj="group")
        
        self.s.snapObj("testgroup", "test", obj="group")


        for transfs, elem_name in zip(self.s.groups["testgroup"]["snapshots"]["test"], self.names):
            _transf = self.s.system[elem_name]["transf"]
            self.assertFalse(id(transfs) == id(_transf))

            for tr, trs in zip(transfs.ravel(), _transf.ravel()): 
                self.assertEqual(tr, trs)

        for fr_n in self.fr_names:
            self.s.translateGrids(fr_n, trans, obj="frame")
            self.s.rotateGrids(fr_n, rot, obj="frame")
            self.s.snapObj(fr_n, "test", obj="frame")

            self.assertFalse(id(self.s.frames[fr_n].snapshots["test"]) == id(self.s.frames[fr_n].transf))
            for tr, trs in zip(self.s.frames[fr_n].snapshots["test"].ravel(), 
                                self.s.frames[fr_n].transf.ravel()):
                
                self.assertEqual(tr, trs)

        for name in self.names:
            self.s.translateGrids(name, trans)
            self.s.rotateGrids(name, rot)
            self.s.revertToSnap(name, "test")

            for tr, trs in zip(self.s.system[name]["snapshots"]["test"].ravel(), 
                                self.s.system[name]["transf"].ravel()):
                
                self.assertEqual(tr, trs)

            self.s.deleteSnap(name, "test")
            self.assertFalse("test" in self.s.system[name]["snapshots"])

            self.s.homeReflector(name, rot=False)
            for tr, trs in zip([0, 0, 0, 1], self.s.system[name]["transf"][:,-1]):
                self.assertEqual(tr, trs)
            
            self.s.homeReflector(name, trans=False)
            for tr, trs in zip(np.eye(4).ravel(), self.s.system[name]["transf"].ravel()):
                self.assertEqual(tr, trs)

        self.s.translateGrids("testgroup", trans, obj="group")
        self.s.rotateGrids("testgroup", rot, obj="group")
        
        self.s.revertToSnap("testgroup", "test", obj="group")
        for transfs, elem_name in zip(self.s.groups["testgroup"]["snapshots"]["test"], self.names):
            _transf = self.s.system[elem_name]["transf"]
            self.assertFalse(id(transfs) == id(_transf))

            for tr, trs in zip(transfs.ravel(), _transf.ravel()): 
                self.assertEqual(tr, trs)

        self.s.deleteSnap("testgroup", "test", obj="group")
        self.assertFalse("test" in self.s.groups["testgroup"]["snapshots"])
        
        for fr_n in self.fr_names:
            self.s.translateGrids(fr_n, trans, obj="frame")
            self.s.rotateGrids(fr_n, rot, obj="frame")
            self.s.revertToSnap(fr_n, "test", obj="frame")

            self.assertFalse(id(self.s.frames[fr_n].snapshots["test"]) == id(self.s.frames[fr_n].transf))
            for tr, trs in zip(self.s.frames[fr_n].snapshots["test"].ravel(), 
                                self.s.frames[fr_n].transf.ravel()):
                
                self.assertEqual(tr, trs)
            
            self.s.deleteSnap(fr_n, "test", obj="frame")
            self.assertFalse("test" in self.s.frames[fr_n].snapshots)
if __name__ == "__main__":
    unittest.main()
