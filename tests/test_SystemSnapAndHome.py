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

        self.names = ["par", "hyp", "ell", "plane"]

        self.fr_names = []
        
        self.s.addParabola(self.validParabola)
        self.s.addHyperbola(self.validHyperbola)
        self.s.addEllipse(self.validEllipse)
        self.s.addPlane(self.validPlane)

        self.groupElements("testgroup", "par", "hyp", "ell", "plane")

    def test_sna

