import sys
import os
import random
import shutil
import unittest
import numpy as np
from pathlib import Path

from PyPO.System import System
from PyPO.FitGauss import fitGaussAbs, generateGauss

##
# @file
# File containing tests for Gaussian fitting in PyPO.
class Test_FitGauss(unittest.TestCase):
    def test_fitGauss(self):
        for i in range(10):
            w0x, w0y, rot, tx, ty = self._makeRandomGauss()
            popt, perr = self.s.fitGaussAbs("Gauss", "Ex", thres=-11, mode="linear", full_output=True, ratio=None)

            total_field = np.sqrt(np.absolute(self.s.fields["Gauss"].Ey)**2 + np.absolute(self.s.fields["Gauss"].Ex)**2)

            for fg, cg in zip(np.absolute(self.s.scalarfields["fitGauss_Gauss"].S).ravel(), total_field.ravel()): 
                self.assertAlmostEqual(fg, cg, delta=1e-3)

            del self.s

    def _makeRandomGauss(self):
        self.s = System(verbose=False)
        w0x = random.uniform(5, 10)
        w0y = random.uniform(5, w0x)
        
        trans_x = random.uniform(-1, 1)
        trans_y = random.uniform(-1, 1)
       
        plane_Gauss = {
                "name"      : "plane_Gauss",
                "gmode"     : "xy",
                "lims_x"    : np.array([-1, 1]) * w0x * 2,
                "lims_y"    : np.array([-1, 1]) * w0y * 2,
                "gridsize"  : np.array([101, 101])
                }

        GDict = {
                "name"      : "Gauss",
                "lam"       : 1,
                "w0x"       : w0x,
                "w0y"       : w0y,
                "E0"        : 1,
                "n"         : 1
                }
        
        self.s.addPlane(plane_Gauss)
        
        rot = random.uniform(-90, 90)

        self.s.createGaussian(GDict, "plane_Gauss")

        return w0x, w0y, rot, trans_x, trans_y

if __name__ == "__main__":
    unittest.main()
