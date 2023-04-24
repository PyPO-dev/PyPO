import sys
import os
import random
import shutil

import unittest
import numpy as np
from pathlib import Path

from src.PyPO.System import System

##
# @file
# File containing tests for the PO efficiencies in PyPO.
class Test_SystemEfficiencies(unittest.TestCase):
    def test_spilloverGauss(self):
        for i in range(100):
            aperDict = self._makeRandomGauss()

            #self.s.plotBeam2D("Gauss", "Ex", aperDict=aperDict, vmin=-30)

            eta_s = self.s.calcSpillover("Gauss", "Ex", aperDict)
            self.assertAlmostEqual(eta_s, 1 - np.exp(-2), delta=1e-3)


    def _makeRandomGauss(self):
        self.s = System(verbose=False)
        w0x = random.uniform(1, 10)
        w0y = w0x
        lam = random.uniform(1, 10)
        
        plane_Gauss = {
                "name"      : "plane_Gauss",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, w0x * 3]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([1001, 1001])
                }

        GDict = {
                "name"      : "Gauss",
                "lam"       : lam,
                "w0x"       : w0x,
                "w0y"       : w0y,
                "n"         : 1
                }

        aperDict = {
                "plot"      : True,
                "center"    : np.array([0, 0]),
                "outer"     : np.array([w0x, w0y]),
                "inner"     : np.array([0, 0])
                }
        
        self.s.addPlane(plane_Gauss)
        self.s.createGaussian(GDict, "plane_Gauss")

        return aperDict

if __name__ == "__main__":
    unittest.main()


