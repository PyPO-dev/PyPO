import sys
import os
import random
import shutil

import unittest
import numpy as np
from pathlib import Path

from PyPO.System import System
from PyPO.Efficiencies import _generateMask

##
# @file
# File containing tests for the PO efficiencies in PyPO.
class Test_SystemEfficiencies(unittest.TestCase):
    def test_spilloverGauss(self):
        for i in range(10):
            aperDict = self._makeRandomGauss()

            eta_s = self.s.calcSpillover("Gauss", "Ex", aperDict)
            self.assertAlmostEqual(eta_s, 1 - np.exp(-2), delta=1e-3)
    
    def test_spilloverUniform(self):
        for i in range(10): 
            aperDict = {
                    "center"    : np.array([random.uniform(0, 10), random.uniform(0, 10)]),
                    "outer"     : np.array([random.uniform(5, 10), random.uniform(5, 10)]),
                    "inner"     : np.array([random.uniform(0, 5), random.uniform(0, 5)])
                    }
            self._makeRandomUniform_xy()

            grid_pl = self.s.generateGrids("plane")
            mask = _generateMask(grid_pl.x, grid_pl.y, aperDict)

            num_tot = self.s.fields["Uniform"].size
            num_mask = len(self.s.fields["Uniform"].Ex[mask])
            
            eta_s = self.s.calcSpillover("Uniform", "Ex", aperDict)

            self.assertAlmostEqual(eta_s, num_mask / num_tot)

    def test_taperUniform(self):
        for i in range(10):
            self._makeRandomUniform_xy()
            eta_t = self.s.calcTaper("Uniform", "Ex")
            self.assertAlmostEqual(eta_t, 1., delta=1e-3)
            
            self._makeRandomUniform_uv()
            eta_t = self.s.calcTaper("Uniform", "Ex")
            self.assertAlmostEqual(eta_t, 1., delta=1e-3)
    
    def test_XpolUniform(self):
        for i in range(10):
            self._makeRandomUniform_xy()
            eta_x = self.s.calcXpol("Uniform", "Ex", "Ex")
            self.assertAlmostEqual(eta_x, .5, delta=1e-3)
            
            self._makeRandomUniform_uv()
            eta_x = self.s.calcXpol("Uniform", "Ex", "Ex")
            self.assertAlmostEqual(eta_x, .5, delta=1e-3)

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
                "gridsize"  : np.array([501, 501])
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
    
    def _makeRandomUniform_xy(self):
        self.s = System(verbose=False)
        lx = random.uniform(10, 100)
        ly = random.uniform(10, 100)
        
        E0 = random.uniform(1, 100)
        ph = random.uniform(-np.pi, np.pi)
        
        lam = random.uniform(1, 10)
        
        plane = {
                "name"      : "plane",
                "gmode"     : "xy",
                "lims_x"    : np.array([-lx, lx]),
                "lims_y"    : np.array([-ly, ly]),
                "gridsize"  : np.array([31, 31])
                }

        UDict = {
                "name"      : "Uniform",
                "lam"       : lam,
                "E0"        : E0,
                "phase"     : ph,
                "pol"       : np.array([1, 0, 0])
                }
        
        self.s.addPlane(plane)
        self.s.createUniformSource(UDict, "plane")
    
    def _makeRandomUniform_uv(self):
        self.s = System(verbose=False)
        lx = random.uniform(0, 100)
        ly = random.uniform(0, 360)
        
        E0 = random.uniform(1, 100)
        ph = random.uniform(-np.pi, np.pi)
        
        lam = random.uniform(1, 10)
        
        plane = {
                "name"      : "plane",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, lx]),
                "lims_v"    : np.array([0, ly]),
                "gridsize"  : np.array([31, 31])
                }

        UDict = {
                "name"      : "Uniform",
                "lam"       : lam,
                "E0"        : E0,
                "phase"     : ph,
                "pol"       : np.array([1, 0, 0])
                }
        
        self.s.addPlane(plane)
        self.s.createUniformSource(UDict, "plane")

if __name__ == "__main__":
    unittest.main()


