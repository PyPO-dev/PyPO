import numpy as np
import unittest
import os

from pathlib import Path
from PyPO.System import System

##
# @file 
# 
# Script for testing the far-field calculations of PyPO by calculating the far-field of 
# a uniformly illuminated disk, with varying wavelength, and calculating the HPBW. 
# This is then compared to the theoretical value of a Bessel function by calculating the relative difference.
# This test is a high level test, but included here such that it can be run automatically..
class Test_Bessel(unittest.TestCase):
    
    def test_Bessel(self):
        lams = np.array([10, 1, 0.1])
        
        E_l = []
        H_l = []
            
        for lam in lams:
            e, h = self._calc_ff(lam)
            E_l.append(e)
            H_l.append(h)

        E_l = np.array(E_l)
        H_l = np.array(H_l)

        diff_lim = lams / 10e3 * 1.025 * 180 / np.pi * 3600

        diff_E = (E_l - diff_lim) / E_l
        diff_H = (H_l - diff_lim) / H_l

        thres = 0.01

        for E, H in zip(diff_E, diff_H):
            self.assertAlmostEqual(E, 0.0, delta=thres)
            self.assertAlmostEqual(H, 0.0, delta=thres)

    def _calc_ff(self, lam):
        s = System(override=False, verbose=False)

        ring = {
                "name"      : "ring",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, 5000]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([501, 501])
                }

        a = 0.2
        b = 40

        ext = (a + b*lam) / 3600

        ff_plane = {
                "name"      : "ff_plane",
                "gmode"     : "AoE",
                "lims_Az"   : np.array([-ext, ext]),
                "lims_El"   : np.array([-ext, ext]),
                "gridsize"  : np.array([201, 201])
                }

        UDict = {
                "name"      : "source",
                "lam"       : lam
                }

        s.addPlane(ring)
        s.addPlane(ff_plane)
        s.createUniformSource(UDict, "ring")

        source_to_ff = {
                "s_current"    : "source",
                "t_name"        : "ff_plane",
                "name_EH"       : "ff_EH",
                "mode"          : "FF",
                "epsilon"       : 10
                }

        s.runPO(source_to_ff)
        E, H = s.calcHPBW("ff_EH", "Ex")

        return E, H

if __name__ == "__main__":
    unittest.main()
