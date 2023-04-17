import numpy as np
import unittest
import os

import matplotlib.pyplot as pt
from pathlib import Path
from src.PyPO.System import System

##
# @file 
# 
# Script for testing the far-field calculations of PyPO by calculating the far-field of 
# a uniformly illuminated disk, with varying wavelength, and calculating the HPBW. 
# This is then compared to the theoretical value of a Bessel function.
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

        self.assertTrue(np.all(diff_E))
        self.assertTrue(np.all(diff_H))
        #print(__file__)
        #self._plot_res(lams, E_l, H_l, diff_lim)

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

    def _plot_res(self, lams, E_l, H_l, diff_lim):
        filePath = Path(__file__).parents[0]
        fig, ax = pt.subplots(1,1, figsize=(5,5))

        ax.plot(np.log10(lams), np.log10(diff_lim), color='black', ls="dashed", label="Bessel")
        ax.scatter(np.log10(lams), np.log10(E_l), color="blue", marker="d", label="E-plane")
        ax.scatter(np.log10(lams), np.log10(H_l), color="red", s=10, label="H-plane")
        ax.set_box_aspect(1)
        ax.set_xlabel(r"$\log_{10}$($\lambda$ / mm)")
        ax.set_ylabel("$\log_{10}$(HPBW / as)")
        ax.legend(frameon=False, prop={'size': 15},handlelength=1)
        pt.savefig(fname=os.path.join(filePath, 'test_Bessel_res.png'), bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    unittest.main()
