import numpy as np
import unittest
import os

from pathlib import Path
from PyPO.System import System

import matplotlib.pyplot as pt

##
# @file 
# 
# Script for manually testing the far-field calculations of PyPO by calculating the far-field of 
# a uniformly illuminated disk, with varying wavelength, and calculating the HPBW.
# To see if it works, the test plots the theoretical HPBW of a Bessel function and the calculated HPBWs.
def Bessel():
    lams = np.linspace(1, 2, 3)
    E_l = []
    H_l = []
        
    for lam in lams:
        e, h = _calc_ff(lam)
        E_l.append(e)
        H_l.append(h)

    E_l = np.array(E_l)
    H_l = np.array(H_l)

    diff_lim = lams / 1e3 * 1.025 * 180 / np.pi * 3600

    fig, ax = pt.subplots(1,1)
    ax.plot(lams, diff_lim, label="Diffraction limit")
    ax.scatter(lams, E_l, label="E-plane, PyPO")
    ax.scatter(lams, H_l, label="H-plane, PyPO")
    ax.set_title("Half-power beamwidth versus wavelength")
    ax.set_xlabel("Wavelength in mm")
    ax.set_ylabel("HPBW in arcseconds")
    ax.legend(frameon=False, prop={'size': 15},handlelength=1)

    pt.show()

def _calc_ff(lam):
    s = System(override=False, verbose=False)

    ring = {
            "name"      : "ring",
            "gmode"     : "uv",
            "lims_u"    : np.array([0, 500]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([201, 201])
            }

    a = 0.2
    b = 400

    ext = (a + b*lam) / 3600

    ff_plane = {
            "name"      : "ff_plane",
            "gmode"     : "AoE",
            "lims_Az"   : np.array([-ext, ext]),
            "lims_El"   : np.array([-ext, ext]),
            "gridsize"  : np.array([101, 101])
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
            "epsilon"       : 10,
            }

    s.runPO(source_to_ff)
    E, H = s.calcHPBW("ff_EH", "Ex")

    return E, H

if __name__ == "__main__":
    Bessel()
