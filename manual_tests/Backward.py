import sys
import os
import random

import matplotlib.pyplot as pt

import numpy as np
from pathlib import Path

from PyPO.System import System

##
# @file
# Manual test to check phase stability of backwards propagation.
# We generate ten point sources with phases between -3 and 3 radians and check the total phase error upon propagating backward.
# The phase error will be plotted to screen.
def Backwards():
    s = System(override=True, verbose=False)
    
    source = {
            "name"      : "source",
            "gmode"     : "xy",
            "lims_x"    : np.array([-0.01, 0.01]),
            "lims_y"    : np.array([-0.01, 0.01]),
            "gridsize"  : np.array([31, 31])
            }
    
    plane_up = {
            "name"      : "plane_up",
            "gmode"     : "uv",
            "lims_u"    : np.array([0, 100]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([101, 101]),
            "flip"      : True
            }

    plane_down = {
            "name"      : "plane_down",
            "gmode"     : "uv",
            "lims_u"    : np.array([0, 0.1]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([101, 101])
            }
    
    s.addPlane(source)
    s.addPlane(plane_up)
    s.addPlane(plane_down)

    ph_diff = []
    phases = np.linspace(-3, 3, 10)
    for ph in phases:
        PSDict = {
                "name"      : "PS_source",
                "lam"       : 1,
                "E0"        : 1,
                "phase"     : ph,
                "pol"       : np.array([1,0,0])
                }
        
        s.createPointSource(PSDict, "source")

        s.translateGrids("plane_up", np.array([0, 0, 100]))
        
        runPODict = {
                "t_name"    : "plane_up",
                "s_current" : "PS_source",
                "epsilon"   : 10,
                "exp"       : "fwd",
                "mode"      : "JMEH",
                "name_JM"   : "JM_up",
                "name_EH"   : "EH_up",
                }
        
        runPODict_bwd = {
                "t_name"    : "plane_down",
                "s_current" : "JM_up",
                "epsilon"   : 10,
                "exp"       : "bwd",
                "mode"      : "JMEH",
                "name_JM"   : "JM_down",
                "name_EH"   : "EH_down",
                }

        s.runPO(runPODict)
        s.runPO(runPODict_bwd)

        phase_Ex = np.mean(np.angle(s.fields["EH_down"].Ex))
        ph_diff.append(ph - phase_Ex)

    fig, ax = pt.subplots(1,1)
    ax.scatter(phases, ph_diff)
    ax.set_title("Backwards propagated phase error versus starting phase")
    ax.set_xlabel("Starting phase")
    ax.set_ylabel("Absolute phase error")
    pt.show()

def get_PS(phase):

    return PSDict

if __name__ == "__main__":
    Backwards()
