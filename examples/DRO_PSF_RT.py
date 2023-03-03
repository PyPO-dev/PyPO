import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

from src.PyPO.System import System

def ex_DRO_RT(device):
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
    """

    parabola = {
            "name"      : "p1",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "flip"      : False,
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0,360]),
            "gridsize"  : np.array([1501,1501])
            }

    RTpar = {
            "name"      : "start",
            "nRays"     : 10,
            "nRing"     : 10,
            "angx"      : 0,
            "angy"      : 0,
            "a"         : 10000,
            "b"         : 10000,
            "tChief"    : np.array([180,0,0]),
            "oChief"    : np.array([0,0,12e3])
            }

    s = System()
    s.addParabola(parabola)

    s.plotSystem()

    s.createTubeFrame(RTpar)

    start_pri_RT = {
            "fr_in"     : "start",
            "t_name"    : "p1",
            "fr_out"    : "pri",
            "device"    : device,
            "tol"       : 1e-6
            }

    s.runRayTracer(start_pri_RT)
    s.findRTfocus("pri")

    s.plotRTframe("focus_pri")
    s.plotSystem(RTframes=["start", "pri", "focus_pri"])

if __name__ == "__main__":
    ex_DRO_RT()
