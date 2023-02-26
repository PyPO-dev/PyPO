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

    plane = {
            "name"      : "plane1",
            "gmode"     : "xy",
            "flip"      : False,
            "lims_x"    : np.array([-100,100]),
            "lims_y"    : np.array([-100,100]),
            "gridsize"  : np.array([3, 3])
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
    s.addPlane(plane)
    s.translateGrids("plane1", np.array([0,0,12e3]))
    
    #s.rotateGrids("p1", np.array([30,0,0]), np.array([0,0,12e3]))

    s.plotSystem()

    s.createTubeFrame(RTpar)

    start_pri_RT = {
            "fr_in"     : "start",
            "t_name"    : "p1",
            "fr_out"    : "pri",
            "tol"       : 1e-3,
            "nThreads"  : 11 if device == "CPU" else 256,
            "device"    : device
            }

    pri_focus_RT = {
            "fr_in"     : "pri",
            "t_name"    : "plane1",
            "fr_out"    : "focus",
            "tol"       : -1e-3,
            "nThreads"  : 11 if device == "CPU" else 256,
            "device"    : device
            }
    
    s.runRayTracer(start_pri_RT)
    s.runRayTracer(pri_focus_RT)

    pt.show()
    s.plotRTframe("focus")
    s.plotSystem(RTframes=["start", "pri", "focus"])

if __name__ == "__main__":
    ex_DRO_RT()
