import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

from src.POPPy.System import System

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
            "nRays"     : 100,
            "nRing"     : 100,
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

    s.plotSystem()

    frame_in = s.createFrame(RTpar)

    if device == "CPU":
        frame_out = s.runRayTracer(frame_in, "p1", nThreads=11, t0=1e4)

        frame_out2 = s.runRayTracer(frame_out, "plane1", nThreads=11, t0=1e4)

    elif device == "GPU":
        frame_out = s.runRayTracer(frame_in, "p1", nThreads=256, t0=1e4, device=device)

        frame_out2 = s.runRayTracer(frame_out, "plane1", nThreads=256, t0=1e4, device=device)

    stack = s.calcRayLen(frame_in, frame_out, frame_out2)

    s.plotRTframe(frame_out2)
    s.plotSystem(RTframes=[frame_in, frame_out, frame_out2])





if __name__ == "__main__":
    ex_DRO()
