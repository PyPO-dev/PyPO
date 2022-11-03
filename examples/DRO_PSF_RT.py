import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

from src.POPPy.System import System

def ex_DRO_RT():
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
            "lims_u"    : [200,12.5e3],
            "lims_v"    : [0,360],
            "gridsize"  : [1501,1501]
            }

    plane = {
            "name"      : "plane1",
            "gmode"     : "xy",
            "flip"      : False,
            "lims_x"    : [-100,100],
            "lims_y"    : [-100,100],
            "gridsize"  : [3, 3]
            }

    RTpar = {
            "nRays"     : 1000,
            "nRing"     : 1000,
            "angx"      : 0,
            "angy"      : 0,
            "a"         : 10000,
            "b"         : 10000,
            "tChief"    : np.array([180,0,0]),
            "oChief"    : np.array([0,0,12e3])
            }

    s = System()
    s.addPlotter()
    s.addParabola(parabola)
    s.addPlane(plane)
    s.translateGrids("plane1", np.array([0,0,12e3]))

    s.plotter.plotSystem(s.system)

    frame_in = s.createFrame(RTpar)

    frame_out = s.runRayTracer(frame_in, "p1", nThreads=11, t0=1e4)

    frame_out2 = s.runRayTracer(frame_out, "plane1", nThreads=11, t0=1e4)

    stack = s.calcRayLen(frame_in, frame_out, frame_out2)

    s.plotter.plotRTframe(frame_out2)
    s.plotter.plotSystem(s.system, RTframes=[frame_in, frame_out, frame_out2])





if __name__ == "__main__":
    ex_DRO()
