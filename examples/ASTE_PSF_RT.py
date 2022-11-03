import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

from src.POPPy.System import System

def ex_ASTE_RT(device):
    parabola = {
            "name"          : "p1",
            "pmode"         : "focus",
            "gmode"         : "uv",
            "flip"          : False,
            "vertex"        : np.zeros(3),
            "focus_1"       : np.array([0,0,3.5e3]),
            "lims_u"        : [200,5e3],
            "lims_v"        : [0,360],
            "gridsize"      : [1501,1501]
            }

    d_foc_h = 5606.286
    hyperbola = {
            "name"          : "h1",
            "pmode"         : "focus",
            "gmode"         : "uv",
            "flip"          : True,
            "focus_1"       : np.array([0,0,3.5e3]),
            "focus_2"       : np.array([0,0,3.5e3 - d_foc_h]),
            "ecc"           : 1.08208248,
            "lims_u"        : [0,310],
            "lims_v"        : [0,360],
            "gridsize"      : [501,501]
            }

    plane = {
            "name"          : "plane1",
            "gmode"         : "xy",
            "flip"          : False,
            "lims_x"        : [-100,100],
            "lims_y"        : [-100,100],
            "gridsize"      : [3, 3]
            }

    RTpar = {
            "nRays"         : 10,
            "nRing"         : 10,
            "angx"          : 0,
            "angy"          : 0,
            "a"             : 4000,
            "b"             : 4000,
            "tChief"        : np.array([0,0,0]),
            "oChief"        : np.array([0,0,3.5e3])
            }

    rotation = np.array([0, 0, 0])

    s = System()
    s.addPlotter()
    s.addParabola(parabola)
    s.addHyperbola(hyperbola)
    s.addPlane(plane)
    s.translateGrids("plane1", np.array([0,0,3.5e3 - d_foc_h]))

    s.rotateGrids("p1", rotation)
    s.rotateGrids("h1", rotation)
    #s.rotateGrids("plane1", rotation)

    s.plotter.plotSystem(s.system)

    frame_in = s.createFrame(argDict=RTpar)

    frame_out = s.runRayTracer(frame_in, "p1", nThreads=11)
    frame_out1 = s.runRayTracer(frame_out, "h1", nThreads=11)
    frame_out2 = s.runRayTracer(frame_out1, "plane1", nThreads=11)

    s.plotter.plotRTframe(frame_out2, project="xy")
    s.plotter.plotSystem(s.system, RTframes=[frame_in, frame_out, frame_out1, frame_out2])

if __name__ == "__main__":
    ex_ASTE()
