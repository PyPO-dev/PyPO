import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt
import matplotlib.cm as cm
import src.POPPy.Colormaps as cmaps

from src.POPPy.System import System
from WO import MakeWO

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
            "name"          : "sec",
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

    #s = System()
    s = MakeWO()
    s.addPlane(plane)

    s.addParabola(parabola)
    s.addHyperbola(hyperbola)
    #s.translateGrids("plane1", np.array([0,0,3.5e3 - d_foc_h]))

    e_f0 = np.array([486.3974883317985, 0.0, 1371.340617233771]) # Warm focus

    s.translateGrids("p1", -np.array([0,0,3.5e3 - d_foc_h]))
    s.translateGrids("h1", -e_f0)
    s.translateGrids("e1", -e_f0)
    s.translateGrids("sec", -np.array([0,0,3.5e3 - d_foc_h]))

    s.rotateGrids("p1", rotation)
    s.rotateGrids("sec", rotation)
    #s.rotateGrids("plane1", rotation)
    
    cm_l = [cm.cool, cm.cool, cm.autumn, cm.cool, cm.cool]

    s.plotSystem(cmap=cm_l)

    frame_in = s.createFrame(argDict=RTpar)

    frame_out = s.runRayTracer(frame_in, "p1", nThreads=11)
    frame_out1 = s.runRayTracer(frame_out, "sec", nThreads=11)
    frame_out2 = s.runRayTracer(frame_out1, "plane1", nThreads=11)

    s.plotRTframe(frame_out2, project="xy")
    s.plotSystem(RTframes=[frame_in, frame_out, frame_out1, frame_out2])

if __name__ == "__main__":
    ex_ASTE()
