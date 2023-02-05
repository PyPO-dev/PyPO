import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt
import matplotlib.cm as cm
import src.PyPO.Colormaps as cmaps

from src.PyPO.System import System
#from WO import MakeWO

def ex_ASTE_RT(device):
    parabola = {
            "name"          : "pri",
            "pmode"         : "focus",
            "gmode"         : "uv",
            "flip"          : False,
            "vertex"        : np.zeros(3),
            "focus_1"       : np.array([0,0,3.5e3]),
            "lims_u"        : np.array([200,5e3]),
            "lims_v"        : np.array([0,360]),
            "gridsize"      : np.array([1501,1501])
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
            "lims_u"        : np.array([0,310]),
            "lims_v"        : np.array([0,360]),
            "gridsize"      : np.array([501,501])
            }


    plane = {
            "name"          : "plane1",
            "gmode"         : "xy",
            "flip"          : False,
            "lims_x"        : np.array([-1000,1000]),
            "lims_y"        : np.array([-1000,1000]),
            "gridsize"      : np.array([3, 3])
            }

    RTpar = {
            "name"          : "start",
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

    s = System()#MakeWO()
    s.addPlane(plane)

    s.addParabola(parabola)
    s.addHyperbola(hyperbola)
    s.translateGrids("plane1", np.array([0,0,3.5e3 - d_foc_h]))
 
    #s.translateGrids("sec", np.array([1064e-6, 1064e-6, 1064e-6]))
    #s.rotateGrids("sec", np.array([1,0,0]), np.array([0,0,3.5e3]))

    s.createFrame(argDict=RTpar)

    if device == "CPU":
        s.runRayTracer("start", "pri", "pri", nThreads=11)
        s.runRayTracer("pri", "sec", "sec", nThreads=11)
        s.runRayTracer("sec", "focus", "plane1", nThreads=11)

    if device == "GPU":
        s.runRayTracer("start", "pri", "pri", nThreads=256, device=device)
        s.runRayTracer("pri", "sec", "sec", nThreads=256, device=device)
        s.runRayTracer("sec", "focus", "plane1", nThreads=256, device=device)

    s.plotRTframe("focus", project="xy")
    s.plotSystem(RTframes=["start", "pri", "sec", "focus"])

if __name__ == "__main__":
    ex_ASTE()
