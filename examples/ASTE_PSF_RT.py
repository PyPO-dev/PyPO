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
            "lims_x"        : np.array([-1000,1000]),
            "lims_y"        : np.array([-1000,1000]),
            "gridsize"      : np.array([3, 3])
            }

    RTpar = {
            "name"          : "start",
            "nRays"         : 10,
            "nRing"         : 10,
            "angx0"         : 0,
            "angy0"         : 0,
            "x0"            : 4000,
            "y0"            : 4000,
            }

    rotation = np.array([0, 0, 0])

    s = System()#MakeWO()
    s.addPlane(plane)

    s.addParabola(parabola)
    s.addHyperbola(hyperbola)
    s.translateGrids("plane1", np.array([0,0,3.5e3 - d_foc_h]))
 
    #s.translateGrids("sec", np.array([1064e-6, 1064e-6, 1064e-6]))
    #s.rotateGrids("sec", np.array([1,0,0]), np.array([0,0,3.5e3]))

    s.createTubeFrame(argDict=RTpar)
    s.translateGrids("start", np.array([0,0,3.5e3]), obj="frame")

    _rotation = np.array([160,45,14])

    s.rotateGrids("start", _rotation, obj="frame", pivot=np.zeros(3))

    s.groupElements("ASTE", "pri", "sec", "plane1")
    s.rotateGrids("ASTE", _rotation, obj="group", pivot=np.zeros(3))

    s.plotSystem()
    start_pri_RT = {
            "fr_in"     : "start",
            "t_name"    : "pri",
            "fr_out"    : "pri",
            "device"    : device
            }

    pri_sec_RT = {
            "fr_in"     : "pri",
            "t_name"    : "sec",
            "fr_out"    : "sec",
            "device"    : device
            }
    
    sec_focus_RT = {
            "fr_in"     : "sec",
            "t_name"    : "plane1",
            "fr_out"    : "focus",
            "device"    : device
            }
    
    s.runRayTracer(start_pri_RT)
    s.runRayTracer(pri_sec_RT)
    s.runRayTracer(sec_focus_RT)
    
    s.plotRTframe("focus", project="xy")
    s.plotSystem(RTframes=["start", "pri", "sec", "focus"])

if __name__ == "__main__":
    ex_ASTE()
