import numpy as np
import sys
#sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.PyPO.System import System

def ex_ASTE_PO(device):
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
    """

    lam = 1 # [mm]

    parabola = {
            "name"      : "pri",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "flip"      : False,
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,3.5e3]),
            "lims_u"    : np.array([200,5e3]),
            "lims_v"    : np.array([0,360]),
            "gridsize"  : np.array([1501,1501])
            }

    d_foc_h = 5606.286
    hyperbola = {
            "name"      : "sec",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "flip"      : True,
            "focus_1"   : np.array([0,0,3.5e3]),
            "focus_2"   : np.array([0,0,3.5e3 - d_foc_h]),
            "ecc"       : 1.08208248,
            "lims_u"    : np.array([0,310]),
            "lims_v"    : np.array([0,360]),
            "gridsize"  : np.array([501,1501])
            }

    plane = {
            "name"      : "plane1",
            "gmode"     : "xy",
            "flip"      : False,
            "lims_x"    : np.array([-0.1,0.1]),
            "lims_y"    : np.array([-0.1,0.1]),
            "gridsize"  : np.array([3, 3])
            }

    planeff = {
            "name"      : "planeff",
            "gmode"     : "AoE",
            "flip"      : False,
            "lims_Az"   : np.array([-0.03,0.03]),
            "lims_El"   : np.array([-0.03,0.03]),
            "gridsize"  : np.array([201, 201])
            }

    s = System()
    s.addParabola(parabola)
    s.addHyperbola(hyperbola)
    s.addPlane(plane)
    s.addPlane(planeff)

    ps = s.generatePointSource("plane1") 
    JM = s.calcCurrents("plane1", ps, mode="PMC")

    translation = np.array([0,0,3.5e3 - d_foc_h])
    s.translateGrids("plane1", translation)

    grids_p = s.generateGrids("pri")
    grids_s = s.generateGrids("sec")
    
    fig, ax = pt.subplots(1,1, figsize=(5,5))
    ax.plot(grids_p.x[:,0], grids_p.z[:,0], color="dimgrey", lw=10)
    ax.plot(grids_p.x[:,750], grids_p.z[:,750], color="dimgrey", lw=10)
    ax.plot(grids_s.x[:,0], grids_s.z[:,0], color="dimgrey", lw=10)
    ax.plot(grids_s.x[:,750], grids_s.z[:,750], color="dimgrey", lw=10)
    ax.set_aspect('equal', adjustable='box')
    pt.show()

    if device == "GPU":
        nThreads = 256

    else:
        nThreads = 11

    plane1_to_sec = {
            "s_name"    : "plane1",
            "t_name"    : "sec",
            "s_current" : JM,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "JM"
            }

    JM1 = s.runPO(plane1_to_sec)

    sec_to_pri = {
            "s_name"    : "sec",
            "t_name"    : "pri",
            "s_current" : JM1,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "JM"
            }

    JM2 = s.runPO(sec_to_pri)

    pri_to_planeff = {
            "s_name"    : "pri",
            "t_name"    : "planeff",
            "s_current" : JM2,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "FF"
            }

    EH = s.runPO(pri_to_planeff)
    
    s.plotBeam2D("planeff", EH.Ex, units="as")

if __name__ == "__main__":
    ex_DRO()
