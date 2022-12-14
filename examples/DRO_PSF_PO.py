import numpy as np
import sys
#sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.PyPO.System import System

def ex_DRO_PO(device):
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
    """

    lam = 210 # mm

    parabola = {
            "name"      : "p1",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "flip"      : False,
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,12e3]),
            "lims_u"    : np.array([200,12.5e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([1501,1501])
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
            "lims_Az"   : np.array([-3,3]),
            "lims_El"   : np.array([-3,3]),
            "gridsize"  : np.array([201, 201])
            }

    s = System()
    s.addParabola(parabola)
    s.addPlane(plane)
    s.addPlane(planeff)

    s.plotSystem()

    ps = s.generatePointSource("plane1") 
    JM = s.calcCurrents("plane1", ps, mode="PMC")

    translation = np.array([0, 0, 12e3])# + np.array([210, 210, -210])
    rotation_plane = np.array([180, 0, 0])
    s.rotateGrids("plane1", rotation_plane)
    s.translateGrids("plane1", translation)
    s.rotateGrids(["p1", "plane1"], np.array([1, 0, 0]))

    if device == "GPU":
        nThreads = 256

    else:
        nThreads = 11

    plane1_to_p1 = {
            "s_name"    : "plane1",
            "t_name"    : "p1",
            "s_current" : JM,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "JM"
            }

    JM1 = s.runPO(plane1_to_p1)

    p1_to_planeff = {
            "s_name"    : "p1",
            "t_name"    : "planeff",
            "s_current" : JM1,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "FF"
            }

    EH = s.runPO(p1_to_planeff)
    
    s.plotBeam2D("planeff", EH.Ex, units="deg")
if __name__ == "__main__":
    ex_DRO()
