import numpy as np
import sys
#sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.POPPy.System import System

def ex_DRO_PO(device):
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
    """

    lam = 210 # [mm]
    k = 2 * np.pi / lam

    parabola = {}
    parabola["name"] = "p1"
    parabola["pmode"] = "focus"
    parabola["gmode"] = "uv"
    parabola["flip"] = False
    parabola["vertex"] = np.zeros(3)
    parabola["focus_1"] = np.array([0,0,12e3])
    parabola["lims_u"] = np.array([200,12.5e3])
    parabola["lims_v"] = np.array([0, 360])
    parabola["gridsize"] = np.array([1501,1501])

    plane = {}
    plane["name"] = "plane1"
    plane["gmode"] = "xy"
    plane["flip"] = False
    plane["lims_x"] = np.array([-0.1,0.1])
    plane["lims_y"] = np.array([-0.1,0.1])
    plane["gridsize"] = np.array([3, 3])

    planeff = {}
    planeff["name"] = "planeff"
    planeff["gmode"] = "AoE"
    planeff["flip"] = False
    planeff["lims_Az"] = np.array([-3,3])
    planeff["lims_El"] = np.array([-3,3])
    planeff["gridsize"] = np.array([201, 201])

    s = System()
    s.addParabola(parabola)
    s.addPlane(plane)
    s.addPlane(planeff)

    s.plotSystem()

    s.saveElement("p1")

    s.setCustomBeamPath(path="ps/", append=True)

    cBeam = "ps"

    translation = np.array([0, 0, 12e3])# + np.array([210, 210, -210])
    rotation_plane = np.array([180, 0, 0])
    s.rotateGrids("plane1", rotation_plane)
    s.translateGrids("plane1", translation)

    JM, EH = s.readCustomBeam(cBeam, "plane1", "Ex", convert_to_current=True, mode="PMC")

    if device == "CPU":

        JM1 = s.propagatePO_CPU("plane1", "p1", JM, k=k,
                        epsilon=10, t_direction=-1, nThreads=11,
                        mode="JM", precision="double")

        EH = s.propagatePO_CPU("p1", "planeff", JM1, k=k,
                        epsilon=10, t_direction=-1, nThreads=11,
                        mode="FF", precision="double")

    elif device == "GPU":

        JM1 = s.propagatePO_GPU("plane1", "p1", JM, k=k,
                        epsilon=10, t_direction=-1, nThreads=256,
                        mode="JM", precision="single")

        EH = s.propagatePO_GPU("p1", "planeff", JM1, k=k,
                        epsilon=10, t_direction=-1, nThreads=256,
                        mode="FF", precision="single")

    eta_Xpol = s.calcXpol(EH.Ex, EH.Ey)
    print(eta_Xpol)

    result = s.fitGaussAbs(EH.Ex, "planeff", thres=-11)
    print(result)

    s.plotBeam2D("planeff", EH.Ex)
if __name__ == "__main__":
    ex_DRO()
