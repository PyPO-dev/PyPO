import numpy as np
import sys
#sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.POPPy.System import System

def ex_DRO_PO():
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
    parabola["lims_u"] = [200,12.5e3]
    parabola["lims_v"] = [0,360]
    parabola["gridsize"] = [1501,1501]

    plane = {}
    plane["name"] = "plane1"
    plane["gmode"] = "xy"
    plane["flip"] = False
    plane["lims_x"] = [-0.1,0.1]
    plane["lims_y"] = [-0.1,0.1]
    plane["gridsize"] = [3, 3]

    planeff = {}
    planeff["name"] = "planeff"
    planeff["gmode"] = "AoE"
    planeff["flip"] = False
    planeff["lims_Az"] = [-3,3]
    planeff["lims_El"] = [-3,3]
    planeff["gridsize"] = [201, 201]

    s = System()
    s.addPlotter()
    s.addParabola(parabola)
    s.addPlane(plane)
    s.addPlane(planeff)

    s.setCustomBeamPath(path="ps/", append=True)

    cBeam = "ps"
    JM = s.readCustomBeam(name=cBeam, comp="Ex", shape=[3,3], convert_to_current=True, mode="PMC")

    translation = np.array([0, 0, 12e3])
    rotation_plane = np.array([180, 0, 0])
    s.rotateGrids("plane1", rotation_plane)
    s.translateGrids("plane1", translation)

    JM1 = s.propagatePO_CPU("plane1", "p1", JM, k=k,
                    epsilon=10, t_direction=-1, nThreads=11,
                    mode="JM", precision="double")

    #EH = s.propagatePO_CPU("p1", "planeff", JM1, k=k,
    #                epsilon=10, t_direction=-1, nThreads=11,
    #                mode="FF", precision="double")

    #JM1 = s.propagatePO_GPU("plane1", "p1", JM, k=k,
    #                epsilon=10, t_direction=-1, nThreads=256,
    #                mode="JM", precision="single")

    EH = s.propagatePO_GPU("p1", "planeff", JM1, k=k,
                    epsilon=10, t_direction=-1, nThreads=256,
                    mode="FF", precision="single")

    pt.imshow(20*np.log10(np.absolute(EH.Ex) / np.max(np.absolute(EH.Ex))), vmin=-30, vmax=0)
    pt.show()

if __name__ == "__main__":
    ex_DRO()
