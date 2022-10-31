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

    RTpar = {
            "nRays"     :       10,
            "nRing"     :       10,
            "angx"      :       6,
            "angy"      :       6,
            "a"         :       10,
            "b"         :       10,
            "tChief"    :       np.zeros(3),
            "oChief"    :       np.array([0,0,12e3])
            }

    s = System()
    s.addParabola(parabola)
    s.addPlane(plane)

    frame_in = s.initRayTracer(nThreads=11, mode="manual", argDict=RTpar)
    frame_out = s.runRayTracer(frame_in, "p1", nThreads=11)

    print(frame_out.size)

    frame_out2 = s.runRayTracer(frame_out, "plane1", nThreads=11)



    pt.scatter(frame_out2.x, frame_out2.y)
    pt.show()



if __name__ == "__main__":
    ex_DRO()
