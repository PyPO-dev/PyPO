import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

from src.POPPy.System import System

def ex_ASTE_RT(device):
    parabola = {}
    parabola["name"] = "p1"
    parabola["pmode"] = "focus"
    parabola["gmode"] = "uv"
    parabola["flip"] = False
    parabola["vertex"] = np.zeros(3)
    parabola["focus_1"] = np.array([0,0,3.5e3])
    parabola["lims_u"] = [200,5e3]
    parabola["lims_v"] = [0,360]
    parabola["gridsize"] = [1501,1501]

    d_foc_h = 5606.286
    hyperbola = {}
    hyperbola["name"] = "h1"
    hyperbola["pmode"] = "focus"
    hyperbola["gmode"] = "uv"
    hyperbola["flip"] = True
    hyperbola["focus_1"] = np.array([0,0,3.5e3])
    hyperbola["focus_2"] = np.array([0,0,3.5e3 - d_foc_h])
    hyperbola["ecc"] = 1.08208248
    hyperbola["lims_u"] = [0,310]
    hyperbola["lims_v"] = [0,360]
    hyperbola["gridsize"] = [501,501]

    plane = {}
    plane["name"] = "plane1"
    plane["gmode"] = "xy"
    plane["flip"] = False
    plane["lims_x"] = [-100,100]
    plane["lims_y"] = [-100,100]
    plane["gridsize"] = [3, 3]

    RTpar = {
            "nRays"     :       10,
            "nRing"     :       10,
            "angx"      :       0,
            "angy"      :       0,
            "a"         :       4000,
            "b"         :       4000,
            "tChief"    :       np.array([0,0,0]),
            "oChief"    :       np.array([0,0,3.5e3])
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

    frame_in = s.createFrame(argDict=RTpar)

    frame_out = s.runRayTracer(frame_in, "p1", nThreads=11)
    frame_out1 = s.runRayTracer(frame_out, "h1", nThreads=11)

    frame_out2 = s.runRayTracer(frame_out1, "plane1", nThreads=11)

    s.plotter.plotRTframe(frame_out2, project="xy")
    s.plotter.plotSystem(s.system, RTframes=[frame_in, frame_out, frame_out1, frame_out2])

if __name__ == "__main__":
    ex_ASTE()
