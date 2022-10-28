from src.POPPy.System import System
import numpy as np
import matplotlib.pyplot as pt

def plotSystem_test():
    parabola = {}
    parabola["name"] = "p1"
    #parabola["pmode"] = "manual"
    parabola["pmode"] = "focus"
    parabola["gmode"] = "uv"
    parabola["flip"] = False
    parabola["coeffs"] = [1, 1, -1]
    parabola["vertex"] = np.zeros(3)
    parabola["focus_1"] = np.array([0,0,3.5e3])
    parabola["lims_x"] = [-1,1]
    parabola["lims_y"] = [-1,1]
    parabola["lims_u"] = [200,5e3]
    parabola["lims_v"] = [0,2*np.pi]
    parabola["gridsize"] = [403,401]

    hyperbola = {}
    hyperbola["name"] = "h1"
    #hyperbola["pmode"] = "manual"
    hyperbola["pmode"] = "focus"
    hyperbola["gmode"] = "xy"
    hyperbola["flip"] = False
    hyperbola["focus_1"] = np.array([0,0,3.5e3])
    hyperbola["focus_2"] = np.array([0,0,3.5e3 - 5606])
    hyperbola["ecc"] = 1.08208248
    hyperbola["lims_x"] = [-310,310]
    hyperbola["lims_y"] = [-310,310]
    hyperbola["lims_u"] = [0,310]
    hyperbola["lims_v"] = [0,2*np.pi]
    hyperbola["gridsize"] = [501,501]

    s = System()
    s.addPlotter()
    s.addHyperbola(hyperbola)
    s.addParabola(parabola)

    rotation=np.array([90, 0, 0])

    #s.rotateGrids("p1", rotation)

    s.plotter.plotSystem(s.system, fine=2, norm=False)

    s.setCustomBeamPath(path="240GHz/", append=True)

    cBeam = "240"
    JM, EH = s.readCustomBeam(name="240", comp="Ex", shape=[403,401], convert_to_current=True, mode="PMC", ret="both")

    pt.imshow(np.absolute(EH.Ex))
    pt.show()

    EH1 = s.propagatePO_GPU(s.system["p1"], s.system["h1"], JM, k=4.9,
                    epsilon=1, t_direction=-1, nThreads=256,
                    mode="EH", precision="single")

    EH1 = s.propagatePO_CPU(s.system["p1"], s.system["h1"], JM, k=4.9,
                    epsilon=1, t_direction=-1, nThreads=11,
                    mode="EH", precision="double")

    pt.imshow(np.absolute(EH1.Ex))
    pt.show()


if __name__ == "__main__":
    plotSystem_test()
