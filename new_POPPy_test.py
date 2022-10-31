from src.POPPy.System import System
import numpy as np
import matplotlib.pyplot as pt

import src.POPPy.Colormaps as cmaps

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
    parabola["lims_x"] = [-5000,5000]
    parabola["lims_y"] = [-5000,5000]
    parabola["lims_u"] = [200,5e3]
    parabola["lims_v"] = [0,360]
    parabola["gridsize"] = [501,501]

    hyperbola = {}
    hyperbola["name"] = "h1"
    #hyperbola["pmode"] = "manual"
    hyperbola["pmode"] = "focus"
    hyperbola["gmode"] = "xy"
    hyperbola["flip"] = False
    hyperbola["focus_1"] = np.array([0,0,3.5e3])
    hyperbola["focus_2"] = np.array([0,0,3.5e3-1000])
    hyperbola["ecc"] = 1.08208248
    hyperbola["lims_x"] = [-310,310]
    hyperbola["lims_y"] = [-310,310]
    hyperbola["lims_u"] = [0,310]
    hyperbola["lims_v"] = [0,360]
    hyperbola["gridsize"] = [501,501]
    """
    plane = {}
    plane["name"] = "plane1"
    plane["gmode"] = "xy"
    plane["flip"] = False
    plane["lims_x"] = [-0.1,0.1]
    plane["lims_y"] = [-0.1,0.1]
    plane["gridsize"] = [3, 3]
    """
    plane = {}
    plane["name"] = "plane1"
    plane["gmode"] = "xy"
    plane["flip"] = False
    plane["lims_x"] = [-80,80]
    plane["lims_y"] = [-80,80]
    plane["gridsize"] = [403, 401]

    plane2 = {}
    plane2["name"] = "plane2"
    plane2["gmode"] = "xy"
    plane2["flip"] = False
    plane2["lims_x"] = [-1000,1000]
    plane2["lims_y"] = [-1000,1000]
    plane2["gridsize"] = [1501, 1501]

    planeff = {}
    planeff["name"] = "planeff"
    planeff["gmode"] = "AoE"
    planeff["flip"] = False
    planeff["lims_Az"] = [-0.3,0.3]
    planeff["lims_El"] = [-0.3,0.3]
    planeff["gridsize"] = [201, 201]

    s = System()
    s.addPlotter()
    #s.addHyperbola(hyperbola)
    s.addParabola(parabola)
    s.addPlane(plane)
    s.addPlane(planeff)


    rotation=np.array([42, 42, 0])

    #s.rotateGrids("p1", rotation)
    #s.rotateGrids("h1", rotation)

    translation = np.array([0, 0, 3.5e3-150])
    rotation_plane = np.array([180, 0, 0])
    s.rotateGrids("plane1", rotation_plane)
    s.translateGrids("plane1", translation)

    s.plotter.plotSystem(s.system, fine=2, norm=False)


    s.setCustomBeamPath(path="240GHz/", append=True)

    cBeam = "240"


    #s.setCustomBeamPath(path="ps/", append=True)

    #cBeam = "ps"
    JM, EH = s.readCustomBeam(name="240", comp="Ex", shape=[403,401], convert_to_current=True, mode="PMC", ret="both")

    #JM, EH = s.readCustomBeam(name=cBeam, comp="Ex", shape=[3,3], convert_to_current=True, mode="PMC", ret="both")

    JM1 = s.propagatePO_GPU("plane1", "p1", JM, k=5,
                    epsilon=10, t_direction=-1, nThreads=256,
                    mode="JM", precision="single")

    #EH2 = s.propagatePO_CPU("p1", "planeff", JM1, k=0.02991993003,
    #                epsilon=10, t_direction=-1, nThreads=11,
    #                mode="FF", precision="double")

    EH2 = s.propagatePO_GPU("p1", "planeff", JM1, k=5,
                    epsilon=10, t_direction=-1, nThreads=256,
                    mode="FF", precision="single")

    pt.imshow(20*np.log10(np.absolute(EH2.Ex) / np.max(np.absolute(EH2.Ex))), vmin=-30, vmax=0, cmap=cmaps.plasma)
    pt.show()


if __name__ == "__main__":
    plotSystem_test()
