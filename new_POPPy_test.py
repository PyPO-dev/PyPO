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
    parabola["lims_x"] = [-5000,5000]
    parabola["lims_y"] = [-5000,5000]
    parabola["lims_u"] = [200,5e3]
    parabola["lims_v"] = [0,360]
    parabola["gridsize"] = [1501,1501]

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

    plane = {}
    plane["name"] = "plane1"
    plane["gmode"] = "xy"
    plane["flip"] = False
    plane["lims_x"] = [-1,1]
    plane["lims_y"] = [-1,1]
    plane["gridsize"] = [3, 3]

    planeff = {}
    planeff["name"] = "planeff"
    planeff["gmode"] = "AoE"
    planeff["flip"] = False
    planeff["lims_Az"] = [-3,3]
    planeff["lims_El"] = [-3,3]
    planeff["gridsize"] = [101, 101]

    s = System()
    s.addPlotter()
    #s.addHyperbola(hyperbola)
    s.addParabola(parabola)
    s.addPlane(plane)


    rotation=np.array([42, 42, 0])

    #s.rotateGrids("p1", rotation)
    #s.rotateGrids("h1", rotation)

    translation = np.array([0, 0, 3500])
    rotation_plane = np.array([180, 0, 0])
    s.rotateGrids("plane1", rotation_plane)
    s.translateGrids("plane1", translation)

    s.plotter.plotSystem(s.system, fine=2, norm=False)

    s.addPlane(planeff)

    #s.setCustomBeamPath(path="240GHz/", append=True)

    #cBeam = "240"


    s.setCustomBeamPath(path="ps/", append=True)

    cBeam = "ps"
    #JM, EH = s.readCustomBeam(name="240", comp="Ex", shape=[403,401], convert_to_current=True, mode="PMC", ret="both")

    JM, EH = s.readCustomBeam(name=cBeam, comp="Ex", shape=[3,3], convert_to_current=True, mode="PMC", ret="both")

    pt.imshow(np.absolute(EH.Ex))
    pt.show()

    JM1, EH1 = s.propagatePO_GPU("plane1", "p1", JM, k=4.9,
                    epsilon=10, t_direction=1, nThreads=256,
                    mode="JMEH", precision="single")

    pt.imshow(np.absolute(JM1.My))
    pt.show()

    #EH1 = s.propagatePO_CPU(s.system["p1"], s.system["h1"], JM, k=4.9,
    #                epsilon=1, t_direction=-1, nThreads=11,
    #                mode="EH", precision="double")

    EH1 = s.propagatePO_CPU("p1", "planeff", JM1, k=4.9,
                    epsilon=10, t_direction=-1, nThreads=11,
                    mode="FF", precision="double")

    pt.imshow(np.absolute(EH1.Ex))
    pt.show()


if __name__ == "__main__":
    plotSystem_test()
