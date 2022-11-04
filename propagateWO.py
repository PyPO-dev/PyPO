import numpy as np
import matplotlib.pyplot as pt

from WO import MakeWO


def propagateWO(mode):
    RTpar = {
            "nRays"     : 10,
            "nRing"     : 10,
            "angx"      : 6,
            "angy"      : 6,
            "a"         : 0,
            "b"         : 0,
            "tChief"    : np.array([0,90,0]),
            "oChief"    : np.array([0,0,0])
            }

    wfp = {
            "name"      : "wfp",
            "gmode"     : "xy",
            "flip"      : False,
            "lims_x"    : [-100,100],
            "lims_y"    : [-100,100],
            "gridsize"  : [301, 301]
            }

    wfpff = {
            "name"      : "wfpff",
            "gmode"     : "AoE",
            "flip"      : False,
            "lims_Az"    : [-7,7],
            "lims_El"    : [-7,7],
            "gridsize"  : [301, 301]
            }

    s = MakeWO()

    s.addPlane(wfp)
    s.addPlane(wfpff)

    wf = np.array([486.3974883317985, 0.0, 1371.340617233771])

    s.translateGrids("wfp", wf)

    if mode == "RT":
        fr_in = s.createFrame(RTpar)
        frame_out = s.runRayTracer(fr_in, "h1", nThreads=11, t0=1e2)
        frame_out2 = s.runRayTracer(frame_out, "e1", nThreads=11, t0=1e1)
        frame_out3 = s.runRayTracer(frame_out2, "wfp", nThreads=11, t0=1e3)

        s.plotSystem(RTframes=[fr_in, frame_out, frame_out2, frame_out3])
        s.plotRTframe(frame_out3)

    elif mode == "PO":
        source = {
                "name"      : "cryo",
                "gmode"     : "xy",
                "flip"      : False,
                "lims_x"    : [-80,80],
                "lims_y"    : [-80,80],
                "gridsize"  : [403, 401]
                }

        s.setCustomBeamPath("./custom/beam/240GHz/")
        freq = 240
        cl = 299792458e3
        lam = cl / (freq*1e9)
        k = 2 * np.pi / lam

        d_cryo = np.array([158.124, 0, 0])

        s.addPlane(source)
        s.rotateGrids("cryo", np.array([0,90,0]))
        s.translateGrids("cryo", d_cryo)

        # MISALIGN BEAM
        s.rotateGrids("cryo", np.array([0,0,0]))

        JM, EH = s.readCustomBeam("240", "cryo", "Ez", convert_to_current=True, mode="PMC")

        s.plotBeam2D("cryo", EH.Ez, project="yz")

        JM1, EH = s.propagatePO_GPU("cryo", "h1", JM, k=k,
                        epsilon=10, t_direction=-1, nThreads=256,
                        mode="JMEH", precision="single")

        s.plotBeam2D("h1", EH.Ez, project="yz")

        JM2 = s.propagatePO_GPU("h1", "e1", JM1, k=k,
                        epsilon=10, t_direction=-1, nThreads=256,
                        mode="JM", precision="single")

        JM_wf, EH_wf = s.propagatePO_GPU("e1", "wfp", JM2, k=k,
                        epsilon=10, t_direction=-1, nThreads=256,
                        mode="JMEH", precision="single")

        s.plotBeam2D("wfp", EH_wf.Ex, project="yx")

        s.translateGrids("wfp", -wf)
        EH_wfff = s.propagatePO_GPU("wfp", "wfpff", JM_wf, k=k,
                        epsilon=10, t_direction=-1, nThreads=256,
                        mode="FF", precision="single")

        pt.imshow(np.absolute(EH_wfff.Ex))
        pt.show()

        s.plotBeam2D("wfpff", EH_wfff.Ex, project="xy")

if __name__ == "__main__":
    propagateWO("PO")
