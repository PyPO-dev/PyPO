import numpy as np

from src.POPPy.System import System 

def Planes():
    plane = {
            "name"      : "hyp",
            "gmode"     : "uv",
            "pmode"     : "manual",
            "coeffs"    : np.array([1,1,1]),
            "lims_u"    : np.array([0, 1]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([101, 101]),
            "gcenter"   : np.array([0.5, 0.5]),
            "ecc_uv"    : 0.9,
            "rot_uv"    : 31.2,
            "flip"      : False
            }
    s = System()
    s.addHyperbola(plane)
    s.plotSystem()

if __name__ == "__main__":
    Planes()
