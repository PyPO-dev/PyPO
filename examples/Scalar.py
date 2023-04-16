import numpy as np

from src.PyPO.System import System

def Scalar(device):
    plane_source = {
            "name"      : "plane_source",
            "gmode"     : "xy",
            "lims_x"    : np.array([-1,1]),
            "lims_y"    : np.array([-1,1]),
            "gridsize"  : np.array([301, 301])
            }

    plane_eval = {
            "name"      : "plane_eval",
            "gmode"     : "xy",
            "lims_x"    : np.array([-500,500]),
            "lims_y"    : np.array([-500,500]),
            "gridsize"  : np.array([1001, 1001])
            }
    
    s = System()
    s.addPlane(plane_source)
    s.addPlane(plane_eval)

    s.translateGrids("plane_eval", np.array([0, 0, 100]))
   
    lam = 1

    PSDict = {
            "name"      : "ps1",
            "lam"       : lam,
            "E0"        : 1,
            "phase"     : 0,
            "w0x"       : 1,
            "w0y"       : 1,
            "pol"       : np.array([1,0,0])
            }
    
    s.createScalarGaussian(PSDict, "plane_source") 
    
    source_to_eval = {
            "t_name"        : "plane_eval",
            "s_scalarfield" : "ps1",
            "name_field"    : "out",
            "epsilon"       : 10,
            "device"        : device,
            "mode"          : "scalar"
            }

    s.runPO(source_to_eval)

    s.plotBeam2D("out")
    
if __name__ == "__main__":
    Scalar("CPU")
