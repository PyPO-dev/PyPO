import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.Python.System import System

def ex_DRO():
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
    """
    
    cpp_path = '../src/C++/'

    # Initialize system
    s = System()
    s.setCustomBeamPath("../custom/beam/")
    s.setCustomReflPath("../custom/reflector/")
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addCustomReflector(location="WO_M1_240GHz/", name="h1")
    s.addCustomReflector(location="WO_M2_240GHz/", name="e1")

    # Warm focus
    wf = np.array([486.3974883317985, 0.0, 1371.340617233771])
    
    

    # Instantiate camera surface. 
    center_cam = wf
    lims_x_cam = [-30, 30]
    lims_y_cam = [-30, 30]
    gridsize_cam = [201, 201]
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1")

    s.plotSystem(focus_1=False, focus_2=False)
    
    # Initialize a plane wave illuminating the primary from above. Place at height of primary focus.
    # Apply mask to plane wave grid corresponding to secondary mirror size. Make slightly oversized to minimize numerical
    # diffraction effects due to plane wave grid edges.
    
    R_pw = 10*R_pri + 10*lam
    
    lims_x_pw = [-R_pw, R_pw]
    lims_y_pw = [-R_pw, R_pw]
    gridsize_pw = [501, 501]
    
    s.addCustomBeamGrid(area=1, n=3, amp=1e16)
    s.inputBeam.calcJM(mode='PMC')
    
    offTrans_ps = np.array([0,0,1e16])
    s.inputBeam.transBeam(offTrans=offTrans_ps)
    
    s.addPlotter(save='../images/')
    #s.addBeam(lims_x_pw, lims_y_pw, gridsize_pw, flip=True)

    #s.inputBeam.calcJM()
    
    #offTrans_pw = foc_pri + np.array([0,0,100])
    #s.inputBeam.transBeam(offTrans=offTrans_pw)
    
    s.initPhysOptics(target=s.system["p1"], k=k, numThreads=11, cpp_path=cpp_path)
    #s.initPhysOptics(target=s.system["cam1"], k=k, numThreads=11)
    s.runPhysOptics(save=2)
    
    s.PO.plotField(s.system["p1"].grid_x, s.system["p1"].grid_y, mode='Ex', polar=True)
    
    s.nextPhysOptics(source=s.system["p1"], target=s.system["cam1"])
    s.runPhysOptics(save=2)
    
    s.plotSystem(focus_1=False, focus_2=False)#, exclude=[0,1,2])
    
    field = s.loadField(s.system["cam1"], mode='Ex')
    #field = s.loadField(s.system["p1"], mode='Ez')
    
    
    
    #s.plotter.plotBeam2D(s.system["p1"], field=field, ff=12e3, vmin=-30, interpolation='lanczos')
    #s.plotter.beamCut(s.system["p1"], field=field)
    s.plotter.plotBeam2D(s.system["cam1"], field=field, ff=12e3, vmin=-30, interpolation='lanczos')
    s.plotter.beamCut(s.system["cam1"], field=field)
    
    #s.PO.FF_fromFocus(s.system["cam1"].grid_x, s.system["cam1"].grid_y)
    
if __name__ == "__main__":
    ex_DRO()

