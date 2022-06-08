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
    
    lam = 210 # [mm]
    k = 2 * np.pi / lam
    
    # Primary parameters
    R_pri           = 12.5e3 # Radius in [mm]
    R_aper          = 0 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,12e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1 = [foc_pri, ver_pri]

    lims_r_p1       = [R_aper, R_pri]
    lims_v_p1       = [0, 2*np.pi]
    
    lims_x_p1 = [-R_pri, R_pri]
    lims_y_p1 = [-R_pri, R_pri]

    gridsize_p1     = [301, 301] # The gridsizes along the x and y axes

    # Initialize system
    s = System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    #s.addParabola(name="p1", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    
    s.addParabola(name="p1", coef=coef_p1, lims_x=lims_x_p1, lims_y=lims_y_p1, gridsize=gridsize_p1, pmode='foc', gmode='xy')
    
    pt.imshow(s.system["p1"].area)
    pt.show()
    
    # Instantiate camera surface. 
    center_cam = foc_pri + np.array([0,0,0])
    lims_x_cam = [-1000, 1000]
    lims_y_cam = [-1000, 1000]
    gridsize_cam = [301, 301]
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1")

    print(s.system["p1"])
    print(s.system["cam1"])
    s.plotSystem(focus_1=True, focus_2=True)
    
    # Initialize a plane wave illuminating the primary from above. Place at height of primary focus.
    # Apply mask to plane wave grid corresponding to secondary mirror size. Make slightly oversized to minimize numerical
    # diffraction effects due to plane wave grid edges.
    
    R_pw = R_pri + 30*lam
    
    lims_x_pw = [-R_pw, R_pw]
    lims_y_pw = [-R_pw, R_pw]
    gridsize_pw = [201, 201]
    
    s.addBeam(lims_x_pw, lims_y_pw, gridsize_pw, flip=True)

    s.inputBeam.calcJM(mode='PMC')
    
    offTrans_pw = foc_pri + np.array([0,0,100])
    s.inputBeam.transBeam(offTrans=offTrans_pw)
    
    s.initPhysOptics(target=s.system["p1"], k=k, numThreads=11, cpp_path=cpp_path)
    #s.initPhysOptics(target=s.system["cam1"], k=k, numThreads=11)
    #s.runPhysOptics()
    #s.nextPhysOptics(source=s.system["p1"], target=s.system["cam1"])
    s.runPhysOptics(save=2)
    
    s.plotSystem(focus_1=False, focus_2=False)#, exclude=[0,1,2])
    
    #field = s.loadField(s.system["cam1"])
    field = s.loadField(s.system["p1"], mode='Jz')
    s.addPlotter(save='../images/')
    
    
    s.plotter.plotBeam2D(s.system["p1"], field=field, ff=12e3, vmin=-30, interpolation='lanczos')
    #s.plotter.plotBeam2D(s.system["cam1"], field=field, ff=12e3, vmin=-30, interpolation='lanczos')
    
    #s.PO.FF_fromFocus(s.system["cam1"].grid_x, s.system["cam1"].grid_y)
    
if __name__ == "__main__":
    ex_DRO()

