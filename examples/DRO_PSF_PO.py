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
    R_aper          = 300 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,12e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1 = [foc_pri, ver_pri]

    lims_r_p1       = [R_aper, R_pri]
    lims_v_p1       = [0, 2*np.pi]

    gridsize_p1     = [801, 501] # The gridsizes along the x and y axes

    # Initialize system
    s = System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name="p1", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    
    # Instantiate camera surface. 
    center_cam = foc_pri
    lims_x_cam = [-800, 800]
    lims_y_cam = [-800, 800]
    gridsize_cam = [401, 401]
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1")

    s.plotSystem(focus_1=True, focus_2=True)
    
    s.addPointSource(area=1, pol='incoherent', n=3, amp=1e16)

    offTrans_ps = np.array([0,0,1e16])
    s.inputBeam.translateBeam(offTrans=offTrans_ps)
    
    s.addPlotter(save='../images/')

    s.initPhysOptics(target=s.system["p1"], k=k, numThreads=11, cpp_path=cpp_path)
    s.runPhysOptics(save=2, material_source='alu')
    
    s.PO.plotField(s.system["p1"].grid_x, s.system["p1"].grid_y, mode='Field', polar=True)
    
    s.nextPhysOptics(source=s.system["p1"], target=s.system["cam1"])
    s.runPhysOptics(save=2, material_source='vac')
    
    s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Field', polar=True)
    
    s.plotSystem(focus_1=False, focus_2=False)
    
    field = s.loadField(s.system["cam1"], mode='Field')
    
    s.plotter.plotBeam2D(s.system["cam1"], field=field, ff=foc_pri[2], vmin=-30, interpolation='lanczos')
    s.plotter.beamCut(s.system["cam1"], field=field)

    
if __name__ == "__main__":
    ex_DRO()

