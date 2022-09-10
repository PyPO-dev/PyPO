import numpy as np
import sys

sys.path.append('../')

import matplotlib.pyplot as pt

import src.Python.System as System



def ex_ASTE():
    """
    In this example script we will build the ASTE telescope.
    For this end, we build a paraboloid and hyperboloid and perform a raytrace through the setup.
    This example showcases how one can construct reflectors by using the functions
    supplied by POPPy
    """
    
    cpp_path = '../src/C++/'

    lam = 12.49135 # [mm]
    k = 2 * np.pi / lam
    
    # Primary parameters
    R_pri           = 5e3 # Radius in [mm]
    R_aper          = 200 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,3.5e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1         = [foc_pri, ver_pri]
    gridsize_p1     = [2501, 801] # The gridsizes along the u and v axes
    
    lims_r_p1       = [R_aper, R_pri]
    lims_v_p1       = [0, 2*np.pi]
    
    # Secondary parameters
    R_sec           = 310
    d_foc           = 5606.286
    foc_1_h1        = np.array([0,0,3.5e3])
    foc_2_h1        = np.array([0,0,3.5e3 -  d_foc])
    ecc_h1          = 1.08208248
    
    # Pack coefficients together for instantiating hyperbola: [focus 1, focus 2, eccentricity]
    coef_h1         = [foc_1_h1, foc_2_h1, ecc_h1]
    gridsize_h1     = [401, 201]
    
    lims_r_h1       = [0, R_sec]
    lims_v_h1       = [0, 2*np.pi]
    
    M_p = 25.366
    f_pri = 3500 #mm
    
    f_sys = M_p * f_pri

    # Initialize system
    s = System.System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "p1", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    s.addHyperbola(name = "h1", coef=coef_h1, lims_x=lims_r_h1, lims_y=lims_v_h1, gridsize=gridsize_h1, pmode='foc', gmode='uv')

    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    center_cam = foc_2_h1#foc_2_h1 # Place the camera at the z coordinate of the hyperbolic secondary focus
    lims_x_cam = [-1000, 1000]
    lims_y_cam = [-1000, 1000]
    gridsize_cam = [201, 201]

    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1")

    s.plotSystem(focus_1=True, focus_2=True)
    
    s.addPointSource(area=1, pol='incoherent', n=3, amp=1e16)

    offTrans_ps = np.array([0,0,1e16])
    s.inputBeam.translateBeam(offTrans=offTrans_ps)
    
    s.addPlotter(save='../images/')

    s.initPhysOptics(target=s.system["p1"], k=k, numThreads=11, cpp_path=cpp_path)
    #'''
    s.runPhysOptics(save=2, material_source='alu')
    
    s.PO.plotField(s.system["p1"].grid_x, s.system["p1"].grid_y, mode='Field', polar=True)
    
    s.nextPhysOptics(source=s.system["p1"], target=s.system["h1"])
    s.runPhysOptics(save=2, material_source='alu')
    
    s.PO.plotField(s.system["h1"].grid_x, s.system["h1"].grid_y, mode='Field', polar=True)

    s.nextPhysOptics(source=s.system["h1"], target=s.system["cam1"])
    s.runPhysOptics(save=2, material_source='vac')
    
    s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Field', polar=True)
    #'''
    field = s.loadField(s.system["cam1"], mode='Field')
    
    s.plotter.plotBeam2D(s.system["cam1"], field=field, ff=f_sys, vmin=-30, interpolation='lanczos')
    s.plotter.beamCut(s.system["cam1"], field=field)
    
if __name__ == "__main__":
    ex_ASTE()

