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
    
    # Primary parameters
    R_pri           = 5e3 # Radius in [mm]
    R_aper          = 0#200 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,3.5e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1         = [foc_pri, ver_pri]
    gridsize_p1     = [201, 201] # The gridsizes along the u and v axes
    
    lims_r_p1       = [R_aper, R_pri]
    lims_v_p1       = [0, 2*np.pi]
    
    # Secondary parameters
    R_sec           = 310
    d_foc           = 5606.286
    foc_1_h1        = np.array([0,0,3.5e3])
    foc_2_h1        = np.array([0,0,3.5e3 -  d_foc])
    ecc_h1          =  1.08208248
    
    # Pack coefficients together for instantiating hyperbola: [focus 1, focus 2, eccentricity]
    coef_h1         = [foc_1_h1, foc_2_h1, ecc_h1]
    gridsize_h1     = [201, 201]
    
    lims_r_h1       = [R_aper, R_sec]
    lims_v_h1       = [0, 2*np.pi]

    # Initialize system
    s = System.System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "p1", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    s.addHyperbola(name = "h1", coef=coef_h1, lims_x=lims_r_h1, lims_y=lims_v_h1, gridsize=gridsize_h1, pmode='foc', gmode='uv')

    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    center_cam = foc_1_h1#foc_2_h1 # Place the camera at the z coordinate of the hyperbolic secondary focus
    lims_x_cam = [-1000, 1000]
    lims_y_cam = [-1000, 1000]
    gridsize_cam = [201, 201]
    
    # Add camera surface to optical system
    s.addCamera(name = "cam1", center=center_cam)
    
    print(s.system["p1"])
    print(s.system["h1"])
    print(s.system["cam1"])

    s.system["cam1"].setGrid(lims_x_cam, lims_y_cam, gridsize_cam)
    #s.system["cam1"].plotCamera()
    
    s.plotSystem(focus_1=True, focus_2=True)
    
    # Initialize a plane wave illuminating the primary from above. Place at height of primary focus.
    # Apply mask to plane wave grid corresponding to secondary mirror size. Make slightly oversized to minimize numerical
    # diffraction effects due to plane wave grid edges.
    lims_x_pw = [-6100, 6100]
    lims_y_pw = [-6100, 6100]
    gridsize_pw = [201, 201]
    
    lam = 1.2 * 1e2 # [mm]
    k = 2 * np.pi / lam
    
    s.addBeam(lims_x_pw, lims_y_pw, gridsize_pw, flip=True)


    s.inputBeam.calcJM(mode='PMC')
    
    offTrans_pw = foc_pri + np.array([0,0,100])
    s.inputBeam.transBeam(offTrans=offTrans_pw)
    
    s.initPhysOptics(target=s.system["p1"], k=k, numThreads=11)
    #s.initPhysOptics(target=s.system["cam1"], k=k, numThreads=11)
    s.runPhysOptics()
    #s.nextPhysOptics(target=s.system["h1"])
    #s.runPhysOptics()
    s.nextPhysOptics(target=s.system["cam1"])
    s.runPhysOptics(save=2)
    
    s.plotSystem(focus_1=False, focus_2=False)#, exclude=[0,1,2])

    #s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Ez')
    s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Ex')
    
if __name__ == "__main__":
    ex_ASTE()

