import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

import src.POPPy.System as System

def ex_ASTE():
    """
    In this example script we will build the ASTE telescope.
    For this end, we build a paraboloid and hyperboloid and perform a raytrace through the setup.
    This example showcases how one can construct reflectors by using the functions
    supplied by POPPy
    """
    
    # Primary parameters
    R_pri           = 5e3 # Radius in [mm]
    R_aper          = 200 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,3.5e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1         = [foc_pri, ver_pri]
    gridsize_p1     = [501, 501] # The gridsizes along the u and v axes
    
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
    gridsize_h1     = [301, 301]
    
    lims_r_h1       = [0, R_sec]
    lims_v_h1       = [0, 2*np.pi]
    
    # Initialize system
    s = System.System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "pri", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    s.addHyperbola(name = "sec", coef=coef_h1, lims_x=lims_r_h1, lims_y=lims_v_h1, gridsize=gridsize_h1, pmode='foc', gmode='uv')

    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    center_cam = foc_2_h1 # Place the camera at the z coordinate of the hyperbolic secondary focus
    lims_x_cam = [-100, 100]
    lims_y_cam = [-100, 100]
    gridsize_cam = [101, 101]
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam")
    
    s.plotSystem(focus_1=True, focus_2=True)
    
    # Illuminate primary from above
    s.initRaytracer(nRays=20, nRing=10, a=R_pri, b=R_pri, originChief=foc_pri, tiltChief=np.array([0,180,0]))
    s.Raytracer.plotRays(frame=0, quiv=False)
    
    s.startRaytracer(target=s.system["pri"])
    s.startRaytracer(target=s.system["sec"])
    s.startRaytracer(target=s.system["cam"])
    
    s.Raytracer.plotRays(frame=-1, quiv=False)

    s.plotSystem(focus_1=False, focus_2=False, plotRaytrace=True)#, exclude=[0,1,2])
    
if __name__ == "__main__":
    ex_ASTE()

