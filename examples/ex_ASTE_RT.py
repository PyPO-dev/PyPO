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
    R_aper          = 200 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,3.5e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1 = [foc_pri, ver_pri]
    
    # Secondary parameters
    R_sec           = 310
    d_foc           = 5606.286
    foc_1_h1        = np.array([0,0,3.5e3])
    foc_2_h1        = np.array([0,0,3.5e3 -  d_foc])
    ecc_h1          =  1.08208248
    gridsize_h1     = [301, 301]
    
    # Pack coefficients together for instantiating hyperbola: [focus 1, focus 2, eccentricity]
    coef_h1 = [foc_1_h1, foc_2_h1, ecc_h1]

    # Initialize system
    s = System.System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "p1", coef=coef_p1, mode='foc')
    s.addHyperbola(name = "h1", coef=coef_h1, mode='foc')
    
    # Calculate upper u-limits for uv initialization of surfaces
    u_max_p1 = s.system["p1"].r_to_u(R_pri)
    u_max_h1 = s.system["h1"].r_to_u(R_sec)
    
    # Calculate lower limit of u for parabola, to plot cabin hole
    u_min_p1 = s.system["p1"].r_to_u(R_aper)
    
    # Use uv definition of parabola & hyperbola for interpolation in ray trace
    lims_u_p1       = [u_min_p1, u_max_p1]
    lims_v_p1       = [0, 2*np.pi]
    gridsize_p1     = [501, 501] # The gridsizes along the u and v axes
    
    lims_u_h1       = [1, u_max_h1]
    lims_v_h1       = [0, 2*np.pi]

    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    center_cam = foc_2_h1 # Place the camera at the z coordinate of the hyperbolic secondary focus
    lims_x_cam = [-100, 100]
    lims_y_cam = [-100, 100]
    gridsize_cam = [101, 101]
    
    # Add camera surface to optical system
    s.addCamera(name = "cam1", center=center_cam)
    
    print(s.system["p1"])
    print(s.system["h1"])
    print(s.system["cam1"])

    s.system["p1"].setGrid(lims_u_p1, lims_v_p1, gridsize_p1, calcArea=False, trunc=False, param='uv')
    s.system["p1"].rotateGrid()
    s.system["p1"].plotReflector(focus_1=False, focus_2=True, norm=True)

    s.system["h1"].setGrid(lims_u_h1, lims_v_h1, gridsize_h1, orientation='outside', trunc=False, param='uv')
    s.system["h1"].rotateGrid()
    s.system["h1"].plotReflector(focus_1=True, focus_2=False, norm=True)

    s.system["cam1"].setGrid(lims_x_cam, lims_y_cam, gridsize_cam)
    s.system["cam1"].plotCamera()
    
    s.plotSystem(focus_1=True, focus_2=True)
    
    # Illuminate primary from above
    s.initRaytracer(rCirc=4500, NraysCirc=20, originChief=foc_pri, nomChief=np.array([0,0,-1]), div_ang_x=0, div_ang_y=0)
    
    s.startRaytracer(surface="p1")
    s.startRaytracer(surface="h1")
    s.startRaytracer(surface="cam1")
    
    s.Raytracer.plotRays(frame=-1, quiv=False)

    s.plotSystem(focus_1=False, focus_2=False, plotRaytrace=True)#, exclude=[0,1,2])
    
if __name__ == "__main__":
    ex_ASTE()

