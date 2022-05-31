import numpy as np
import matplotlib.pyplot as pt

import src.Python.System as System

def ex_DRO():
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
    """
    
    lam = 210 # [mm]
    k = 2 * np.pi / lam
    
    # Primary parameters
    R_pri           = 12.5e3 # Radius in [mm]
    R_aper          = 0 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,12e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1 = [foc_pri, ver_pri]

    # Initialize system
    s = System.System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "p1", coef=coef_p1, mode='foc')
    
    # Calculate upper u-limits for uv initialization of surfaces
    u_max_p1 = s.system["p1"].r_to_u(R_pri)
    
    # Calculate lower limit of u for parabola, to plot cabin hole
    u_min_p1 = s.system["p1"].r_to_u(R_aper)
    
    # Use uv definition of parabola & hyperbola for interpolation in ray trace
    lims_u_p1       = [u_min_p1, u_max_p1]
    lims_v_p1       = [0, 2*np.pi]
    gridsize_p1     = [201, 201] # The gridsizes along the u and v axes

    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    center_cam = foc_pri + np.array([0,0,0])
    lims_x_cam = [-2000, 2000]
    lims_y_cam = [-2000, 2000]
    gridsize_cam = [201, 201]
    
    # Add camera surface to optical system
    s.addCamera(name = "cam1", center=center_cam)
    
    print(s.system["p1"])
    print(s.system["cam1"])

    s.system["p1"].setGrid(lims_u_p1, lims_v_p1, gridsize_p1, calcArea=False, trunc=False, param='uv')
    s.system["p1"].rotateGrid()

    s.system["cam1"].setGrid(lims_x_cam, lims_y_cam, gridsize_cam)
    
    s.plotSystem(focus_1=True, focus_2=True)
    
    # Initialize a raytrace beam illuminating the parabolic reflector from above
    
    R_rt = R_pri - lam
    
    s.initRaytracer(rCirc=R_rt, NraysCirc=20, originChief=foc_pri, nomChief=np.array([0,0,-1]), div_ang_x=0, div_ang_y=0)
    
    s.startRaytracer(surface="p1")
    s.startRaytracer(surface="cam1")
    
    s.Raytracer.plotRays(frame=-1, quiv=False)
    
    s.plotSystem(focus_1=False, focus_2=False, plotRaytrace=True)#, exclude=[0,1,2])
    
if __name__ == "__main__":
    ex_DRO()

