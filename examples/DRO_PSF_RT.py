import numpy as np
import sys
sys.path.append('../')

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
    R_aper          = 300 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,12e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1 = [foc_pri, ver_pri]

    lims_r_p1       = [R_aper, R_pri]
    lims_v_p1       = [0, 2*np.pi]
    
    gridsize_p1     = [801, 501] # The gridsizes along the x and y axes
    
    coef_p1 = [foc_pri, ver_pri]

    # Initialize system
    s = System.System()

    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name="p1", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    
    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    center_cam = foc_pri
    lims_x_cam = [-20, 20]
    lims_y_cam = [-20, 20]
    gridsize_cam = [201, 201]
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1")

    s.plotSystem(focus_1=True, focus_2=True)
    
    # Initialize a raytrace beam illuminating the parabolic reflector from above
    
    R_rt = R_pri - lam
    
    s.initRaytracer(rCirc=R_rt, nRays=20, nCirc=10, originChief=foc_pri, nomChief=np.array([0,0,-1]), div_ang_x=0, div_ang_y=0)
    
    s.startRaytracer(surface="p1")
    s.startRaytracer(surface="cam1")
    
    s.Raytracer.plotRays(frame=-1, quiv=False)
    
    s.plotSystem(focus_1=False, focus_2=False, plotRaytrace=True)#, exclude=[0,1,2])
    
if __name__ == "__main__":
    ex_DRO()

