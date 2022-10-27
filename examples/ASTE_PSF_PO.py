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
    
    cpp_path = '../src/C++/'

    lam = 1.249135 # [mm]
    k = 2 * np.pi / lam
    
    # Primary parameters
    R_pri           = 5e3 # Radius in [mm]
    R_aper          = 200 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,3.5e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1         = [foc_pri, ver_pri]
    gridsize_p1     = [3501, 1501] # The gridsizes along the u and v axes
    
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
    gridsize_h1     = [801, 501]
    
    lims_r_h1       = [0, R_sec]
    lims_v_h1       = [0, 2*np.pi]
    
    M_p = 25.366
    f_pri = 3500 #mm
    
    f_sys = M_p * f_pri

    # Initialize system
    s = System.System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "pri", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    s.addHyperbola(name = "sec", coef=coef_h1, lims_x=lims_r_h1, lims_y=lims_v_h1, gridsize=gridsize_h1, pmode='foc', gmode='uv')

    # Make far-field camera
    center_cam = np.zeros(3)
    
    lim = 3 * 31
    
    lims_x_cam = [-lim, lim]
    lims_y_cam = [-lim, lim]
    gridsize_cam = [201, 201]
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1", gmode='AoE', units=['as', 'mm'])
    
    s.plotSystem(focus_1=True, focus_2=True)
    s.addPlotter(save='../images/')
    s.addPointSource(area=1, pol=np.array([1,0,0]), n=51, amp=1, units='mm')
    s.inputBeam.calcJM(mode='PMC')
    
    fieldps = [s.inputBeam.Ex, 'Ex']
    s.plotter.plotBeam2D(s.inputBeam, field=fieldps, vmin=-30, interpolation='none', units='mm', project='xy', save=True)
    offTrans_ps = foc_2_h1
    s.inputBeam.translateBeam(offTrans=offTrans_ps)

    s.initPhysOptics(target=s.system["sec"], k=k, numThreads=11, cpp_path=cpp_path)
    s.runPhysOptics(save=2, material_source='alu')

    s.nextPhysOptics(source=s.system["sec"], target=s.system["pri"])
    s.runPhysOptics(save=2, material_source='alu')

    s.ffPhysOptics(source=s.system["pri"], target=s.system["cam1"])
    s.runPhysOptics(save=2, material_source='vac', prop_mode=1)

    field = s.loadField(s.system["cam1"], mode='Ex')
    field2 = s.loadField(s.system["cam1"], mode='Ey')
    
    s.plotter.plotBeam2D(s.system["cam1"], field=field, vmin=-30, interpolation='lanczos', units='as', save=True)
    s.plotter.beamCut(s.system["cam1"], field=field, cross=field2, save=True)
    
if __name__ == "__main__":
    ex_ASTE()

