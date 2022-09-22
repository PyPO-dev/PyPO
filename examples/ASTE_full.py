import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.POPPy.System import System
import matplotlib.pyplot as pt
from examples.BuildWO import MakeWO

def ASTE_full():
    """
    In this example script we will build the full setup at ASTE.
    """
    
    cpp_path = '../src/C++/'

    lam = 1.249135 # [mm]
    k = 2 * np.pi / lam

    # Initialize system
    s = System()
    
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s = MakeWO()
    s.setCustomBeamPath("../custom/beam/240GHz/")
    s.setCustomReflPath("../custom/reflector/")
    
    # Warm focus
    wf = np.array([486.3974883317985, 0.0, 1371.340617233771])
    
    d_cryo = np.array([158.124, 0, 0])
    
    trans_cam_cf = d_cryo# + np.array([-200, 0, 0])
    #trans_cam_cf = np.array([700, 0, 0])

    # Instantiate camera surface. 
    center_cam = wf
    lims_x_cam = [-5e2, 5e2]
    lims_y_cam = [-5e2, 5e2]
    gridsize_cam = [101, 101]
    #rotcam = np.array([0, 90, 0])
    
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
    gridsize_h1     = [601, 401]
    
    lims_r_h1       = [0, R_sec]
    lims_v_h1       = [0, 2*np.pi]
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "pri", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    s.addHyperbola(name = "sec", coef=coef_h1, lims_x=lims_r_h1, lims_y=lims_v_h1, gridsize=gridsize_h1, pmode='foc', gmode='uv')
    
    translate_ASTE = wf - foc_2_h1
    
    s.system["pri"].translateGrid(translate_ASTE)
    s.system["sec"].translateGrid(translate_ASTE)
    
    # Add camera surface to optical system in primary aperture
    h_cam = foc_pri + translate_ASTE
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=h_cam, name = "cam1")
    #s.system["cam1"].rotateGrid(rotcam)
    #s.system["cam1"].translateGrid(trans_cam_cf)
    #s.plotSystem(focus_1=True, focus_2=True, plotRaytrace=False)
    '''
    s.initRaytracer(nRays=10, nRing=4, 
                 a=2, b=2, angx=0, angy=0,
                 originChief=np.array([0,0,0]), 
                 tiltChief=np.array([0,90,0]))
    
    s.Raytracer.plotRays(mode='x', quiv=True)
    
    s.startRaytracer(target=s.system["h1"])
    s.startRaytracer(target=s.system["e1"])
    s.startRaytracer(target=s.system["sec"])
    s.startRaytracer(target=s.system["pri"])
    s.startRaytracer(target=s.system["cam1"])
    
    
    s.Raytracer.plotRays(mode='z', frame=-1)

    s.plotSystem(focus_1=True, focus_2=True, plotRaytrace=True)

    #s.plotSystem(focus_1=False, focus_2=False, plotRaytrace=False)
    
    '''
    bpath = '240GHz/'

    
    lims_x = [-80, 80]
    lims_y = [-80, 80]
    gridsize_beam = [403, 401]
    
    s.addBeam(lims_x=lims_x, lims_y=lims_y, gridsize=gridsize_beam, name='240.txt', beam='custom', comp='Ez')
    
    beam_rot = np.array([0, 90, 0])
    d_cryo = np.array([158.124, 0, 0])
    
    s.inputBeam.rotateBeam(beam_rot)
    s.inputBeam.translateBeam(d_cryo)

    s.inputBeam.calcJM(mode='PMC')

    s.addPlotter(save='../images/')
    
    lim_ff = 3 * 31 # as
    center_ff = np.zeros(3)
    lims_x_ff = [-lim_ff, lim_ff]
    lims_y_ff = [-lim_ff, lim_ff]
    gridsize_ff = [201, 201]
    
    s.addCamera(lims_x_ff, lims_y_ff, gridsize_ff, center=center_ff, name = "ff", gmode='AoE', units=['as', 'mm'])
    
    s.initPhysOptics(target=s.system["h1"], k=k, numThreads=11, cpp_path=cpp_path)
    
    
    #s.initPhysOptics(k=k, numThreads=11, cpp_path=cpp_path, cont=True)
    #s.folderffPhysOptics(folder='ASTE/pri/', source=s.system["pri"], target=s.system["ff"])
    #s.runPhysOptics(save=2, material_source='vac', prop_mode=1)
    #s.runPhysOptics(save=2, material_source='alu')
    #s.PO.plotField(s.system["pri"].grid_y, s.system["pri"].grid_x, mode='Ex', polar=True, show=False)
    '''
    s.runPhysOptics(save=2, material_source='alu', folder='ASTE/h1/')
    s.PO.plotField(s.system["h1"].grid_y, s.system["h1"].grid_z, mode='Ez', polar=False, show=False, save="h1")

    s.nextPhysOptics(source=s.system["h1"], target=s.system["e1"])
    s.runPhysOptics(save=2, material_source='alu', folder='ASTE/e1/')
    s.PO.plotField(s.system["e1"].grid_y, s.system["e1"].grid_x, mode='Ez', polar=False, show=False, save="e1")

    s.nextPhysOptics(source=s.system["e1"], target=s.system["sec"])
    s.runPhysOptics(save=2, material_source='alu', folder='ASTE/sec/')
    s.PO.plotField(s.system["sec"].grid_y, s.system["sec"].grid_x, mode='Ex', polar=True, show=False, save="sec")
    
    s.nextPhysOptics(source=s.system["sec"], target=s.system["pri"])
    s.runPhysOptics(save=2, material_source='alu', folder='ASTE/pri/')
    s.PO.plotField(s.system["pri"].grid_y, s.system["pri"].grid_x, mode='Ex', polar=True, show=False, save="pri")
    
    s.system["pri"].homeReflector()
    
    s.ffPhysOptics(source=s.system["pri"], target=s.system["ff"])
    s.runPhysOptics(save=2, material_source='vac', prop_mode=1)
    
    #s.nextPhysOptics(source=s.system["pri"], target=s.system["cam1"])
    
    #s.runPhysOptics(save=2, material_source='vac', folder='ASTE/cam1/')
    '''
    #s.PO.plotField(s.system["cam1"].grid_y, s.system["cam1"].grid_x, mode='Ey', polar=False, show=False, save="cam1")
    
    

    field = s.loadField(s.system["ff"], mode='Ex')
    field2 = s.loadField(s.system["ff"], mode='Ey')
    
    s.plotter.plotBeam2D(s.system["ff"], field=field, vmin=-30, interpolation='none', units='as', save=True, project='')
    s.plotter.beamCut(s.system["ff"], field=field, cross=field2, save=True)

if __name__ == "__main__":
    ASTE_full()

 
