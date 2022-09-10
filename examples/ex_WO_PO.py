import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.Python.System import System
import matplotlib.pyplot as pt
from examples.BuildWO import MakeWO

def ex_DRO():
    """
    In this example script we will build the Dwingeloo Radio Observatory (DRO).
    The setup consists of a parabolic reflector and feed.
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
    lims_x_cam = [-30, 30]
    lims_y_cam = [-30, 30]
    gridsize_cam = [301, 301]
    #rotcam = np.array([0, 90, 0])
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam, lims_y_cam, gridsize_cam, center=center_cam, name = "cam1")
    #s.system["cam1"].rotateGrid(rotcam)
    #s.system["cam1"].translateGrid(trans_cam_cf)
    
    '''
    s.initRaytracer(nRays=1, nCirc=2, 
                 rCirc=2, div_ang_x=5, div_ang_y=5,
                 originChief=np.array([0,0,0]), 
                 tiltChief=np.array([0,0,0]), nomChief = np.array([1,0,0]))
    
    s.startRaytracer(surface="h1")
    s.startRaytracer(surface="e1")
    s.startRaytracer(surface="cam1")
    
    s.Raytracer.plotRays(mode='x')
    s.Raytracer.plotRays(mode='z', frame=-1)
    '''
    s.plotSystem(focus_1=False, focus_2=False, plotRaytrace=False)
    
    
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
    
    s.initPhysOptics(target=s.system["h1"], k=k, numThreads=11, cpp_path=cpp_path)
    '''
    s.runPhysOptics(save=2, material_source='alu')
    
    s.PO.plotField(s.system["h1"].grid_y, s.system["h1"].grid_z, mode='Ez', polar=False)

    s.nextPhysOptics(source=s.system["h1"], target=s.system["e1"])
    #s.nextPhysOptics(source=s.system["e1"], target=s.system["cam1"])
    s.runPhysOptics(save=2, material_source='alu')
    #s.PO.plotField(s.system["cam1"].grid_y, s.system["cam1"].grid_z, mode='Ez', polar=False)
    
    s.PO.plotField(s.system["e1"].grid_y, s.system["e1"].grid_x, mode='Ex', polar=False)

    s.nextPhysOptics(source=s.system["e1"], target=s.system["cam1"])
    s.runPhysOptics(save=2, material_source='vac')
    
    s.PO.plotField(s.system["cam1"].grid_y, s.system["cam1"].grid_x, mode='Ex', polar=False)
    '''
    field = s.loadField(s.system["cam1"], mode='Ex')
    s.PO.FF_fromFocus(s.system["cam1"].grid_y, s.system["cam1"].grid_x)

if __name__ == "__main__":
    ex_DRO()

