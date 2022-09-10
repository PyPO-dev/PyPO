import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.Python.System import System
import matplotlib.pyplot as pt

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
    s.setCustomBeamPath("../custom/beam/240GHz/")
    s.setCustomReflPath("../custom/reflector/")
    
    # Warm focus
    wf = np.array([486.3974883317985, 0.0, 1371.340617233771])
    
    d_cryo = np.array([158.124, 0, 0])
    
    trans_cam_cf = np.zeros(3)
    trans_cam_cw = d_cryo

    # Instantiate camera surface at 300 mm from cf
    center_cam_cf = np.zeros(3)
    lims_x_cam_cf = [-30, 30]
    lims_y_cam_cf = [-30, 30]
    gridsize_cam_cf = [301, 301]
    rot_cam_cf = np.array([0, 90, 0])
    
    # camera at window
    center_cam_cw = np.zeros(3)
    lims_x_cam_cw = [-80, 80]
    lims_y_cam_cw = [-80, 80]
    gridsize_cam_cw = [403, 401]
    rot_cam_cw = np.array([0, 270, 0])
    
    # Add camera surface to optical system
    s.addCamera(lims_x_cam_cf, lims_y_cam_cf, 
                gridsize_cam_cf, center=center_cam_cf, 
                name = "cam_cf")
    
    s.system["cam_cf"].rotateGrid(rot_cam_cf)
    s.system["cam_cf"].translateGrid(trans_cam_cf)
    
    s.addCamera(lims_x_cam_cw, lims_y_cam_cw, 
                gridsize_cam_cw, center=center_cam_cw, 
                name = "cam_cw")
    
    s.system["cam_cw"].rotateGrid(rot_cam_cw)
    s.system["cam_cw"].translateGrid(trans_cam_cw)
    
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
    
    s.initPhysOptics(target=s.system["cam_cf"], k=k, numThreads=11, cpp_path=cpp_path)
    s.runPhysOptics(save=2, t_direction='backward')
    
    s.PO.plotField(s.system["cam_cf"].grid_y, s.system["cam_cf"].grid_z, mode='Ez', polar=False)
    
    s.nextPhysOptics(source=s.system["cam_cf"], target=s.system["cam_cw"])
    s.runPhysOptics(save=2, material_source='alu', t_direction='backward')
    
    s.PO.plotField(s.system["cam_cw"].grid_y, s.system["cam_cw"].grid_z, mode='Ez', polar=False)
    
    
if __name__ == "__main__":
    ex_DRO()

 
