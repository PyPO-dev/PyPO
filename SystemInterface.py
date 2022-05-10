import numpy as np
import matplotlib.pyplot as pt

import src.Python.Reflectors as Reflectors
import src.Python.System as System

def SystemInterface():
    
    rot_p1 = np.radians([0, 0, 0])
    cRot_p1 = np.array([0, 0, 0])
    
    rot_h1 = np.radians([0, 0, 0])
    cRot_h1 = np.array([0, 0, 0])
    offTrans_h1 = np.array([0, 0, 1000])
    
    # Parabola p1 initialization
    lims_x_p1 = 500
    lims_y_p1 = 500
    
    range_x_p1 = np.linspace(-lims_x_p1, lims_x_p1, 1000)
    range_y_p1 = np.linspace(-lims_y_p1, lims_y_p1, 1000)
    
    grid_x_p1, grid_y_p1 = np.meshgrid(range_x_p1, range_y_p1)
    
    # Hyperbola h1 initialization
    lims_x_h1 = 100
    lims_y_h1 = 100
    
    range_x_h1 = np.linspace(-lims_x_h1, lims_x_h1, 100)
    range_y_h1 = np.linspace(-lims_y_h1, lims_y_h1, 100)
    
    grid_x_h1, grid_y_h1 = np.meshgrid(range_x_h1, range_y_h1)
    
    ####
    s = System.System()
    s.addParabola(name = "p1", a = 50, b = 50, cRot = cRot_p1, offRot = rot_p1)
    s.addHyperbola(name = "h1",a = 10, b = 10, c = 2, cRot = cRot_h1, offRot = rot_h1, offTrans = offTrans_h1)
    
    print(s.system["p1"])
    print(s.system["h1"])
    
    s.system["p1"].setGrid(grid_x_p1, grid_y_p1)
    s.system["p1"].rotateGrid()
    s.system["p1"].plotReflector()
    
    s.system["h1"].setGrid(grid_x_h1, grid_y_h1)
    s.system["h1"].rotateGrid()
    s.system["h1"].plotReflector(color='red')
    
    s.plotSystem()
    
if __name__ == "__main__":
    SystemInterface()
