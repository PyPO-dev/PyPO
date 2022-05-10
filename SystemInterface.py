import numpy as np
import matplotlib.pyplot as pt

import src.Python.Reflectors as Reflectors
import src.Python.System as System

def SystemInterface():
    
    rot_p1 = np.radians([0, 0, 0])
    cRot_p1 = np.array([0, 0, 0])
    
    rot_e1 = np.radians([0, 45, 0])
    offTrans_e1 = np.array([0, 0, 1000])
    cRot_e1 = offTrans_e1
    
    # Parabola p1 initialization
    lims_x_p1 = 500
    lims_y_p1 = 500
    
    range_x_p1 = np.linspace(-lims_x_p1, lims_x_p1, 1000)
    range_y_p1 = np.linspace(-lims_y_p1, lims_y_p1, 1000)
    
    grid_x_p1, grid_y_p1 = np.meshgrid(range_x_p1, range_y_p1)
    
    # Hyperbola h1 initialization
    lims_x_e1 = 100
    lims_y_e1 = 100
    
    range_x_e1 = np.linspace(-lims_x_e1, lims_x_e1, 500)
    range_y_e1 = np.linspace(-lims_y_e1, lims_y_e1, 500)
    
    grid_x_e1, grid_y_e1 = np.meshgrid(range_x_e1, range_y_e1)
    
    ####
    s = System.System()
    s.addParabola(name = "p1", a = 50, b = 50, cRot = cRot_p1, offRot = rot_p1)
    s.addEllipse(name = "e1",a = 100, b = 100, c = 50, cRot = cRot_e1, offRot = rot_e1, offTrans = offTrans_e1)
    
    print(s.system["p1"])
    print(s.system["e1"])
    
    s.system["p1"].setGrid(grid_x_p1, grid_y_p1)
    s.system["p1"].rotateGrid()
    s.system["p1"].plotReflector()
    
    s.system["e1"].setGrid(grid_x_e1, grid_y_e1)
    s.system["e1"].rotateGrid()
    s.system["e1"].plotReflector(color='red')
    
    s.plotSystem()
    
if __name__ == "__main__":
    SystemInterface()
