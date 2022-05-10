import numpy as np
import matplotlib.pyplot as pt

import src.Python.Reflectors as Reflectors

def SystemInterface():
    
    rot = np.radians([25, 70.33, 4])
    cRot = np.array([25, 70.33, 4])
    
    p = Reflectors.Parabola(b = 1000, cRot = cRot, offRot = rot)
    h = Reflectors.Hyperbola()#offRot = rot)
    e = Reflectors.Ellipse(b = 1000)
    #print(help(e))
    print(p)
    print(h)
    print(e)
    limsa = 1
    limsb = 10
    
    
    range_x = np.linspace(-limsa, limsa, 100)
    range_y = np.linspace(-limsb, limsb, 100)
    
    grid_x, grid_y = np.meshgrid(range_x, range_y)
    
    p.setGrid(grid_x, grid_y)
    p.rotateGrid()
    p.plotReflector()
    
    h.setGrid(grid_x, grid_y)
    h.rotateGrid()
    h.plotReflector()
    
    e.setGrid(grid_x, grid_y)
    e.rotateGrid()
    e.plotReflector()
    
if __name__ == "__main__":
    SystemInterface()
