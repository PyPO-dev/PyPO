import numpy as np
import matplotlib.pyplot as pt

import src.Python.MatRotate as MatRotate

class Reflector(object):
    """
    Base class for reflector objects.
    """
    
    _counter = 0
    
    def __init__(self, a, b, cRot, offTrans, offRot):
        
        self.a = a
        self.b = b
        
        self.cRot     = cRot
        self.offTrans = offTrans
        self.offRot   = offRot
        
        Reflector._counter += 1
        self.id = Reflector._counter
        
    def __str__(self):#, reflectorType, reflectorId, a, b, offTrans, offRot):
        offRotDeg = np.degrees(self.offRot)
        s = """\n######################### REFLECTOR INFO #########################
Reflector type      : {}
Reflector ID        : {}

Semi-major axis     : {} [mm]
Semi-minor axis     : {} [mm]
Foci distance       : {} [mm]
Off [x, y, z]       : [{:.3f}, {:.3f}, {:.3f}] [mm]
Rot [rx, ry, rz]    : [{:.3f}, {:.3f}, {:.3f}] [deg]
COR [x, y, z]       : [{:.3f}, {:.3f}, {:.3f}] [mm]
######################### REFLECTOR INFO #########################\n""".format(self.reflectorType, self.reflectorId, 
                                                                               self.a, self.b, self.c,
                                              self.offTrans[0], self.offTrans[1], self.offTrans[2],
                                              offRotDeg[0], offRotDeg[1], offRotDeg[2],
                                              self.cRot[0], self.cRot[1], self.cRot[2])
        
        return s
    
    def rotateGrid(self):
        gridRot = MatRotate.MatRotate(self.offRot, [self.grid_x, self.grid_y, self.grid_z])
        
        self.grid_x = gridRot[0]
        self.grid_y = gridRot[1]
        self.grid_z = gridRot[2]
        
    
    def plotReflector(self):
        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})

        reflector = ax.plot_surface(self.grid_x, self.grid_y, self.grid_z,
                       linewidth=0, antialiased=False, alpha=0.5)
        
        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        pt.show()
        
        

class Parabola(Reflector):
    """
    Derived class from Reflector. Creates a paraboloid mirror.
    """
    
    def __init__(self, a = 1, b = 1, cRot = np.zeros(3), offTrans = np.zeros(3), offRot = np.zeros(3)):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = "N/A"
        
        self.reflectorId = self.id
        self.reflectorType = "Paraboloid"

    def setGrid(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_x**2 / self.a**2 + grid_y**2 / self.b**2
    

class Hyperbola(Reflector):
    """
    Derived class from Reflector. Creates a Hyperboloid mirror.
    """
    
    def __init__(self, a = 1, b = 1, c = 2, cRot = np.zeros(3), offTrans = np.zeros(3), offRot = np.zeros(3)):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = c
        
        self.reflectorId = self.id
        self.reflectorType = "Hyperboloid"
        
    def setGrid(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = self.c * np.sqrt(grid_x ** 2 / self.a ** 2 + grid_y ** 2 / self.b ** 2 + 1)
        
class Ellipse(Reflector):
    """
    Derived class from Reflector. Creates an Ellipsoid mirror.
    """
    
    def __init__(self, a = 2, b = 3, c = 5, cRot = np.zeros(3), offTrans = np.zeros(3), offRot = np.zeros(3)):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = c
        
        self.reflectorId = self.id
        self.reflectorType = "Ellipsoid"
        
    def setGrid(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = -self.c * (np.sqrt(1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2) + 1)
        


if __name__ == "__main__":
    rot = np.radians([25, 70.33, 4])
    cRot = np.array([25, 70.33, 4])
    
    p = Parabola(b = 1000, cRot = cRot, offRot = rot)
    h = Hyperbola()#offRot = rot)
    e = Ellipse(b = 1000)
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
    
    
    

