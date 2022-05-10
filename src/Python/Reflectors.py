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
        
    
    def plotReflector(self, color='blue', returns=False, ax_append=False):
        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        reflector = ax_append.plot_surface(self.grid_x, self.grid_y, self.grid_z,
                       linewidth=0, antialiased=False, alpha=0.5, color=color)
        
        if not returns:
            ax_append.set_ylabel(r"$y$ / [mm]", labelpad=20)
            ax_append.set_xlabel(r"$x$ / [mm]", labelpad=10)
            ax_append.set_zlabel(r"$z$ / [mm]", labelpad=50)
            world_limits = ax_append.get_w_lims()
            ax_append.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
            ax_append.tick_params(axis='x', which='major', pad=-3)
            pt.show()
            
        else:
            return ax_append
        
        

class Parabola(Reflector):
    """
    Derived class from Reflector. Creates a paraboloid mirror.
    """
    
    def __init__(self, a, b, cRot, offTrans, offRot):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = "N/A"
        
        self.reflectorId = self.id
        self.reflectorType = "Paraboloid"

    def setGrid(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_x**2 / self.a**2 + grid_y**2 / self.b**2 + self.offTrans[2]
        
        self.grid_x = self.grid_x + self.offTrans[0]
        self.grid_y = self.grid_y + self.offTrans[1]
    

class Hyperbola(Reflector):
    """
    Derived class from Reflector. Creates a Hyperboloid mirror.
    """
    
    def __init__(self, a, b, c, cRot = np.zeros(3), offTrans = np.zeros(3), offRot = np.zeros(3)):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = c
        
        self.reflectorId = self.id
        self.reflectorType = "Hyperboloid"
        
    def setGrid(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = self.c * np.sqrt(grid_x ** 2 / self.a ** 2 + grid_y ** 2 / self.b ** 2 + 1) + self.offTrans[2] - 1
        
        self.grid_x = self.grid_x + self.offTrans[0]
        self.grid_y = self.grid_y + self.offTrans[1]
        
class Ellipse(Reflector):
    """
    Derived class from Reflector. Creates an Ellipsoid mirror.
    """
    
    def __init__(self, a, b, c, cRot, offTrans, offRot):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = c
        
        self.reflectorId = self.id
        self.reflectorType = "Ellipsoid"
        
    def setGrid(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = -self.c * (np.sqrt(1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2) + 1) + self.offTrans[2]
        
        self.grid_x = self.grid_x + self.offTrans[0]
        self.grid_y = self.grid_y + self.offTrans[1]
        


if __name__ == "__main__":
    print("These classes represent reflectors that can be used in POPPy simulations.")
    
    
    

