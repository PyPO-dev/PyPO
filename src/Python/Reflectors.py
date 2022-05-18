# Standard imports
import math
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.tri as mtri
import scipy.interpolate as interp

# POPPy-specific modules
import src.Python.MatRotate as MatRotate

class Reflector(object):
    """
    Base class for reflector objects.
    All reflector types are quadric surfaces.
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

Focus 1 position    : [{:.3f}, {:.3f}, {:.3f}] [mm]
Focus 2 position    : [{:.3f}, {:.3f}, {:.3f}] [mm]

Semi-major axis     : {} [mm]
Semi-minor axis     : {} [mm]
Foci distance       : {} [mm]
Off [x, y, z]       : [{:.3f}, {:.3f}, {:.3f}] [mm]
Rot [rx, ry, rz]    : [{:.3f}, {:.3f}, {:.3f}] [deg]
COR [x, y, z]       : [{:.3f}, {:.3f}, {:.3f}] [mm]
######################### REFLECTOR INFO #########################\n""".format(self.reflectorType, self.reflectorId, 
                                                                               self.focus_1[0], self.focus_1[1], self.focus_1[2],
                                                                               self.focus_2[0], self.focus_2[1], self.focus_2[2],
                                                                               self.a, self.b, self.c,
                                              self.offTrans[0], self.offTrans[1], self.offTrans[2],
                                              offRotDeg[0], offRotDeg[1], offRotDeg[2],
                                              self.cRot[0], self.cRot[1], self.cRot[2])
        
        return s
    
    def setGrid(self, lims_x, lims_y, gridsize, trunc=True):
        """
        Takes limits in x and y directions, gridsize in x and y. 
        Per default, truncates the grid.
        """
        
        range_x = np.linspace(lims_x[0], lims_x[1], gridsize[0])
        range_y = np.linspace(lims_y[0], lims_y[1], gridsize[1])
        
        self.dx = np.absolute(range_x[0] - range_x[1]) # Store dx for possible future usage
        self.dy = np.absolute(range_y[0] - range_y[1]) # Store dy for possible future usage
    
        grid_x, grid_y = np.meshgrid(range_x, range_y)
        
        # Create flattened structure, easier to work with
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        
        # Oversized grid not flattened to work easy with current algorithm
        range_x_o = np.linspace(lims_x[0] - self.dx/2, lims_x[1] + self.dx/2, gridsize[0]+1)
        range_y_o = np.linspace(lims_y[0] - self.dy/2, lims_y[1] + self.dy/2, gridsize[1]+1)
        grid_x_o, grid_y_o = np.meshgrid(range_x_o, range_y_o)
        
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        
        if self.reflectorType == "Paraboloid":
            grid_z_o = grid_x_o**2 / self.a**2 + grid_y_o**2 / self.b**2
            
            self.grid_z = grid_x**2 / self.a**2 + grid_y**2 / self.b**2 + self.offTrans[2]
            grid_nx = 2 * grid_x / self.a**2
            grid_ny = 2 * grid_y / self.b**2
            
        elif self.reflectorType == "Hyperboloid":
            grid_z_o = self.c * np.sqrt(grid_x_o**2 / self.a**2 + grid_y_o**2 / self.b**2 + 1)
            
            self.grid_z = self.c * np.sqrt(grid_x ** 2 / self.a ** 2 + grid_y ** 2 / self.b ** 2 + 1) + self.offTrans[2]
            grid_nx = self.c * 2 * grid_x / self.a**2 * (grid_x ** 2 / self.a ** 2 + grid_y ** 2 / self.b ** 2 + 1)**(-1/2)
            grid_ny = self.c * 2 * grid_y / self.b**2 * (grid_x ** 2 / self.a ** 2 + grid_y ** 2 / self.b ** 2 + 1)**(-1/2)
            
        elif self.reflectorType == "Ellipsoid":
            grid_z_o = -self.c * (np.sqrt(1 - grid_x_o ** 2 / self.a ** 2 - grid_y_o ** 2 / self.b ** 2) - 1)
            
            self.grid_z = -self.c * (np.sqrt(1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2) - 1) + self.offTrans[2]
            grid_nx = self.c * 2 * grid_x / self.a**2 * (1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2)**(-1/2)
            grid_ny = self.c * 2 * grid_y / self.b**2 * (1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2)**(-1/2)
            
        grid_nz = -np.ones(grid_nx.shape)
        
        self.grid_nx = grid_nx.flatten()
        self.grid_ny = grid_ny.flatten()
        self.grid_nz = grid_nz.flatten()
        
        norm = np.sqrt(self.grid_nx**2 + self.grid_ny**2 + self.grid_nz**2)
        
        self.grid_nx /= norm
        self.grid_ny /= norm
        self.grid_nz /= norm
        
        grids_o = [grid_x_o, grid_y_o, grid_z_o]
        self.calcArea(grids_o)

        if trunc:
            self.truncateGrid()

        self.grid_x = self.grid_x + self.offTrans[0]
        self.grid_y = self.grid_y + self.offTrans[1]
    
    def rotateGrid(self):
        gridRot = MatRotate.MatRotate(self.offRot, [self.grid_x, self.grid_y, self.grid_z], self.cRot)
        
        self.grid_x = gridRot[0]
        self.grid_y = gridRot[1]
        self.grid_z = gridRot[2]
        
        grid_nRot = MatRotate.MatRotate(self.offRot, [self.grid_nx, self.grid_ny, self.grid_nz], self.cRot, vecRot=True)
        
        self.grid_nx = grid_nRot[0]
        self.grid_ny = grid_nRot[1]
        self.grid_nz = grid_nRot[2]
        
        self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1, self.cRot)
        self.focus_2 = MatRotate.MatRotate(self.offRot, self.focus_2, self.cRot)
    
    def calcArea(self, grids_o):
        # Calculate surface approximations from oversized grids
        # Do this before grid truncation!

        x = grids_o[0]
        y = grids_o[1]
        z = grids_o[2]
        
        A = np.zeros((x.shape[0]-1, x.shape[1]-1))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v1 = np.zeros(3)
                v2 = np.zeros(3)
            
                v1[0] = x[i,j] - x[i+1,j+1]
                v1[1] = y[i,j] - y[i+1,j+1]
                v1[2] = z[i,j] - z[i+1,j+1]
                
                v2[0] = x[i,j+1] - x[i+1,j]
                v2[1] = y[i,j+1] - y[i+1,j]
                v2[2] = z[i,j+1] - z[i+1,j]
                
                outer = np.cross(v1, v2)
                
                A[i,j] = np.sqrt(np.dot(outer, outer))
        
        # Flatten array containing area elements now for easier analysis
        self.area = A.flatten()
    
    # Function to truncate with an ellipse in xy plane
    # TODO: plane with any orientation wrt xy plane
    def truncateGrid(self):
        lim_x_neg = np.amin(self.grid_x)
        lim_x_pos = np.amax(self.grid_x)
        
        lim_y_neg = np.amin(self.grid_y)
        lim_y_pos = np.amax(self.grid_y)
        
        to_check = 4 * (self.grid_x**2 / (lim_x_neg - lim_x_pos)**2 + self.grid_y**2 / (lim_y_neg - lim_y_pos)**2)
        
        idx_in_ellipse = to_check < 1
        
        self.grid_x = self.grid_x[idx_in_ellipse]
        self.grid_y = self.grid_y[idx_in_ellipse]
        self.grid_z = self.grid_z[idx_in_ellipse]
        
        self.grid_nx = self.grid_nx[idx_in_ellipse]
        self.grid_ny = self.grid_ny[idx_in_ellipse]
        self.grid_nz = self.grid_nz[idx_in_ellipse]
        
        self.area = self.area[idx_in_ellipse]
        
    def interpReflector(self, res=100):
        tcks_p = interp.bisplrep(self.grid_x, self.grid_y, self.grid_z)
        
        u_interp = interp.bisplrep(self.grid_x, self.grid_y, self.grid_nx)
        v_interp = interp.bisplrep(self.grid_x, self.grid_y, self.grid_ny)
        w_interp = interp.bisplrep(self.grid_x, self.grid_y, self.grid_nz)
        
        tcks = [tcks_p, u_interp, v_interp, w_interp]
        
        # Store interpolation parameters as members
        self.tcks = tcks
    
    def plotReflector(self, color='blue', returns=False, ax_append=False, focus_1=False, focus_2=False, fine=50):
        
        skip = slice(None,None,fine)
        
        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        reflector = ax_append.plot_trisurf(self.grid_x[skip], self.grid_y[skip], self.grid_z[skip],
                       linewidth=0, antialiased=False, alpha=0.5, color=color)
        
        if focus_1:
            ax_append.scatter(self.focus_1[0], self.focus_1[1], self.focus_1[2], color='black')
            
        if focus_2:
            ax_append.scatter(self.focus_2[0], self.focus_2[1], self.focus_2[2], color='black')

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
        self.c = float("NaN")
        
        if math.isclose(a, b, rel_tol=1e-6):
            self.focus_1 = np.array([offTrans[0], offTrans[1], offTrans[2] + a**2 / 4])
            #self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1)
            self.focus_2 = np.ones(len(self.focus_1)) * float("NaN")
        else:
            self.focus_1 = np.ones(len(offTrans)) * float("NaN")
            self.focus_2 = np.ones(len(offTrans)) * float("NaN")
            
            #self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1)
            #self.focus_2 = MatRotate.MatRotate(self.offRot, self.focus_2)

        self.reflectorId = self.id
        self.reflectorType = "Paraboloid"

class Hyperbola(Reflector):
    """
    Derived class from Reflector. Creates a Hyperboloid mirror.
    """
    
    def __init__(self, a, b, c, cRot = np.zeros(3), offTrans = np.zeros(3), offRot = np.zeros(3)):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = c
        
        self.ecc = np.sqrt(1 - self.a**2/self.c**2)
        # TODO: correct implementation of secondary ecc and focus
        
        if math.isclose(a, b, rel_tol=1e-6):
            self.focus_1 = np.array([offTrans[0], offTrans[1], offTrans[2] + self.c])
            self.focus_2 = np.array([offTrans[0], offTrans[1], offTrans[2] - self.c])
            
            #self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1)
            #self.focus_2 = MatRotate.MatRotate(self.offRot, self.focus_2)

        else:
            self.focus_1 = np.ones(len(offTrans)) * float("NaN")
            self.focus_2 = np.ones(len(offTrans)) * float("NaN")
            
            #self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1)
            #self.focus_2 = MatRotate.MatRotate(self.offRot, self.focus_2)
        
        self.reflectorId = self.id
        self.reflectorType = "Hyperboloid"
        
class Ellipse(Reflector):
    """
    Derived class from Reflector. Creates an Ellipsoid mirror.
    """
    
    def __init__(self, a, b, c, cRot, offTrans, offRot):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot)
        self.c = c
                
        self.ecc = np.sqrt(1 - self.b**2/self.a**2)
        
        if math.isclose(a, b, rel_tol=1e-6):
            self.focus_1 = np.array([offTrans[0], offTrans[1], offTrans[2] + self.a*(1 + self.ecc)])
            self.focus_2 = self.focus_1 + np.array([0,0,self.c])
            
            self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1)
            self.focus_2 = MatRotate.MatRotate(self.offRot, self.focus_2)
            
        else:
            self.focus_1 = np.ones(len(offTrans)) * float("NaN")
            self.focus_2 = np.ones(len(offTrans)) * float("NaN")
            
            #self.focus_1 = MatRotate.MatRotate(self.offRot, self.focus_1)
            #self.focus_2 = MatRotate.MatRotate(self.offRot, self.focus_2)
        
        self.reflectorId = self.id
        self.reflectorType = "Ellipsoid"
        


if __name__ == "__main__":
    print("These classes represent reflectors that can be used in POPPy simulations.")
    
    
    

