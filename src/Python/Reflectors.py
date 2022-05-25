# Standard imports
import math
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import cm
import scipy.interpolate as interp

# POPPy-specific modules
import src.Python.MatRotate as MatRotate

class Reflector(object):
    """
    Base class for reflector objects.
    All reflector types are quadric surfaces.
    """
    
    _counter = 0
    
    def __init__(self, a, b, cRot, offTrans, offRot, name):
        
        self.a = a
        self.b = b
        
        self.cRot     = cRot
        self.offTrans = offTrans
        self.offRot   = offRot
        
        Reflector._counter += 1
        self.id = Reflector._counter
        self.elType = "Reflector"
        self.name = name
        
    def __str__(self):#, reflectorType, reflectorId, a, b, offTrans, offRot):
        offRotDeg = np.degrees(self.offRot)
        s = """\n######################### REFLECTOR INFO #########################
Reflector type      : {}
Reflector ID        : {}

Focus 1 position    : [{:.3f}, {:.3f}, {:.3f}] [mm]
Focus 2 position    : [{:.3f}, {:.3f}, {:.3f}] [mm]

3D:
a                   : {} [mm]
b                   : {} [mm]
c                   : {} [mm]

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
    
    #### SETTERS ###
    def set_cRot(self, cRot):
        self.cRot = cRot
        
    def set_offTrans(self, offTrans):
        self.offTrans = offTrans
        
    def set_offRot(self, offRot):
        self.offRot = offRot
    
    def setGrid(self, lims_x, lims_y, gridsize, trunc=False, orientation='inside', calcArea=False, verbose=False, param='xy'):
        """
        Takes limits in x and y directions, gridsize in x and y. 
        Per default, truncates the grid.
        
        @param lims_x The negative and positive extents of the reflector in the x-direction: list, arr
        @param lims_y The negative and positive extents of the reflector in the x-direction: list, arr
        @param gridsize Gridding resolutions along x and y axes: list, arr
        @param trunc Whether to truncate grid by elliptic aperture: bool
            TODO: make elliptic from circular truncation
        @param orientation Whether normal vectors of reflectors point inside or outside: 'inside'/'outside'
            'inside' (default) refers to the normal vectors pointing upward.
            'outside' refers to downward pointing normal vectors.
            TODO: fix normal vectors of Hyperbola
            TODO: test Ellipsoid normal vectors
        """
        
        if orientation == 'inside':
            mult = 1
        else:
            mult = -1
            
        if verbose:
            print("Setting grids and normal vectors to {}".format(self.name))
        
        if param == 'xy':
            range_x = np.linspace(lims_x[0], lims_x[1], gridsize[0])
            range_y = np.linspace(lims_y[0], lims_y[1], gridsize[1])
            
            self.dx = np.absolute(range_x[0] - range_x[1]) # Store dx for possible future usage
            self.dy = np.absolute(range_y[0] - range_y[1]) # Store dy for possible future usage
            
            #grid_x, grid_y = np.meshgrid(range_x, range_y)
            grid_x, grid_y = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            if self.reflectorType == "Paraboloid":
                grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz = self.xyParabola(grid_x, grid_y)
                
            elif self.reflectorType == "Hyperboloid":
                grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz = self.xyHyperbola(grid_x, grid_y)
                
            elif self.reflectorType == "Ellipsoid":
                grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz = self.xyParabola(grid_x, grid_y)
                
            norm = np.sqrt(grid_nx**2 + grid_ny**2 + grid_nz**2)
                
            grid_nx *= mult / norm
            grid_ny *= mult / norm
            grid_nz *= mult / norm
            
        elif param == 'uv':
            range_u = np.linspace(lims_x[0], lims_x[1], gridsize[0])# TODO remove this hack
            range_v = np.linspace(lims_y[0], lims_y[1], gridsize[1])
            
            #grid_u, grid_v = np.meshgrid(range_u, range_v)
            grid_u, grid_v = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            if self.reflectorType == "Paraboloid":
                grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz = self.uvParabola(grid_u, grid_v)
                
            if self.reflectorType == "Hyperboloid":
                grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz = self.uvHyperbola(grid_u, grid_v)
            
            grid_nx *= mult
            grid_ny *= mult
            grid_nz *= mult
            
            '''
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax.plot_surface(grid_x, grid_y, grid_z)
            pt.show()
            '''
            range_x = grid_x[:,0]
            range_y = grid_y[0,:]
            
        # Oversized grid not flattened to work easy with current algorithm
        #range_x_o = np.linspace(lims_x[0] - self.dx/2, lims_x[1] + self.dx/2, gridsize[0]+1)
        #range_y_o = np.linspace(lims_y[0] - self.dy/2, lims_y[1] + self.dy/2, gridsize[1]+1)
        #grid_x_o, grid_y_o = np.meshgrid(range_x_o, range_y_o)
            
        elif self.reflectorType == "Ellipsoid":
            #grid_z_o = -self.c * (np.sqrt(1 - grid_x_o ** 2 / self.a ** 2 - grid_y_o ** 2 / self.b ** 2) - 1)
            
            grid_z = -self.c * (np.sqrt(1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2) - 1)
            grid_nx = self.c * 2 * grid_x / self.a**2 * (1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2)**(-1/2)
            grid_ny = self.c * 2 * grid_y / self.b**2 * (1 - grid_x ** 2 / self.a ** 2 - grid_y ** 2 / self.b ** 2)**(-1/2)

        #grids_o = [grid_x_o, grid_y_o, grid_z_o]

        if calcArea:
            #self.calcArea(grids_o, verbose)
            pass
        else:
            self.area = np.ones(grid_x.shape)

        if trunc:
            self.truncateGrid(verbose=verbose)
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.grid_nz = grid_nz
        
        self.grid_x += self.offTrans[0]
        self.grid_y += self.offTrans[1]
        self.grid_z += self.offTrans[2]
        
        # save edges of rectangular grid
        self.edge_x = range_x + self.offTrans[0]
        self.edge_y = range_y + self.offTrans[1]
    
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
    
    def calcArea(self, grids_o, verbose=False):
        # Calculate surface approximations from oversized grids
        # Do this before grid truncation!
        if verbose:
            print("Calculating surface area of elements on {}".format(self.name))
            
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
    def truncateGrid(self, verbose=False):
        if verbose:
            print("Truncating grids of {}".format(self.name))
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
        
    def interpReflector(self, res=1, verbose=False):
        skip = slice(None,None,res)
        if verbose:
            print("Calculating interpolation of {}".format(self.name))
            
        posInterp = interp.bisplrep(self.grid_x, self.grid_y, self.grid_z, kx=3, ky=3, s=0.000001)
        
        nxInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nx.ravel()[skip], kx=3, ky=3, s=0.000001)
        nyInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_ny.ravel()[skip], kx=3, ky=3, s=0.000001)
        nzInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nz.ravel()[skip], kx=3, ky=3, s=0.000001)
        
        tcks = [posInterp, nxInterp, nyInterp, nzInterp]
        # Store interpolations as members
        self.tcks = tcks

    def plotReflector(self, color='blue', returns=False, ax_append=False, focus_1=False, focus_2=False, fine=2, norm=False):
        
        skip = slice(None,None,fine)
        
        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        reflector = ax_append.plot_surface(self.grid_x[skip], self.grid_y[skip], self.grid_z[skip],
                       linewidth=0, antialiased=False, alpha=0.5, cmap=cm.cool)
        
        if focus_1:
            ax_append.scatter(self.focus_1[0], self.focus_1[1], self.focus_1[2], color='black')
            
        if focus_2:
            ax_append.scatter(self.focus_2[0], self.focus_2[1], self.focus_2[2], color='black')
            
        if norm:
            skipn = slice(None,None,10*fine)
            ax_append.quiver(self.grid_x[skipn,skipn], self.grid_y[skipn,skipn], self.grid_z[skipn,skipn], self.grid_nx[skipn,skipn], self.grid_ny[skipn,skipn], self.grid_nz[skipn,skipn], color='black', length=100, normalize=True)

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
    
    def __init__(self, a, b, cRot, offTrans, offRot, name):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot, name)
        self.c = float("NaN")
        
        if math.isclose(a, b, rel_tol=1e-6):
            self.focus_1 = np.array([offTrans[0], offTrans[1], offTrans[2] + a**2 / 4])
            self.focus_2 = np.ones(len(self.focus_1)) * float("NaN")
        else:
            self.focus_1 = np.ones(len(offTrans)) * float("NaN")
            self.focus_2 = np.ones(len(offTrans)) * float("NaN")

        self.reflectorId = self.id
        self.reflectorType = "Paraboloid"
        
    def uvParabola(self, u, v):
        """
        Create paraboloid from uv points.
        @param u Height parameter (number/array).
        @param v Angular parameter (number/array).
        
        @return x, y, z Parabola points (number/array)
        @return nx, ny, nz Parabola normal vector components (number/array)
        """
        
        x = self.a * u * np.cos(v)
        y = self.b * u * np.sin(v)
        z = u**2
        
        prefac = 1 / np.sqrt(4 * self.b**2 * u**2 * (np.cos(v)**2) + 4 * self.a**2 * u**2 * (np.sin(v))**2 + self.a**2 * self.b**2)
        
        nx = -2 * self.b * u * np.cos(v) * prefac
        ny = -2 * self.a * u * np.sin(v) * prefac
        nz = self.a * self.b * prefac
        
        return x, y, z, nx, ny, nz
    
    def xyParabola(self, x, y):
        """
        Create paraboloid from xy points.
        @param x X coordinate(s) (number/array).
        @param y Y coordinate(s) (number/array).
        
        @return x, y, z Parabola points (number/array)
        @return nx, ny, nz Parabola normal vector components (number/array)
        """
        
        z = x**2 / self.a**2 + y**2 / self.b**2

        nx = 2 * x / self.a**2
        ny = 2 * y / self.b**2
        
        if hasattr(nx, 'shape'):
            nz = -np.ones(nx.shape)
            
        else:
            nz = -1
        
        return x, y, z, nx, ny, nz

class Hyperbola(Reflector):
    """
    Derived class from Reflector. Creates a Hyperboloid mirror.
    """
    
    def __init__(self, a, b, c, cRot, offTrans, offRot, name):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot, name)
        self.c = c
        
        self.ecc = np.sqrt(1 - self.a**2/self.c**2)
        # TODO: correct implementation of secondary ecc and focus
        
        if math.isclose(a, b, rel_tol=1e-6):
            self.focus_1 = np.array([offTrans[0], offTrans[1], offTrans[2] + self.c])
            self.focus_2 = np.array([offTrans[0], offTrans[1], offTrans[2] - self.c])

        else:
            self.focus_1 = np.ones(len(offTrans)) * float("NaN")
            self.focus_2 = np.ones(len(offTrans)) * float("NaN")
        
        self.reflectorId = self.id
        self.reflectorType = "Hyperboloid"
    
    def uvHyperbola(self, u, v):
        
        x = self.a * np.sqrt(u**2 - 1) * np.cos(v)
        y = self.b * np.sqrt(u**2 - 1) * np.sin(v)
        z = self.c * u
        
        prefac = 1 / np.sqrt(self.b**2 * self.c**2 * (u**2 - 1) * np.cos(v)**2 + self.a**2 * self.c**2 * (u**2 - 1) * np.sin(v)**2 + self.a**2 * self.b**2 * u**2)
        
        nx = -self.b * self.c * np.sqrt(u**2 - 1) * np.cos(v) * prefac
        ny = -self.a * self.c * np.sqrt(u**2 - 1) * np.sin(v) * prefac
        nz = self.a * self.b * u * prefac
        
        return x, y, z, nx, ny, nz
        
    
    def xyHyperbola(self, x, y):
        z = self.c * np.sqrt(x ** 2 / self.a ** 2 + y ** 2 / self.b ** 2 + 1)
        '''
        nx = self.c * 2 * x / self.a**2 * (x ** 2 / self.a ** 2 + y ** 2 / self.b ** 2 + 1)**(-1/2)
        ny = self.c * 2 * y / self.b**2 * (x ** 2 / self.a ** 2 + y ** 2 / self.b ** 2 + 1)**(-1/2)
        '''
        
        nx = 2 * x / self.a**2
        ny = 2 * y / self.b**2
        
        if hasattr(nx, 'shape'):
            #nz = -np.ones(nx.shape)
            nz = -2 * z / self.c**2
            
        else:
            #nz = -1
            nz = -2 * z / self.c**2
        
        return x, y, z, nx, ny, nz
        
class Ellipse(Reflector):
    """
    Derived class from Reflector. Creates an Ellipsoid mirror.
    """
    
    def __init__(self, a, b, c, cRot, offTrans, offRot, name):
        Reflector.__init__(self, a, b, cRot, offTrans, offRot, name)
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
    
    
    

