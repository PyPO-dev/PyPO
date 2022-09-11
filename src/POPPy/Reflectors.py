# Standard imports
import math
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import cm
import scipy.interpolate as interp

# POPPy-specific modules
import src.POPPy.MatRotate as MatRotate

class Reflector(object):
    """
    Base class for reflector objects.
    All reflector types are quadric surfaces.
    """
    
    _counter = 0
    
    def __init__(self, a, b, cRot, name):
        
        self.a = a
        self.b = b
        
        self.focus_1 = np.ones(3) * float("NaN")
        self.focus_2 = np.ones(3) * float("NaN")
        
        self.cRot     = cRot
        
        Reflector._counter += 1
        self.id = Reflector._counter
        self.elType = "Reflector"
        self.name = name
        
        # Use internal list of references to iterable attributes
        self._iterList = [0 for _ in range(7)]
        
    def __str__(self):#, reflectorType, reflectorId, a, b, offTrans, offRot):
        s = """\n######################### REFLECTOR INFO #########################
Reflector type      : {}
Reflector ID        : {}

Focus 1 position    : [{:.3f}, {:.3f}, {:.3f}] [mm]
Focus 2 position    : [{:.3f}, {:.3f}, {:.3f}] [mm]
Vertex position     : [{:.3f}, {:.3f}, {:.3f}] [mm]

3D reflector parameters:
a                   : {} [mm]
b                   : {} [mm]
c                   : {} [mm]

COR [x, y, z]       : [{:.3f}, {:.3f}, {:.3f}] [mm]
######################### REFLECTOR INFO #########################\n""".format(self.reflectorType, self.reflectorId, 
                                                                               self.focus_1[0], self.focus_1[1], self.focus_1[2],
                                                                               self.focus_2[0], self.focus_2[1], self.focus_2[2],
                                                                               self.vertex[0], self.vertex[1], self.vertex[2],
                                                                               self.a, self.b, self.c,
                                                                               self.cRot[0], self.cRot[1], self.cRot[2])
        
        return s
    
    def __iter__(self):
        self._iterIdx = 0
        return self
    
    def __next__(self):
        if self._iterIdx < len(self._iterList):
            result = self._iterList[self._iterIdx]
            self._iterIdx += 1
            
            return result
        
        else:
            raise StopIteration
        
    
    #### SETTERS ###
    def set_cRot(self, cRot):
        self.cRot = cRot

    def setGrid(self, lims_x, lims_y, gridsize, gmode, trunc, flip, axis):
        """
        Takes limits in x and y directions, gridsize in x and y. 
        Per default, truncates the grid.
        
        @param lims_x The negative and positive extents of the reflector in the x-direction: list, arr
        @param lims_y The negative and positive extents of the reflector in the y-direction: list, arr
        @param gridsize Gridding resolutions along x and y axes: list, arr
        @param trunc Whether to truncate grid by elliptic aperture: bool
            TODO: make elliptic from circular truncation
        @param flip Whether normal vectors of reflectors point inside or outside: True/False
            'inside' (default) refers to the normal vectors pointing upward.
            'outside' refers to downward pointing normal vectors.
            TODO: fix normal vectors of Hyperbola
            TODO: test Ellipsoid normal vectors
        """
        
        if flip:
            mult = -1
        else:
            mult = 1

        self.shape = gridsize
        
        if gmode == 'xy':
            grid_x, grid_y = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            self.dx = grid_x[1,0] - grid_x[0,0]
            self.dy = grid_y[0,1] - grid_y[0,0]
            
            grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz = self.xyGrid(grid_x, grid_y)
            
            norm = np.sqrt(grid_nx**2 + grid_ny**2 + grid_nz**2)
            
            self.area = norm * self.dx * self.dy
                
            grid_nx *= mult / norm
            grid_ny *= mult / norm
            grid_nz *= mult / norm
            
        elif gmode == 'uv':
            # Assume now lims_x represent smallest aperture and largest aperture radii
            # Convert radii to u values
            lims_x = [self.r_to_u(lims_x[0], axis=axis), self.r_to_u(lims_x[1], axis=axis)]
            
            v_open = (lims_y[1] - lims_y[0]) / gridsize[1]
            
            #lims_y[1] -= v_open
            
            grid_u, grid_v = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            du = grid_u[1,0] - grid_u[0,0]
            dv = grid_v[0,1] - grid_v[0,0]

            grid_x, grid_y, grid_z, grid_nx, grid_ny, grid_nz, area = self.uvGrid(grid_u, grid_v, du, dv)

            self.area = area
            
            grid_nx *= mult
            grid_ny *= mult
            grid_nz *= mult

            range_x = grid_x[:,0]
            range_y = grid_y[0,:]

        if trunc:
            self.truncateGrid()
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.grid_nz = grid_nz

        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        self._iterList[4] = self.grid_nx
        self._iterList[5] = self.grid_ny
        self._iterList[6] = self.grid_nz
        
    def updateIterlist(self):
        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        self._iterList[4] = self.grid_nx
        self._iterList[5] = self.grid_ny
        self._iterList[6] = self.grid_nz
        
    def translateGrid(self, offTrans):
        self.grid_x += offTrans[0]
        self.grid_y += offTrans[1]
        self.grid_z += offTrans[2]
        
        self.focus_1 += offTrans
        self.focus_2 += offTrans
        self.vertex += offTrans
        
        self.updateIterlist()
        
    def rotateGrid(self, offRot, radians=False):
        gridRot = MatRotate.MatRotate(offRot, [self.grid_x, self.grid_y, self.grid_z], self.cRot, radians=radians)
        
        self.grid_x = gridRot[0]
        self.grid_y = gridRot[1]
        self.grid_z = gridRot[2]
        
        grid_nRot = MatRotate.MatRotate(offRot, [self.grid_nx, self.grid_ny, self.grid_nz], self.cRot, vecRot=True, radians=radians)
        
        self.grid_nx = grid_nRot[0]
        self.grid_ny = grid_nRot[1]
        self.grid_nz = grid_nRot[2]
        
        self.focus_1 = MatRotate.MatRotate(offRot, self.focus_1, self.cRot, radians=radians)
        self.focus_2 = MatRotate.MatRotate(offRot, self.focus_2, self.cRot, radians=radians)
        self.vertex = MatRotate.MatRotate(offRot, self.vertex, self.cRot, radians=radians)
        
        self.updateIterlist()
    '''
    def calcArea(self, grids_o):
        # Calculate surface approximations from oversized grids
        # Only for xy mode
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
        self.area = A
    '''
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
        
    def interpReflector(self, res, mode):
        skip = slice(None,None,res)
        
        if mode == 'z':
            posInterp = interp.bisplrep(self.grid_x, self.grid_y, self.grid_z, kx=3, ky=3, s=1e-6)
        
            nxInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nx.ravel()[skip], kx=3, ky=3, s=1e-6)
            nyInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_ny.ravel()[skip], kx=3, ky=3, s=1e-6)
            nzInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nz.ravel()[skip], kx=3, ky=3, s=1e-6)

        elif mode == 'x':
            posInterp = interp.bisplrep(self.grid_y, self.grid_z, self.grid_x, kx=3, ky=3, s=1e-6)
        
            nxInterp = interp.bisplrep(self.grid_y.ravel()[skip], self.grid_z.ravel()[skip], self.grid_nx.ravel()[skip], kx=3, ky=3, s=1e-6)
            nyInterp = interp.bisplrep(self.grid_y.ravel()[skip], self.grid_z.ravel()[skip], self.grid_ny.ravel()[skip], kx=3, ky=3, s=1e-6)
            nzInterp = interp.bisplrep(self.grid_y.ravel()[skip], self.grid_z.ravel()[skip], self.grid_nz.ravel()[skip], kx=3, ky=3, s=1e-6)
            
        elif mode == 'y':
            posInterp = interp.bisplrep(self.grid_z, self.grid_x, self.grid_y, kx=3, ky=3, s=1e-6)
        
            nxInterp = interp.bisplrep(self.grid_z.ravel()[skip], self.grid_x.ravel()[skip], self.grid_nx.ravel()[skip], kx=3, ky=3, s=1e-6)
            nyInterp = interp.bisplrep(self.grid_z.ravel()[skip], self.grid_x.ravel()[skip], self.grid_ny.ravel()[skip], kx=3, ky=3, s=1e-6)
            nzInterp = interp.bisplrep(self.grid_z.ravel()[skip], self.grid_x.ravel()[skip], self.grid_nz.ravel()[skip], kx=3, ky=3, s=1e-6)

        
        
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
    
    def __init__(self, a, b, cRot, name):
        Reflector.__init__(self, a, b, cRot, name)
        self.c = float("NaN")

        self.vertex = np.zeros(3)

        self.reflectorId = self.id
        self.reflectorType = "Paraboloid"
        
    def uvGrid(self, u, v, du, dv):
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
        
        area = u * np.sqrt(4 * self.b**2 * u**2 * np.cos(v)**2 + 4 * self.a**2 * u**2 * np.sin(v)**2 + self.a**2 * self.b**2) * du * dv
        
        return x, y, z, nx, ny, nz, area
    
    def xyGrid(self, x, y):
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
            
        #self.area = np.sqrt(1 + 4/self.a**4 * x**2 + 4/self.b**4 * y**2) * self.dx * self.dy
        
        return x, y, z, nx, ny, nz
    
    def r_to_u(self, r, axis='a'):
        """
        Convert aperture radius to u parameter.
        @param r Radius of aperture in mm
        @param axis Axis along which to set the radius. Default is 'a', semi major axis.
        
        @return u Value of u corresponding to r
        """
        if r == 0:
            r = np.finfo(float).eps
        
        if axis == 'a':
            u = r / self.a
        elif axis == 'b':
            u = r / self.b
            
        return u

class Hyperbola(Reflector):
    """
    Derived class from Reflector. Creates a Hyperboloid mirror.
    """
    
    def __init__(self, a, b, c, cRot, name, sec):
        Reflector.__init__(self, a, b, cRot, name)
        self.c = c
        
        self.vertex = np.array([0,0,self.c])
        
        self.reflectorId = self.id
        self.reflectorType = "Hyperboloid"
        if sec == 'upper':
            self.section = 1
        elif sec == 'lower':
            self.section = -1
    
    def uvGrid(self, u, v, du, dv):
        
        x = self.section * self.a * np.sqrt(u**2 - 1) * np.cos(v)
        y = self.section * self.b * np.sqrt(u**2 - 1) * np.sin(v)
        z = self.section * self.c * u
        
        prefac = self.section / np.sqrt(self.b**2 * self.c**2 * (u**2 - 1) * np.cos(v)**2 + self.a**2 * self.c**2 * (u**2 - 1) * np.sin(v)**2 + self.a**2 * self.b**2 * u**2)
        
        nx = -self.b * self.c * np.sqrt(u**2 - 1) * np.cos(v) * prefac
        ny = -self.a * self.c * np.sqrt(u**2 - 1) * np.sin(v) * prefac
        nz = self.a * self.b * u * prefac
        
        area = np.sqrt(self.b**2 * self.c**2 * (u**2 - 1) * np.cos(v)**2 + self.a**2 * self.c**2 * (u**2 - 1) * np.sin(v)**2 + self.a**2 * self.b**2 * u**2) * du * dv
        
        return x, y, z, nx, ny, nz, area
        
    
    def xyGrid(self, x, y):
        z = self.section * self.c * np.sqrt(x ** 2 / self.a ** 2 + y ** 2 / self.b ** 2 + 1)
        
        nx = self.section * 2 * x / self.a**2
        ny = self.section * 2 * y / self.b**2
        
        if hasattr(nx, 'shape'):
            #nz = -np.ones(nx.shape)
            nz = -self.section * 2 * z / self.c**2
            
        else:
            #nz = -1
            nz = -self.section * 2 * z / self.c**2
        
        return x, y, z, nx, ny, nz
    
    def r_to_u(self, r, axis='a'):
        if r == 0:
            r = 0.0001#np.finfo(float).eps
            
        if axis == 'a':
            u = np.sqrt((r / self.a)**2 + 1)
        elif axis == 'b':
            u = np.sqrt((r / self.b)**2 + 1)
            
        return u
        
class Ellipse(Reflector):
    """
    Derived class from Reflector. Creates an Ellipsoid mirror.
    """
    
    def __init__(self, a, b, c, cRot, name, ori):
        Reflector.__init__(self, a, b, cRot, name)
        self.c = c
        
        self.vertex = np.array([0,0,self.c])
        
        self.reflectorId = self.id
        self.reflectorType = "Ellipse"
        
        if ori == 'vertical':
            self.section = -1
        
        elif ori == 'horizontal':
            self.section = 1
    
    def uvGrid(self, u, v, du, dv):
        
        x = self.a * np.sqrt(u**2 - 1) * np.cos(v)
        y = self.b * np.sqrt(u**2 - 1) * np.sin(v)
        z = self.c * u
        
        prefac = 1 / np.sqrt(self.b**2 * self.c**2 * (u**2 - 1) * np.cos(v)**2 + self.a**2 * self.c**2 * (u**2 - 1) * np.sin(v)**2 + self.a**2 * self.b**2 * u**2)
        
        nx = -self.b * self.c * np.sqrt(u**2 - 1) * np.cos(v) * prefac
        ny = -self.a * self.c * np.sqrt(u**2 - 1) * np.sin(v) * prefac
        nz = self.a * self.b * u * prefac
        
        area = np.sqrt(self.b**2 * self.c**2 * (u**2 - 1) * np.cos(v)**2 + self.a**2 * self.c**2 * (u**2 - 1) * np.sin(v)**2 + self.a**2 * self.b**2 * u**2) * du * dv
        
        return x, y, z, nx, ny, nz, area
        
    
    def xyGrid(self, x, y):
        z = self.section * self.c * np.sqrt(1 - x ** 2 / self.a ** 2 - y ** 2 / self.b ** 2)
        
        nx = self.section * 2 * x / self.a**2
        ny = self.section * 2 * y / self.b**2
        
        if hasattr(nx, 'shape'):
            #nz = -np.ones(nx.shape)
            nz = self.section * 2 * z / self.c**2
            
        else:
            #nz = -1
            nz = self.section * 2 * z / self.c**2
        
        return x, y, z, nx, ny, nz
    
    def r_to_u(self, r, axis='a'):
        if axis == 'a':
            u = np.sqrt((r / self.a)**2 + 1)
        elif axis == 'b':
            u = np.sqrt((r / self.b)**2 + 1)
            
        return u
    
class Custom(Reflector):
    """
    Derived from reflector. Creates a custom reflector from supplied x,y,z and nx,ny,nz and area grids.
    The grids should be supplied in flattened format, in the custom/reflector folder
    Requires a gridsize.txt to be present, which contains the gridsizes for reshaping
    """
    
    def __init__(self, cRot, name, path):
        a = 0
        b = 0
        
        Reflector.__init__(self, a, b, cRot, name)
        self.c = 0
        
        self.vertex = np.array([0,0,self.c])
        
        self.reflectorId = self.id
        self.reflectorType = "Custom"
        
        gridsize = np.loadtxt(path + "gridsize.txt")
        gridsize = [int(gridsize[0]), int(gridsize[1])]
        self.shape = gridsize
            
        self.grid_x = np.loadtxt(path + "x.txt").reshape(gridsize)
        self.grid_y = np.loadtxt(path + "y.txt").reshape(gridsize)
        self.grid_z = np.loadtxt(path + "z.txt").reshape(gridsize)
        
        self.area = np.loadtxt(path + "A.txt").reshape(gridsize)
        
        self.grid_nx = np.loadtxt(path + "nx.txt").reshape(gridsize)
        self.grid_ny = np.loadtxt(path + "ny.txt").reshape(gridsize)
        self.grid_nz = np.loadtxt(path + "nz.txt").reshape(gridsize)

        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        self._iterList[4] = self.grid_nx
        self._iterList[5] = self.grid_ny
        self._iterList[6] = self.grid_nz
            
        

if __name__ == "__main__":
    print("These classes represent reflectors that can be used in POPPy simulations.")
    
    
    

