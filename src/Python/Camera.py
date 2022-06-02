import numpy as np
import matplotlib.pyplot as pt
import scipy.interpolate as interp

import src.Python.MatRotate as MatRotate

class Camera(object):
    
    def __init__(self, center, name):
        self.center = center
        self.cRot = center
        
        self.elType = "Camera"
        self.name = name
        
        # Use internal list of references to iterable attributes
        self._iterList = [0 for _ in range(7)]
        
    def __str__(self):
        s = """\n######################### CAMERA INFO #########################
Center position     : [{:.3f}, {:.3f}, {:.3f}] [mm]
######################### CAMERA INFO #########################\n""".format(self.center[0], self.center[1], self.center[2])
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
        
    def setGrid(self, lims_x, lims_y, gridsize):
        grid_x, grid_y = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
        
        dx = grid_x[1,0] - grid_x[0,0]
        dy = grid_y[0,1] - grid_y[0,0]
        
        self.grid_x = grid_x + self.center[0]
        self.grid_y = grid_y + self.center[1]
        self.grid_z = np.zeros(self.grid_x.shape) + self.center[2]
        
        self.grid_nx = np.zeros(self.grid_x.shape)
        self.grid_ny = np.zeros(self.grid_y.shape)
        self.grid_nz = np.ones(self.grid_z.shape)
        
        self.area = np.ones(self.grid_x.shape) * dx * dy
        
        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        self._iterList[4] = self.grid_nx
        self._iterList[5] = self.grid_ny
        self._iterList[6] = self.grid_nz
       
    def set_cRot(self, cRot):
        self.cRot = cRot   
    
    def translateGrid(self, offTrans):
        self.grid_x += offTrans[0]
        self.grid_y += offTrans[1]
        self.grid_z += offTrans[2]
        
        self.center += offTrans
        
    def rotateGrid(self, offRot):
        gridRot = MatRotate.MatRotate(offRot, [self.grid_x, self.grid_y, self.grid_z], self.cRot)
        
        self.grid_x = gridRot[0]
        self.grid_y = gridRot[1]
        self.grid_z = gridRot[2]
        
        grid_nRot = MatRotate.MatRotate(offRot, [self.grid_nx, self.grid_ny, self.grid_nz], self.cRot, vecRot=True)
        
        self.grid_nx = grid_nRot[0]
        self.grid_ny = grid_nRot[1]
        self.grid_nz = grid_nRot[2]
        
        self.center = MatRotate.MatRotate(offRot, self.center, self.cRot)
    
    def interpCamera(self, res=100):
        skip = slice(None,None,res)

        posInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_z.ravel()[skip])
        
        nxInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nx.ravel()[skip])
        nyInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_ny.ravel()[skip])
        nzInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nz.ravel()[skip])
        
        tcks = [posInterp, nxInterp, nyInterp, nzInterp]
        
        # Store interpolations as members
        self.tcks = tcks
        
    def plotCamera(self, color='gold', returns=False, ax_append=False, norm=False):
        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        camera = ax_append.plot_surface(self.grid_x, self.grid_y, self.grid_z,
                       linewidth=0, antialiased=False, alpha=0.5, color=color)

        if norm:
            skipn = slice(None,None,10*fine)
            ax_append.quiver(self.grid_x[skipn], self.grid_y[skipn], self.grid_z[skipn], self.grid_nx[skipn], self.grid_ny[skipn], self.grid_nz[skipn], color='black', length=100, normalize=True)

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
