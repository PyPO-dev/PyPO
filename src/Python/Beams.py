import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.Python.Colormaps as cmaps
import src.Python.Plotter as Plotter

# POPPy-specific modules
import src.Python.MatRotate as MatRotate

class Beams(object):
    """
    Class that contains templates for commonly used beam patterns.
    Currently supports plane waves and Gaussian beams.
    
    Beams are always defined on rectangular grids, oriented in the xy plane and centered at (0,0,0).
    In order to translate and rotate beams, the positions of the grid are rotated. 
    The beam values are unchanged.
    """
    
    compList_eh = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
    
    def __init__(self, x_lims, y_lims, gridsize, flip, name, cRot):
        self.cl = 299792458e3 # [mm]
        # Use internal list of references to iterable attributes
        self._iterList = [0 for _ in range(10)]
        self._compList = [0 for _ in range(6)]
        self.name = name
        self.cRot = cRot
        
        # Default to coherent field as input
        self.status = 'coherent'
        
        grid_x, grid_y = np.mgrid[x_lims[0]:x_lims[1]:gridsize[0]*1j, y_lims[0]:y_lims[1]:gridsize[1]*1j]
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = np.zeros(grid_x.shape)

        dx = grid_x[1,0] - grid_x[0,0]
        dy = grid_y[0,1] - grid_y[0,0]
        
        self.area = dx * dy * np.ones(grid_x.shape)
        
        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        if flip:
            self.norm = np.array([0,0,-1])
        else:
            self.norm = np.array([0,0,1])

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

    def calcJM(self, mode='None'):
        temp = np.zeros(self.grid_x.shape, dtype=np.ndarray)
        
        norm_arr = temp.fill(self.norm)

        e_arr = temp
        h_arr = temp
        
        self.Jx = np.zeros(self.grid_x.shape, dtype=complex)
        self.Jy = np.zeros(self.grid_x.shape, dtype=complex)
        self.Jz = np.zeros(self.grid_x.shape, dtype=complex)
        
        self.Mx = np.zeros(self.grid_x.shape, dtype=complex)
        self.My = np.zeros(self.grid_x.shape, dtype=complex)
        self.Mz = np.zeros(self.grid_x.shape, dtype=complex)

        comps_M = np.zeros((self.grid_x.shape[0], self.grid_x.shape[1], 3), dtype=complex)
        np.stack((self._compList[0], self._compList[1], self._compList[2]), axis=2, out=comps_M)
        
        comps_J = np.zeros((self.grid_x.shape[0], self.grid_x.shape[1], 3), dtype=complex)
        np.stack((self._compList[3], self._compList[4], self._compList[5]), axis=2, out=comps_J)

        if mode == 'None':
            M = -np.cross(self.norm, comps_M, axisb=2)
            J = np.cross(self.norm, comps_J, axisb=2)

            self.Mx = M[:,:,0]
            self.My = M[:,:,1]
            self.Mz = M[:,:,2]
            
            self.Jx = J[:,:,0]
            self.Jy = J[:,:,1]
            self.Jz = J[:,:,2]

        elif mode == 'PMC':
            M = -2 * np.cross(self.norm, comps_M, axisb=2)

            self.Mx = M[:,:,0]
            self.My = M[:,:,1]
            self.Mz = M[:,:,2]
            
        elif mode == 'PEC':
            J = 2 * np.cross(self.norm, comps_J, axisb=2)

            self.Jx = J[:,:,0]
            self.Jy = J[:,:,1]
            self.Jz = J[:,:,2]
        
        self._iterList[4] = self.Jx
        self._iterList[5] = self.Jy
        self._iterList[6] = self.Jz
        
        self._iterList[7] = self.Mx
        self._iterList[8] = self.My
        self._iterList[9] = self.Mz
        
    def translateBeam(self, offTrans):
        self.grid_x += offTrans[0]
        self.grid_y += offTrans[1]
        self.grid_z += offTrans[2]
        
        self.updateIterlist()
        
    def rotateBeam(self, offRot, radians=False):
        gridRot = MatRotate.MatRotate(offRot, [self.grid_x, self.grid_y, self.grid_z], self.cRot, radians=radians)
        
        self.grid_x = gridRot[0]
        self.grid_y = gridRot[1]
        self.grid_z = gridRot[2]
        
        self.norm = MatRotate.MatRotate(offRot, self.norm, self.cRot, vecRot=True, radians=radians)
        print(self.norm)
        
        self.updateIterlist()
        
    def updateIterlist(self):
        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z

class PlaneWave(Beams):
    def __init__(self, x_lims, y_lims, gridsize, pol, amp, phase, flip, name, cRot):
        Beams.__init__(self, x_lims, y_lims, gridsize, flip, name, cRot)
        
        amp_B = amp / self.cl * np.cross(self.norm, pol)
        
        if not isinstance(pol, str):
            for i, co in enumerate(self.compList_eh):
                if i <= 2:
                    self._compList[i] = pol[i] * amp * np.exp(1j * phase) * np.ones(self.grid_x.shape)
            
                else:
                    self._compList[i] = amp_B[i-3] * np.exp(1j * phase) * np.ones(self.grid_x.shape)
                
            self.Ex = self._compList[0]
            self.Ey = self._compList[1]
            self.Ez = self._compList[2]
        
            self.Hx = self._compList[3]
            self.Hy = self._compList[4]
            self.Hz = self._compList[5]
        
class PointSource(Beams):
    def __init__(self, area, pol, amp, phase, flip, name, n, cRot):
        x_lims = [-np.sqrt(area)*1.5, np.sqrt(area)*1.5]
        y_lims = [-np.sqrt(area)*1.5, np.sqrt(area)*1.5]
        
        gridsize = [n, n]
        
        Beams.__init__(self, x_lims, y_lims, gridsize, flip, name, cRot)
        
        idx = int((n - 1) / 2)
        
        field = np.zeros((n,n))
        field[idx, idx] = 1
        
        if not isinstance(pol, str):
            amp_B = amp / self.cl * np.cross(self.norm, pol)
            
            for i, co in enumerate(self.compList_eh):
                if i <= 2:
                    self._compList[i] = pol[i] * amp * np.exp(1j * phase) * field
            
                else:
                    self._compList[i] = amp_B[i-3] * np.exp(1j * phase) * field
                
            self.Ex = self._compList[0]
            self.Ey = self._compList[1]
            self.Ez = self._compList[2]
        
            self.Hx = self._compList[3]
            self.Hy = self._compList[4]
            self.Hz = self._compList[5]
            
        if pol == 'incoherent':
            # For incoherent source, immediately adjust _iterlist. No need for calculating J or M
            self._compList[0] = amp * np.exp(1j * phase) * field
            self.Field = self._compList[0]
            self._iterList[4] = self._compList[0]

            self._compList = self._compList[:len(self._compList)-5]
            self._iterList = self._iterList[:len(self._iterList)-5]
            
            self.status = 'incoherent'
            

class CustomBeam(Beams):
    def __init__(self, x_lims, y_lims, gridsize, comp, pathsToField, flip, name, cRot):
        idxComp = self.compList_eh.index(comp)
        
        Beams.__init__(self, x_lims, y_lims, gridsize, flip, name, cRot)
        
        rfield = np.loadtxt(pathsToField[0])
        ifield = np.loadtxt(pathsToField[1])
        
        field = rfield + 1j * ifield
        
        for i, co in enumerate(self.compList_eh):
            if i == idxComp:
                self._compList[i] = field
            
            else:
                self._compList[i] = np.zeros(self.grid_x.shape)
                
        self.Ex = self._compList[0]
        self.Ey = self._compList[1]
        self.Ez = self._compList[2]
        
        self.Hx = self._compList[3]
        self.Hy = self._compList[4]
        self.Hz = self._compList[5]

