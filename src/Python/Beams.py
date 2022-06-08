import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.Python.Colormaps as cmaps
import src.Python.Plotter as Plotter

class Beams(object):
    """
    Class that contains templates for commonly used beam patterns.
    Currently supports plane waves and Gaussian beams.
    
    Beams are always defined on rectangular grids, oriented in the xy plane and centered at (0,0,0).
    In order to translate and rotate beams, the positions of the grid are rotated. 
    The beam values are unchanged.
    """
    
    compList_eh = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
    
    def __init__(self, x_lims, y_lims, gridsize, flip):
        self.cl = 299792458e3 # [mm]
        # Use internal list of references to iterable attributes
        self._iterList = [0 for _ in range(10)]
        self._compList = [0 for _ in range(6)]
        
        grid_x, grid_y = np.mgrid[x_lims[0]:x_lims[1]:gridsize[0]*1j, y_lims[0]:y_lims[1]:gridsize[1]*1j]
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = np.zeros(grid_x.shape)
        
        range_x = np.linspace(x_lims[0], x_lims[1], gridsize[0])
        range_y = np.linspace(y_lims[0], y_lims[1], gridsize[1])
        
        dx = range_x[1] - range_x[0]
        dy = range_y[1] - range_y[0]
        
        #dx = grid_x[1,0] - grid_x[0,0]
        #dy = grid_y[1,0] - grid_y[0,0]
        
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
    
    def transBeam(self, offTrans):
        self.grid_x += offTrans[0]
        self.grid_y += offTrans[1]
        self.grid_z += offTrans[2]
        
    def calcJM(self, mode='None'):
        Jx = np.zeros(self._compList[0].ravel().shape).astype(complex)
        Jy = np.zeros(self._compList[0].ravel().shape).astype(complex)
        Jz = np.zeros(self._compList[0].ravel().shape).astype(complex)
        
        Mx = np.zeros(self._compList[0].ravel().shape).astype(complex)
        My = np.zeros(self._compList[0].ravel().shape).astype(complex)
        Mz = np.zeros(self._compList[0].ravel().shape).astype(complex)
        
        print(type(My))
        
        if mode == 'None':
            for i,ex,ey,ez,hx,hy,hz in enumerate(zip(self._compList[0].ravel(), self._compList[1].ravel(), self._compList[2].ravel(), self._compList[3].ravel(), self._compList[4].ravel(), self._compList[5].ravel())):
                e_arr = np.array([ex,ey,ez])
                h_arr = np.array([hx,hy,hz])
                
                js = np.cross(self.norm, h_arr)
                ms = -np.cross(self.norm, e_arr)
                
                Jx.append(js[0])
                Jy.append(js[1])
                Jz.append(js[2])
                
                Mx.append(ms[0])
                My.append(ms[1])
                Mz.append(ms[2])
                
        elif mode == 'PMC':
            for i,(ex,ey,ez) in enumerate(zip(self._compList[0].ravel(), self._compList[1].ravel(), self._compList[2].ravel())):
                e_arr = np.array([ex,ey,ez])

                ms = -2 * np.cross(self.norm, e_arr)
                
                Mx[i] = ms[0]
                My[i] = ms[1]
                Mz[i] = ms[2]

        self.Jx = Jx.reshape(self.grid_x.shape).astype(complex)
        self.Jy = Jy.reshape(self.grid_x.shape).astype(complex)
        self.Jz = Jz.reshape(self.grid_x.shape).astype(complex)
        
        self.Mx = Mx.reshape(self.grid_x.shape).astype(complex)
        self.My = My.reshape(self.grid_x.shape).astype(complex)
        self.Mz = Mz.reshape(self.grid_x.shape).astype(complex)
        
        self._iterList[4] = self.Jx
        self._iterList[5] = self.Jy
        self._iterList[6] = self.Jz
        
        self._iterList[7] = self.Mx
        self._iterList[8] = self.My
        self._iterList[9] = self.Mz

class PlaneWave(Beams):
    def __init__(self, x_lims, y_lims, gridsize, pol, amp, phase, flip):
        Beams.__init__(self, x_lims, y_lims, gridsize, flip)
        
        for i, co in enumerate(self.compList_eh):
            if i <= 2:
                self._compList[i] = pol[i] * amp * np.exp(1j * phase) * np.ones(self.grid_x.shape)
            
            else:
                self._compList[i] = np.zeros(self.grid_x.shape)
                
        self.Ex = self._compList[0]
        self.Ey = self._compList[1]
        self.Ez = self._compList[2]
        
        self.Hx = self._compList[3]
        self.Hy = self._compList[4]
        self.Hz = self._compList[5]

class CustomBeam(Beams):
    def __init__(self, x_lims, y_lims, gridsize, comp, pathsToField, flip):
        idxComp = self.compList_eh.index(comp)
        
        Beams.__init__(self, x_lims, y_lims, gridsize, flip)
        
        rfield = np.loadtxt(pathsToField[0])
        ifield = np.loadtxt(pathsToField[1])
        
        field = rfield.reshape(self.grid_x.shape) + 1j * ifield.reshape(self.grid_x.shape)
        
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
        
    
        
        
        
        

