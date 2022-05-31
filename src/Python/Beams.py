import numpy as np
import matplotlib.pyplot as pt

class Beams(object):
    """
    Class that contains templates for commonly used beam patterns.
    Currently supports plane waves and Gaussian beams.
    
    Beams are always defined on rectangular grids, oriented in the xy plane and centered at (0,0,0).
    In order to translate and rotate beams, the positions of the grid are rotated. 
    The beam values are unchanged.
    """
    
    def __init__(self, x_lims, y_lims, gridsize, flip):
        self.cl = 299792458e3 # [mm]
        # Use internal list of references to iterable attributes
        self._iterList = [0 for _ in range(10)]
        
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
        Jx = np.zeros(self.Ex.ravel().shape).astype(complex)
        Jy = np.zeros(self.Ex.ravel().shape).astype(complex)
        Jz = np.zeros(self.Ex.ravel().shape).astype(complex)
        
        Mx = np.zeros(self.Ex.ravel().shape).astype(complex)
        My = np.zeros(self.Ex.ravel().shape).astype(complex)
        Mz = np.zeros(self.Ex.ravel().shape).astype(complex)
        
        print(type(My))
        
        if mode == 'None':
            for i,ex,ey,ez,hx,hy,hz in enumerate(zip(self.Ex.ravel(), self.Ey.ravel(), self.Ez.ravel(), self.Hx.ravel(), self.Hy.ravel(), self.Hz.ravel())):
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
            for i,(ex,ey,ez) in enumerate(zip(self.Ex.ravel(), self.Ey.ravel(), self.Ez.ravel())):
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
        
        print(type(self.My[0,0]))

class PlaneWave(Beams):
    
    def __init__(self, x_lims, y_lims, gridsize, pol, amp, phase, flip):
        Beams.__init__(self, x_lims, y_lims, gridsize, flip)
        
        self.Ex = pol[0] * amp * np.exp(1j * phase) * np.ones(self.grid_x.shape)
        self.Ey = pol[1] * amp * np.exp(1j * phase) * np.ones(self.grid_x.shape)
        self.Ez = pol[2] * amp * np.exp(1j * phase) * np.ones(self.grid_x.shape)
        
        self.Hx = np.zeros(self.grid_x.shape)
        self.Hy = np.zeros(self.grid_x.shape)
        self.Hz = np.zeros(self.grid_x.shape)
        
    def plotBeam(self):
        fig, ax = pt.subplots(1,2)
        ax[0].imshow(np.absolute(self.Ex))
        ax[1].imshow(np.angle(self.Ex))
        
        pt.show()
        
        
        
        

