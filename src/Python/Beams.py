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
    
    def __init__(self, x_lims, y_lims, gridsize):
        self.cl = 299792458e3 # [mm]
        
        grid_x, grid_y = np.mgrid[x_lims[0]:x_lims[1]:gridsize[0]*1j, y_lims[0]:y_lims[1]:gridsize[1]*1j]
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = np.zeros(grid_x.shape)
        
        self.norm = np.array([0,0,1])
        
    def calc_Js(self, mode='None'):
        Jx = []
        Jy = []
        Jz = []
        
        Mx = []
        My = []
        Mz = []
        
        if mode == 'None':
            for ex,ey,ez,hx,hy,hz in zip(self.Ex.ravel(), self.Ey.ravel(), self.Ez.ravel(), self.Hx.ravel(), self.Hy.ravel(), self.Hz.ravel()):
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
            for ex,ey,ez in zip(self.Ex.ravel(), self.Ey.ravel(), self.Ez.ravel()):
                e_arr = np.array([ex,ey,ez])
                
                js = 0
                ms = -2 * np.cross(self.norm, e_arr)
                
                Jx.append(js)
                Jy.append(js)
                Jz.append(js)
                
                Mx.append(ms[0])
                My.append(ms[1])
                Mz.append(ms[2])
                
        Jx = np.array(Jx).reshape(self.grid_x.shape)
        Jy = np.array(Jy).reshape(self.grid_x.shape)
        Jz = np.array(Jz).reshape(self.grid_x.shape)
                
        Mx = np.array(Mx).reshape(self.grid_x.shape)
        My = np.array(My).reshape(self.grid_x.shape)
        Mz = np.array(Mz).reshape(self.grid_x.shape)

class PlaneWave(Beams):
    
    def __init__(self, x_lims, y_lims, gridsize, pol=np.array([1,0,0]), amp=1, phase=0):
        Beams.__init__(self, x_lims, y_lims, gridsize)
        
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
        
        
        
        

