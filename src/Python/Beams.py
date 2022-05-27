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
        
        grid_x, grid_y = np.mgrid[x_lims[0]:x_lims[1]:gridsize[0]*1j, y_lims[0]:y_lims[1]:gridsize[1]*1j]
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = np.zeros(grid_x.shape)
        
        if flip:
            self.norm = np.array([0,0,-1])
        else:
            self.norm = np.array([0,0,1])
            
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
        
        self.Jx = np.array(Jx).reshape(self.grid_x.shape)
        self.Jy = np.array(Jy).reshape(self.grid_x.shape)
        self.Jz = np.array(Jz).reshape(self.grid_x.shape)
        
        self.Mx = np.array(Mx).reshape(self.grid_x.shape)
        self.My = np.array(My).reshape(self.grid_x.shape)
        self.Mz = np.array(Mz).reshape(self.grid_x.shape)

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
        
        
        
        

