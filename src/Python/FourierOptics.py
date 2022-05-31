import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.fft as ft

class FourierOptics(object):
    """
    Class to handle Fourier related propagation techniques. 
    Current propagation modes:
    APWS: decompose field on a plane into a plane wave spectrum and propagate along specified axis.
    TODO: Fresnel and Fraunhofer
    TODO: generally oriented planes
    """
    
    def __init__(self, k):
        """
        Constructor of FourierOptics object.
        @param k wavenumber of field to be propagated
        """
        
        self.k = k
        
    def padGrids(self, grid_x, grid_y, field, pad_range=(1000,1000), noise=1e-12):
        noise_level = noise + 1j * noise
        
        field_pad = np.pad(field, padding_range, 'constant', constant_values=(noise_level, noise_level))
        
        grid_x_pad = np.pad(grid_x, padding_range, 'reflect', reflect_type='odd')
        grid_y_pad = np.pad(grid_y, padding_range, 'reflect', reflect_type='odd')
        
        return grid_x_pad, grid_y_pad, field_pad
        
    def ft2(self, grid_x, grid_y, field):
        ft2_field = ft.fftshift(ft.fft2(ft.ifftshift(field)))
        
        # Calculate Fourier space axes
        dx = grid_x[1,0] - grid_x[0,0]
        dy = grid_y[0,1] - grid_y[0,0]
        
        Lx = np.max(grid_x[:,0]) - np.min(grid_x[:,0])
        Ly = np.max(grid_y[0,:]) - np.min(grid_y[0,:])
        
        fx_min = -1 / (2*dx)
        fx_max = 1 / (2*dx) - 1 / Lx
        dfx = grid_x.shape[0]
        
        fy_min = -1 / (2*dy)
        fy_max = 1 / (2*dy) - 1 / Ly
        dfy = grid_y.shape[1]
        
        fx, fy = np.mgrid[fx_min:fx_max:dfx*1j, fy_min:fy_max:dfy*1j]
        
        return ft2_field, fx, fy
    
    def ift2(self, field):
        ift2_field = ft.fftshift(ft.ifft2(ft.ifftshift(field)))
        
        return ift2_field
    
    def propagateAPWS(self, grid_x, grid_y, field, z, returns='field'):
        
        APWS_source, fx, fy = self.ft2(grid_x, grid_y, field)
        
        kx = 2 * np.pi * fx
        ky = 2 * np.pi * fy
        
        check = np.sqrt(self.k**2 - kx**2 - ky**2)
        check[check < 0] = 0
        kz = np.sqrt(check)
        H = np.exp(1j * z * kz)
        
        APWS_target = APWS_source * H
        
        if returns == 'field':
            return self.ift2(APWS_target)
        
        else:
            return kx, ky, APWS_target
        
        
        
        
