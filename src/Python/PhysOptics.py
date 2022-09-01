import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.fft as ft

import src.Python.Colormaps as cmaps

class PhysOptics(object):
    """
    Class for running physical optics simulations.
    Methods in this class are mostly for interaction with the C++ .exe located in /src/C++.
    """
    
    def __init__(self, k, numThreads, thres, cpp_path):
        self.inputPath = cpp_path + "input/"
        self.outputPath = cpp_path + "output/"
        
        existInput = os.path.isdir(self.inputPath)
        existOutput = os.path.isdir(self.outputPath)
        
        if not existInput:
            os.makedirs(inputPath)
    
        if not existOutput:
            os.makedirs(outputPath)
        
        self.k = k
        self.numThreads = numThreads
        self.thres = thres
        self.cpp_path = cpp_path
        
    def writeInput(self, name, toWrite):
        if np.any(np.iscomplexobj(toWrite)):
            re = np.real(toWrite)
            im = np.imag(toWrite)
            
            np.savetxt(self.inputPath + "r" + name, re)
            np.savetxt(self.inputPath + "i" + name, im)
        
        else:
            np.savetxt(self.inputPath + name, toWrite)
        
    def copyToInput(self, name, nameNew):
        os.system('cp {} {}'.format(self.outputPath + "r" + name, self.inputPath + "r" + nameNew))
        os.system('cp {} {}'.format(self.outputPath + "i" + name, self.inputPath + "i" + nameNew))
        
    def runPhysOptics(self, save, material_source, prop_mode):
        """
        Call PhysBeam.exe and do hardcore PO stuff.
        @param save 
        @param prop_mode 
        """
        if material_source == 'vac':
            epsilon = 1.0
        elif material_source == 'alu':
            epsilon = 10.0

        cwd = os.getcwd()
        os.chdir(self.cpp_path)
        os.system('./PhysBeam.exe {} {} {} {} {} {}'.format(self.numThreads, self.k, self.thres, save, epsilon, prop_mode))
        os.chdir(cwd)
        
    def loadField(self, shape, mode='Ex'):
        mode = list(mode)

        re = np.loadtxt(self.outputPath + "r{}t_{}.txt".format(mode[0], mode[1]))
        im = np.loadtxt(self.outputPath + "i{}t_{}.txt".format(mode[0], mode[1]))
        
        re = re.reshape(shape)
        im = im.reshape(shape)
        
        field = re.T + 1j * im.T # Transpose to respect original orientation of grids

        return [field, mode]
    
    def FF_fromFocus(self, grid_x, grid_y, padding_range=(1000,1000)):
        noise_level = 1e-12 + 1j * 1e-12
        
        field_pad = np.pad(self.field, padding_range, 'constant', constant_values=(noise_level, noise_level))
        
        grid_x_pad = np.pad(grid_x, padding_range, 'reflect', reflect_type='odd')
        grid_y_pad = np.pad(grid_y, padding_range, 'reflect', reflect_type='odd')
        
        # Update gridsize_foc correspondingly
        gridsize_pad = field_pad.shape
        
        ff_field = ft.fftshift(ft.ifft2(ft.ifftshift(field_pad)))
        
        pt.imshow(20 * np.log10(np.absolute(ff_field) / np.max(np.absolute(ff_field))), cmap=cmaps.parula, vmax=0, vmin=-50)
        pt.show()
        
        theta_x = np.degrees(grid_foc_pad[1]*1e-3 / f_sys) * 3600
        theta_y = np.degrees((grid_foc_pad[2] - z0)*1e-3 / f_sys) * 3600
        
    def plotField(self, grid_x, grid_y, mode='Ex', title='', vmax=0, vmin=-30, ff=0, show=True, save=False, polar=False):
        '''
        if polar:
            fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5}, subplot_kw=dict(projection='polar'))
        else:
            fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})
        '''
        fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})
    
        divider1 = make_axes_locatable(ax[0])
        divider2 = make_axes_locatable(ax[1])

        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
        field = self.loadField(shape=grid_x.shape, mode=mode)
        max_field = np.max(np.absolute(field[0]))
        
        if not ff:
            extent = [np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y)]
            ax[0].set_ylabel(r"$y$ / [mm]")
            ax[0].set_xlabel(r"$x$ / [mm]")
            ax[1].set_ylabel(r"$y$ / [mm]")
            ax[1].set_xlabel(r"$x$ / [mm]")
            
        else:
            extent = [grid_x[0,0] / ff * 3600, grid_x[-1,0] / ff * 3600, grid_y[0,0] / ff * 3600, grid_y[0,-1] / ff * 3600]
            ax[0].set_ylabel(r"El / [as]")
            ax[0].set_xlabel(r"Az / [as]")
            ax[1].set_ylabel(r"El / [as]")
            ax[1].set_xlabel(r"Az / [as]")
        
        if polar:
            x_vals = grid_x# * np.cos(grid_y)
            y_vals = grid_y# * np.sin(grid_y)
            
            ampfig = ax[0].pcolormesh(x_vals, y_vals, 20 * np.log10(np.absolute(field[0].T) / max_field), vmin=vmin, vmax=vmax, cmap=cmaps.parula)#, origin='lower', extent=extent, cmap=cmaps.parula, vmin=vmin, vmax=vmax)
            phasefig = ax[1].pcolormesh(x_vals, y_vals, np.angle(field[0].T), cmap=cmaps.parula)#, origin='lower', extent=extent, cmap=cmaps.parula)
        
        else:
            ampfig = ax[0].imshow(20 * np.log10(np.absolute(field[0]) / max_field), origin='lower', extent=extent, cmap=cmaps.parula, vmin=vmin, vmax=vmax)
            phasefig = ax[1].imshow(np.angle(field[0]), origin='lower', extent=extent, cmap=cmaps.parula)

        ax[0].set_title("PNA / [dB]", y=1.08)
        ax[0].set_aspect(1)
        ax[1].set_title("Phase / [rad]", y=1.08)
        ax[1].set_aspect(1)
    
        c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
        c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')
        pt.suptitle(title)
        #pt.savefig(fname="PhysBeam/images/beam_focus",bbox_inches='tight', dpi=300)
        if save:
            pt.savefig(save + '.jpg',bbox_inches='tight', dpi=300)
            
        if show:
            pt.show()

        pt.close()
