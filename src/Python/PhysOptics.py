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
    
    def __init__(self, inputPath, outputPath, k, numThreads, thres):
        existInput = os.path.isdir(inputPath)
        existOutput = os.path.isdir(outputPath)
        
        if not existInput:
            os.makedirs(inputPath)
    
        if not existOutput:
            os.makedirs(outputPath)
            
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.k = k
        self.numThreads = numThreads
        self.thres = thres
        
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
        
    def runPhysOptics(self, save=0):
        os.chdir('./src/C++/')
        os.system('./PhysBeam.exe {} {} {} {}'.format(self.numThreads, self.k, self.thres, save))
        os.chdir('../../')
        
    def loadField(self, shape, mode='Ex'):
        mode = list(mode)

        re = np.loadtxt("./src/C++/output/r{}t_{}.txt".format(mode[0], mode[1]))
        im = np.loadtxt("./src/C++/output/i{}t_{}.txt".format(mode[0], mode[1]))
        
        re = re.reshape(shape)
        im = im.reshape(shape)
        
        self.field = re + 1j * im

        return self.field
    
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
        
    def plotField(self, grid_x, grid_y, mode='Ex', vmax=0, vmin=-30, ff=0):
        fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})
    
        divider1 = make_axes_locatable(ax[0])
        divider2 = make_axes_locatable(ax[1])

        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
        field = self.loadField(shape=grid_x.shape, mode=mode)
        max_field = np.max(np.absolute(field))
        
        if not ff:
            extent = [grid_x[0,0], grid_x[-1,0], grid_y[0,0], grid_y[0,-1]]
            ax[0].set_ylabel(r"$z$ / [mm]")
            ax[0].set_xlabel(r"$y$ / [mm]")
            ax[1].set_ylabel(r"$z$ / [mm]")
            ax[1].set_xlabel(r"$y$ / [mm]")
            
        else:
            extent = [grid_x[0,0] / ff * 3600, grid_x[-1,0] / ff * 3600, grid_y[0,0] / ff * 3600, grid_y[0,-1] / ff * 3600]
            ax[0].set_ylabel(r"El / [as]")
            ax[0].set_xlabel(r"Az / [as]")
            ax[1].set_ylabel(r"El / [as]")
            ax[1].set_xlabel(r"Az / [as]")
        
        
        ampfig = ax[0].imshow(20 * np.log10(np.absolute(field) / max_field), origin='lower', extent=extent, cmap=cmaps.parula, vmin=vmin, vmax=vmax)
        phasefig = ax[1].imshow(np.angle(field), origin='lower', extent=extent, cmap=cmaps.parula)

        ax[0].set_title("PNA / [dB]", y=1.08)
        ax[0].set_box_aspect(1)
        ax[1].set_title("Phase / [rad]", y=1.08)
        ax[1].set_box_aspect(1)
    
        c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
        c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')
        #pt.savefig(fname="PhysBeam/images/beam_focus",bbox_inches='tight', dpi=300)
    
        pt.show()
        pt.close()
