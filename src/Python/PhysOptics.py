import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        np.savetxt(self.inputPath + name, toWrite)
        
    def copyToInput(self, name, nameNew, grid=False):
        if grid:
            os.system('cp {} {}'.format(self.inputPath + name, self.inputPath + nameNew))
        else:
            os.system('cp {} {}'.format(self.outputPath + name, self.inputPath + nameNew))
        
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

        return re + 1j * im
        
    def plotField(self, grid_x, grid_y, mode='Ex'):
        fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})
    
        divider1 = make_axes_locatable(ax[0])
        divider2 = make_axes_locatable(ax[1])

        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
        field = self.loadField(shape=grid_x.shape, mode=mode)
        max_field = np.max(np.absolute(field))
    
        extent = [grid_x[0,0], grid_x[-1,0], grid_y[0,0], grid_y[0,-1]]
        ampfig = ax[0].imshow(20 * np.log10(np.absolute(field) / max_field), origin='lower', extent=extent, cmap=cmaps.parula, vmin=-30, vmax=0)
        phasefig = ax[1].imshow(np.angle(field), origin='lower', extent=extent, cmap=cmaps.parula)

        ax[0].set_title("PNA / [dB]", y=1.08)
        ax[0].set_box_aspect(1)
        ax[0].set_ylabel(r"$z$ / [mm]")
        ax[0].set_xlabel(r"$y$ / [mm]")
        ax[1].set_title("Phase / [rad]", y=1.08)
        ax[1].set_ylabel(r"$z$ / [mm]")
        ax[1].set_xlabel(r"$y$ / [mm]")
        ax[1].set_box_aspect(1)
    
        c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
        c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')
        #pt.savefig(fname="PhysBeam/images/beam_focus",bbox_inches='tight', dpi=300)
    
        pt.show()
        pt.close()
