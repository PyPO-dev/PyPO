import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft as ft

import src.POPPy.Colormaps as cmaps

class PhysOptics(object):
    """
    Class for running physical optics simulations.
    Methods in this class are mostly for interaction with the C++ .exe located in /src/C++.
    """
    
    #### DUNDER METHODS ###
    
    def __init__(self, k, numThreads, thres, cpp_path):
        self.inputPath = cpp_path + "input/"
        self.outputPath = cpp_path + "output/"
        
        existInput = os.path.isdir(self.inputPath)
        existOutput = os.path.isdir(self.outputPath)
        
        if not existInput:
            os.makedirs(self.inputPath)
    
        if not existOutput:
            os.makedirs(self.outputPath)
        
        self.k = k
        self.numThreads = numThreads
        self.thres = thres
        self.cpp_path = cpp_path
        
        self.propType = 'coherent'
        
    #### PUBLIC METHODS ###
        
    def runPhysOptics(self, save, material_source, prop_mode, t_direction):
        """
        (PUBLIC)
        Call PhysBeam.exe and perform PO calculation.
        @param  ->
            save        :   Which fields to save to disk after PO calculation.
                        :   Options: 0 (J,M), 1 (E,H), 2 (J,M,E,H), 3 (E,H,Pr)
            m_t         :   Material of target surface
            prop_mode   :   Propagate to surface (0) or to far-field (1)
            t_direction :   Reverse time by changing sign in Greens function
        """
        
        if material_source == 'vac':
            epsilon = 1.0
        elif material_source == 'alu':
            epsilon = 10.0
            
        if t_direction == 'forward':
            t_dir = -1
        elif t_direction == 'backward':
            t_dir = 1
        
        cwd = os.getcwd()
        os.chdir(self.cpp_path)
        if self.propType == 'coherent':
            os.system('./PhysBeam.exe {} {} {} {} {} {} {}'.format(self.numThreads, self.k, self.thres, save, epsilon, prop_mode, t_dir))
            
        elif self.propType == 'incoherent':
            os.system('./PhysBeamScalar.exe {} {} {}'.format(self.numThreads, self.k, epsilon))
        os.chdir(cwd)
        
    def loadField(self, shape, mode='Ex'):
        """
        (PUBLIC)
        Load field from output folder, reshape and make complex.
        
        @param  ->
            shape       :   Tuple containing shape of field
            mode        :   Polarization of loaded field
            
        @return ->  (list)
            field       :   Loaded field as numpy array
            mode        :   Polarization of loaded field
        """
        
        if mode != 'Field':
            mode = list(mode)
        
            re = np.loadtxt(self.outputPath + "r{}t_{}.txt".format(mode[0], mode[1]))
            im = np.loadtxt(self.outputPath + "i{}t_{}.txt".format(mode[0], mode[1]))
        
        elif mode == 'Field':
            re = np.loadtxt(self.outputPath + "rFt.txt")
            im = np.loadtxt(self.outputPath + "iFt.txt")
        
        re = re.reshape(shape)
        im = im.reshape(shape)
        
        field = re + 1j * im

        return [field, mode]
    
    def loadPr(self, shape):
        """
        (PUBLIC)
        Load reflected Poynting vector from file.
        
        @param  ->
            shape       :   Tuple containing shape of field
            
        @return ->  (list)
            Pr          :   Loaded Poynting vectors as a list of three numpy array
        """
        
        Pr = []
        
        for c in ['x','y','z']:
            pr = np.loadtxt(self.outputPath + "Pr_" + c+ ".txt")
            pr = pr.reshape(shape)
            
            Pr.append(pr)
        
        return Pr
    
    def writeInput(self, name, toWrite):
        """
        (PUBLIC)
        Write numpy arrays containg source grids and currents to input folder
        
        @param  ->
            name        :   Name of array to write
            toWrite     :   Array which to write
        """
        
        if np.any(np.iscomplexobj(toWrite)):
            re = np.real(toWrite)
            im = np.imag(toWrite)
            
            np.savetxt(self.inputPath + "r" + name, re.ravel())
            np.savetxt(self.inputPath + "i" + name, im.ravel())
        
        else:
            np.savetxt(self.inputPath + name, toWrite.ravel())
        
    def copyToInput(self, name, nameNew, inpath=None, outpath=None):
        """
        (PUBLIC)
        Copy currents in output folder to input folder.
        Used for continuation of a PO calculation.
        
        @param  ->
            name        :   Name in output folder of currents to write
            nameNew     :   Name in input folder of currents to write
            inpath      :   Path (from cwd) where to write current files
            outpath     :   Path (from cwd) where to copy from
        """
        
        if inpath is None:
            inpath = self.inputPath
        
        if outpath is None:
            outpath = self.outputPath
        
        os.system('cp {} {}'.format(outpath + "r" + name, inpath + "r" + nameNew))
        os.system('cp {} {}'.format(outpath + "i" + name, inpath + "i" + nameNew))
        
    def copyToFolder(self, folder):
        """
        (PUBLIC)
        Copy content in input/ and output/ folder in cpp_path to a local save
        
        @param  ->
            folder      :   Folder to which to copy cpp_path input/ and output/.
                        :   If folder or path_to_folder does not exist, 
                        :   create automatically.
        """
        
        exists_folder = os.path.isdir(folder)

        if not exists_folder:
            os.makedirs(folder)
        
        os.system('cp -R {} {}'.format(self.outputPath, folder + 'output/'))
        os.system('cp -R {} {}'.format(self.inputPath, folder + 'input/'))
        
    def copyFromFolder(self, folder):
        """
        (PUBLIC)
        Copy content in input/ folder of a local save to the input/ folder in
        cpp_path.
        
        @param  ->
            folder      :   Path/folder containing input/ subfolder to be copied to input/ in cpp_path
        """
        
        os.system('cp -R {} {}'.format(folder + 'input/', self.cpp_path))
    
    '''
    def FF_fromFocus(self, grid_x, grid_y, padding_range=(1000,1000)):
        noise_level = np.finfo(float).eps + 1j * np.finfo(float).eps
        
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
    '''
        
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
            '''
            ax[0].set_ylabel(r"$y$ / [mm]")
            ax[0].set_xlabel(r"$x$ / [mm]")
            ax[1].set_ylabel(r"$y$ / [mm]")
            ax[1].set_xlabel(r"$x$ / [mm]")
            '''
            ax[0].set_ylabel(r"El / [deg]")
            ax[0].set_xlabel(r"Az / [deg]")
            ax[1].set_ylabel(r"El / [deg]")
            ax[1].set_xlabel(r"Az / [deg]")
            
        else:
            extent = [grid_x[0,0] / ff * 3600, grid_x[-1,0] / ff * 3600, np.degrees(grid_y[0,0] / ff) * 3600, np.degrees(grid_y[0,-1] / ff) * 3600]
            ax[0].set_ylabel(r"El / [as]")
            ax[0].set_xlabel(r"Az / [as]")
            ax[1].set_ylabel(r"El / [as]")
            ax[1].set_xlabel(r"Az / [as]")
        
        if polar:
            x_vals = grid_x# * np.cos(grid_y)
            y_vals = grid_y# * np.sin(grid_y)
            
            ampfig = ax[0].pcolormesh(x_vals, y_vals, 20 * np.log10(np.absolute(field[0]) / max_field), vmin=vmin, vmax=vmax, cmap=cmaps.parula)#, origin='lower', extent=extent, cmap=cmaps.parula, vmin=vmin, vmax=vmax)
            phasefig = ax[1].pcolormesh(x_vals, y_vals, np.angle(field[0]), cmap=cmaps.parula)#, origin='lower', extent=extent, cmap=cmaps.parula)
        
        else:
            ampfig = ax[0].imshow(20 * np.log10(np.absolute(np.flip(field[0])) / max_field), origin='lower', extent=extent, cmap=cmaps.parula, vmin=vmin, vmax=vmax)
            phasefig = ax[1].imshow(np.angle(np.flip(field[0])), origin='lower', extent=extent, cmap=cmaps.parula)

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
        
    #### PRIVATE METHODS ###
        
    
