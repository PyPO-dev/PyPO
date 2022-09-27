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
        
    def runPhysOptics(self, save, material_source, prop_mode, t_direction, prec, device):
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
        
        if prec == 'float':
            if self.propType == 'coherent':
                if device == 'cpu':
                    os.system('./PhysBeamf.exe {} {} {} {} {} {} {}'.format(self.numThreads, self.k, self.thres, save, epsilon, prop_mode, t_dir))
                    
                elif device == 'gpu':
                    pass
            
            elif self.propType == 'incoherent':
                os.system('./PhysBeamScalarf.exe {} {} {}'.format(self.numThreads, self.k, epsilon))
                
        elif prec == 'double':
            if self.propType == 'coherent':
                if device == 'cpu':
                    os.system('./PhysBeam.exe {} {} {} {} {} {} {}'.format(self.numThreads, self.k, self.thres, save, epsilon, prop_mode, t_dir))
                    
                elif device == 'gpu':
                    pass
            
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
