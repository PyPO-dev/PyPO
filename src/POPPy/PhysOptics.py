import numpy as np
import os
import json
import math
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fftpack import fft as ft
from scipy.interpolate import bisplev

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
        
        # Detect if GPU benchmark exists
        '''
        files = os.listdir('./save/benchmark/')

        if files:
            self.GPUbar = True
            for file in files:
                file = file.replace(".json", "")
                self.bm_file = file
                print("Found GPU benchmark file for device {}. Enabling GPU progressbar.\n".format(file.replace("_", " ").replace(" GPU", "")))
        
        else:
        '''
        self.GPUbar = False
        self.bm_file = ''
        #print("No GPU benchmark file found. Disabling GPU progressbar.\n")

        
    #### PUBLIC METHODS ###
        
    def runPhysOptics(self, save, material_source, prop_mode, t_direction, prec, device):
        """
        (PUBLIC)
        Call PhysBeam.exe and perform PO calculation.
        @param  ->
            save            :   Which fields to save to disk after PO calculation.
                            :   Options: 0 (J,M), 1 (E,H), 2 (J,M,E,H), 3 (E,H,Pr)
            material_source :   Material of source surface
            prop_mode       :   Propagate to surface (0) or to far-field (1)
            t_direction     :   Reverse time by changing sign in Greens function
            prec            :   Run simulation in single or double precision
            device          :   Run on either CPU or GPU. For GPU, needs CUDA
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
        
        if prec == 'single':
            if self.propType == 'coherent':
                if device == 'cpu':
                    os.chdir(self.cpp_path)
                    os.system('./PhysBeamf.exe {} {} {} {} {} {} {}'.format(self.numThreads, self.k, self.thres, 
                                                                            save, epsilon, prop_mode, t_dir))
                    
                elif device == 'gpu':
                    if self.GPUbar:
                        time = self._get_GPUtime()
                        wsleep = math.ceil(time * 1e4)
                        print("Estd. time : {:.2f} s".format(time))
                        
                    else:
                        wsleep = 0.
                        
                    nThread = 256
                    nBlock = math.ceil(self.gt / nThread)
                    os.chdir(self.cpp_path)
                    os.system('./GPhysBeamf.exe {} {} {} {} {} {} {} {} {} {}'.format(nThread, nBlock, self.k, 
                                                                                   save, epsilon, prop_mode, 
                                                                                   t_dir, self.gs, self.gt, wsleep))
            
            elif self.propType == 'incoherent':
                os.chdir(self.cpp_path)
                os.system('./PhysBeamScalarf.exe {} {} {}'.format(self.numThreads, self.k, epsilon))
                
        elif prec == 'double':
            if self.propType == 'coherent':
                if device == 'cpu':
                    os.chdir(self.cpp_path)
                    os.system('./PhysBeam.exe {} {} {} {} {} {} {}'.format(self.numThreads, self.k, self.thres, 
                                                                           save, epsilon, prop_mode, t_dir))
                    
                elif device == 'gpu':
                    if self.GPUbar:
                        time = self._get_GPUtime()
                        print("Estd. time : {} s".format(time))
                        
                    else:
                        time = 0.
                        
                    nThread = 256
                    nBlock = math.ceil(self.gt / nThread)
                    os.chdir(self.cpp_path)
                    os.system('./GPhysBeam.exe {} {} {} {} {} {} {} {} {}'.format(nThread, nBlock, self.k, 
                                                                                   save, epsilon, prop_mode, 
                                                                                   t_dir, self.gs, self.gt))
            
            elif self.propType == 'incoherent':
                os.chdir(self.cpp_path)
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
        
    def set_gs(self, gs):
        self.gs = gs[0] * gs[1]
        
    def set_gt(self, gt):
        self.gt = gt[0] * gt[1]
        
    def _get_GPUtime(self):
        """
        (PRIVATE)
        Obtain estimated GPU time from previous benchmark.
        
        @param  ->
            ns              :   Total number of cells on source grid.
            nt              :   Total number of cells on target grid.
            name            :   Name of .json file containing GPU benchmark.
            
        @return ->
            time            :   Estimated GPU time for calculation.
        """

        with open('save/benchmark/{}.json'.format(self.bm_file), 'r') as f:
            gpu_dict = json.loads(f.read())
            
        '''
        tcks_arr = []
        for ll in gpu_dict["tcks"]:
            if isinstance(ll, list):
                ll = np.array(ll)
            
            tcks_arr.append(ll)
        '''
        coeff = gpu_dict["coeff"]
        #time = bisplev(self.gs, self.gt, tcks_arr)
        time = coeff[0] + coeff[1] * self.gs + (coeff[2] + coeff[3] * self.gs) * self.gt
        
        return time
