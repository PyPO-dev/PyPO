# Standard Python imports
import numbers
import numpy as np
import matplotlib.pyplot as pt
import time
import psutil
import os
import sys

# POPPy-specific modules
import src.POPPy.Reflectors as Reflectors
import src.POPPy.RayTrace as RayTrace
import src.POPPy.Camera as Camera
import src.POPPy.Beams as Beams
import src.POPPy.PhysOptics as PO
import src.POPPy.FourierOptics as FO
from src.POPPy.Plotter import Plotter
import src.POPPy.Aperture as Aperture
from src.POPPy.Efficiencies import Efficiencies as EF
#from src.Python.Fitting import Fitting

class System(object):
    
    # Coherent propagation labels
    fileNames_s = ['grid_s_x.txt', 'grid_s_y.txt', 'grid_s_z.txt', 'As.txt', 'Js_x.txt', 'Js_y.txt', 'Js_z.txt', 'Ms_x.txt', 'Ms_y.txt', 'Ms_z.txt']
    fileNames_t = ['grid_t_x.txt', 'grid_t_y.txt', 'grid_t_z.txt', 'As.txt', 'norm_t_nx.txt', 'norm_t_ny.txt', 'norm_t_nz.txt']
    fileNames_tc = ['Jt_x.txt', 'Jt_y.txt', 'Jt_z.txt', 'Mt_x.txt', 'Mt_y.txt', 'Mt_z.txt']
    
    # Incoherent propagation labels
    fileNames_ss = ['grid_s_x.txt', 'grid_s_y.txt', 'grid_s_z.txt', 'As.txt', 'Fs.txt']
    fileNames_ts = ['grid_t_x.txt', 'grid_t_y.txt', 'grid_t_z.txt', 'As.txt']
    fileNames_tcs = ['Ft.txt']
    
    # Far field propagation labels
    fileNames_t_ff = ['grid_t_ph.txt', 'grid_t_th.txt']
    
    customBeamPath = './custom/beam/'
    customReflPath = './custom/reflector/'
    
    def __init__(self):
        self.num_ref = 0
        self.num_cam = 0
        self.system = {}
        
        self.EF = EF()
        
    def __str__(self):
        pass
    
    def setCustomBeamPath(self, path, append=False):
        if append:
            self.customBeamPath += path
        else:
            self.customBeamPath = path
            
    def setCustomReflPath(self, path, append=False):
        if append:
            self.customReflPath += path
        else:
            self.customReflPath = path
            
    def addPlotter(self, save='./images/'):
        self.plotter = Plotter(save='./images/')
    
    #### ADD REFLECTOR METHODS
    def addParabola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Parabola", axis='a', units='mm', trunc=False, flip=False):
        """
        Function for adding paraboloid reflector to system. If gmode='uv', lims_x should contain the smallest and largest radius and lims_y 
        should contain rotation.
        """
        if name == "Parabola":
            name = name + "_{}".format(self.num_ref)
            
        if pmode == 'man':
            a = coef[0]
            b = coef[1]
            
            p = Reflectors.Parabola(a, b, cRot, name, units)
            
            p.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
            
        elif pmode == 'foc':
            f1 = coef[0] # Focal point position
            ve = coef[1] # Vertex position
            
            diff = f1 - ve

            df = np.sqrt(np.dot(diff, diff))
            a = 2 * np.sqrt(df)
            b = a
            
            orientation = diff / np.sqrt(np.dot(diff, diff))
            offTrans = ve
            
            # Find rotation in frame of vertex
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0#np.arccos(np.dot(np.array([0,0,1]), orientation))
        
            offRot = np.array([rx, ry, rz])
            cRot = offTrans
        
            p = Reflectors.Parabola(a, b, cRot, name, units)
            
            p.focus_1 = f1
            p.focus_2 = np.ones(3) * float("NaN")
            
            p.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
            
            p.rotateGrid(offRot)
            p.translateGrid(offTrans)
        
        self.system["{}".format(name)] = p
        self.num_ref += 1

    def addHyperbola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Hyperbola", axis='a', units='mm', sec='upper', trunc=False, flip=False):
        if name == "Hyperbola":
            name = name + "_{}".format(self.num_ref)
        
        if pmode == 'man':
            a = coef[0]
            b = coef[1]
            c = coef[2]
            
            h = Reflectors.Hyperbola(a, b, c, cRot, name, units, sec)

            h.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
        
        elif pmode == 'foc':
            # Calculate a, b, c of hyperbola
            f1 = coef[0] # Focal point 1 position
            f2 = coef[1] # Focal point 2 position
            ecc = coef[2] # Eccentricity of hyperbola
            
            diff = f1 - f2
            c = np.sqrt(np.dot(diff, diff)) / 2
            a = c / ecc
            b = np.sqrt(c**2 - a**2)
            
            # Convert 2D hyperbola a,b,c to 3D hyperboloid a,b,c
            a3 = b
            b3 = b
            c3 = a
        
            # Find direction between focii
            orientation = diff / np.sqrt(np.dot(diff, diff))
        
            # Find offset from center. Use offset as rotation origin for simplicity
            center = (f1 + f2) / 2
            offTrans = center
        
            # Find rotation in frame of center
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0#np.arccos(np.dot(np.array([0,0,1]), orientation))
        
            offRot = np.array([rx, ry, rz])
            cRot = offTrans
        
            h = Reflectors.Hyperbola(a3, b3, c3, cRot, name, units, sec)
            
            h.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
            
            h.rotateGrid(offRot)
            h.translateGrid(offTrans)
            
            h.focus_1 = f1
            h.focus_2 = f2
            
        self.system["{}".format(name)] = h
        self.num_ref += 1
        
    def addEllipse(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Ellipse", axis='a', units='mm', trunc=False, flip=False, ori='vertical'):
        if name == "Ellipse":
            name = name + "_{}".format(self.num_ref)
            
        if pmode == 'man':
            a = coef[0]
            b = coef[1]
            c = coef[2]
            
            e = Reflectors.Ellipse(a, b, c, cRot, name, ori, units)

            e.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)

        self.system["{}".format(name)] = e
        self.num_ref += 1
    
    def addCustomReflector(self, location, name="Custom", units='mm', cRot=np.zeros(3), offTrans=np.zeros(3)):
        if name == "Custom":
            name = name + "_{}".format(self.num_ref)
            
        path = self.customReflPath + location
        custom = Reflectors.Custom(cRot, name, path, units)
        
        self.system["{}".format(name)] = custom
        self.num_ref += 1
    
    def addCamera(self, lims_x, lims_y, gridsize, center=np.zeros(3), name="Camera", gmode='xy', units='mm'):
        cam = Camera.Camera(center, name, units)
        
        cam.setGrid(lims_x, lims_y, gridsize, gmode)
        
        self.system["{}".format(name)] = cam
        self.num_cam += 1
        
    def removeElement(self, name):
        del self.system[name]
        self.num_ref -= 1
    
    #### PLOTTING OPTICAL SYSTEM
    def plotSystem(self, focus_1=False, focus_2=False, plotRaytrace=False, norm=False, save=False, ret=False):
        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        
        for elem in self.system.values():
            if elem.elType == "Reflector":
                ax = elem.plotReflector(returns=True, ax_append=ax, focus_1=focus_1, focus_2=focus_2, norm=norm)
            
            elif elem.elType == "Camera":
                ax = elem.plotCamera(returns=True, ax_append=ax)
                
            elif elem.elType == "Aperture":
                ax = elem.plotAperture(returns=True, ax_append=ax)
            
        if plotRaytrace:
            ax = self.Raytracer.plotRaysSystem(ax_append=ax)
            
        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        
        if save:
            pt.savefig(fname='system.jpg',bbox_inches='tight', dpi=300)
        
        if ret:
            return fig
        
        pt.show()
        
    def initRaytracer(self, nRays=0, nRing=0, 
                 a=0, b=0, angx=0, angy=0,
                 originChief=np.zeros(3), 
                 tiltChief=np.zeros(3)):
        
        rt = RayTrace.RayTrace()
        
        rt.initRaytracer(nRays, nRing, a, b, angx, angy, originChief, tiltChief)
        
        self.Raytracer = rt
        
    def POtoRaytracer(self, source):
        
        # Load reflected Poynting vectors
        Pr = self.loadPr(source)
        
        rt = RayTrace.RayTrace()
        
        rt.POtoRaytracer(source, Pr)

        self.Raytracer = rt
        
    def startRaytracer(self, target, a0=100, workers=1, res=1, mode='auto'):
        """
        Propagate rays in RayTracer to a surface.
        Adds new frame to rays, containing point of intersection and reflected direction.
        """
                
        print("Raytracing to {}".format(target.name))
        print("Total amount of rays: {}".format(self.Raytracer.nTot))
        
        # Check if ray-trace size is OK
        workers = self._memCheckRT(workers)
        
        start = time.time()
        
        if mode == 'auto':
            mode = self.Raytracer.getMode()
        
        if not hasattr(target, 'tcks'):
            
            if target.elType == "Reflector":
                target.interpReflector(res, mode)
                
            elif target.elType == "Camera":
                target.interpCamera(res, mode)
        
        self.Raytracer.set_tcks(target.tcks)

        self.Raytracer.propagateRays(a0=a0, mode=mode, workers=workers)
        end = time.time()
        print("Elapsed time: {:.2f} [s]\n".format(end - start))
        print(len(self.Raytracer))
        
    def fieldRaytracer(self, target, field, k, a0=100, workers=1, res=1, mode='auto'):
        self.startRaytracer(target, a0, workers, res, mode)
        
        self.Raytracer.calcPathlength()
        
        f_prop = []
        for i, ((key, ray), f) in enumerate(zip(self.Raytracer.rays.items(), field[0].flatten())):
            f_prop.append(f * np.exp(-1j * k * ray["length"]))
            
        f_prop = np.array(f_prop).reshape(field[0].shape)
        return [f_prop, field[1]]
        
    def emptyRaytracer(self):
        self.Raytracer.rays.clear()
        
    def addBeam(self, lims_x, lims_y, gridsize, beam='pw', pol=np.array([1,0,0]), amp=1, phase=0, flip=False, name='', comp='Ex', units='mm', cRot=np.zeros(3)):
        if beam == 'pw':
            self.inputBeam = Beams.PlaneWave(lims_x, lims_y, gridsize, pol, amp, phase, flip, name, units, cRot)
            
        elif beam == 'custom':
            pathsToFields = [self.customBeamPath + 'r' + name, self.customBeamPath + 'i' + name, self.customBeamPath]
            self.inputBeam = Beams.CustomBeam(lims_x, lims_y, gridsize, comp, pathsToFields, flip, name, units, cRot)

    def addPointSource(self, area=1, pol='incoherent', amp=1, phase=0, flip=False, name='', units='mm', n=3, cRot=np.zeros(3)):
        self.inputBeam = Beams.PointSource(area, pol, amp, phase, flip, name, units, n, cRot)
            
    def initPhysOptics(self, target=None, k=1, thres=-50, numThreads=1, cpp_path='./src/C++/', cont=False):
        """
        Create a PO object that will act as the 'translator' between POPPy and PhysBeam.
        Also performs the initial propagation from beam to target.
        Should be called once per system.
        """
        
        self.PO = PO.PhysOptics(k, numThreads, thres, cpp_path)
        self.PO.propType = self.inputBeam.status
        
        if self.PO.propType == 'coherent':
            if not cont:
                # Write input beam to input
                for i, attr in enumerate(self.inputBeam):
                    self.PO.writeInput(self.fileNames_s[i], attr)
            
                for i, attr in enumerate(target):
                    if i == 3:
                        # Dont write area of target
                        pass
                    else:
                        self.PO.writeInput(self.fileNames_t[i], attr)
                        
        elif self.PO.propType == 'incoherent':
            if not cont:
                # Write input beam to input
                for i, attr in enumerate(self.inputBeam):
                    self.PO.writeInput(self.fileNames_ss[i], attr)
            
                for i, attr in enumerate(target):
                    if i >= 3:
                        # Dont write area of target
                        pass
                    else:
                        self.PO.writeInput(self.fileNames_ts[i], attr)
        
        if target != None:
            print("Calculating currents on {} from {}".format(target.name, self.inputBeam.name))
        else:
            print("Initialized PO object")

    def nextPhysOptics(self, source, target):
        """
        Perform a physical optics propagation from source to target.
        Automatically continues from last propagation by copying currents and gird to input
        """
        
        if self.PO.propType == 'coherent':
            for i, attr in enumerate(source):
                # Only write xyz and area
                if i <= 3:
                    self.PO.writeInput(self.fileNames_s[i], attr)
                else:
                    continue
        
            for i, field in enumerate(self.fileNames_tc):
                # Copy J and M from output to input
                self.PO.copyToInput(self.fileNames_tc[i], self.fileNames_s[i+4])
        
            # Write target grid
            for i, attr in enumerate(target):
                if i == 3:
                    # Dont write area of target
                    pass
                else:
                    self.PO.writeInput(self.fileNames_t[i], attr)
                    
        elif self.PO.propType == 'incoherent':
            for i, attr in enumerate(source):
                # Only write xyz and area
                if i <= 3:
                    self.PO.writeInput(self.fileNames_ss[i], attr)
                else:
                    continue
        
            for i, field in enumerate(self.fileNames_tcs):
                # Copy J and M from output to input
                self.PO.copyToInput(self.fileNames_tcs[i], self.fileNames_ss[i+4])
        
            # Write target grid
            for i, attr in enumerate(target):
                if i >= 3:
                    # Dont write area of target
                    pass
                else:
                    self.PO.writeInput(self.fileNames_ts[i], attr)
                    
        print("Calculating currents on {} from {}".format(target.name, source.name))
                    
    def ffPhysOptics(self, source, target):
        for i, attr in enumerate(source):
            # Only write xyz and area
            if i <= 3:
                self.PO.writeInput(self.fileNames_s[i], attr)
            else:
                continue
        
        for i, field in enumerate(self.fileNames_tc):
            # Copy J and M from output to input
            self.PO.copyToInput(self.fileNames_tc[i], self.fileNames_s[i+4])
        
        # Write target grid
        for i, attr in enumerate(target):
            if i >= 2:
                pass
            else:
                self.PO.writeInput(self.fileNames_t_ff[i], attr)
                
        print("Calculating far-field on {} from {}".format(target.name, source.name))
                
    def folderPhysOptics(self, folder, source, target):
        """
        Perform a physical optics propagation from folder with saved input/output to target.
        Automatically continues from last propagation by copying currents and grid to input
        """
        
        # First, copy input/ to cpp_path. Do this first so that copy operations are performed
        # in the cpp_path input/ folder. This preserves the saved data
        self.PO.copyFromFolder(folder)
        
        # Now, copy J and M from local output/ to input/ in cpp_path
        outpath = folder + 'output/'

        for i, field in enumerate(self.fileNames_tc):
            self.PO.copyToInput(self.fileNames_tc[i], self.fileNames_s[i+4], outpath=outpath)
        
        for i, attr in enumerate(source):
            # Only write xyz and area
            if i <= 3:
                self.PO.writeInput(self.fileNames_s[i], attr)
            else:
                continue
        
        # Write target grid in copied input/ folder
        for i, attr in enumerate(target):
            if i == 3:
                # Dont write area of target
                pass
            else:
                self.PO.writeInput(self.fileNames_t[i], attr)
        
        print("Calculating currents on {} from {}".format(target.name, source.name))
        
    def folderffPhysOptics(self, folder, source, target):
        """
        Perform a physical optics propagation from folder with saved input/output to target.
        Automatically continues from last propagation by copying currents and grid to input
        """
        
        # First, copy input/ to cpp_path. Do this first so that copy operations are performed
        # in the cpp_path input/ folder. This preserves the saved data
        self.PO.copyFromFolder(folder)
        
        # Now, copy J and M from local output/ to input/ in cpp_path
        outpath = folder + 'output/'

        for i, field in enumerate(self.fileNames_tc):
            self.PO.copyToInput(self.fileNames_tc[i], self.fileNames_s[i+4], outpath=outpath)
        
        for i, attr in enumerate(source):
            # Only write xyz and area
            if i <= 3:
                self.PO.writeInput(self.fileNames_s[i], attr)
            else:
                continue
        
        # Write target grid
        for i, attr in enumerate(target):
            if i >= 2:
                pass
            else:
                self.PO.writeInput(self.fileNames_t_ff[i], attr)
                
        print("Calculating far-field on {} from {}".format(target.name, source.name))
    
    def runPhysOptics(self, save=0, material_source='vac', prop_mode=0, t_direction='forward', folder=''):
        self.PO.runPhysOptics(save, material_source, prop_mode, t_direction)
        
        if folder != '':
            self.PO.copyToFolder(folder)
        
    def loadField(self, surface, mode='Ex'):
        field = self.PO.loadField(surface.grid_x.shape, mode)
        
        return field
    
    def loadPr(self, surface, mode='Ex'):
        Pr = self.PO.loadPr(surface.grid_x.shape)
        
        return Pr
        
    def initFourierOptics(self, k):
        self.FO = FO.FourierOptics(k=k)
        
    def addCircAper(self, r_max, gridsize, r_min=1e-3, cRot=np.zeros(3), name=''):
        ap = Aperture.Aperture(cRot, name)
        ap.makeCircAper(r_max, r_min, gridsize)
        self.system["{}".format(name)] = ap
        
    def calcSpillover(self, surfaceObject, field, R_aper, ret_field=False):
        eta_s = self.EF.calcSpillover(surfaceObject, field, R_aper, ret_field)
        return eta_s
    
    def _countCPU(self):
        cpu_count = os.cpu_count()
        return cpu_count
    
    def _systemMem(self):
        # Only return available memory
        available = psutil.virtual_memory()[1] * 1e-9
        return available
    
    def _memCheckRT(self, workers, buff=2):
        """
        For determining whether a Ray-trace has enough resources.
        To be used inside calls to startRaytracer etc.
        If memory needs too large, manually decrease amount of workers.
        Function will return the amount of workers appropriate for your memory.
        """
        
        s_r = self.Raytracer.sizeOf(units='gb') * 2 * workers
        s_ram = self._systemMem()
        
        w_init = workers
        
        if s_r >= (s_ram - buff):
            print("""WARNING! You are attempting to start a ray-trace requiring {:.2f} gb of memory.
Your system currently has {:.2f} gb of available RAM.
Automatically reducing # of workers...""".format(s_r, s_ram))
        
        while s_r >= s_ram:
            if workers == 1:
                sys.exit("Insufficient RAM for ray-trace with single worker. Exiting POPPy.")
            
            workers -= 1
            s_r = self.Raytracer.sizeOf(units='gb') * 2 * workers
        
        if w_init != workers:
            print("Reduced # of workers from {} to {} due to RAM constraints.".format(w_init, workers))
            
        return workers

    '''
    def addFitter(self):
        self.FI = Fitting()
        
    def fitGauss(self, mode='abs', thres=-11):
        #TODO Implement fitting object
        pass
    '''    
if __name__ == "__main__":
    print("Please run System.py from the SystemInterface.py, located in the POPPy directory.")
