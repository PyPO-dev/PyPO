# Standard Python imports
import numbers
import numpy as np
import matplotlib.pyplot as pt
import time
import psutil
import os
import sys
import json

# POPPy-specific modules
from src.POPPy.BindRefl import *
from src.POPPy.BindGPU import *
from src.POPPy.BindCPU import *
from src.POPPy.Copy import copyGrid
from src.POPPy.MatTransform import *
from src.POPPy.Plotter import Plotter
from src.POPPy.POPPyTypes import *

class System(object):
    customBeamPath = './custom/beam/'
    customReflPath = './custom/reflector/'

    savePathElem = './save/elements/'

    def __init__(self):
        self.num_ref = 0
        self.num_cam = 0
        self.system = {}

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
    # Parabola takes as input the reflectordict from now on.
    # Should build correctness test!
    #def addParabola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Parabola", axis='a', units='mm', trunc=False, flip=False):
    def addParabola(self, reflDict):
        """
        Function for adding paraboloid reflector to system. If gmode='uv', lims_x should contain the smallest and largest radius and lims_y
        should contain rotation.
        """
        if reflDict["name"] == "Parabola":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 0
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)

        if reflDict["pmode"] == "focus":
            self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

            ve = reflDict["vertex"] # Vertex point position
            f1 = reflDict["focus_1"] # Focal point position

            diff = f1 - ve

            df = np.sqrt(np.dot(diff, diff))
            a = 2 * np.sqrt(df)
            b = a

            orientation = diff / np.sqrt(np.dot(diff, diff))
            offTrans = ve

            # Find rotation in frame of vertex
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0

            offRot = np.array([rx, ry, rz])
            cRot = offTrans

            self.system[reflDict["name"]]["transf"] = MatRotate(offRot, reflDict["transf"], cRot)
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, reflDict["transf"])

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            # Convert v in degrees to radians
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addHyperbola(self, reflDict):
        if reflDict["name"] == "Hyperbola":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 1
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)

        if reflDict["pmode"] == "focus":
            self.system[reflDict["name"]]["coeffs"] = np.zeros(3)
            # Calculate a, b, c of hyperbola
            f1 = reflDict["focus_1"] # Focal point 1 position
            f2 = reflDict["focus_2"] # Focal point 2 position
            ecc = reflDict["ecc"] # Eccentricity of hyperbola

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
            rz = 0

            offRot = np.array([rx, ry, rz])
            cRot = offTrans

            self.system[reflDict["name"]]["transf"] = MatRotate(offRot, reflDict["transf"], cRot)
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, reflDict["transf"])

            self.system[reflDict["name"]]["coeffs"][0] = a3
            self.system[reflDict["name"]]["coeffs"][1] = b3
            self.system[reflDict["name"]]["coeffs"][2] = c3

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addEllipse(self, reflDict):
        if reflDict["name"] == "Ellipse":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 2
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)

        if reflDict["pmode"] == "focus":
            pass

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addPlane(self, reflDict):
        if reflDict["name"] == "Plane":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 3
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)
        self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

        self.system[reflDict["name"]]["coeffs"][0] = -1
        self.system[reflDict["name"]]["coeffs"][1] = -1
        self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["gmode"] = 1

        elif reflDict["gmode"] == "AoE":
            # Assume is given in degrees
            # Convert Az and El to radians

            self.system[reflDict["name"]]["lims_Az"] = [self.system[reflDict["name"]]["lims_Az"][0],
                                                        self.system[reflDict["name"]]["lims_Az"][1]]

            self.system[reflDict["name"]]["lims_El"] = [self.system[reflDict["name"]]["lims_El"][0],
                                                        self.system[reflDict["name"]]["lims_El"][1]]

            self.system[reflDict["name"]]["gmode"] = 2

        self.num_ref += 1
<<<<<<< HEAD
=======
    
    def addCamera(self, lims_x, lims_y, gridsize, center=np.zeros(3), name="Camera", gmode='xy', units='mm'):
        cam = Camera.Camera(center, name, units)
        
        cam.setGrid(lims_x, lims_y, gridsize, gmode)
        
        self.system["{}".format(name)] = cam
        self.num_cam += 1
        
    def saveElement(self, element):
        saveExist = os.path.isdir(self.savePathElem)
        
        if not saveExist:
            os.makedirs(self.savePathElem)
        
        paramdict = element.paramTojson()
        
        with open('./{}{}.json'.format(self.savePathElem, element.name), 'w') as f:
            json.dump(paramdict, f)
            
    def loadElement(self, name):
        """
        (PUBLIC)
        Load an element from save.
        
        @param  ->
            name            :   Name of reflector to load
        """
        
        with open('./{}{}.json'.format(self.savePathElem, name), 'r') as f:
            paramdict = json.loads(f.read())
            
        name = paramdict["name"]
            
        coef = [paramdict["a"], paramdict["b"], paramdict["c"]]
        
        lims_x = paramdict["lims_x"]
        lims_y = paramdict["lims_y"]
        gridsize = paramdict["gridsize"]
        
        gmode = paramdict["gmode"]
        units = paramdict["units"]
        
        flip = paramdict["flip"]
        
        cRot = np.array(paramdict["cRot"])
        uvaxis = paramdict["uvaxis"]
        
        sec = paramdict["sec"]
            
        if paramdict["type"] == "Paraboloid":

            self.addParabola(coef=coef, lims_x=lims_x, lims_y=lims_y, 
                             gridsize=gridsize, cRot=cRot, 
                             pmode='man', gmode=gmode, name=name, axis=uvaxis, 
                             units=units, trunc=False, flip=flip)
        
        elif paramdict["type"] == "Hyperboloid":

            self.addHyperbola(coef=coef, lims_x=lims_x, lims_y=lims_y, 
                             gridsize=gridsize, cRot=cRot, 
                             pmode='man', gmode=gmode, name=name, axis=uvaxis, 
                             units=units, sec=sec, trunc=False, flip=flip)
        
        history = []    
        for ll in paramdict["history"]:
            history.append([np.array(x) if isinstance(x, list) else x for x in ll])
            
        self.system[name].history = history
        
        for action in history:
            a_t = list(action[0])
            
            if a_t[0] == "r":
                self.system[name].cRot = action[2]
                self.system[name].rotateGrid(action[1], save=False)
                
            elif a_t[0] == "t":
                self.system[name].translateGrid(action[1], units, save=False)
        
        
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
        self.PO.set_gs(self.inputBeam.shape)
        
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
            self.PO.set_gt(target.shape)
            print("Calculating currents on {} from {}".format(target.name, self.inputBeam.name))
        else:
            print("Initialized PO object")
>>>>>>> 8700665f2c7b45d188a03bfc9af5ee937a51c77b

    def rotateGrids(self, name, rotation, cRot=np.zeros(3)):
        self.system[name]["transf"] = MatRotate(rotation, self.system[name]["transf"], cRot)

    def translateGrids(self, name, translation):
        self.system[name]["transf"] = MatTranslate(translation, self.system[name]["transf"])

    def generateGrids(self, name):
        grids = generateGrid(self.system[name])
        return grids

    def readCustomBeam(self, name, comp, shape, convert_to_current=True, mode="PMC", ret="currents"):
        rfield = np.loadtxt(self.customBeamPath + "r" + name + ".txt")
        ifield = np.loadtxt(self.customBeamPath + "i" + name + ".txt")

        field = rfield.reshape(shape) + 1j*ifield.reshape(shape)

        fields_c = self._compToFields(comp, field)

        if convert_to_current:

            comps_M = np.zeros((shape[0], shape[1], 3), dtype=complex)
            np.stack((fields_c.Ex, fields_c.Ey, fields_c.Ez), axis=2, out=comps_M)

            comps_J = np.zeros((shape[0], shape[1], 3), dtype=complex)
            np.stack((fields_c.Hx, fields_c.Hy, fields_c.Hz), axis=2, out=comps_J)

            null = np.zeros(field.shape, dtype=complex)

            # Because we initialize in xy plane, take norm as z
            norm = np.array([0,0,1])
            if mode == 'None':
                M = -np.cross(norm, comps_M, axisb=2)
                J = np.cross(norm, comps_J, axisb=2)

                Mx = M[:,:,0]
                My = M[:,:,1]
                Mz = M[:,:,2]

                Jx = J[:,:,0]
                Jy = J[:,:,1]
                Jz = J[:,:,2]

                currents_c = currents(Jx, Jy, Jz, Mx, My, Mz)

            elif mode == 'PMC':
                M = -2 * np.cross(norm, comps_M, axisb=2)

                Mx = M[:,:,0]
                My = M[:,:,1]
                Mz = M[:,:,2]

                currents_c = currents(null, null, null, Mx, My, Mz)

            elif mode == 'PEC':
                J = 2 * np.cross(norm, comps_J, axisb=2)

                Jx = J[:,:,0]
                Jy = J[:,:,1]
                Jz = J[:,:,2]

                currents_c = currents(Jx, Jy, Jz, null, null, null)

        if ret == "currents":
            return currents_c

        elif ret == "fields":
            return fields_c

        elif ret == "both":
            return currents_c, fields_c

    def propagatePO_CPU(self, source_name, target_name, s_currents, k,
                    epsilon=1, t_direction=-1, nThreads=1,
                    mode="JM", precision="single"):

        source = self.system[source_name]
        target = self.system[target_name]

        if precision == "double":
            if mode == "JM":
                out = calcJM_CPUd(source, target, s_currents, k, epsilon, t_direction, nThreads)

            elif mode == "EH":
                out = calcEH_CPUd(source, target, s_currents, k, epsilon, t_direction, nThreads)

            elif mode == "JMEH":
                out1, out2 = calcJMEH_CPUd(source, target, s_currents, k, epsilon, t_direction, nThreads)
                out = [out1, out2]

            elif mode == "EHP":
                out1, out2 = calcEHP_CPUd(source, target, s_currents, k, epsilon, t_direction, nThreads)
                out = [out1, out2]

            elif mode == "FF":
                out = calcFF_CPUd(source, target, s_currents, k, epsilon, t_direction, nThreads)

        return out

    def propagatePO_GPU(self, source_name, target_name, s_currents, k,
                    epsilon=1, t_direction=-1, nThreads=256,
                    mode="JM", precision="single"):

        source = self.system[source_name]
        target = self.system[target_name]

        if precision == "single":
            if mode == "JM":
                out = calcJM_GPUf(source, target, s_currents, k, epsilon, t_direction, nThreads)

            elif mode == "EH":
                out = calcEH_GPUf(source, target, s_currents, k, epsilon, t_direction, nThreads)

            elif mode == "JMEH":
                out1, out2 = calcJMEH_GPUf(source, target, s_currents, k, epsilon, t_direction, nThreads)
                out = [out1, out2]

            elif mode == "EHP":
                out1, out2 = calcEHP_GPUf(source, target, s_currents, k, epsilon, t_direction, nThreads)
                out = [out1, out2]

            elif mode == "FF":
                out = calcFF_GPUf(source, target, s_currents, k, epsilon, t_direction, nThreads)

        return out

    def _compToFields(self, comp, field):
        null = np.zeros(field.shape, dtype=complex)

        if comp == "Ex":
            field_c = fields(field, null, null, null, null, null)
        elif comp == "Ey":
            field_c = fields(null, field, null, null, null, null)
        elif comp == "Ez":
            field_c = fields(null, null, field, null, null, null)
        elif comp == "Hx":
            field_c = fields(null, null, null, field, null, null)
        elif comp == "Hy":
            field_c = fields(null, null, null, null, field, null)
        elif comp == "Hz":
            field_c = fields(null, null, null, null, null, field)

        return field_c

if __name__ == "__main__":
    print("Please run System.py from the SystemInterface.py, located in the POPPy directory.")
