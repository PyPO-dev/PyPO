# Standard Python imports
import numbers
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import time
import os
import sys
import copy
import json
from pathlib import Path
from contextlib import contextmanager
from scipy.interpolate import griddata
import pickle 

# PyPO-specific modules
from src.PyPO.BindRefl import *
from src.PyPO.BindGPU import *
from src.PyPO.BindCPU import *
from src.PyPO.BindBeam import *
from src.PyPO.MatTransform import *
from src.PyPO.PyPOTypes import *
from src.PyPO.Checks import *
import src.PyPO.Config as Config

import src.PyPO.Plotter as plt
import src.PyPO.Efficiencies as effs
import src.PyPO.FitGauss as fgs

# Set PyPO absolute root path
sysPath = Path(__file__).parents[2]

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

##
# @file
# System interface for PyPO.
#
# This script contains the System class definition.

class System(object):
    customBeamPath = os.path.join(sysPath, "custom", "beam")
    customReflPath = os.path.join(sysPath, "custom", "reflector")

    savePathElem = os.path.join(sysPath, "save", "elements")
    savePathFields = os.path.join(sysPath, "save", "fields")
    savePathCurrents = os.path.join(sysPath, "save", "currents")
    savePathSystems = os.path.join(sysPath, "save", "systems")

    def __init__(self, redirect=None):
        self.num_ref = 0
        self.num_cam = 0
        Config.initPrint(redirect)
        # Internal dictionaries
        self.system = {}
        self.frames = {}
        self.fields = {}
        self.currents = {}

        self.EHcomplist = np.array(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
        self.JMcomplist = np.array(["Jx", "Jy", "Jz", "Mx", "My", "Mz"])

        self.cl = 2.99792458e11 # mm / s
        #self.savePathElem = "./save/elements/"

        saveElemExist = os.path.isdir(self.savePathElem)
        saveFieldsExist = os.path.isdir(self.savePathFields)
        saveCurrentsExist = os.path.isdir(self.savePathCurrents)
        saveSystemsExist = os.path.isdir(self.savePathSystems)

        if not saveElemExist:
            os.makedirs(self.savePathElem)

        elif not saveFieldsExist:
            os.makedirs(self.savePathFields)

        elif not saveCurrentsExist:
            os.makedirs(self.savePathCurrents)

        elif not saveSystemsExist:
            os.makedirs(self.savePathSystems)
        
        self.savePath = os.path.join(sysPath, "images")

        existSave = os.path.isdir(self.savePath)

        if not existSave:
            os.makedirs(self.savePath)

    def __str__(self):
        s = "Reflectors in system:\n"
        for key, item in self.system.items():
            s += f"{key}\n"
        return s

    def setCustomBeamPath(self, path, append=False):
        if append:
            self.customBeamPath = os.path.join(self.customBeamPath, path)
        else:
            self.customBeamPath = path

    def setSavePath(self, path, append=False):
        if append:
            self.savePath = os.path.join(self.savePath, path)
        else:
            self.savePath = path
    
    def setCustomReflPath(self, path, append=False):
        if append:
            self.customReflPath = os.path.join(self.customReflPath, path)
        else:
            self.customReflPath = path

    def mergeSystem(self, *args):
        for sysObject in args:
            sys_copy = self.copyObj(sysObject.system)
            self.system.update(sys_copy)

    #### ADD REFLECTOR METHODS
    # Parabola takes as input the reflectordict from now on.
    # Should build correctness test!
    #def addParabola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Parabola", axis='a', units='mm', trunc=False, flip=False):
    def addParabola(self, reflDict):
        """
        Function for adding paraboloid reflector to system. If gmode='uv', lims_x should contain the smallest and largest radius and lims_y
        should contain rotation.
        """
        if not "name" in reflDict:
            reflDict["name"] = "Parabola"

        if not "gcenter" in reflDict:
            reflDict["gcenter"] = np.zeros(2)

        if reflDict["name"] == "Parabola":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 0
        reflDict["transf"] = np.eye(4)

        if not "flip" in reflDict:
            reflDict["flip"] = False

        check_ElemDict(reflDict, self.system.keys()) 
        if not "ecc_uv" in reflDict:
            reflDict["ecc_uv"] = 0

        if not "rot_uv" in reflDict:
            reflDict["rot_uv"] = 0

        self.system[reflDict["name"]] = self.copyObj(reflDict)

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

        elif reflDict["pmode"] == "manual":
            self.system[reflDict["name"]]["coeffs"] = np.array([reflDict["coeffs"][0], reflDict["coeffs"][1], -1])

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            # Convert v in degrees to radians
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addHyperbola(self, reflDict):

        if not "name" in reflDict:
            reflDict["name"] = "Hyperbola"
        if reflDict["name"] == "Hyperbola":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)
        if not "gcenter" in reflDict:
            reflDict["gcenter"] = np.zeros(2)

        reflDict["type"] = 1
        reflDict["transf"] = np.eye(4)
        if not "flip" in reflDict:
            reflDict["flip"] = False
        check_ElemDict(reflDict, self.system.keys()) 
        if not "ecc_uv" in reflDict:
            reflDict["ecc_uv"] = 0

        if not "rot_uv" in reflDict:
            reflDict["rot_uv"] = 0
        self.system[reflDict["name"]] = self.copyObj(reflDict)

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

            _transf = MatRotate(offRot, reflDict["transf"])
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, _transf)

            self.system[reflDict["name"]]["coeffs"][0] = a3
            self.system[reflDict["name"]]["coeffs"][1] = b3
            self.system[reflDict["name"]]["coeffs"][2] = c3

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addEllipse(self, reflDict):
        if not "name" in reflDict:
            reflDict["name"] = "Ellipse"

        if reflDict["name"] == "Ellipse":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        if not "gcenter" in reflDict:
            reflDict["gcenter"] = np.zeros(2)
        reflDict["type"] = 2
        reflDict["transf"] = np.eye(4)
        if not "flip" in reflDict:
            reflDict["flip"] = False

        check_ElemDict(reflDict, self.system.keys()) 
        if not "ecc_uv" in reflDict:
            reflDict["ecc_uv"] = 0

        if not "rot_uv" in reflDict:
            reflDict["rot_uv"] = 0
        self.system[reflDict["name"]] = self.copyObj(reflDict)

        if reflDict["pmode"] == "focus":
            self.system[reflDict["name"]]["coeffs"] = np.zeros(3)
            f1 = reflDict["focus_1"]
            f2 = reflDict["focus_2"]
            ecc = reflDict["ecc"]

            diff = f1 - f2

            trans = (f1 + f2) / 2

            f_norm = diff / np.sqrt(np.dot(diff, diff))
            fxy = np.array([f_norm[0], f_norm[1], 0])
            fxy /= np.sqrt(np.dot(fxy, fxy))

            rot_z = np.degrees(np.arccos(np.dot(fxy, np.array([1, 0, 0]))))
            fxz = np.array([f_norm[0], 0, f_norm[2]])
            fxz /= np.sqrt(np.dot(fxz, fxz))

            rot_y = np.degrees(np.arccos(np.dot(fxz, np.array([1, 0, 0]))))
            print(trans)
            rotation = np.array([0, rot_y, rot_z])

            a = np.sqrt(np.dot(diff, diff)) / (2 * ecc)
            b = a * np.sqrt(1 - ecc**2)
            
            _transf = MatRotate(rotation, reflDict["transf"])
            self.system[reflDict["name"]]["transf"] = MatTranslate(trans, _transf)

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = b


        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addPlane(self, reflDict):
        if not "name" in reflDict:
            reflDict["name"] = "plane"

        if not "gcenter" in reflDict:
            reflDict["gcenter"] = np.zeros(2)

        if reflDict["name"] == "plane":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 3
        reflDict["transf"] = np.eye(4)
        if not "flip" in reflDict:
            reflDict["flip"] = False
        check_ElemDict(reflDict, self.system.keys()) 

        if not "ecc_uv" in reflDict:
            reflDict["ecc_uv"] = 0

        if not "rot_uv" in reflDict:
            reflDict["rot_uv"] = 0
        
        self.system[reflDict["name"]] = self.copyObj(reflDict)
        self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

        self.system[reflDict["name"]]["coeffs"][0] = -1
        self.system[reflDict["name"]]["coeffs"][1] = -1
        self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
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
    
    ##
    # Rotate reflector grids.
    #
    # Apply a rotation, around a center of rotation, to a (selection of) reflector(s).
    #
    # @param name Reflector name or list of reflector names.
    # @param rotation Numpy ndarray of length 3, containing rotation angles around x,y and z axes, in degrees.
    # @param cRot Numpy ndarray of length 3, containing center of rotation x,y and z co-ordinates, in mm.
    
    def rotateGrids(self, name, rotation, cRot=np.zeros(3)):
        if isinstance(name, list):
            for _name in name:
                self.system[_name]["transf"] = MatRotate(rotation, self.system[_name]["transf"], cRot)

        else:
            self.system[name]["transf"] = MatRotate(rotation, self.system[name]["transf"], cRot)

    ##
    # Translate reflector grids.
    #
    # Apply a translation to a (selection of) reflector(s).
    #
    # @param name Reflector name or list of reflector names.
    # @param translation Numpy ndarray of length 3, containing translation x,y and z co-ordinates, in mm.
    
    def translateGrids(self, name, translation):
        if isinstance(name, list):
            for _name in name:
                self.system[_name]["transf"] = MatTranslate(translation, self.system[_name]["transf"])
        else:
            self.system[name]["transf"] = MatTranslate(translation, self.system[name]["transf"])

    def homeReflector(self, name):
        if isinstance(name, list):
            for _name in name:
                self.system[_name]["transf"] = np.eye(4)
        else:
            self.system[name]["transf"] = np.eye(4)

    def generateGrids(self, name, transform=True, spheric=True):
        grids = generateGrid(self.system[name], transform, spheric)
        return grids

    def saveElement(self, name):
        jsonDict = self.copyObj(self.system[name])
        
        with open('{}.json'.format(os.path.join(self.savePathElem, name)), 'w') as f:
            json.dump(jsonDict, f, cls=NpEncoder)

    def saveSystem(self, name):
        path = os.path.join(self.savePathSystems, name)
        saveExist = os.path.isdir(path)

        if not saveExist:
            os.makedirs(path)
        
        with open(os.path.join(path, "system"), 'wb') as file: 
            pickle.dump(self.system, file)
        
       # for key, item in self.frames.items():
        with open(os.path.join(path, "frames"), 'wb') as file: 
            pickle.dump(self.frames, file)
        
        #for key, item in self.fields.items():
        with open(os.path.join(path, "fields"), 'wb') as file: 
            pickle.dump(self.fields, file)
        
        #for key, item in self.currents.items():
        with open(os.path.join(path, "currents"), 'wb') as file: 
            pickle.dump(self.currents, file)

    def loadSystem(self, name):
        path = os.path.join(self.savePathSystems, name)
        loadExist = os.path.isdir(path)

        if not loadExist:
            print("Not here...")
        
        with open(os.path.join(path, "system"), 'rb') as file: 
            self.system = pickle.load(file)
        
       # for key, item in self.frames.items():
        with open(os.path.join(path, "frames"), 'rb') as file: 
            self.frames = pickle.load(file)
        
        #for key, item in self.fields.items():
        with open(os.path.join(path, "fields"), 'rb') as file: 
            self.fields = pickle.load(file)
        
        #for key, item in self.currents.items():
        with open(os.path.join(path, "currents"), 'rb') as file: 
            self.currents = pickle.load(file)
    
    def removeElement(self, name):
        del self.system[name]
    
    def removeFrame(self, frameName):
        del self.frames[frameName]
    
    def removeField(self, fieldName):
        del self.fields[fieldName]
    
    def removeCurrent(self, currentName):
        del self.currents[currentName]

    def saveFields(self, fields, name_fields):
        saveDir = os.path.join(self.savePathFields, name_fields)

        saveDirExist = os.path.isdir(saveDir)

        if not saveDirExist:
            os.makedirs(saveDir)

        np.save(os.path.join(saveDir, "Ex.npy"), fields.Ex)
        np.save(os.path.join(saveDir, "Ey.npy"), fields.Ey)
        np.save(os.path.join(saveDir, "Ez.npy"), fields.Ez)

        np.save(os.path.join(saveDir, "Hx.npy"), fields.Hx)
        np.save(os.path.join(saveDir, "Hy.npy"), fields.Hy)
        np.save(os.path.join(saveDir, "Hz.npy"), fields.Hz)

    def saveCurrents(self, currents, name_currents):
        saveDir = os.path.join(self.savePathCurrents, name_currents)

        saveDirExist = os.path.isdir(saveDir)

        if not saveDirExist:
            os.makedirs(saveDir)

        np.save(os.path.join(saveDir, "Jx.npy"), currents.Jx)
        np.save(os.path.join(saveDir, "Jy.npy"), currents.Jy)
        np.save(os.path.join(saveDir, "Jz.npy"), currents.Jz)

        np.save(os.path.join(saveDir, "Mx.npy"), currents.Mx)
        np.save(os.path.join(saveDir, "My.npy"), currents.My)
        np.save(os.path.join(saveDir, "Mz.npy"), currents.Mz)

    def loadCurrents(self, name_currents):
        try:
            loadDir = os.path.join(self.savePathCurrents, name_currents)

            Jx = np.load(os.path.join(loadDir, "Jx.npy"))
            Jy = np.load(os.path.join(loadDir, "Jy.npy"))
            Jz = np.load(os.path.join(loadDir, "Jz.npy"))

            Mx = np.load(os.path.join(loadDir, "Mx.npy"))
            My = np.load(os.path.join(loadDir, "My.npy"))
            Mz = np.load(os.path.join(loadDir, "Mz.npy"))

            out = currents(Jx, Jy, Jz, Mx, My, Mz)
            return out

        except:
            print(f"Could not find {name_currents} in {self.savePathCurrents}!")
            return 1

    def loadFields(self, name_fields):
        try:
            loadDir = os.path.join(self.savePathFields, name_fields)

            Ex = np.load(os.path.join(loadDir, "Ex.npy"))
            Ey = np.load(os.path.join(loadDir, "Ey.npy"))
            Ez = np.load(os.path.join(loadDir, "Ez.npy"))

            Hx = np.load(os.path.join(loadDir, "Hx.npy"))
            Hy = np.load(os.path.join(loadDir, "Hy.npy"))
            Hz = np.load(os.path.join(loadDir, "Hz.npy"))

            out = fields(Ex, Ey, Ez, Hx, Hy, Hz)
            return out

        except:
            print(f"Could not find {name_fields} in {self.savePathFields}!")
            return 1
    
    def loadElement(self, name):
        with open('{}.json'.format(os.path.join(self.savePathElem, name)), 'r') as f:
            elem = json.load(f)

        for key, value in elem.items():
            if type(value) == list:
                elem[key] = np.array(value)

            else:
                elem[key] = value

        return elem

    def readCustomBeam(self, name_beam, name_source, comp, convert_to_current=True, normalise=True, mode="PMC", scale=1000):
        rfield = np.loadtxt(os.path.join(self.customBeamPath, "r" + name_beam + ".txt"))
        ifield = np.loadtxt(os.path.join(self.customBeamPath, "i" + name_beam + ".txt"))

        field = rfield + 1j*ifield

        if normalise:
            maxf = np.max(field)
            field /= maxf
            field *= scale


        shape = self.system[name_source]["gridsize"]

        fields_c = self._compToFields(comp, field)
        currents_c = calcCurrents(fields_c, self.system[name_source], mode)

        self.fields[name_beam] = fields_c
        self.currents[name_beam] = currents_c

    def calcCurrents(self, name_source, fields, mode="PMC"):
        currents = calcCurrents(fields, self.system[name_source], mode)
        return currents

    def runPO(self, PODict):
        PODict["s_current"] = self.currents[PODict["s_current"]]
        
        source = self.system[PODict["s_current"].surf]
        target = self.system[PODict["t_name"]]
        PODict["k"] = PODict["s_current"].k
        # Default exponent to -1
        if not "exp" in PODict:
            PODict["exp"] = "fwd"

        # TODO: insert check for PODict
        if PODict["device"] == "CPU":
            out = PyPO_CPUd(source, target, PODict)

        elif PODict["device"] == "GPU":
            out = PyPO_GPUf(source, target, PODict)

        if PODict["mode"] == "JM":
            out.setMeta(PODict["t_name"], PODict["k"])
            self.currents[PODict["name_JM"]] = out
        
        elif PODict["mode"] == "EH" or PODict["mode"] == "FF":
            out.setMeta(PODict["t_name"], PODict["k"])
            self.fields[PODict["name_EH"]] = out
        
        elif PODict["mode"] == "JMEH":
            out[0].setMeta(PODict["t_name"], PODict["k"])
            out[1].setMeta(PODict["t_name"], PODict["k"])
            self.currents[PODict["name_JM"]] = out[0]
            self.fields[PODict["name_EH"]] = out[1]
        
        elif PODict["mode"] == "EHP":
            out[0].setMeta(PODict["t_name"], PODict["k"])
            self.fields[PODict["name_EH"]] = out[0]

            frame = self.loadFramePoynt(out[1], PODict["t_name"])
            self.frames[PODict["name_P"]] = frame


        return out

    def createFrame(self, argDict):
        if not argDict["name"]:
            argDict["name"] = f"Frame_{len(self.frames)}"
        
        check_RTDict(argDict, self.frames.keys())
        self.frames[argDict["name"]] = makeRTframe(argDict)

    def loadFramePoynt(self, Poynting, name_source):
        grids = generateGrid(self.system[name_source])

        nTot = Poynting.x.shape[0] * Poynting.x.shape[1]
        frame_in = frame(nTot, grids.x.ravel(), grids.y.ravel(), grids.z.ravel(),
                        Poynting.x.ravel(), Poynting.y.ravel(), Poynting.z.ravel())

        return frame_in

    def calcRayLen(self, *args, start=0):
        if isinstance(start, np.ndarray):
            frame0 = self.frames[args[0]]

            out = []
            sumd = np.zeros(len(frame0.x))

            diffx = frame0.x - start[0]
            diffy = frame0.y - start[1]
            diffz = frame0.z - start[2]

            lens = np.sqrt(diffx**2 + diffy**2 + diffz**2)
            out.append(lens)

            sumd += lens

            for i in range(len(args) - 1):
                diffx = self.frames[args[i+1]].x - frame0.x
                diffy = self.frames[args[i+1]].y - frame0.y
                diffz = self.frames[args[i+1]].z - frame0.z

                lens = np.sqrt(diffx**2 + diffy**2 + diffz**2)
                out.append(lens)

                frame0 = self.frames[args[i+1]]
                sumd += lens

            out.append(sumd)

        else:
            frame0 = self.frames[args[0]]

            out = []
            sumd = np.zeros(len(frame0.x))

            for i in range(len(args) - 1):
                diffx = self.frames[args[i+1]].x - frame0.x
                diffy = self.frames[args[i+1]].y - frame0.y
                diffz = self.frames[args[i+1]].z - frame0.z

                lens = np.sqrt(diffx**2 + diffy**2 + diffz**2)
                out.append(lens)

                frame0 = self.frames[args[i+1]]
                sumd += lens

            out.append(sumd)

        return out

    # Beam!   
    def createGauss(self, gaussDict, name_source):
        gauss_in = makeGauss(gaussDict, self.system[name_source])

        k = 2 * np.pi / gaussDict["lam"]
        gauss_in[0].setMeta(name_source, k)
        gauss_in[1].setMeta(name_source, k)

        self.fields[gaussDict["name"]] = gauss_in[0]
        self.currents[gaussDict["name"]] = gauss_in[1]
        return gauss_in

    def runRayTracer(self, fr_in, fr_out, name_target, epsilon=1e-3, nThreads=1, t0=100, device="CPU", verbose=True):
        if verbose:
            if device == "CPU":
                frameObj = RT_CPUd(self.system[name_target], self.frames[fr_in], epsilon, t0, nThreads)

            elif device == "GPU":
                frameObj = RT_GPUf(self.system[name_target], self.frames[fr_in], epsilon, t0, nThreads)

        else:
            with suppress_stdout():
                if device == "CPU":
                    frameObj = RT_CPUd(self.system[name_target], self.frames[fr_in], epsilon, t0, nThreads)

                elif device == "GPU":
                    frameObj = RT_GPUf(self.system[name_target], self.frames[fr_in], epsilon, t0, nThreads)
        
        self.frames[fr_out] = frameObj

    def interpFrame(self, name_fr_in, name_field, name_target, name_out, comp, method="nearest"):
        grids = generateGrid(self.system[name_target])

        points = (self.frames[name_fr_in].x, self.frames[name_fr_in].y, self.frames[name_fr_in].z)

        rfield = np.real(getattr(self.fields[name_field], comp))
        ifield = np.imag(getattr(self.fields[name_field], comp))

        grid_interp = (grids.x, grids.y, grids.z)

        rout = griddata(points, rfield, grid_interp, method=method)
        iout = griddata(points, ifield, grid_interp, method=method)

        out = rout.reshape(self.system[name_target]["gridsize"]) + 1j * iout.reshape(self.system[name_target]["gridsize"])

        field = self._compToFields(comp, out)
        field.setMeta(name_target, self.fields[name_field].k)

        self.fields[name_out] = field 

        return out

    def calcSpotRMS(self, name_frame):
        frame = self.frames[name_frame]
        rms = effs.calcRMS(frame)
        return rms

    def calcSpillover(self, name_field, comp, aperDict):
        field = self.fields[name_field]
        field_comp = getattr(field, comp)
        surfaceObj = self.system[field.surf]

        return effs.calcSpillover(field_comp, surfaceObj, aperDict)

    def calcTaper(self, name_field, comp, aperDict={}):
        field = self.fields[name_field]
        field_comp = getattr(field, comp)
        surfaceObj = self.system[field.surf]

        return effs.calcTaper(field_comp, surfaceObj, aperDict)

    def calcXpol(self, name_field, comp_co, comp_X):
        field = self.fields[name_field]
        field_co = getattr(field, comp_co)
        
        field_X = getattr(field, comp_X)
        
        return effs.calcXpol(field_co, field_X)

    def calcDirectivity(self, eta_t, name_target, k):
        surfaceObj = self.system[name_target]
        return effs.calcDirectivity(eta_t, surfaceObj, k)

    def fitGaussAbs(self, field, name_surface, thres, mode):
        surfaceObj = self.system[name_surface]
        out = fgs.fitGaussAbs(field, surfaceObj, thres, mode)

        return out

    def generateGauss(self, fgs_out, name_surface, mode="dB"):
        surfaceObj = self.system[name_surface]
        Psi = fgs.generateGauss(fgs_out, surfaceObj, mode)
        return Psi

    def generatePointSource(self, PSDict, name_surface):
        surfaceObj = self.system[name_surface]
        ps = np.zeros(surfaceObj["gridsize"], dtype=complex)

        xs_idx = int((surfaceObj["gridsize"][0] - 1) / 2)
        ys_idx = int((surfaceObj["gridsize"][1] - 1) / 2)

        ps[xs_idx, ys_idx] = PSDict["E0"] * np.exp(1j * PSDict["phase"])

        Ex = ps * PSDict["pol"][0]
        Ey = ps * PSDict["pol"][1]
        Ez = ps * PSDict["pol"][2]

        Hx = ps * 0
        Hy = ps * 0
        Hz = ps * 0

        field = fields(Ex, Ey, Ez, Hx, Hy, Hz) 
        current = self.calcCurrents(name_surface, field)

        k =  2 * np.pi / PSDict["lam"]

        field.setMeta(name_surface, k)
        current.setMeta(name_surface, k)

        self.fields[PSDict["name"]] = field
        self.currents[PSDict["name"]] = current

    def plotBeam2D(self, name_field, comp, ptype="field",
                    vmin=-30, vmax=0, show=True, amp_only=False,
                    save=False, polar=False, interpolation=None,
                    aperDict={"plot":False}, mode='dB', project='xy',
                    units="", name='', titleA="Amp", titleP="Phase",
                    unwrap_phase=False, returns=False):


        if ptype == "field":
            field = self.fields[name_field]
        
            if comp in self.EHcomplist:
                field_comp = field[np.argwhere(self.EHcomplist == comp)[0]]


        elif ptype == "current":
            field = self.currents[name_field] 
            
            if comp in self.JMcomplist:
                field_comp = field[np.argwhere(self.JMcomplist == comp)[0]]

        name_surface = field.surf
        plotObject = self.system[name_surface]
        
        default = "mm"
        if plotObject["gmode"] == 2 and not units:
            default = "deg"

        unitl = self._units(units, default)
        
        fig, ax = plt.plotBeam2D(plotObject, field_comp,
                        vmin, vmax, show, amp_only,
                        save, polar, interpolation,
                        aperDict, mode, project,
                        unitl, name, titleA, titleP, self.savePath, unwrap_phase)

        if returns:
            return fig, ax

        elif save:
            pt.savefig(fname=self.savePath + '{}_{}.jpg'.format(plotObject["name"], name),
                        bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

    def plot3D(self, name_surface, fine=2, cmap=cm.cool,
            norm=False, show=True, foc1=False, foc2=False, save=False, ret=False):
        
        #pt.rcParams['xtick.minor.visible'] = False
        #pt.rcParams['ytick.minor.visible'] = False

        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        plotObject = self.system[name_surface]

        plt.plot3D(plotObject, ax, fine, cmap, norm, foc1, foc2)
        if ret:
            return fig, ax
        
        elif save:
            pt.savefig(fname=savePath + '{}.jpg'.format(plotObject["name"]),bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

        #pt.rcParams['xtick.minor.visible'] = True
        #pt.rcParams['ytick.minor.visible'] = True

    def plotSystem(self, fine=2, cmap=cm.cool,
                norm=False, show=True, foc1=False, foc2=False, save=False, ret=False, select=[], RTframes=[]):

        #pt.rcParams['xtick.minor.visible'] = False
        #pt.rcParams['ytick.minor.visible'] = False
        
        plotDict = {}
        if select:
            for name in select:
                plotDict[name] = self.system[name]
        else:
            plotDict = self.system
        
        _RTframes = []
        if RTframes:
            for name in RTframes:
                _RTframes.append(self.frames[name])


        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        plt.plotSystem(plotDict, ax, fine, cmap,norm,
                    foc1, foc2, _RTframes)

        if ret:
            return fig, ax
        
        elif save:
            pt.savefig(fname=self.savePath + 'system.jpg',bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()
        
        #pt.rcParams['xtick.minor.visible'] = True
        #pt.rcParams['ytick.minor.visible'] = True
        

    def plotRTframe(self, frame, project="xy", returns=False, aspect=1):
        if returns:
            return plt.plotRTframe(self.frames[frame], project, self.savePath, returns, aspect)
        else:
            plt.plotRTframe(self.frames[frame], project, self.savePath, returns, aspect)

    def copyObj(self, obj):
        return copy.deepcopy(obj)

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

    def _units(self, unit, default="mm"):
        if unit == "m":
            return [unit, 1e-3]

        elif unit == "mm":
            return [unit, 1.]

        elif unit == "cm":
            return [unit, 1e-2]

        elif unit == "deg":
            return [unit, 1.]

        elif unit == "am":
            return [unit, 60]

        elif unit == "as":
            return [unit, 3600]

        else:
            return [default, 1.]

if __name__ == "__main__":
    print("System interface for PyPO.")
