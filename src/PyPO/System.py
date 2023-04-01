# Standard Python imports
import numbers
import scipy.optimize as opt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import time
import os
import sys
import copy
import logging
import json
from pathlib import Path
from contextlib import contextmanager
from scipy.interpolate import griddata, interpn
import pickle 

# PyPO-specific modules
from src.PyPO.BindRefl import *
from src.PyPO.BindGPU import *
from src.PyPO.BindCPU import *
from src.PyPO.BindBeam import *
from src.PyPO.BindTransf import *
from src.PyPO.MatTransform import *
from src.PyPO.PyPOTypes import *
from src.PyPO.Checks import *
#import src.PyPO.Config as Config
from src.PyPO.CustomLogger import CustomLogger
import src.PyPO.Plotter as plt
import src.PyPO.Efficiencies as effs
import src.PyPO.FitGauss as fgs

import src.PyPO.WorldParam as world

# Set PyPO absolute root path
sysPath = Path(__file__).parents[2]
logging.getLogger(__name__)

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
    savePathScalarFields = os.path.join(sysPath, "save", "scalarfields")
    savePathCurrents = os.path.join(sysPath, "save", "currents")
    savePathSystems = os.path.join(sysPath, "save", "systems")


    ##
    # Constructor. Initializes system state.
    #
    # @param redirect Redirect all print statements within system to given stdout.
    # @param context Whether system is created in script or in GUI.
    def __init__(self, redirect=None, context=None, threadmgr=None, verbose=True):
        self.num_ref = 0
        self.num_cam = 0
        self.nThreads_cpu = os.cpu_count()
        self.context = context
        
        Config.setContext(context)
        
        # Internal dictionaries
        self.system = {}
        self.frames = {}
        self.fields = {}
        self.currents = {}
        self.scalarfields = {}

        self.groups = {}

        self.EHcomplist = np.array(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
        self.JMcomplist = np.array(["Jx", "Jy", "Jz", "Mx", "My", "Mz"])

        self.cl = 2.99792458e11 # mm / s
        #self.savePathElem = "./save/elements/"

        saveElemExist = os.path.isdir(self.savePathElem)
        saveFieldsExist = os.path.isdir(self.savePathFields)
        saveScalarFieldsExist = os.path.isdir(self.savePathScalarFields)
        saveCurrentsExist = os.path.isdir(self.savePathCurrents)
        saveSystemsExist = os.path.isdir(self.savePathSystems)

        if not saveElemExist:
            os.makedirs(self.savePathElem)

        elif not saveFieldsExist:
            os.makedirs(self.savePathFields)
        
        elif not saveScalarFieldsExist:
            os.makedirs(self.savePathScalarFields)

        elif not saveCurrentsExist:
            os.makedirs(self.savePathCurrents)

        elif not saveSystemsExist:
            os.makedirs(self.savePathSystems)
        
        self.savePath = os.path.join(sysPath, "images")

        existSave = os.path.isdir(self.savePath)

        if not existSave:
            os.makedirs(self.savePath)
        
        #redirect = None
        if redirect is None:
            self.clog_mgr = CustomLogger(os.path.basename(__file__))
            self.clog = self.clog_mgr.getCustomLogger() if verbose else self.clog_mgr.getCustomLogger(open(os.devnull, "w"))

        else:
            self.clog = redirect
        #print(self.clog)

        if context == "S":
            self.clog.info("INITIALIZED EMPTY SYSTEM.")

       
    ##
    # Destructor. Deletes any reference to the logger assigned to current system.
    def __del__(self):
        if self.context != "G":
            self.clog.info("EXITING SYSTEM.")
            del self.clog_mgr
            del self.clog

    def getSystemLogger(self):
        return self.clog

    def __str__(self):
        s = "Reflectors in system:\n"
        for key, item in self.system.items():
            s += f"{key}\n"
        return s

    ##
    # Set path to folder containing custom beams.
    #
    # @param path Path to custom beams. Beams should be stored in r<name_of_beam>.txt and i<name_of_beam>.txt format, containing real and imaginary parts respectively.
    # @param append Whether path is relative to ./custom/beams/ or absolute.
    def setCustomBeamPath(self, path, append=False):
        if append:
            self.customBeamPath = os.path.join(self.customBeamPath, path)
        else:
            self.customBeamPath = path

    ##
    # Set path to folder were to save output plots.
    #
    # @param path Path to save directory.
    # @param append Whether path is relative to ./images/ or absolute.
    def setSavePath(self, path, append=False):
        if append:
            self.savePath = os.path.join(self.savePath, path)

        else:
            self.savePath = path

        if not os.path.isdir(self.savePath):
            os.makedirs(self.savePath)

    ##
    # Merge multiple systems together into current system.
    # 
    # @param systems Systems to be merged into current system
    def mergeSystem(self, *systems):
        for sysObject in systems:
            sys_copy = self.copyObj(sysObject.system)
            fie_copy = self.copyObj(sysObject.fields)
            cur_copy = self.copyObj(sysObject.currents)
            fra_copy = self.copyObj(sysObject.frames)
            gro_copy = self.copyObj(sysObject.groups)
            sfi_copy = self.copyObj(sysObject.scalarfields)
            
            self.system.update(sys_copy)
            self.fields.update(sys_copy)
            self.currents.update(sys_copy)
            self.frames.update(sys_copy)
            self.groups.update(sys_copy)
            self.scalarfields.update(sys_copy)

    ##
    # Set the verbosity of the logging from within the system.
    #
    # @param verbose Whether to enable logging or not.
    # @param handler If multiple handlers are present, select which handler to adjust.
    def setLoggingVerbosity(self, verbose=True, handler=None):
        if handler is None:
            for fstream in self.clog.handlers:
                fstream.setStream(sys.stdout) if verbose else fstream.setStream(open(os.devnull, "w"))

        else:
            self.clog.handlers[handler].setStream() if verbose else self.clog.handlers[handler].setStream(open(os.devnull, "w"))

    ##
    # Add a paraboloid reflector to the System.
    #
    # Take a reflectordictionary and append to self.system list.
    # If "pmode" == "focus", convert focus and vertex to a, b, c coefficients.
    #
    # @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
    def addParabola(self, reflDict):

        reflDict["type"] = 0

        check_ElemDict(reflDict, self.system.keys(), self.num_ref, self.clog) 

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

            R = self.findRotation(world.IAX, orientation)
            
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, R)
            self.system[reflDict["name"]]["pos"] = (self.system[reflDict["name"]]["transf"] @ np.append(self.system[reflDict["name"]]["pos"], 1))[:-1]
            self.system[reflDict["name"]]["ori"] = R[:-1, :-1] @ self.system[reflDict["name"]]["ori"]

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = -1

        elif reflDict["pmode"] == "manual":
            self.system[reflDict["name"]]["coeffs"] = np.array([reflDict["coeffs"][0], reflDict["coeffs"][1], -1])

        if reflDict["gmode"] == "xy" or reflDict["gmode"] == 0:
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv" or reflDict["gmode"] == 1:
            self.system[reflDict["name"]]["gmode"] = 1
        
        self.system[reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added paraboloid {reflDict['name']} to system.")
        self.num_ref += 1

    ##
    # Add a hyperboloid reflector to the System.
    #
    # Take a reflectordictionary and append to self.system list.
    # If "pmode" == "focus", convert focus_1 and focus_2 to a, b, c coefficients.
    #
    # @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
    def addHyperbola(self, reflDict):

        reflDict["type"] = 1
        check_ElemDict(reflDict, self.system.keys(), self.num_ref, self.clog) 
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
            orientation = diff / np.linalg.norm(diff)

            # Find offset from center. Use offset as rotation origin for simplicity
            center = (f1 + f2) / 2
            offTrans = center

            # Find rotation in frame of center
            R = self.findRotation(world.IAX, orientation)
            
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, R)

            self.system[reflDict["name"]]["pos"] = (self.system[reflDict["name"]]["transf"] @ np.append(self.system[reflDict["name"]]["pos"], 1))[:-1]
            self.system[reflDict["name"]]["ori"] = R[:-1, :-1] @ self.system[reflDict["name"]]["ori"]

            self.system[reflDict["name"]]["coeffs"][0] = a3
            self.system[reflDict["name"]]["coeffs"][1] = b3
            self.system[reflDict["name"]]["coeffs"][2] = c3

        if reflDict["gmode"] == "xy" or reflDict["gmode"] == 0:
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv" or reflDict["gmode"] == 1:
            self.system[reflDict["name"]]["gmode"] = 1

        self.system[reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added hyperboloid {reflDict['name']} to system.")
        self.num_ref += 1

    ##
    # Add an ellipsoid reflector to the System.
    #
    # Take a reflectordictionary and append to self.system list.
    # If "pmode" == "focus", convert focus_1 and focus_2 to a, b, c coefficients.
    #
    # @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
    def addEllipse(self, reflDict):

        reflDict["type"] = 2
        check_ElemDict(reflDict, self.system.keys(), self.num_ref, self.clog) 
        self.system[reflDict["name"]] = self.copyObj(reflDict)

        if reflDict["pmode"] == "focus":
            self.system[reflDict["name"]]["coeffs"] = np.zeros(3)
            f1 = reflDict["focus_1"]
            f2 = reflDict["focus_2"]
            ecc = reflDict["ecc"]

            diff = f1 - f2

            trans = (f1 + f2) / 2

            f_norm = diff / np.linalg.norm(diff)

            R = self.findRotation(np.array([1,0,0]), f_norm)

            a = np.sqrt(np.dot(diff, diff)) / (2 * ecc)
            b = a * np.sqrt(1 - ecc**2)
            
            #_transf = MatRotate(rotation, reflDict["transf"])
            self.system[reflDict["name"]]["transf"] = MatTranslate(trans, R)
            
            self.system[reflDict["name"]]["pos"] = (self.system[reflDict["name"]]["transf"] @ np.append(self.system[reflDict["name"]]["pos"], 1))[:-1]
            self.system[reflDict["name"]]["ori"] = R[:-1, :-1] @ self.system[reflDict["name"]]["ori"]

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = b

        if reflDict["gmode"] == "xy" or reflDict["gmode"] == 0:
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv" or reflDict["gmode"] == 1:
            self.system[reflDict["name"]]["gmode"] = 1

        check_ellipseLimits(self.system[reflDict["name"]], self.clog)

        self.system[reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added ellipsoid {reflDict['name']} to system.")
        self.num_ref += 1

    ##
    # Add a planar surface to the System.
    #
    # Take a reflectordictionary and append to self.system list.
    # If "gmode" == "AoE", the surface is evaluated as an angular far-field grid.
    #
    # @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
    def addPlane(self, reflDict):

        reflDict["type"] = 3
        check_ElemDict(reflDict, self.system.keys(), self.num_ref, self.clog) 

        self.system[reflDict["name"]] = self.copyObj(reflDict)
        self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

        self.system[reflDict["name"]]["coeffs"][0] = -1
        self.system[reflDict["name"]]["coeffs"][1] = -1
        self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy" or reflDict["gmode"] == 0:
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv" or reflDict["gmode"] == 1:
            self.system[reflDict["name"]]["gmode"] = 1

        elif reflDict["gmode"] == "AoE" or reflDict["gmode"] == 2:
            self.system[reflDict["name"]]["gmode"] = 2

        self.system[reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added plane {reflDict['name']} to system.")
        self.num_ref += 1
   

    ##
    # Rotate reflector grids.
    #
    # Apply a rotation, around a center of rotation, to a reflector or group. 
    # Note that an absolute orientation rotates the orientation such that it is oriented w.r.t. the z-axis.
    # In this case, the pivot defaults to the origin.
    #
    # @param name Reflector name or list of reflector names.
    # @param rotation Numpy ndarray of length 3, containing rotation angles around x, y and z axes, in degrees.
    # @param pivot Numpy ndarray of length 3, containing pivot x, y and z co-ordinates, in mm. Defaults to pos. 
    # @param obj Whether the name corresponds to a single element or group.
    # @param mode Apply rotation relative ('relative') to current orientation, or rotate to specified orientation ('absolute').
    def rotateGrids(self, name, rotation, obj="element", mode="relative", pivot=None):

        if obj == "element":
            check_elemSystem(name, self.system, self.clog, extern=True)
            pivot = self.system[name]["pos"] if pivot is None else pivot
            
            if mode == "absolute":
                match = world.IAX
                match_rot = (MatRotate(rotation))[:-1, :-1] @ match
                R = self.findRotation(self.system[name]["ori"], match_rot)

                Tp = world.INITM
                Tpm = world.INITM
                Tp[:-1,-1] = pivot
                Tpm[:-1,-1] = -pivot
                
                Rtot = Tp @ R @ Tpm

                self.system[name]["transf"] = Rtot @ self.system[name]["transf"]
                self.system[name]["transf"][:-1, :-1] = (MatRotate(rotation, pivot=pivot))[:-1, :-1]

                #print(np.linalg.det(self.system[name]["transf"]))

                self.system[name]["pos"] = (Rtot @ np.append(self.system[name]["pos"], 1))[:-1]
                self.system[name]["ori"] = Rtot[:-1, :-1] @ self.system[name]["ori"]

                self._checkBoundPO(name, Rtot)

                self.clog.info(f"Rotated element {name} to {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

            elif mode == "relative":
                self.system[name]["transf"] = MatRotate(rotation, self.system[name]["transf"], pivot)
                #print(np.linalg.det(self.system[name]["transf"]))
                
                self.system[name]["pos"] = (MatRotate(rotation, pivot=pivot) @ np.append(self.system[name]["pos"], 1))[:-1]
                self.system[name]["ori"] = MatRotate(rotation)[:-1, :-1] @ self.system[name]["ori"]
                
                self._checkBoundPO(name, MatRotate(rotation))
            
                self.clog.info(f"Rotated element {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

        elif obj == "group":
            check_groupSystem(name, self.groups, self.clog, extern=True)
            pivot = self.group[name]["pos"] if pivot is None else pivot
            
            if mode == "absolute":
                match = world.IAX
                match_rot = (MatRotate(rotation))[:-1, :-1] @ match
                R = self.findRotation(self.system[name]["ori"], match_rot)

                Tp = world.INITM
                Tpm = world.INITM
                Tp[:-1,-1] = pivot
                Tpm[:-1,-1] = -pivot
                
                Rtot = Tp @ R @ Tpm
                
                for self.system[elem] in self.groups[name]["members"]:
                    self.system[elem]["transf"] = Rtot @ self.system[elem]["transf"]
                    self.system[elem]["transf"][:-1, :-1] = (MatRotate(rotation, pivot=pivot))[:-1, :-1]

                    self.system[elem]["pos"] = (Rtot @ np.append(self.system[elem]["pos"], 1))[:-1]
                    self.system[elem]["ori"] = Rtot[:-1, :-1] @ self.system[elem]["ori"]
                    
                    self._checkBoundPO(elem, Rtot)

                self.groups[name]["pos"] = (Rtot @ np.append(self.groups[name]["pos"], 1))[:-1]
                self.groups[name]["ori"] = Rtot[:-1, :-1] @ self.groups[name]["ori"]
                
                self.clog.info(f"Rotated group {name} to {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

            elif mode == "relative":
                for elem in self.groups[name]["members"]:
                    self.system[elem]["transf"] = MatRotate(rotation, self.system[elem]["transf"], pivot)
                    
                    self.system[elem]["pos"] = (MatRotate(rotation, pivot=pivot) @ np.append(self.system[elem]["pos"], 1))[:-1]
                    self.system[elem]["ori"] = MatRotate(rotation)[:-1, :-1] @ self.system[elem]["ori"]
                    
                    self._checkBoundPO(elem, MatRotate(rotation))

                self.groups[name]["pos"] = (MatRotate(rotation, pivot=pivot) @ np.append(self.groups[name]["pos"], 1))[:-1]
                self.groups[name]["ori"] = MatRotate(rotation)[:-1, :-1] @ self.groups[name]["ori"]
                
                self.clog.info(f"Rotated group {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

        if obj == "frame":
            check_frameSystem(name, self.frames, self.clog, extern=True)
            pivot = self.frames[name].pos if pivot is None else pivot
            
            if mode == "absolute":
                match = world.IAX
                match_rot = (MatRotate(rotation))[:-1, :-1] @ match
                R = self.findRotation(self.frames[name].ori, match_rot)

                Tp = world.INITM
                Tpm = world.INITM
                Tp[:-1,-1] = pivot
                Tpm[:-1,-1] = -pivot
                
                Rtot = Tp @ R @ Tpm

                self.frames[name].transf = Rtot
                #self.frames[name].transf[:-1, :-1] = (MatRotate(rotation, pivot=pivot))[:-1, :-1]
                _fr = transformRays(self.frames[name])
                self.frames[name] = self.copyObj(_fr)

                #print(np.linalg.det(self.frames[name].transf))

                self.frames[name].pos = (Rtot @ np.append(self.frames[name].pos, 1))[:-1]
                self.frames[name].ori = Rtot[:-1, :-1] @ self.frames[name].ori
                self.clog.info(f"Rotated element {name} to {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

            elif mode == "relative":
                self.frames[name].transf = MatRotate(rotation, pivot=pivot)
                _fr = transformRays(self.frames[name])
                self.frames[name] = self.copyObj(_fr)
                #print(np.linalg.det(self.frames[name].transf))
                
                self.frames[name].pos = (MatRotate(rotation, pivot=pivot) @ np.append(self.frames[name].pos, 1))[:-1]
                self.frames[name].ori = MatRotate(rotation)[:-1, :-1] @ self.frames[name].ori
            
                self.clog.info(f"Rotated frame {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")
    
    ##
    # Translate reflector grids.
    #
    # Apply a translation to a reflector or a group.
    #
    # @param name Reflector name or list of reflector names.
    # @param translation Numpy ndarray of length 3, containing translation x, y and z co-ordinates, in mm.
    # @param obj Whether the name corresponds to a single element or group.
    # @param mode Apply translation relative ('relative') to current position, or move to specified position ('absolute').
    def translateGrids(self, name, translation, obj="element", mode="relative"):

        _translation = self.copyObj(translation)
        
        if obj == "element":
            if mode == "absolute":
                _translation -= self.system[name]["pos"]# - translation
            
            check_elemSystem(name, self.system, self.clog, extern=True)
            self.system[name]["transf"] = MatTranslate(_translation, self.system[name]["transf"])
            self.system[name]["pos"] += _translation
            
            if mode == "absolute":
                self.clog.info(f"Translated element {name} to {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
            else:
                self.clog.info(f"Translated element {name} by {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
        
        elif obj == "group":
            if mode == "absolute":
                _translation -= self.groups[name]["pos"]# - translation
            
            check_groupSystem(name, self.groups, self.clog, extern=True)
            for elem in self.groups[name]["members"]:
                self.system[elem]["transf"] = MatTranslate(_translation, self.system[elem]["transf"])
                self.system[elem]["pos"] += _translation
            
            self.groups[name]["pos"] += _translation
            
            if mode == "absolute":
                self.clog.info(f"Translated group {name} to {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
            
            else:
                self.clog.info(f"Translated group {name} by {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")

        elif obj == "frame":
            if mode == "absolute":
                _translation -= self.frames[name].pos# - translation
            
            check_frameSystem(name, self.frames, self.clog, extern=True)
            

            self.frames[name].transf = MatTranslate(_translation)
            _fr = transformRays(self.frames[name])
            self.frames[name] = self.copyObj(_fr)
            self.frames[name].pos += _translation
            
            if mode == "absolute":
                self.clog.info(f"Translated frame {name} to {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
            else:
                self.clog.info(f"Translated frame {name} by {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
    
    ##
    # Home a reflector or a group back into default configuration.
    #
    # Set internal transformation matrix of a (selection of) reflector(s) to identity.
    #
    # @param name Reflector name or list of reflector names to be homed.
    def homeReflector(self, name, obj="element", trans=True, rot=True):
        if obj == "group":
            check_groupSystem(name, self.groups, self.clog, extern=True)
            if trans:
                for elem in self.groups[name]:
                    _transf = world.INITM
                    _transf[:-1, :-1] = elem["transf"][:-1, :-1]
                    elem["transf"] = _transf
                    elem["pos"] = world.ORIGIN

                self.groups[name]["pos"] = world.ORIGIN
            
            if rot:
                for elem in self.groups[name]:
                    _transf = elem["transf"]
                    _transf[:-1, :-1] = np.eye(3)
                    elem["transf"] = _transf
                
                self.groups[name]["ori"] = world.IAX
            
            self.clog.info(f"Transforming group {name} to home position.")

                    
        else:
            check_elemSystem(name, self.system, self.clog, extern=True)
            if trans:
                _transf = world.INITM
                _transf[:-1, :-1] = self.system[name]["transf"][:-1, :-1]
                self.system[name]["transf"] = _transf
                self.system[name]["pos"] = world.ORIGIN 

            
            if rot:
                _transf = self.system[name]["transf"]
                _transf[:-1, :-1] = np.eye(3)
                self.system[name]["transf"] = _transf
                self.system[name]["ori"] = world.IAX
            
            self.clog.info(f"Transforming element {name} to home position.")
 
    ##
    # Take and store snapshot of object's current configuration.
    # 
    # @param name Name of object to be snapped.
    # @param snap_name Name of snapshot to save.
    # @param obj Whether object is an element or a group.
    def snapObj(self, name, snap_name, obj="element"):
        if obj == "group":
            check_groupSystem(name, self.groups, self.clog, extern=True)
            self.groups[name]["snapshots"][snap_name] = []

            for elem in self.groups[name]["members"]:
                self.groups[name]["snapshots"][snap_name].append(self.copyObj(self.system[elem]["transf"]))
            
            self.clog.info(f"Saved snapshot {snap_name} for group {name}.")
        
        elif obj == "element":
            check_elemSystem(name, self.system, self.clog, extern=True)
            self.system[name]["snapshots"][snap_name] = self.copyObj(self.system[name]["transf"])

            self.clog.info(f"Saved snapshot {snap_name} for element {name}.")
        
        elif obj == "frame":
            check_frameSystem(name, self.frames, self.clog, extern=True)
            self.frames[name].snapshots[snap_name] = self.copyObj(self.frames[name].transf)

            self.clog.info(f"Saved snapshot {snap_name} for frame {name}.")
    
    ##
    # Revert object configuration to a saved snapshot.
    # 
    # @param name Name of obj to revert.
    # @param snap_name Name of snapshot to revert to.
    # @param obj Whether object is an element or a group.
    def revertToSnap(self, name, snap_name, obj="element"):
        if obj == "group":
            check_groupSystem(name, self.groups, self.clog, extern=True)

            for elem, snap in zip(self.groups[name]["members"], self.groups[name]["snapshots"][snap_name]):
                self._checkBoundPO(elem, InvertMat(self.system[elem]["transf"]))
                self.system[elem]["transf"] = self.copyObj(snap)
                self._checkBoundPO(elem, self.system[elem]["transf"])
            
            self.clog.info(f"Reverted group {name} to snapshot {snap_name}.")

        elif obj == "element":
            check_elemSystem(name, self.system, self.clog, extern=True)
            self._checkBoundPO(name, InvertMat(self.system[name]["transf"]))
            self.system[name]["transf"] = self.copyObj(self.system[name]["snapshots"][snap_name])
            self._checkBoundPO(name, self.system[name]["transf"])
        
            self.clog.info(f"Reverted element {name} to snapshot {snap_name}.")
        
        elif obj == "frame":
            check_frameSystem(name, self.frames, self.clog, extern=True)
            self.frames[name].transf = self.copyObj(self.frames[name].snapshots[snap_name])
            
            _fr = transformRays(self.frames[name])
            self.frames[name] = self.copyObj(_fr)
            
            self.clog.info(f"Reverted frame {name} to snapshot {snap_name}.")
    
    ##
    # Delete a saved snapshot belonging to an object.
    # 
    # @param name Name of object.
    # @param snap_name Name of snapshot to delete.
    # @param obj Whether object is an element or a group.
    def deleteSnap(self, name, snap_name, obj="element"):
        if obj == "group":
            check_groupSystem(name, self.groups, self.clog, extern=True)
           
            del self.groups[name]["snapshots"][snap_name]

            self.clog.info(f"Deleted snapshot {snap_name} belonging to group {name}.")

        elif obj == "element":
            del self.system[name]["snapshots"][snap_name]

            self.clog.info(f"Deleted snapshot {snap_name} belonging to element {name}.")
        
        elif obj == "frame":
            del self.frames[name].snapshots[snap_name]

            self.clog.info(f"Deleted snapshot {snap_name} belonging to frame {name}.")
    
    ##
    # Group elements together into a single block. After grouping, can translate, rotate and characterise as one.
    #
    # @param names Names of elements to put in group.
    # @param name_group Name of the group.
    # @param pos Position tracer for the group.
    # @param ori Orientation tracker for group.
    def groupElements(self, name_group, *names, pos=None, ori=None):
        pos = world.ORIGIN if pos is None else pos
        ori = world.IAX if ori is None else ori

        for _name in names:
            check_elemSystem(_name, self.system, self.clog, extern=True)

        self.groups[name_group] = {
                "members"   : list(names),
                "pos"       : pos,
                "ori"       : ori,
                "snapshots" : {}
                }
        self.clog.info(f"Grouped elements {names} into group {name_group}.")

    ##
    # Remove a group of elements from system. Note that this does not remove the elements inside the group.
    #
    # @param name_group Name of the group to be removed.
    def removeGroup(self, *names):
        for ng in names:
            check_groupSystem(ng, self.groups, self.clog, extern=True)
            del self.groups[ng]
        self.clog.info(f"Removed group {names} from system.")

    ##
    # Generate reflector grids and normals.
    # 
    # Evaluate a stored reflDict and return the x, y, z grids, area and normals.
    #
    # @param name Name of reflector to be gridded.
    # @param transform Apply internal transformation matrix to reflector.
    # @param spheric Return spheric or square far-field grid (far-field only).
    #
    # @return grids A reflGrids object containing the grids, area and normals.
    #
    # @see reflGrids
    def generateGrids(self, name, transform=True, spheric=True):
        check_elemSystem(name, self.system, self.clog, extern=True)
        grids = generateGrid(self.system[name], transform, spheric)
        return grids
    
    ##
    # Save a system object to /save/systems/. This saves all reflectors, fields, currents and frames in the system to disk.
    #
    # @param name Save the current system under this name.
    def saveSystem(self, name):
        path = os.path.join(self.savePathSystems, name)
        saveExist = os.path.isdir(path)

        if not saveExist:
            os.makedirs(path)
        
        with open(os.path.join(path, "system.pys"), 'wb') as file: 
            pickle.dump(self.system, file)
        
        with open(os.path.join(path, "groups.pys"), 'wb') as file: 
            pickle.dump(self.groups, file)
        
        with open(os.path.join(path, "frames.pys"), 'wb') as file: 
            pickle.dump(self.frames, file)
        
        with open(os.path.join(path, "fields.pys"), 'wb') as file: 
            pickle.dump(self.fields, file)
        
        with open(os.path.join(path, "currents.pys"), 'wb') as file: 
            pickle.dump(self.currents, file)
        
        with open(os.path.join(path, "scalarfields.pys"), 'wb') as file: 
            pickle.dump(self.scalarfields, file)
        
        self.clog.info(f"Saved current system to {name}.")

    ##
    # Load a system object from /save/systems/. This loads all reflectors, fields, currents and frames in the system to disk.
    #
    # @param name Load the system under this name.
    def loadSystem(self, name):
        self.clog.info(f"Loading system {name} from {self.savePathSystems} into current system.")
        path = os.path.join(self.savePathSystems, name)
        loadExist = os.path.isdir(path)

        if not loadExist:
            self.clog.error("Specified system does not exist.")
            exit(1)

        with open(os.path.join(path, "system.pys"), 'rb') as file: 
            self.system = pickle.load(file)
        
        with open(os.path.join(path, "groups.pys"), 'rb') as file: 
            self.groups = pickle.load(file)
        
        with open(os.path.join(path, "frames.pys"), 'rb') as file: 
            self.frames = pickle.load(file)
        
        with open(os.path.join(path, "fields.pys"), 'rb') as file: 
            self.fields = pickle.load(file)
        
        with open(os.path.join(path, "currents.pys"), 'rb') as file: 
            self.currents = pickle.load(file)
        
        with open(os.path.join(path, "scalarfields.pys"), 'rb') as file: 
            self.scalarfields = pickle.load(file)
    
    ##
    # Remove reflector from system.
    #
    # @ param name Name of reflector to be removed.
    def removeElement(self, *name):
        for n in name:
            check_elemSystem(n, self.system, self.clog, extern=True)
            print(self.groups)
            for group in self.groups.values():
                print(type(group))
                if n in group["members"]:
                    group["members"].remove(n)
            del self.system[n]
        self.clog.info(f"Removed element {name} from system.")
    
    ##
    # Copy reflector.
    #
    # @ param name Name of reflector to be copied.
    def copyElement(self, name, name_copy):
        check_elemSystem(name, self.system, self.clog, extern=True)
        self.system[name_copy] = self.copyObj(self.system[name])
        self.clog.info(f"Copied element {name} to {name_copy}.")
    
    ##
    # Copy group.
    #
    # @ param name Name of group to be copied.
    def copyGroup(self, name, name_copy):
        check_groupSystem(name, self.groups, self.clog, extern=True)
        self.groups[name_copy] = self.copyObj(self.groups[name])
        print(id(self.groups[name_copy]))
        print(id(self.groups[name]))
        self.clog.info(f"Copied group {name} to {name_copy}.")
    
    ##
    # Remove a ray-trace frame from system
    #
    # @param frameName Name of frame to be removed.
    def removeFrame(self, *frameName):
        for fn in frameName:
            check_frameSystem(fn, self.frames, self.clog, extern=True)
            del self.frames[fn]
        self.clog.info(f"Removed frame {frameName} from system.")
    
    ##
    # Remove a PO field from system
    #
    # @param fieldName Name of field to be removed.
    def removeField(self, *fieldName):
        for fn in fieldName:
            check_fieldSystem(fn, self.fields, self.clog, extern=True)
            del self.fields[fn]
        self.clog.info(f"Removed PO field {fieldName} from system.")
    
    ##
    # Remove a PO current from system
    #
    # @param curentName Name of current to be removed.
    def removeCurrent(self, *currentName):
        for cn in currentName:
            check_currentSystem(cn, self.currents, self.clog, extern=True)
            del self.currents[cn]
        self.clog.info(f"Removed PO current {currentName} from system.")

    ##
    # Remove a scalar PO field from system
    #
    # @param fieldName Name of scalar field to be removed.
    def removeScalarField(self, *fieldName):
        for fn in fieldName:
            check_scalarfieldSystem(fn, self.scalarfields, self.clog, extern=True)
            del self.scalarfields[fn]
        self.clog.info(f"Removed scalar PO field {fieldName} from system.")
    
    ##
    # Read a custom beam from disk into the system. 
    #
    # @param name_beam Name of the beam (without the 'r' or 'i' prefixes or '.txt' suffix).
    # @param name_source Name of source surface on which to define the beam. 
    # @param comp Polarisation of beam.
    # @param convert_to_current Whether or not to also calculate PO currents associated with the beam.
    # @param normalise Whether or not to normalise beam to its maximum amplitude.
    # @para mode Which approximation to use. Can choose between Perfect Electrical Conductor ('PEC'), Perfect Magnetic Conductor ('PMC') or full calculation ('full'). Defaults to 'PMC'.
    # @param scale Scale factor for beam. Defaults to 1.
    #
    # @see setCustomBeamPath
    def readCustomBeam(self, name_beam, name_source, comp, convert_to_current=True, normalise=True, mode="PMC", scale=1):
        check_elemSystem(name_source, self.system, self.clog, extern=True)
        
        rfield = np.loadtxt(os.path.join(self.customBeamPath, "r" + name_beam + ".txt"))
        ifield = np.loadtxt(os.path.join(self.customBeamPath, "i" + name_beam + ".txt"))

        field = rfield + 1j*ifield

        if normalise:
            maxf = np.max(field)
            field /= maxf
            field *= scale

 
        shape = self.system[name_source]["gridsize"]

        fields_c = self._compToFields(comp, field)
        self.fields[name_beam] = fields_c#.H()
        currents_c = calcCurrents(fields_c, self.system[name_source], mode)

        self.currents[name_beam] = currents_c

    ##
    # Calculate currents on a surface given a field object. Sort of a private method.
    # 
    # @param name_source Name of surface in which to calculate currents.
    # @param fields Fields object from which to calculate currents.
    # @para mode Which approximation to use. Can choose between Perfect Electrical Conductor ('PEC'), Perfect Magnetic Conductor ('PMC') or full calculation ('full'). Defaults to 'PMC'.
    #
    # @see fields
    # @see currents
    def calcCurrents(self, name_source, fields, mode="PMC"):
        check_elemSystem(name_source, self.system, self.clog, extern=True)
        currents = calcCurrents(fields, self.system[name_source], mode)
        return currents

    ##
    # Instantiate a PO propagation. Stores desired output in the system.fields and/or system.currents lists.
    # If the 'EHP' mode is selected, the reflected Poynting frame is stored in system.frames.
    #
    # @param PODict Dictionary containing the PO propagation instructions.
    #
    # @see PODict
    def runPO(self, runPODict):
        self.clog.info("*** Starting PO propagation ***")
       
        check_runPODict(runPODict, self.system.keys(), self.fields.keys(), self.currents.keys(),
                    self.scalarfields.keys(), self.frames.keys(), self.clog)

        _runPODict = self.copyObj(runPODict)

        if _runPODict["mode"] != "scalar":
            sc_name = _runPODict["s_current"]
            _runPODict["s_current"] = self.currents[_runPODict["s_current"]]
            self.clog.info(f"Propagating {sc_name} on {_runPODict['s_current'].surf} to {_runPODict['t_name']}, propagation mode: {_runPODict['mode']}.")
            source = self.system[_runPODict["s_current"].surf]
            _runPODict["k"] = _runPODict["s_current"].k

        else:
            sc_name = _runPODict["s_scalarfield"]
            _runPODict["s_scalarfield"] = self.scalarfields[_runPODict["s_scalarfield"]]
            self.clog.info(f"Propagating {sc_name} on {_runPODict['s_scalarfield'].surf} to {_runPODict['t_name']}, propagation mode: {_runPODict['mode']}.")
            source = self.system[_runPODict["s_scalarfield"].surf]
            _runPODict["k"] = _runPODict["s_scalarfield"].k
       

        target = self.system[_runPODict["t_name"]]
        
        start_time = time.time()
        
        if _runPODict["device"] == "CPU":
            self.clog.info(f"Hardware: running {_runPODict['nThreads']} CPU threads.")
            self.clog.info(f"... Calculating ...")
            out = PyPO_CPUd(source, target, _runPODict)

        elif _runPODict["device"] == "GPU":
            self.clog.info(f"Hardware: running {_runPODict['nThreads']} CUDA threads per block.")
            self.clog.info(f"... Calculating ...")
            out = PyPO_GPUf(source, target, _runPODict)

        dtime = time.time() - start_time
        
        if _runPODict["mode"] == "JM":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.currents[_runPODict["name_JM"]] = out
        
        elif _runPODict["mode"] == "EH" or _runPODict["mode"] == "FF":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.fields[_runPODict["name_EH"]] = out
        
        elif _runPODict["mode"] == "JMEH":
            out[0].setMeta(_runPODict["t_name"], _runPODict["k"])
            out[1].setMeta(_runPODict["t_name"], _runPODict["k"])
            self.currents[_runPODict["name_JM"]] = out[0]
            self.fields[_runPODict["name_EH"]] = out[1]
        
        elif _runPODict["mode"] == "EHP":
            out[0].setMeta(_runPODict["t_name"], _runPODict["k"])
            self.fields[_runPODict["name_EH"]] = out[0]

            frame = self.loadFramePoynt(out[1], _runPODict["t_name"])
            self.frames[_runPODict["name_P"]] = frame

        elif _runPODict["mode"] == "scalar":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.scalarfields[_runPODict["name_field"]] = out

        self.clog.info(f"*** Finished: {dtime:.3f} seconds ***")
        return out

    ##
    # Create a tube of rays from a TubeRTDict.
    #
    # @param argDict A TubeRTDict, filled. If not filled properly, will raise an exception.
    #
    # @see TubeRTDict
    def createTubeFrame(self, argDict):
        if not argDict["name"]:
            argDict["name"] = f"Frame_{len(self.frames)}"
        
        check_TubeRTDict(argDict, self.frames.keys(), self.clog)
        
        self.frames[argDict["name"]] = makeRTframe(argDict)

        self.frames[argDict["name"]].setMeta(world.ORIGIN, world.IAX, world.INITM)
        self.frames[argDict["name"]].pos = world.ORIGIN
        self.frames[argDict["name"]].ori = world.IAX
        self.frames[argDict["name"]].transf = world.INITM

        self.clog.info(f"Added tubular frame {argDict['name']} to system.")
    
    ##
    # Create a Gaussian beam distribution of rays from a GRTDict.
    #
    # @param argDict A GRTDict, filled. If not filled properly, will raise an exception.
    #
    # @see GRTDict
    def createGRTFrame(self, argDict):
        check_GRTDict(argDict, self.frames.keys(), self.clog)
        
        self.clog.info(f"Generating Gaussian ray-trace beam.")
        self.clog.info(f"... Sampling ...")
        
        if not argDict["name"]:
            argDict["name"] = f"Frame_{len(self.frames)}"
       
        start_time = time.time()
        argDict["angx0"] = np.degrees(argDict["lam"] / (np.pi * argDict["n"] * argDict["x0"]))
        argDict["angy0"] = np.degrees(argDict["lam"] / (np.pi * argDict["n"] * argDict["y0"]))

        #check_RTDict(argDict, self.frames.keys())
        self.frames[argDict["name"]] = makeGRTframe(argDict)
        
        self.frames[argDict["name"]].pos = world.ORIGIN
        self.frames[argDict["name"]].ori = world.IAX
        self.frames[argDict["name"]].transf = world.INITM
        
        dtime = time.time() - start_time
        self.clog.info(f"Succesfully sampled {argDict['nRays']} rays: {dtime} seconds.")
        self.clog.info(f"Added Gaussian frame {argDict['name']} to system.")

    ##
    # Convert a Poynting vector grid to a frame object. Sort of private method
    # 
    # @param Poynting An rfield object containing reflected Poynting vectors.
    # @param name_source Name of reflector on which reflected Poynting vectors are defined
    #
    # @returns frame_in Frame object containing the Poynting vectors and base points.
    #
    # @see rfield
    # @see frame
    def loadFramePoynt(self, Poynting, name_source):
        check_elemSystem(name_source, self.system, self.clog, extern=True)
        grids = generateGrid(self.system[name_source])

        nTot = Poynting.x.shape[0] * Poynting.x.shape[1]
        frame_in = frame(nTot, grids.x.ravel(), grids.y.ravel(), grids.z.ravel(),
                        Poynting.x.ravel(), Poynting.y.ravel(), Poynting.z.ravel())

        return frame_in

    ##
    # Calculate total length of a ray-trace beam.
    # Takes multiple frames and calculates the distance for each ray between frames.
    #
    # @param frames Frames between which to calculate total pathlength.
    # @param start Point from which to start the calculation, len-3 Numpy array. If given, also calculates distance between point and the first frame. Defaults to None.
    #
    # @returns out List containing the distances between frames. Can be summed over to obtain total distance.
    def calcRayLen(self, *frames, start=None):
        for fr in frames:
            check_frameSystem(fr, self.frames, self.clog, extern=True)

        if isinstance(start, np.ndarray):
            frame0 = self.frames[frames[0]]

            out = []
            sumd = np.zeros(len(frame0.x))

            diffx = frame0.x - start[0]
            diffy = frame0.y - start[1]
            diffz = frame0.z - start[2]

            lens = np.sqrt(diffx**2 + diffy**2 + diffz**2)
            out.append(lens)

            sumd += lens

            for i in range(len(frames) - 1):
                diffx = self.frames[frames[i+1]].x - frame0.x
                diffy = self.frames[frames[i+1]].y - frame0.y
                diffz = self.frames[frames[i+1]].z - frame0.z

                lens = np.sqrt(diffx**2 + diffy**2 + diffz**2)
                out.append(lens)

                frame0 = self.frames[frames[i+1]]
                sumd += lens

            out.append(sumd)

        else:
            frame0 = self.frames[frames[0]]

            out = []
            sumd = np.zeros(len(frame0.x))

            for i in range(len(frames) - 1):
                diffx = self.frames[frames[i+1]].x - frame0.x
                diffy = self.frames[frames[i+1]].y - frame0.y
                diffz = self.frames[frames[i+1]].z - frame0.z

                lens = np.sqrt(diffx**2 + diffy**2 + diffz**2)
                out.append(lens)

                frame0 = self.frames[frames[i+1]]
                sumd += lens

            out.append(sumd)

        return out

    ##
    # Create a vectorial Gaussian beam.
    #
    # @param argDict A GDict containing parameters for the Gaussian beam.
    # @param name_source Name of plane on which to define Gaussian.
    #
    # @see GDict
    def createGaussian(self, gaussDict, name_source):
        check_elemSystem(name_source, self.system, self.clog, extern=True)
        check_GPODict(gaussDict, self.fields, self.clog)
        
        gauss_in = makeGauss(gaussDict, self.system[name_source])

        k = 2 * np.pi / gaussDict["lam"]
        gauss_in[0].setMeta(name_source, k)
        gauss_in[1].setMeta(name_source, k)

        self.fields[gaussDict["name"]] = gauss_in[0]
        self.currents[gaussDict["name"]] = gauss_in[1]
        #return gauss_in
    
    ##
    # Create a scalar Gaussian beam.
    #
    # @param argDict A GDict containing parameters for the Gaussian beam.
    # @param name_source Name of plane on which to define Gaussian.
    #
    # @see GDict
    def createScalarGaussian(self, gaussDict, name_source):
        check_elemSystem(name_source, self.system, self.clog, extern=True)
        check_GPODict(gaussDict, self.scalarfields, self.clog)
        
        gauss_in = makeScalarGauss(gaussDict, self.system[name_source])

        k = 2 * np.pi / gaussDict["lam"]
        gauss_in.setMeta(name_source, k)

        self.scalarfields[gaussDict["name"]] = gauss_in

    ##
    # Run a ray-trace propagation from a frame to a surface.
    #
    # @param runRTDict A runRTDict object specifying the ray-trace.
    def runRayTracer(self, runRTDict):
        self.clog.info("*** Starting RT propagation ***")
        
        _runRTDict = self.copyObj(runRTDict)

        check_runRTDict(_runRTDict, self.system, self.frames, self.clog)

        _runRTDict["fr_in"] = self.frames[_runRTDict["fr_in"]]
        _runRTDict["t_name"] = self.system[_runRTDict["t_name"]]

        start_time = time.time()
       
        if _runRTDict["device"] == "CPU":
            self.clog.info(f"Hardware: running {_runRTDict['nThreads']} CPU threads.")
            self.clog.info(f"... Calculating ...")
            frameObj = RT_CPUd(_runRTDict)

        elif _runRTDict["device"] == "GPU":
            self.clog.info(f"Hardware: running {_runRTDict['nThreads']} CUDA threads per block.")
            self.clog.info(f"... Calculating ...")
            frameObj = RT_GPUf(_runRTDict)
        
        dtime = time.time() - start_time
        
        self.clog.info(f"*** Finished: {dtime:.3f} seconds ***")
        self.frames[runRTDict["fr_out"]] = frameObj
        
        self.frames[runRTDict["fr_out"]].setMeta(self.calcRTcenter(runRTDict["fr_out"]), self.calcRTtilt(runRTDict["fr_out"]), world.INITM)


    def interpFrame(self, name_fr_in, name_field, name_target, name_out, comp, method="nearest"):
        check_frameSystem(name_fr_in, self.frames, self.clog, extern=True)
        check_elemSystem(name_target, self.system, self.clog, extern=True)
        
        grids = generateGrid(self.system[name_target])

        points = (self.frames[name_fr_in].x, self.frames[name_fr_in].y, self.frames[name_fr_in].z)

        rfield = np.real(getattr(self.fields[name_field], comp)).ravel()
        ifield = np.imag(getattr(self.fields[name_field], comp)).ravel()

        grid_interp = (grids.x, grids.y, grids.z)

        rout = griddata(points, rfield, grid_interp, method=method)
        iout = griddata(points, ifield, grid_interp, method=method)

        out = rout.reshape(self.system[name_target]["gridsize"]) + 1j * iout.reshape(self.system[name_target]["gridsize"])

        field = self._compToFields(comp, out)
        field.setMeta(name_target, self.fields[name_field].k)

        self.fields[name_out] = field 

        return out

    ##
    # Calculate the geometric center of a ray-trace frame.
    #
    # @param name_frame Name of frame to calculate center of.
    #
    # @returns c_f Len-3 Numpy array containing x, y and z co-ordinates of frame center.
    def calcRTcenter(self, name_frame):
        check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        frame = self.frames[name_frame]
        c_f = effs.calcRTcenter(frame)
        return c_f

    ##
    # Calculate the mean direction normal of a ray-trace frame.
    #
    # @param name_frame Name of frame to calculate tilt of.
    #
    # @returns t_f Len-3 Numpy array containing x, y and z components of frame tilt direction.
    def calcRTtilt(self, name_frame):
        check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        frame = self.frames[name_frame]
        t_f = effs.calcRTtilt(frame)
        return t_f
    
    ##
    # Calculate the RMS spot size of a ray-trace frame.
    #
    # @param name_frame Name of frame to calculate RMS of.
    #
    # @returns rms RMS spot size of frame in mm.
    def calcSpotRMS(self, name_frame):
        check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        frame = self.frames[name_frame]
        rms = effs.calcRMS(frame)
        return rms

    ##
    # Calculate spillover efficiency of a beam defined on a surface.
    # The method calculates the spillover using the fraction of the beam that illuminates the region defined in aperDict versus the total beam.
    #
    # @param name_field Name of the PO field.
    # @param comp Component of field to calculate spillover of.
    # @param aperDict An aperDict dictionary containing the parameters for defining the spillover aperture.
    #
    # @returns spill The spillover efficiency.
    #
    # @see aperDict
    def calcSpillover(self, name_field, comp, aperDict):
        check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        check_aperDict(aperDict, self.clog)

        field = self.fields[name_field]
        field_comp = getattr(field, comp)
        surfaceObj = self.system[field.surf]

        return effs.calcSpillover(field_comp, surfaceObj, aperDict)

    ##
    # Calculate taper efficiency of a beam defined on a surface.
    # The method calculates the taper efficiency using the fraction of the beam that illuminates the region defined in aperDict versus the total beam.
    # If aperDict is not given, it will calculate the taper efficiency on the entire beam.
    #
    # @param name_field Name of the PO field.
    # @param comp Component of field to calculate taper efficiency of.
    # @param aperDict An aperDict dictionary containing the parameters for defining the taper aperture. Defaults to None.
    #
    # @returns taper The taper efficiency.
    #
    # @see aperDict
    def calcTaper(self, name_field, comp, aperDict=None):
        check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        aperDict = {} if aperDict is None else aperDict

        if aperDict:
            check_aperDict(aperDict, self.clog)

        field = self.fields[name_field]
        field_comp = getattr(field, comp)
        surfaceObj = self.system[field.surf]

        return effs.calcTaper(field_comp, surfaceObj, aperDict)

    ##
    # Calculate cross-polar efficiency of a field defined on a surface.
    # The cross-polar efficiency is calculated over the entire field extent.
    #
    # @param name_field Name of the PO field.
    # @param comp_co Co-polar component of field.
    # @param comp_cr Cross-polar component of field.
    #
    # @returns crp The cross-polar efficiency.
    def calcXpol(self, name_field, comp_co, comp_cr):
        check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        field = self.fields[name_field]
        field_co = getattr(field, comp_co)
        
        field_cr = getattr(field, comp_cr)
        
        return effs.calcXpol(field_co, field_cr)

    ##
    # Fit a Gaussian profile to the amplitude of a field component and adds the result to scalar field in system.
    #
    # @param name_field Name of field object.
    # @param comp Component of field object.
    # @param thres Threshold to fit to, in decibels.
    # @param mode Fit to amplitude in decibels, linear or logarithmically.
    # @param full_output Return fitted parameters and standard deviations.
    #
    # @returns popt Fitted beam parameters.
    # @returns perr Standard deviation of fitted parameters.
    def fitGaussAbs(self, name_field, comp, thres=None, mode=None, full_output=False):
        check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        thres = -11 if thres is None else thres
        mode = "dB" if mode is None else mode

        field = getattr(self.fields[name_field], comp)
        surfaceObj = self.system[self.fields[name_field].surf]
        popt, perr = fgs.fitGaussAbs(field, surfaceObj, thres, mode)

        Psi = scalarfield(fgs.generateGauss(popt, surfaceObj, mode="linear"))
        Psi.setMeta(self.fields[name_field].surf, self.fields[name_field].k)

        self.scalarfields[f"fitGauss_{name_field}"] = Psi
 
        if full_output:
            return popt, perr

    def calcMainBeam(self, name_field, comp, thres=None, mode=None):
        check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        thres = -11 if thres is None else thres
        mode = "dB" if mode is None else mode

        _thres = self.copyObj(thres)

        eff = 1

        while eff >= 1:
            if thres > _thres:
                self.clog.warning(f"Could not fit at {_thres} dB level. Raising by 1 dB.")
            self.fitGaussAbs(name_field, comp, thres, mode)
            field = getattr(self.fields[name_field], comp)
            surfaceObj = self.system[self.fields[name_field].surf]

            #self.plotBeam2D(name_field, comp="Ey", vmin=-30, vmax=0)
            #self.plotBeam2D(f"fitGauss_{name_field}", vmin=-30, vmax=0)
            
            eff = effs.calcMainBeam(field, surfaceObj, self.scalarfields[f"fitGauss_{name_field}"].S)
            thres += 1
        return eff
    
    def calcBeamCuts(self, name_field, comp, phi=0, center=True, align=True):
        check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        name_surf = self.fields[name_field].surf
        field = np.absolute(getattr(self.fields[name_field], comp))

        self.snapObj(name_surf, "__pre")
        
        if center or align:
            popt, perr = self.fitGaussAbs(name_field, comp, mode="linear", full_output=True)
            print(popt) 
            if popt[0] > popt[1]:
                angf = 90
            else:
                angf = 0
            
            if center:
                self.translateGrids(name_surf, np.array([-popt[2], -popt[3], 0]))
            
            if align:
                self.rotateGrids(name_surf, np.array([0, 0, angf + np.degrees(-popt[4])]), pivot=world.ORIGIN)
        
        grids_transf = self.generateGrids(name_surf, spheric=False)

        idx_c = np.unravel_index(np.argmax(field), field.shape)
        x_cut = self.copyObj(20 * np.log10(field[:, idx_c[1]] / np.max(field)))
        y_cut = self.copyObj(20 * np.log10(field[idx_c[0], :] / np.max(field)))

        x_strip = self.copyObj(grids_transf.x[:, idx_c[1]])
        y_strip = self.copyObj(grids_transf.y[idx_c[0], :])

        pt.plot(x_cut)
        pt.plot(y_cut)
        pt.show()

        self.revertToSnap(name_surf, "__pre")
        self.deleteSnap(name_surf, "__pre")

        return x_cut, y_cut, x_strip, y_strip
    
    def calcHPBW(self, name_field, comp, interp=50):
        x_cut, y_cut, x_strip, y_strip = self.calcBeamCuts(name_field, comp)#, center=False, align=False)

        x_interp = np.linspace(np.min(x_strip), np.max(x_strip), num=len(x_strip) * interp)
        y_interp = np.linspace(np.min(y_strip), np.max(y_strip), num=len(y_strip) * interp)

        x_cut_interp = interp1d(x_strip, x_cut, kind="cubic")(x_interp)
        y_cut_interp = interp1d(y_strip, y_cut, kind="cubic")(y_interp)

        mask_x = (x_cut_interp > -3.1) & (x_cut_interp < -2.9)
        mask_y = (y_cut_interp > -3.1) & (y_cut_interp < -2.9)

        HPBW_E = np.mean(np.absolute(x_interp[mask_x])) * 2 * 3600
        HPBW_H = np.mean(np.absolute(y_interp[mask_y])) * 2 * 3600

        self.clog.info(f"E-plane HPBW = {HPBW_E} degrees.")
        self.clog.info(f"H-plane HPBW = {HPBW_H} degrees.")

    ##
    # Generate point-source PO fields and currents.
    #
    # @param PSDict A PSDict dictionary, containing parameters for the point source.
    # @param name_surface Name of surface on which to define the point-source.
    #
    # @see PSDict
    def generatePointSource(self, PSDict, name_surface):
        check_elemSystem(name_surface, self.system, self.clog, extern=True)
        check_PSDict(PSDict, self.fields, self.clog)

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

    ##
    # Generate point-source scalar PO field.
    #
    # @param PSDict A PSDict dictionary, containing parameters for the point source.
    # @param name_surface Name of surface on which to define the point-source.
    #
    # @see PSDict
    def generatePointSourceScalar(self, PSDict, name_surface):
        check_elemSystem(name_surface, self.system, self.clog, extern=True)
        check_PSDict(PSDict, self.scalarfields, self.clog)
        
        surfaceObj = self.system[name_surface]
        ps = np.zeros(surfaceObj["gridsize"], dtype=complex)

        xs_idx = int((surfaceObj["gridsize"][0] - 1) / 2)
        ys_idx = int((surfaceObj["gridsize"][1] - 1) / 2)

        ps[xs_idx, ys_idx] = PSDict["E0"] * np.exp(1j * PSDict["phase"])
        sfield = scalarfield(ps)

        k =  2 * np.pi / PSDict["lam"]

        sfield.setMeta(name_surface, k)

        self.scalarfields[PSDict["name"]] = sfield
   
    ##
    # Interpolate a PO beam. Only for beams defined on planar surfaces.
    # Can interpolate fields and currents separately.
    # Results are stored in a new fields/currents object with the original name appended by 'interp'.
    # Also, a new plane will be created with the updated gridsize and name appended by 'interp'.
    #
    # @param name Name of beam to be interpolated.
    # @param gridsize_new New gridsizes for interpolation.
    # @param obj Whether to interpolate currents or fields.
    def interpBeam(self, name, gridsize_new, obj_t="fields"):
        if obj_t == "fields":
            check_fieldSystem(name, self.fields, self.clog, extern=True)
            obj = self.fields[name]

        elif obj_t == "currents":
            check_currentSystem(name, self.currents, self.clog, extern=True)
            obj = self.currents[name]

        self.copyElement(obj.surf, obj.surf + "_interp")
        self.system[obj.surf + "_interp"]["gridsize"] = gridsize_new
        self.system[obj.surf + "_interp"]["name"] = obj.surf + "_interp"

        grids = self.generateGrids(obj.surf)
        grids_interp = self.generateGrids(obj.surf + "_interp")
        
        points = (grids.x.ravel(), grids.y.ravel())#, grids.z.ravel())
        points_interp = (grids_interp.x.ravel(), grids_interp.y.ravel())#, grids_interp.z.ravel())
        comp_l = []

        for i in range(6):
            _comp = self.copyObj(obj[i])
            _cr = np.real(_comp)
            _ci = np.imag(_comp)

            _cr_interp = griddata(points, _cr.ravel(), points_interp)
            _ci_interp = griddata(points, _ci.ravel(), points_interp)
       
            _comp_interp = _cr_interp + 1j * _ci_interp

            comp_l.append(_comp_interp.reshape(gridsize_new))

        if obj_t == "fields":
            obj_interp = fields(comp_l[0], comp_l[1], comp_l[2], comp_l[3], comp_l[4], comp_l[5])
            obj_interp.setMeta(obj.surf + "_interp", obj.k)
            self.fields[name + "_interp"] = obj_interp
        
        elif obj_t == "currents":
            obj_interp = currents(comp_l[0], comp_l[1], comp_l[2], comp_l[3], comp_l[4], comp_l[5])
            obj_interp.setMeta(obj.surf + "_interp", obj.k)
            self.currents[name + "_interp"] = obj_interp

    ##
    # Generate a 2D plot of a field or current.
    #
    # @param name_obj Name of field or current to plot.
    # @param comp Component of field or current to plot.
    # @param vmin Minimum amplitude value to display. Default is -30.
    # @param vmax Maximum amplitude value to display. Default is 0.
    # @param show Show plot. Default is True.
    # @param amp_only Only plot amplitude pattern. Default is False.
    # @param save Save plot to /images/ folder.
    # @param interpolation What interpolation to use for displaying amplitude pattern. Default is None.
    # @param norm Normalise field (only relevant when plotting linear scale). Default is True.
    # @param aperDict Plot an aperture defined in an aperDict object along with the field or current patterns. Default is None.
    # @param mode Plot amplitude in decibels ("dB") or on a linear scale ("linear"). Default is "dB".
    # @param project Set abscissa and ordinate of plot. Should be given as a string. Default is "xy".
    # @param units The units of the axes. Default is "", which is millimeters.
    # @param name Name of .png file where plot is saved. Only when save=True. Default is "".
    # @param titleA Title of the amplitude plot. Default is "Amp".
    # @param titleP Title of the phase plot. Default is "Phase".
    # @param unwrap_phase Unwrap the phase patter. Prevents annular structure in phase pattern. Default is False.
    # @param ret Return the Figure and Axis object. Only called by GUI. Default is False.
    #
    # @see aperDict
    def plotBeam2D(self, name_obj, comp=None,
                    vmin=None, vmax=None, show=True, amp_only=False,
                    save=False, interpolation=None, norm=True,
                    aperDict=None, mode='dB', project='xy',
                    units="", name="", titleA="Amp", titleP="Phase",
                    unwrap_phase=False, ret=False):

        aperDict = {"plot":False} if aperDict is None else aperDict

        if comp is None:
            field_comp = self.scalarfields[name_obj].S
            name_surface = self.scalarfields[name_obj].surf
        
        elif comp[0] == "E" or comp[0] == "H":
            check_fieldSystem(name_obj, self.fields, self.clog, extern=True)
            field = self.fields[name_obj]
            name_surface = field.surf
        
            if comp in self.EHcomplist:
                field_comp = getattr(field, comp)

        elif comp[0] == "J" or comp[0] == "M":
            check_currentSystem(name_obj, self.currents, self.clog, extern=True)
            field = self.currents[name_obj] 
            name_surface = field.surf
            
            if comp in self.JMcomplist:
                field_comp = getattr(field, comp)


        plotObject = self.system[name_surface]
        
        default = "mm"
        if plotObject["gmode"] == 2 and not units:
            default = "deg"

        unitl = self._units(units, default)
        
        fig, ax = plt.plotBeam2D(plotObject, field_comp,
                        vmin, vmax, show, amp_only,
                        save, interpolation, norm,
                        aperDict, mode, project,
                        unitl, name, titleA, titleP, self.savePath, unwrap_phase)

        if ret:
            return fig, ax

        elif save:
            pt.savefig(fname=self.savePath + '{}_{}.jpg'.format(plotObject["name"], name),
                        bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

    ##
    # Plot a 3D reflector.
    #
    # @param name_surface Name of reflector to plot.
    # @param cmap Colormap of reflector. Default is cool.
    # @param norm Plot reflector normals. Default is False.
    # @param fine Spacing of normals for plotting. Default is 2.
    # @param show Show plot. Default is True.
    # @param foc1 Plot focus 1. Default is False.
    # @param foc2 Plot focus 2. Default is False.
    # @param save Save the plot.
    # @param ret Return Figure and Axis. Only used in GUI.
    def plot3D(self, name_surface, cmap=cm.cool,
            norm=False, fine=2, show=True, foc1=False, foc2=False, save=False, ret=False):
        
        #pt.rcParams['xtick.minor.visible'] = False
        #pt.rcParams['ytick.minor.visible'] = False

        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        
        if isinstance(name_surface, list) or isinstance(name_surface, np.ndarray):
            for n_s in name_surface:
                check_elemSystem(n_s, self.system, self.clog, extern=True)
                plotObject = self.system[n_s]
                plt.plot3D(plotObject, ax, fine, cmap, norm, foc1, foc2)
        
        else:
            check_elemSystem(name_surface, self.system, self.clog, extern=True)
            plotObject = self.system[name_surface]
            plt.plot3D(plotObject, ax, fine, cmap, norm, foc1, foc2)

        if ret:
            return fig, ax
        
        elif save:
            pt.savefig(fname=self.savePath + '{}.jpg'.format(plotObject["name"]),bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

        #pt.rcParams['xtick.minor.visible'] = True
        #pt.rcParams['ytick.minor.visible'] = True

    ##
    # Plot the current system. Plots the reflectors and optionally ray-trace frames in a 3D plot.
    #
    # @param name_surface Name of reflector to plot.
    # @param cmap Colormap of reflector. Default is cool.
    # @param norm Plot reflector normals. Default is False.
    # @param fine Spacing of normals for plotting. Default is 2.
    # @param show Show plot. Default is True.
    # @param foc1 Plot focus 1. Default is False.
    # @param foc2 Plot focus 2. Default is False.
    # @param save Save the plot.
    # @param ret Return Figure and Axis. Only used in GUI.
    # @param select A list of names of reflectors to plot. If not given, plot all reflectors.
    # @param RTframes A list of names of frame to plot. If not given, plot no ray-trace frames.
    def plotSystem(self, cmap=cm.cool,
                norm=False, fine=2, show=True, foc1=False, foc2=False, save=False, ret=False, select=None, RTframes=None, RTcolor="black"):

        select = [] if select is None else select
        RTframes = [] if RTframes is None else RTframes
        #pt.rcParams['xtick.minor.visible'] = False
        #pt.rcParams['ytick.minor.visible'] = False
        
        plotDict = {}
        if select:
            for name in select:
                check_elemSystem(name, self.system, self.clog, extern=True)
                plotDict[name] = self.system[name]
        else:
            plotDict = self.system
        
        _RTframes = []
        if RTframes:
            for name in RTframes:
                check_frameSystem(name, self.frames, self.clog, extern=True)
                _RTframes.append(self.frames[name])


        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        plt.plotSystem(plotDict, ax, fine, cmap,norm,
                    foc1, foc2, _RTframes, RTcolor)

        if ret:
            return fig, ax
        
        elif save:
            pt.savefig(fname=self.savePath + 'system.pdf',bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()
        
        #pt.rcParams['xtick.minor.visible'] = True
        #pt.rcParams['ytick.minor.visible'] = True
    
    ##
    # Plot a group of reflectors.
    #
    # @param name_group Name of group to be plotted.
    def plotGroup(self, name_group, show=True, ret=False):
        print(self.groups[name_group]["members"])
        select = [x for x in self.groups[name_group]["members"]]

        if ret:
            fig, ax = self.plotSystem(select=select, show=False, ret=True)
            return fig,ax
        else:
            self.plotSystem(select=select, show=show)

    ##
    # Create a spot diagram of a ray-trace frame.
    #
    # @param name_frame Name of frame to plot.
    # @param project Set abscissa and ordinate of plot. Should be given as a string. Default is "xy".
    # @param ret Return Figure and Axis. Default is False.
    # @param aspect Aspect ratio of plot. Default is 1.
    def plotRTframe(self, name_frame, project="xy", ret=False, aspect=1):
        check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        if ret:
            return plt.plotRTframe(self.frames[name_frame], project, self.savePath, ret, aspect)
        else:
            plt.plotRTframe(self.frames[name_frame], project, self.savePath, ret, aspect)

    ##
    # Create a deep copy of any object.
    # 
    # @param obj Object do be deepcopied.
    #
    # @returns copy A deepcopy of obj.
    def copyObj(self, obj=None):
        obj = self if obj is None else obj
        return copy.deepcopy(obj)

    ##
    # Find rotation matrix to rotate v onto u.
    #
    # @param v Numpy array of length 3. 
    # @param u Numpy array of length 3.
    def findRotation(self, v, u):
        I = np.eye(3)
        if np.array_equal(v, u):
            return world.INITM

        lenv = np.linalg.norm(v)
        lenu = np.linalg.norm(u)

        if lenv == 0 or lenu == 0:
            self.clog.error("Encountered 0-length vector. Cannot proceed.")
            exit(0)

        w = np.cross(v/lenv, u/lenu)

        lenw = np.linalg.norm(w)
        
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        R = I + K + K @ K * (1 - np.dot(v, u)) / lenw**2

        R_transf = world.INITM
        R_transf[:-1, :-1] = R
        
        return R_transf

    ##
    # Find the focus of a ray-trace frame.
    # Adds a new plane to the System, perpendicular to the mean ray-trace tilt of the input frame.
    # After completion, the new plane is centered at the ray-trace focus.
    #
    # @param name_frame Name of the input frame.
    # @param f0 Initial try for focal distance.
    # @param verbose Allow verbose System logging.
    #
    # @returns out The focus co-ordinate.
    def findRTfocus(self, name_frame, f0=None, verbose=False):
        check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        f0 = 0 if f0 is None else f0
       
        if not verbose:
            self.setLoggingVerbosity(verbose=False)
        
        tilt = self.calcRTtilt(name_frame)
        center = self.calcRTcenter(name_frame)
        match = world.IAX

        R = self.findRotation(match, tilt)

        t_name = f"focal_plane_{name_frame}"
        fr_out = f"focus_{name_frame}"

        target = {
                "name"      : t_name,
                "gmode"     : "xy",
                "lims_x"    : np.array([-42, 42]),
                "lims_y"    : np.array([-42, 42]),
                "gridsize"  : np.array([3, 3])
                }

        self.addPlane(target)
        self.system[t_name]["transf"] = R 
        self.translateGrids(t_name, center)
        
        runRTDict = {
                "fr_in"     : name_frame,
                "fr_out"    : fr_out,
                "t_name"    : t_name,
                "device"    : "CPU",
                "nThreads"  : 1
                }

        self.clog.info(f"Finding focus of {name_frame}...")
        res = opt.fmin(self._optimiseFocus, f0, args=(runRTDict, tilt), full_output=True, disp=False)

        out = res[0] * tilt + center
        self.clog.info(f"Focus: {*['{:0.3e}'.format(x) for x in out],}, RMS: {res[1]:.3e}")

        if not verbose:
            self.setLoggingVerbosity(verbose=True)

        return out

        
    ##
    # Find x, y and z rotation angles from general rotation matrix.
    #
    # @param M Numpy array of shape (3,3) containg a general rotation matrix.
    #
    # @returns r Numpy array of length 3 containing rotation angles around x, y and z.
    def getAnglesFromMatrix(self, M):
        if M[2,0] < 1:
            if M[2,0] > -1:
                ry = np.arcsin(-M[2,0])
                rz = np.arctan2(M[1,0], M[0,0])
                rx = np.arctan2(M[2,1], M[2,2])

            else:
                ry = np.pi / 2
                rz = -np.arctan2(-M[1,2], M[1,1])
                rx = 0

        else:
            ry = -np.pi / 2
            rz = np.arctan2(-M[1,2], M[1,1])
            rx = 0

        r = np.degrees(np.array([rx, ry, rz]))

        testM = MatRotate(r, world.INITM, pivot=None, radians=False)

        return r#, testM
    
    #############################################################
    #                                                           #
    #                         GUI METHODS                       #
    #                                                           #
    #############################################################
    
    ##
    # Instantiate a GUI PO propagation. Stores desired output in the system.fields and/or system.currents lists.
    # If the 'EHP' mode is selected, the reflected Poynting frame is stored in system.frames.
    #
    # @param PODict Dictionary containing the PO propagation instructions.
    #
    # @see PODict
    def runGUIPO(self, runPODict):
        _runPODict = self.copyObj(runPODict)

        if _runPODict["mode"] != "scalar":
            sc_name = _runPODict["s_current"]
            _runPODict["s_current"] = self.currents[_runPODict["s_current"]]
            source = self.system[_runPODict["s_current"].surf]
            _runPODict["k"] = _runPODict["s_current"].k

        else:
            sc_name = _runPODict["s_scalarfield"]
            _runPODict["s_scalarfield"] = self.scalarfields[_runPODict["s_scalarfield"]]
            source = self.system[_runPODict["s_scalarfield"].surf]
            _runPODict["k"] = _runPODict["s_scalarfield"].k
       
        target = self.system[_runPODict["t_name"]]
        
        if _runPODict["device"] == "CPU":
            out = PyPO_CPUd(source, target, _runPODict)

        elif _runPODict["device"] == "GPU":
            out = PyPO_GPUf(source, target, _runPODict)
        
        if _runPODict["mode"] == "JM":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.currents[_runPODict["name_JM"]] = out
        
        elif _runPODict["mode"] == "EH" or _runPODict["mode"] == "FF":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.fields[_runPODict["name_EH"]] = out
        
        elif _runPODict["mode"] == "JMEH":
            out[0].setMeta(_runPODict["t_name"], _runPODict["k"])
            out[1].setMeta(_runPODict["t_name"], _runPODict["k"])
            self.currents[_runPODict["name_JM"]] = out[0]
            self.fields[_runPODict["name_EH"]] = out[1]
        
        elif _runPODict["mode"] == "EHP":
            out[0].setMeta(_runPODict["t_name"], _runPODict["k"])
            self.fields[_runPODict["name_EH"]] = out[0]

            frame = self.loadFramePoynt(out[1], _runPODict["t_name"])
            self.frames[_runPODict["name_P"]] = frame

        elif _runPODict["mode"] == "scalar":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.scalarfields[_runPODict["name_field"]] = out

        return out
    
    #############################################################
    #                                                           #
    #                       PRIVATE METHODS                     #
    #                                                           #
    #############################################################
    
    ##
    # Cost function for finding a ray-trace frame focus.
    # Optimises RMS spot size as function of tilt multiple f0.
    #
    # @param f0 Tilt multiple for finding focus.
    # @param args The runRTDict for propagation and ray-trace tilt of input frame.
    #
    # @returns RMS The RMS spot size of the frame at f0 times the tilt.
    def _optimiseFocus(self, f0, *args):
        runRTDict, tilt = args

        trans = f0 * tilt

        self.translateGrids(f"focal_plane_{runRTDict['fr_in']}", trans)
        
        self.runRayTracer(runRTDict)
        RMS = self.calcSpotRMS(f"focus_{runRTDict['fr_in']}")
        self.translateGrids(f"focal_plane_{runRTDict['fr_in']}", -trans)
        #self.removeFrame() 
        return RMS
    
    ##
    # Check if an element to be rotated is bound to a PO field/current.
    # If so, rotate vectorial field/current components along.
    #
    # @param name Name of reflector to be rotated.
    # @param rotation Array containing the rotation of the reflector.
    def _checkBoundPO(self, name, transf):

        bound_fields = []
        bound_currents = []

        for key, item in self.fields.items():
            if hasattr(item, "surf"):
                if item.surf == name:
                    bound_fields.append(key)
        
        for key, item in self.currents.items():
            if hasattr(item, "surf"):
                if item.surf == name:
                    bound_currents.append(key)

        if bound_fields:
            self.clog.debug("hi")
            for field in bound_fields:
                out = transformPO(self.fields[field], transf)
                self.fields[field] = self.copyObj(out)

        if bound_currents:
            for current in bound_currents:
                out = transformPO(self.currents[current], transf)
                self.currents[current] = self.copyObj(out)
   
    ##
    # Transform a single component to a filled fields object by setting all other components to zero.
    #
    # @param comp Name of component.
    # @param field Array to be inserted in fields object.
    #
    # @returns field_c Filled fields object with one component filled.
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

    ##
    # Convert a string representation of a unit to a list containing the unit and conversion factor.
    # The conversion is done with respect of the standard PyPO units, which are millimeters.
    # This method is only used for plotting.
    #
    # @param unit String representation of the unit.
    # @param default Default unit, millimeters.
    #
    # @returns out List containing the string unit and the corresponding conversion factor.
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
