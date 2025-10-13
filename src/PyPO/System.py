"""!
@file
System interface for PyPO.

This script contains the System class definition.
"""

# Standard Python imports
from scipy.optimize import fmin
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import numpy as np
import time
import os
import copy
import logging
from pathlib import Path
import pickle 
from typing import Self, Union

# PyPO-specific modules
import PyPO.BindRefl as BRefl
import PyPO.BindGPU as BGPU
import PyPO.BindCPU as BCPU
import PyPO.BindBeam as BBeam
import PyPO.BindTransf as BTransf
import PyPO.MatTransform as MatTransf
import PyPO.PyPOTypes as PTypes
import PyPO.Checks as PChecks
import PyPO.Config as Config
from PyPO.CustomLogger import CustomLogger
import PyPO.Plotter as PPlot
import PyPO.Efficiencies as Effs
import PyPO.FitGauss as FGauss
from PyPO.Enums import Projections, FieldComponents, CurrentComponents, Units, Scales, Objects, Modes

import PyPO.WorldParam as world

from traceback import print_tb

logging.getLogger(__name__)

# type hinting objects
fieldOrCurrentComponents = Union[FieldComponents, CurrentComponents]
contourLevels = Union[int, np.ndarray]

class System(object):
    customBeamPath = os.getcwd()
    customReflPath = os.getcwd()
    savePathSystems = os.getcwd()

    def __init__(self, redirect : logging.RootLogger = None, verbose : bool = True, override : bool = True):
        """!
        Constructor. Initializes system state.
        
        @param redirect Redirect all print statements within system to given stdout. To use, pass an instance of logging.RootLogger object or an object that inherrits from logging.RootLogger
        @param verbose Enable system logger.
        @param override Allow overriding names.
        """
        self.verbosity = verbose

        Config.setOverride(override)

        self.system = {}
        self.frames = {}
        self.fields = {}
        self.currents = {}
        self.scalarfields = {}
        self.groups = {}
        self.assoc = {}

        saveSystemsExist = os.path.isdir(self.savePathSystems)

        if not saveSystemsExist:
            os.makedirs(self.savePathSystems)
        
        self.savePath = os.getcwd() 

        existSave = os.path.isdir(self.savePath)

        if not existSave:
            os.makedirs(self.savePath)
        
        #redirect = None
        if redirect is None:
            self.clog_mgr = CustomLogger(os.path.basename(__file__))
            self.clog = self.clog_mgr.getCustomLogger()

        else:
            self.clog = redirect

        if not verbose:
            self.clog.setLevel(logging.CRITICAL)

        # if context == "S":
        #     self.clog.info("INITIALIZED EMPTY SYSTEM.")
 
        if override == True:
            self.clog.warning("System override set to True.")
       
    #def __del__(self):
    #    """!
    #    Destructor. Deletes any reference to the logger assigned to current system.
    #    """
    #    if self.context != "G":
    #        self.clog.info("EXITING SYSTEM.")
    #        del self.clog_mgr
    #        del self.clog
    
    def getSystemLogger(self) -> logging.RootLogger:
        """!
        Obtain a reference to the custom logger used by system.
        
        This method can be called to obtain a reference to the logging object that PyPO uses internally.
        Can be convenient in cases one wants to log their own information in the layout of the PyPO logger.
        
        @ingroup public_api_sysio
        
        @returns clog Reference to system logger.
        """
        return self.clog

    def __str__(self) -> str:
        """!
        Print system contents.
        """
        
        s = "Contents of system:\n"
        s += f"Reflectors {len(self.system)}:\n"
        for key in self.system.keys():
            s += f"    {key}\n"
        s += f"Groups {len(self.groups)}:\n"
        for key in self.groups.keys():
            s += f"    {key}\n"
        s += f"Frames {len(self.frames)}:\n"
        for key in self.frames.keys():
            s += f"    {key}\n"
        s += f"Currents {len(self.currents)}:\n"
        for key in self.currents.keys():
            s += f"    {key}\n"
        s += f"Fields {len(self.fields)}:\n"
        for key in self.fields.keys():
            s += f"    {key}\n"
        s += f"Scalar Fields {len(self.scalarfields)}:\n"
        for key in self.scalarfields.keys():
            s += f"    {key}\n"
        return s

    def setCustomBeamPath(self, path: str , append : bool = False) -> None:
        """!
        Set path to folder containing custom beam patterns.
        
        Set the path to the directory where PyPO looks for custom beam patterns.
        When a custom beam pattern is imported using the readCustomBeam() method, PyPO will look in the specified path for the beam patterns. 
        
        @ingroup public_api_sysio
        
        @param path Path to custom beams.
        @param append Whether path is relative to current working directory or absolute from home.
        """
        if append:
            self.customBeamPath = os.path.join(self.customBeamPath, path)
        else:
            self.customBeamPath = path

    def setSavePath(self, path : str, append : bool = False):
        """!
        Set path to folder were to save output plots.
        
        Set the path to the directory where PyPO saves output plots.
        
        @ingroup public_api_sysio
        
        @param path Path to save directory.
        @param append Whether path is relative to current working directory or absolute from home.
        """
        
        if append:
            self.savePath = os.path.join(self.savePath, path)

        else:
            self.savePath = path

        if not os.path.isdir(self.savePath):
            os.makedirs(self.savePath)
 
    def setSavePathSystems(self, path, append=False):
        """!
        Set path to folder were to save systems.
        
        Set the path to the directory where PyPO saves systems.
        
        @ingroup public_api_sysio
        
        @param path Path to save directory.
        @param append Whether path is relative to current working directory or absolute from home.
        """
        
        if append:
            self.savePathSystems = os.path.join(self.savePathSystems, path)

        else:
            self.savePathSystems = path

        if not os.path.isdir(self.savePathSystems):
            os.makedirs(self.savePathSystems)

    def setLoggingVerbosity(self, verbose : bool = True):
        """!
        Set the verbosity of the logging from within the system.
        
        Sometimes it is preferrable to not generate logging output to the console, for example when PyPO methods are called in for loops.
        This method allows for on/off switching of the logging unit so that the console is not flooded with logging messages.
        
        @ingroup public_api_sysio
        
        @param verbose Whether to enable logging or not.
        """

        self.verbosity = verbose
        if verbose == True:
            self.clog.setLevel(logging.DEBUG)
        else:
            self.clog.setLevel(logging.CRITICAL)

    def setOverride(self, override : bool = True):
        """!
        Set the override toggle.
        
        By setting "override" to False, PyPO will start appending number of occurences of a duplicate name instead of overwriting it.
        
        @ingroup public_api_sysio
        
        @param override Whether to override duplicate reflector names or not.
        """

        Config.setOverride(override)

    def addParabola(self, reflDict : dict):
        """!
        Add a paraboloid reflector to the System.
        
        This method takes a dictionary filled with the parameters for a paraboloid reflector and generates an internal dictionary for generating the reflectorgrids. 
        
        @ingroup public_api_reflmeths
        
        @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
        """
        
        reflDict["type"] = 0

        _reflDict = self.copyObj(reflDict)
        PChecks.check_ElemDict(_reflDict, self.system.keys(), self.clog) 
        self.system[_reflDict["name"]] = _reflDict

        if _reflDict["pmode"] == "focus":
            self.system[_reflDict["name"]]["coeffs"] = np.zeros(3)

            ve = _reflDict["vertex"] # Vertex point position
            f1 = _reflDict["focus_1"] # Focal point position

            diff = f1 - ve

            df = np.sqrt(np.dot(diff, diff))
            a = 2 * np.sqrt(df)
            b = a

            orientation = diff / np.sqrt(np.dot(diff, diff))
            offTrans = ve

            R = self.findRotation(world.IAX(), orientation)
            
            self.system[_reflDict["name"]]["transf"] = MatTransf.MatTranslate(offTrans, R)
            self.system[_reflDict["name"]]["pos"] = (self.system[_reflDict["name"]]["transf"] @ np.append(self.system[_reflDict["name"]]["pos"], 1))[:-1]
            self.system[_reflDict["name"]]["ori"] = R[:-1, :-1] @ self.system[_reflDict["name"]]["ori"]

            self._fillCoeffs(_reflDict["name"], a, b, -1)

        elif _reflDict["pmode"] == "manual":
            self.system[_reflDict["name"]]["coeffs"] = np.array([_reflDict["coeffs"][0], _reflDict["coeffs"][1], -1])

        if _reflDict["gmode"] == "xy" or _reflDict["gmode"] == 0:
            self.system[_reflDict["name"]]["gmode"] = 0

        elif _reflDict["gmode"] == "uv" or _reflDict["gmode"] == 1:
            self.system[_reflDict["name"]]["gmode"] = 1
        
        self.system[_reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added paraboloid {_reflDict['name']} to system.")

    def addHyperbola(self, reflDict : dict):
        """!
        Add a hyperboloid reflector to the System.
        
        This method takes a dictionary filled with the parameters for a hyperboloid reflector and generates an internal dictionary for generating the reflectorgrids. 
        
        @ingroup public_api_reflmeths
        
        @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
        """
        
        reflDict["type"] = 1
        _reflDict = self.copyObj(reflDict)
        PChecks.check_ElemDict(_reflDict, self.system.keys(), self.clog) 
        self.system[_reflDict["name"]] = _reflDict
        
        if _reflDict["pmode"] == "focus":
            self.system[_reflDict["name"]]["coeffs"] = np.zeros(3)
            # Calculate a, b, c of hyperbola
            f1 = _reflDict["focus_1"] # Focal point 1 position
            f2 = _reflDict["focus_2"] # Focal point 2 position
            ecc = _reflDict["ecc"] # Eccentricity of hyperbola

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
            R = self.findRotation(world.IAX(), orientation)
            
            self.system[_reflDict["name"]]["transf"] = MatTransf.MatTranslate(offTrans, R)

            self.system[_reflDict["name"]]["pos"] = (self.system[_reflDict["name"]]["transf"] @ np.append(self.system[_reflDict["name"]]["pos"], 1))[:-1]
            self.system[_reflDict["name"]]["ori"] = R[:-1, :-1] @ self.system[_reflDict["name"]]["ori"]

            self._fillCoeffs(_reflDict["name"], a3, b3, c3)

        if _reflDict["gmode"] == "xy" or _reflDict["gmode"] == 0:
            self.system[_reflDict["name"]]["gmode"] = 0

        elif _reflDict["gmode"] == "uv" or _reflDict["gmode"] == 1:
            self.system[_reflDict["name"]]["gmode"] = 1

        self.system[_reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added hyperboloid {_reflDict['name']} to system.")

    def addEllipse(self, reflDict : dict):
        """!
        Add an ellipsoid reflector to the System.
        
        This method takes a dictionary filled with the parameters for an ellipsoid reflector and generates an internal dictionary for generating the reflectorgrids. 
        
        @ingroup public_api_reflmeths
        
        @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
        """
        
        reflDict["type"] = 2
        _reflDict = self.copyObj(reflDict)
        PChecks.check_ElemDict(_reflDict, self.system.keys(), self.clog) 
        self.system[_reflDict["name"]] = _reflDict
        
        if _reflDict["pmode"] == "focus":
            self.system[_reflDict["name"]]["coeffs"] = np.zeros(3)
            f1 = _reflDict["focus_1"]
            f2 = _reflDict["focus_2"]
            ecc = _reflDict["ecc"]

            diff = f1 - f2

            trans = (f1 + f2) / 2

            f_norm = diff / np.linalg.norm(diff)
            
            if _reflDict["orient"] == "x":
                R = self.findRotation(np.array([1,0,0]), f_norm)
            
            if _reflDict["orient"] == "z":
                R = self.findRotation(np.array([0,0,1]), f_norm)

            a = np.sqrt(np.dot(diff, diff)) / (2 * ecc)
            b = a * np.sqrt(1 - ecc**2)
            
            self.system[_reflDict["name"]]["transf"] = MatTransf.MatTranslate(trans, R)
            
            self.system[_reflDict["name"]]["pos"] = (self.system[_reflDict["name"]]["transf"] @ np.append(self.system[_reflDict["name"]]["pos"], 1))[:-1]
            self.system[_reflDict["name"]]["ori"] = R[:-1, :-1] @ self.system[_reflDict["name"]]["ori"]

            if _reflDict["orient"] == "x":
                self._fillCoeffs(_reflDict["name"], a, b, b)
            
            if _reflDict["orient"] == "z":
                self._fillCoeffs(_reflDict["name"], b, b, a)

        if _reflDict["gmode"] == "xy" or _reflDict["gmode"] == 0:
            self.system[_reflDict["name"]]["gmode"] = 0

        elif _reflDict["gmode"] == "uv" or _reflDict["gmode"] == 1:
            self.system[_reflDict["name"]]["gmode"] = 1

        PChecks.check_ellipseLimits(self.system[_reflDict["name"]], self.clog)

        self.system[_reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added ellipsoid {_reflDict['name']} to system.")

    def addPlane(self, reflDict : dict):
        """!
        Add a planar surface to the System.
        
        This method takes a dictionary filled with the parameters for a planar reflector and generates an internal dictionary for generating the reflectorgrids. 
        
        @ingroup public_api_reflmeths
        
        @param reflDict A filled reflectordictionary. Will raise an exception if not properly filled.
        """
        
        reflDict["type"] = 3
        _reflDict = self.copyObj(reflDict)
        PChecks.check_ElemDict(_reflDict, self.system.keys(), self.clog) 
        
        self.system[_reflDict["name"]] = _reflDict
        self.system[_reflDict["name"]]["coeffs"] = np.zeros(3)

        if _reflDict["gmode"] == "xy" or _reflDict["gmode"] == 0:
            self.system[_reflDict["name"]]["gmode"] = 0

        elif _reflDict["gmode"] == "uv" or _reflDict["gmode"] == 1:
            self.system[_reflDict["name"]]["gmode"] = 1

        elif _reflDict["gmode"] == "AoE" or _reflDict["gmode"] == 2:
            self.system[_reflDict["name"]]["gmode"] = 2

        self.system[_reflDict["name"]]["snapshots"] = {}
        self.clog.info(f"Added plane {_reflDict['name']} to system.")

    def rotateGrids(self, 
                    name : str, 
                    rotation : np.ndarray, 
                    obj : Objects = Objects.ELEMENT, 
                    mode : Modes = Modes.REL, 
                    pivot : np.ndarray = None, 
                    keep_pol : bool = False):
        """!
        Rotate reflector grids.
        
        Apply a rotation, around a center of rotation, to a reflector, group or frame.
        Note that an absolute orientation rotates the orientation such that it is oriented w.r.t. the z-axis.
        In this case, the pivot defaults to the origin and not to the specified pivot.
        In the case that a PO field and/or a PO current is associated with the reflector, the polarisation of the field and/or current is rotated along as well.
        This can be disabled by setting the "keep_pol" parameter to "True".
        
        @ingroup public_api_common 
        
        @param name Reflector name or list of reflector names.
        @param rotation Numpy ndarray of length 3, containing rotation angles around x, y and z axes, in degrees.
        @param obj Whether the name corresponds to a single element, group, or frame. Fields and currents are translated by translating the associated surface. Choose from Objects enum.
        @param mode Apply rotation relative to current orientation (Modes.REL), or rotate to specified orientation (Modes.ABS).
        @param pivot Numpy ndarray of length 3, containing pivot x, y and z co-ordinates, in mm. Defaults to pos. 
        @param keep_pol Keep polarisation of a field/current defined on the surface, if present.
            If True, does not rotate polarisation of any defined fields/currents along with the rotation of reflector.
            If False, rotates polarisation along with reflector. Defaults to False.
        """

        PChecks.check_array(rotation, self.clog)

        if obj == Objects.ELEMENT:
            PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
            pivot = self.system[name]["pos"] if pivot is None else pivot
            PChecks.check_array(pivot, self.clog)
            
            if mode == Modes.ABS:
                Rtot = self._absRotationMat(rotation, self.system[name]["ori"], pivot)
                self.system[name]["transf"] = Rtot @ self.system[name]["transf"]
                self.system[name]["transf"][:-1, :-1] = (MatTransf.MatRotate(rotation, pivot=pivot))[:-1, :-1]

                self.system[name]["pos"] = (Rtot @ np.append(self.system[name]["pos"], 1))[:-1]
                self.system[name]["ori"] = Rtot[:-1, :-1] @ self.system[name]["ori"]

                if not keep_pol:
                    self._checkBoundPO(name, Rtot)

                self.clog.info(f"Rotated element {name} to {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

            elif mode == Modes.REL:
                self.system[name]["transf"] = MatTransf.MatRotate(rotation, self.system[name]["transf"], pivot)
                
                self.system[name]["pos"] = (MatTransf.MatRotate(rotation, pivot=pivot) @ np.append(self.system[name]["pos"], 1))[:-1]
                self.system[name]["ori"] = MatTransf.MatRotate(rotation)[:-1, :-1] @ self.system[name]["ori"]
                
                if not keep_pol:
                    self._checkBoundPO(name, MatTransf.MatRotate(rotation))
            
                self.clog.info(f"Rotated element {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

        elif obj == Objects.GROUP:
            PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)
            pivot = self.groups[name]["pos"] if pivot is None else pivot
            PChecks.check_array(pivot, self.clog)
            
            if mode == Modes.ABS:
                Rtot = self._absRotationMat(rotation, self.groups[name]["ori"], pivot)
                
                for elem in self.groups[name]["members"]:
                    self.system[elem]["transf"] = Rtot @ self.system[elem]["transf"]
                    self.system[elem]["transf"][:-1, :-1] = (MatTransf.MatRotate(rotation, pivot=pivot))[:-1, :-1]

                    self.system[elem]["pos"] = (Rtot @ np.append(self.system[elem]["pos"], 1))[:-1]
                    self.system[elem]["ori"] = Rtot[:-1, :-1] @ self.system[elem]["ori"]
                    
                    if not keep_pol:
                        self._checkBoundPO(elem, Rtot)

                self.groups[name]["pos"] = (Rtot @ np.append(self.groups[name]["pos"], 1))[:-1]
                self.groups[name]["ori"] = Rtot[:-1, :-1] @ self.groups[name]["ori"]
                
                self.clog.info(f"Rotated group {name} to {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

            elif mode == Modes.REL:
                for elem in self.groups[name]["members"]:
                    self.system[elem]["transf"] = MatTransf.MatRotate(rotation, self.system[elem]["transf"], pivot)
                    
                    self.system[elem]["pos"] = (MatTransf.MatRotate(rotation, pivot=pivot) @ np.append(self.system[elem]["pos"], 1))[:-1]
                    self.system[elem]["ori"] = MatTransf.MatRotate(rotation)[:-1, :-1] @ self.system[elem]["ori"]
                    
                    if not keep_pol:
                        self._checkBoundPO(elem, MatTransf.MatRotate(rotation))

                self.groups[name]["pos"] = (MatTransf.MatRotate(rotation, pivot=pivot) @ np.append(self.groups[name]["pos"], 1))[:-1]
                self.groups[name]["ori"] = MatTransf.MatRotate(rotation)[:-1, :-1] @ self.groups[name]["ori"]
                
                self.clog.info(f"Rotated group {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

        if obj == Objects.FRAME:
            PChecks.check_frameSystem(name, self.frames, self.clog, extern=True)
            pivot = self.frames[name].pos if pivot is None else pivot
            PChecks.check_array(pivot, self.clog)
            
            if mode == Modes.ABS:
                Rtot = self._absRotationMat(rotation, self.frames[name].ori, pivot)

                self.frames[name].transf = Rtot
                _fr = BTransf.transformRays(self.frames[name])
                self.frames[name] = self.copyObj(_fr)

                self.frames[name].pos = (Rtot @ np.append(self.frames[name].pos, 1))[:-1]
                self.frames[name].ori = Rtot[:-1, :-1] @ self.frames[name].ori
                self.clog.info(f"Rotated element {name} to {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

            elif mode == Modes.REL:
                self.frames[name].transf = MatTransf.MatRotate(rotation, pivot=pivot)
                _fr = BTransf.transformRays(self.frames[name])
                self.frames[name] = self.copyObj(_fr)
                
                self.frames[name].pos = (MatTransf.MatRotate(rotation, pivot=pivot) @ np.append(self.frames[name].pos, 1))[:-1]
                self.frames[name].ori = MatTransf.MatRotate(rotation)[:-1, :-1] @ self.frames[name].ori
            
                self.clog.info(f"Rotated frame {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

    def translateGrids(self, 
                       name : str, 
                       translation : np.ndarray, 
                       obj : Objects = Objects.ELEMENT, 
                       mode : Modes = Modes.REL) -> None:
        """!
        Translate reflector grids.
        
        Apply a translation to a reflector, group or frame.
        If the translation is absolute, the object will be translated such that its internal position parameter coincides with the specified translation.
        
        @ingroup public_api_common 
        
        @param name Reflector name or list of reflector names.
        @param translation Numpy ndarray of length 3, containing translation x, y and z co-ordinates, in mm.
        @param obj Whether the name corresponds to a single element, group, or frame. Fields and currents are translated by translating the associated surface. Choose from Objects enum.
        @param mode Apply translation relative ('relative') to current position, or move to specified position ('absolute').
        """
        
        PChecks.check_array(translation, self.clog)

        _translation = self.copyObj(translation)
        
        if obj == Objects.ELEMENT:
            if mode == Modes.ABS:
                _translation -= self.system[name]["pos"]# - translation
            
            PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
            self.system[name]["transf"] = MatTransf.MatTranslate(_translation, self.system[name]["transf"])
            self.system[name]["pos"] += _translation
            
            if mode == Modes.ABS:
                self.clog.info(f"Translated element {name} to {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
            else:
                self.clog.info(f"Translated element {name} by {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
        
        elif obj == Objects.GROUP:
            if mode == Modes.ABS:
                _translation -= self.groups[name]["pos"]# - translation
            
            PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)
            for elem in self.groups[name]["members"]:
                self.system[elem]["transf"] = MatTransf.MatTranslate(_translation, self.system[elem]["transf"])
                self.system[elem]["pos"] += _translation
            
            self.groups[name]["pos"] += _translation
            
            if mode == Modes.ABS:
                self.clog.info(f"Translated group {name} to {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
            
            else:
                self.clog.info(f"Translated group {name} by {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")

        elif obj == Objects.FRAME:
            if mode == Modes.ABS:
                _translation -= self.frames[name].pos# - translation
            
            PChecks.check_frameSystem(name, self.frames, self.clog, extern=True)
            

            self.frames[name].transf = MatTransf.MatTranslate(_translation)
            _fr = BTransf.transformRays(self.frames[name])
            self.frames[name] = self.copyObj(_fr)
            self.frames[name].pos += _translation
            
            if mode == Modes.ABS:
                self.clog.info(f"Translated frame {name} to {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
            else:
                self.clog.info(f"Translated frame {name} by {*['{:0.3e}'.format(x) for x in list(_translation)],} millimeters.")
    
    def homeReflector(self, name : str, obj : Objects = Objects.ELEMENT, trans : bool = True, rot : bool = True):
        """!
        Home a reflector or a group back into default configuration.
        
        Set internal transformation matrix of a reflector or group to identity. 
        
        @ingroup public_api_reflmeths
        
        @param name Reflector name or list of reflector or group to be homed.
        @param obj Type of object to be homed.
        @param trans Home (translate) back to home position.
        @param rot Home (rotate) back to home orientation.
        """

        if obj == Objects.GROUP:
            PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)
            if trans:
                for elem in self.groups[name]["members"]:
                    _transf = self.copyObj(world.INITM())
                    _transf[:-1, :-1] = self.system[elem]["transf"][:-1, :-1]
                    self.system[elem]["transf"] = _transf
                    self.system[elem]["pos"] = self.copyObj(world.ORIGIN())

                self.groups[name]["pos"] = self.copyObj(world.ORIGIN())
            
            if rot:
                for elem in self.groups[name]["members"]:
                    _transf = self.system[elem]["transf"]
                    _transf[:-1, :-1] = np.eye(3)
                    self.system[elem]["transf"] = _transf
                
                self.groups[name]["ori"] = self.copyObj(world.IAX())
            
            self.clog.info(f"Transforming group {name} to home position.")

                    
        else:
            PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
            if trans:
                _transf = self.copyObj(world.INITM())
                _transf[:-1, :-1] = self.system[name]["transf"][:-1, :-1]
                self.system[name]["transf"] = _transf
                self.system[name]["pos"] = self.copyObj(world.ORIGIN())

            
            if rot:
                _transf = self.system[name]["transf"]
                _transf[:-1, :-1] = np.eye(3)
                self.system[name]["transf"] = _transf
                self.system[name]["ori"] = self.copyObj(world.IAX())
            
            self.clog.info(f"Transforming element {name} to home position.")
 
    def snapObj(self, name : str, snap_name : str, obj : Objects = Objects.ELEMENT):
        """!
        Take and store snapshot of object's current configuration.
        
        A snapshot consists of the transformation matrix of an element, group of frame.
        
        @ingroup public_api_reflmeths
        
        @param name Name of object to be snapped.
        @param snap_name Name of snapshot to save.
        @param obj Whether object is an element, group or frame.
        """

        if obj == Objects.GROUP:
            PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)
            self.groups[name]["snapshots"][snap_name] = []

            for elem in self.groups[name]["members"]:
                self.groups[name]["snapshots"][snap_name].append(self.copyObj(self.system[elem]["transf"]))
            
            self.clog.info(f"Saved snapshot {snap_name} for group {name}.")
        
        elif obj == Objects.ELEMENT:
            PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
            self.system[name]["snapshots"][snap_name] = self.copyObj(self.system[name]["transf"])

            self.clog.info(f"Saved snapshot {snap_name} for element {name}.")
        
        elif obj == Objects.FRAME:
            PChecks.check_frameSystem(name, self.frames, self.clog, extern=True)
            self.frames[name].snapshots[snap_name] = self.copyObj(self.frames[name].transf)

            self.clog.info(f"Saved snapshot {snap_name} for frame {name}.")
    
    def revertToSnap(self, name : str, snap_name : str, obj : Objects = Objects.ELEMENT):
        """!
        Revert object configuration to a saved snapshot.
        
        Reverting an object to a snapshot replaces the internal transformation matrix with the matrix stored in the snapshot.
        
        @ingroup public_api_reflmeths
        
        @param name Name of obj to revert.
        @param snap_name Name of snapshot to revert to.
        @param obj Whether object is an element, group or frame.
        """

        if obj == Objects.GROUP:
            PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)

            for elem, snap in zip(self.groups[name]["members"], self.groups[name]["snapshots"][snap_name]):
                self._checkBoundPO(elem, MatTransf.InvertMat(self.system[elem]["transf"]))
                self.system[elem]["transf"] = self.copyObj(snap)
                self._checkBoundPO(elem, self.system[elem]["transf"])
            
            self.clog.info(f"Reverted group {name} to snapshot {snap_name}.")

        elif obj == Objects.ELEMENT:
            PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
            self._checkBoundPO(name, MatTransf.InvertMat(self.system[name]["transf"]))
            self.system[name]["transf"] = self.copyObj(self.system[name]["snapshots"][snap_name])
            self._checkBoundPO(name, self.system[name]["transf"]) 
            self.clog.info(f"Reverted element {name} to snapshot {snap_name}.")
        
        elif obj == Objects.FRAME:
            PChecks.check_frameSystem(name, self.frames, self.clog, extern=True)
            self.frames[name].transf = self.copyObj(self.frames[name].snapshots[snap_name])
            
            _fr = BTransf.transformRays(self.frames[name])
            self.frames[name] = self.copyObj(_fr)
            
            self.clog.info(f"Reverted frame {name} to snapshot {snap_name}.")
    
    def deleteSnap(self, name : str, snap_name : str, obj : Objects = Objects.ELEMENT):
        """!
        Delete a saved snapshot belonging to an object.
        
        This deletes the stored snapshot, including the associated transformation matrix.
        
        @ingroup public_api_reflmeths
        
        @param name Name of object.
        @param snap_name Name of snapshot to delete.
        @param obj Whether object is an element or a group.
        """

        if obj == Objects.GROUP:
            PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)
           
            del self.groups[name]["snapshots"][snap_name]

            self.clog.info(f"Deleted snapshot {snap_name} belonging to group {name}.")

        elif obj == Objects.ELEMENT:
            del self.system[name]["snapshots"][snap_name]

            self.clog.info(f"Deleted snapshot {snap_name} belonging to element {name}.")
        
        elif obj == Objects.FRAME:
            del self.frames[name].snapshots[snap_name]

            self.clog.info(f"Deleted snapshot {snap_name} belonging to frame {name}.")
    
    def groupElements(self, name_group : str, *names : str, pos : np.ndarray = None, ori : np.ndarray = None):
        """!
        Group elements together into a single block. After grouping, can translate, rotate and characterise as one.
        
        This method adds a new field to the internal groups dictionary, with the key being the given name of the group.
        The item corresponding to this key is, again, a dictionary. The first key, "members", corresponds to a list containing the names of the reflectors added to the group.
        The position and orientation trackers of the group are set to the origin and the z-axis, respectively, if not passed as arguments.
        
        @ingroup public_api_reflmeths
        
        @param names Names of elements to put in group.
        @param name_group Name of the group.
        @param pos numpy.ndarray of length 3, position tracker for the group. Defaults to origin.
        @param ori numpy.ndarray of length 3, orientation tracker for group. Deraults to unit vector in z direction. 
        """
        
        num = PChecks.getIndex(name_group, self.groups.keys())

        if num > 0:
            name_group = name_group + "_{}".format(num)
        pos = world.ORIGIN() if pos is None else pos
        ori = world.IAX() if ori is None else ori

        for _name in names:
            PChecks.check_elemSystem(_name, self.system, self.clog, extern=True)

        self.groups[name_group] = {
                "members"   : list(names),
                "pos"       : pos,
                "ori"       : ori,
                "snapshots" : {}
                }
        self.clog.info(f"Grouped elements {names} into group {name_group}.")

    def removeGroup(self, name_group : str):
        """!
        Remove a group of elements from system. Note that this does not remove the elements inside the group.
        
        Removing a group only removes the key and item in the internal groups dictionary and does not remove the contained elements.
        
        @ingroup public_api_reflmeths
        
        @param name_group Name of the group to be removed.
        """
        
        PChecks.check_groupSystem(name_group, self.groups, self.clog, extern=True)
        del self.groups[name_group]
        
        self.clog.info(f"Removed group {name_group} from system.")

    def generateGrids(self, 
                      name : str, 
                      transform :bool = True, 
                      spheric : bool = True
                      ) -> PTypes.reflGrids:
        """!
        Generate reflector grids and normals.
        
        Evaluate a stored reflector dictionary and return the x, y, z grids, area and normals.
        The outputs are grouped together in a grids object.
        
        @ingroup public_api_reflmeths
        
        @param name Name of reflector to be gridded.
        @param transform Apply internal transformation matrix to reflector.
        @param spheric Return spheric or square far-field grid (far-field only).
        
        @return grids A reflGrids object containing the grids, area and normals.
        
        @see reflGrids
        """

        PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
        grids = BRefl.generateGrid(self.system[name], transform, spheric)
        return grids
    
    def saveSystem(self, name : str):
        """!
        Save a system object to disk. 
        
        The system from which this method is called will be saved in its entirety, e.g. all reflectors, fields, currents and frames, to disk.
        The directory to which the system will be saved will either be the current working directory or the directory specified with setSavePathSystems().
        
        @ingroup public_api_sysio
        
        @param name Save the current system under this name.
        """
       
        with open(os.path.join(self.savePathSystems, f"{name}.pyposystem"), 'wb') as file:
            pickle.dump(self.__dict__, file)
        
        self.clog.info(f"Saved current system to {name}.")

    def loadSystem(self, name : str):
        """!
        Load a system object from the savePathSystems path. This loads all reflectors, fields, currents and frames in the system from disk.
        
        The system from which this method is called will be overwritten in its entirety, e.g. all reflectors, fields, currents and frames will be replaced with the 
        internal dictionaries of the system to load.
        The directory from which the system will be saved will either be the current working directory or the directory specified with setSavePathSystems().
        
        @ingroup public_api_sysio
        
        @param name Load the system under this name.
        """

        self.clog.info(f"Loading system {name} from {self.savePathSystems} into current system.")
        path = os.path.join(self.savePathSystems, f"{name}.pyposystem")
        loadExist = os.path.isfile(path)

        if not loadExist:
            self.clog.error("Specified system does not exist.")
            exit(1)

        try:
            with open(os.path.join(self.savePathSystems, f"{name}.pyposystem"), 'rb') as file: 
                self.__dict__ = pickle.load(file)
        except Exception as err:
            print_tb(err.__traceback__)

    def mergeSystem(self, *systems : Self):
        """!
        Merge multiple systems together into current system.
        
        This method takes a arbitrary amount of system objects and updates the internal dictionaries of the system from which this method is called.
        This means that the calling system will keep its internal dictionaries. However, if the systems contain a key in the internal dictionaries
        matching a key in the calling system, this key and item will be overwritten.
        
        @ingroup public_api_sysio
        
        @param systems Systems to be merged into current system
        """

        if len(set(systems)) < len(systems):
            raise Exception("Cannot merge duplicate systems.")
        for sysObject in systems:
            sys_copy = self.copyObj(sysObject)
            
            self.system.update(sys_copy.system)
            self.fields.update(sys_copy.fields)
            self.currents.update(sys_copy.currents)
            self.frames.update(sys_copy.frames)
            self.groups.update(sys_copy.groups)
            self.scalarfields.update(sys_copy.scalarfields)
    
    def removeElement(self, name : str):
        """!
        Remove reflector from system.
        
        This removes the key and corresponding reflector dictionary in the internal system dictionary.
        It also check whether the element is included in a group.
        If it is included in a group, the method will also remove the element from the "members" list of the group.
        
        @ingroup public_api_reflmeths
        
        @ param name Name of reflector to be removed.
        """

        PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
        for group in self.groups.values():
            if name in group["members"]:
                group["members"].remove(name)
        del self.system[name]
        self.clog.info(f"Removed element {name} from system.")
    
    def copyElement(self, name : str, name_copy : str):
        """!
        Copy reflector.
        
        This method takes an internal reflector dictionary and generates a deepcopy, i.e. a true copy.
        The copy can be adjusted safely without changing the contents of the original.
        
        @ingroup public_api_reflmeths
        
        @ param name Name of reflector to be copied.
        @ param name_copy Name of new reflector.
        """

        PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
        self.system[name_copy] = self.copyObj(self.system[name])
        self.clog.info(f"Copied element {name} to {name_copy}.")
    
    def copyGroup(self, name : str, name_copy : str):
        """!
        Copy group.
        
        This method takes a group name and generates a deepcopy, i.e. a true copy.
        The copy can be adjusted safely without changing the contents of the original.
        Note however that the elements themselves are not deepcopied.
        This might change later on.
        
        @ingroup public_api_reflmeths
        
        @ param name Name of group to be copied.
        @ param name_copy Name of new group.
        """

        PChecks.check_groupSystem(name, self.groups, self.clog, extern=True)
        self.groups[name_copy] = self.copyObj(self.groups[name])
        self.clog.info(f"Copied group {name} to {name_copy}.")
    
    def removeFrame(self, frameName : str):
        """!
        Remove a ray-trace frame from system.
        
        This method takes the name of an internally stored frame object and removes it from the internal dictionary.
        
        @ingroup public_api_frames
        
        @param frameName Name of frame to be removed.
        """

        PChecks.check_frameSystem(frameName, self.frames, self.clog, extern=True)
        del self.frames[frameName]
        
        self.clog.info(f"Removed frame {frameName} from system.")
    
    def removeField(self, fieldName : str):
        """!
        Remove a PO field from system.
        
        This method takes the name of an internally stored PO field object and removes it from the internal dictionary.
        
        @ingroup public_api_po
        
        @param fieldName Name of field to be removed.
        """

        PChecks.check_fieldSystem(fieldName, self.fields, self.clog, extern=True)
        del self.fields[fieldName]
        
        self.clog.info(f"Removed PO field {fieldName} from system.")
    
    def removeCurrent(self, currentName : str):
        """!
        Remove a PO current from system.
        
        This method takes the name of an internally stored PO current object and removes it from the internal dictionary.
        
        @ingroup public_api_po
        
        @param curentName Name of current to be removed.
        """

        PChecks.check_currentSystem(currentName, self.currents, self.clog, extern=True)
        del self.currents[currentName]
        
        self.clog.info(f"Removed PO current {currentName} from system.")

    def removeScalarField(self, fieldName : str):
        """!
        Remove a scalar PO field from system.
        
        This method takes the name of an internally stored PO scalarfield object and removes it from the internal dictionary.
        
        @ingroup public_api_po
        
        @param fieldName Name of scalar field to be removed.
        """

        PChecks.check_scalarfieldSystem(fieldName, self.scalarfields, self.clog, extern=True)
        del self.scalarfields[fieldName]
        
        self.clog.info(f"Removed scalar PO field {fieldName} from system.")
    
    def readCustomBeam(self, name_beam : str, name_source : str, comp : FieldComponents, lam : float, normalise : bool = True, scale : float = 1, outname : str = None):
        """!
        Read a custom beam from disk into the system. 
        
        The system will look in the customBeamPath, which defaults to the current working directory and can be set with the setCustomBeamPath() method.
        Note that the custom beam pattern needs to contain a real and imaginary part, and that these need to be stored in separate .txt files, stored as
        such: r<name_beam>.txt and i<name_beam>.txt, where 'r' and 'i' refer to the real and imaginary part, respectively.
        If the beam pattern is a component of the E-field, the currents will be calculated assuming a PMC surface.
        If, on the other hand, the beam pattern is a component of the H-field, the currents will be calculated assuming a PEC surface.

        @ingroup public_api_po
        
        @param name_beam Name of the beam (without the 'r' or 'i' prefixes or '.txt' suffix).
        @param name_source Name of source surface on which to define the beam. 
        @param comp Polarisation component of beam. Instance of FieldComponents enum object.
        @param lam Wavelength of beam, in mm.
        @param normalise Whether or not to normalise beam to its maximum amplitude.
        @param scale Scale factor for beam. Defaults to 1.
        @param outname Name of field/current objects written to system. Defaults to name_beam
        
        @see setCustomBeamPath
        """

        outname = name_beam if outname is None else outname

        PChecks.check_elemSystem(name_source, self.system, self.clog, extern=True)
        
        rfield = np.loadtxt(os.path.join(self.customBeamPath, "r" + name_beam + ".txt"))
        ifield = np.loadtxt(os.path.join(self.customBeamPath, "i" + name_beam + ".txt"))

        field = (rfield + 1j*ifield).T

        if normalise:
            maxf = np.max(field)
            field /= maxf
            field *= scale

        k = 2 * np.pi / lam
 
        shape = self.system[name_source]["gridsize"]

        if comp.value < 4:
            mode = "PMC"
        
        else:
            mode = "PEC"

        fields_c = self._compToFields(comp, field)
        fields_c.setMeta(name_source, k)
        self.fields[outname] = fields_c#.H()
        currents_c = BBeam.calcCurrents(fields_c, self.system[name_source], mode)
        currents_c.setMeta(name_source, k)

        self.currents[outname] = currents_c

    def runPO(self, runPODict : dict):
        """!
        Instantiate a PO propagation. 
        
        Stores desired output in the internal fields and/or internal currents dictionary.
        If the 'EHP' mode is selected, the reflected Poynting frame is stored in the internal frame dictionary.
        
        @ingroup public_api_po
        
        @param PODict Dictionary containing the PO propagation instructions.
        
        @see PODict
        """

        self.clog.work("*** Starting PO propagation ***")
       
        PChecks.check_runPODict(runPODict, self.system.keys(), self.fields.keys(), self.currents.keys(),
                    self.scalarfields.keys(), self.frames.keys(), self.clog)

        _runPODict = self.copyObj(runPODict)

        if _runPODict["mode"] != "scalar":
            sc_name = _runPODict["s_current"]
            _runPODict["s_current"] = self.currents[_runPODict["s_current"]]
            self.clog.work(f"Propagating {sc_name} on {_runPODict['s_current'].surf} to {_runPODict['t_name']}, propagation mode: {_runPODict['mode']}.")
            source = self.system[_runPODict["s_current"].surf]
            _runPODict["k"] = _runPODict["s_current"].k

        else:
            sc_name = _runPODict["s_scalarfield"]
            _runPODict["s_scalarfield"] = self.scalarfields[_runPODict["s_scalarfield"]]
            self.clog.work(f"Propagating {sc_name} on {_runPODict['s_scalarfield'].surf} to {_runPODict['t_name']}, propagation mode: {_runPODict['mode']}.")
            source = self.system[_runPODict["s_scalarfield"].surf]
            _runPODict["k"] = _runPODict["s_scalarfield"].k
       

        target = self.system[_runPODict["t_name"]]
        
        start_time = time.time()
        
        if _runPODict["device"] == "CPU":
            self.clog.work(f"Hardware: running {_runPODict['nThreads']} CPU threads.")
            self.clog.work(f"... Calculating ...")
            out = BCPU.PyPO_CPUd(source, target, _runPODict)

        elif _runPODict["device"] == "GPU":
            self.clog.work(f"Hardware: running {_runPODict['nThreads']} CUDA threads per block.")
            self.clog.work(f"... Calculating ...")
            out = BGPU.PyPO_GPUf(source, target, _runPODict)

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

            frame = self._loadFramePoynt(out[1], _runPODict["t_name"])
            self.frames[_runPODict["name_P"]] = frame
       
            self.assoc[_runPODict["t_name"]] = [_runPODict["name_EH"], _runPODict["name_P"]]

        elif _runPODict["mode"] == "scalar":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.scalarfields[_runPODict["name_field"]] = out

        self.clog.work(f"*** Finished: {dtime:.3f} seconds ***")
        return out

    def mergeBeams(self, *beams : str, obj : Objects = Objects.FIELD, merged_name : str="combined"):
        """!
        Merge multiple beams that are defined on the same surface.
        
        The beams to be merged are first checked to see if they are all defined on the same surface.
        Then, a new PO field or current is defined in the internal dictionary with the new name.
        
        @ingroup public_api_po
        
        @param beams Fields or currents objects to merge.
        @param obj Whether the beams are PO fields or currents.
        @param merged_name Name of merged object.
        """

        PChecks.check_sameBound(beams, checkDict=getattr(self, obj.value), clog=self.clog) 
        ex = getattr(self, obj.value)[beams[0]]

        x1 = np.zeros(ex[0].shape, dtype=complex)
        x2 = np.zeros(ex[0].shape, dtype=complex)
        x3 = np.zeros(ex[0].shape, dtype=complex)
        
        y1 = np.zeros(ex[0].shape, dtype=complex)
        y2 = np.zeros(ex[0].shape, dtype=complex)
        y3 = np.zeros(ex[0].shape, dtype=complex)
        
        for beam in beams:
            if obj == Objects.FIELD:
                PChecks.check_fieldSystem(beam, self.fields, self.clog, extern=True)
            if obj == Objects.CURRENT:
                PChecks.check_currentSystem(beam, self.currents, self.clog, extern=True)
            
            beam = getattr(self, obj.value)[beam]
            x1 += beam[0]
            x2 += beam[1]
            x3 += beam[2]
            
            y1 += beam[3]
            y2 += beam[4]
            y3 += beam[5]

        if obj == Objects.FIELD:
            field = PTypes.fields(x1, x2, x3, y1, y2, y3)
        if obj == Objects.CURRENT:
            field = PTypes.currents(x1, x2, x3, y1, y2, y3)
        
        field.setMeta(ex.surf, ex.k)
        getattr(self, obj.value)[merged_name] = field

    def createTubeFrame(self, argDict : dict):
        """!
        Create a tube of rays from a TubeRTDict.
        
        The tube of rays will be placed in the internal frame dictionary. 
        Position and orientation trackers for the frame are initialised to the origin and z-axis, respectively.
        
        @ingroup public_api_frames
        
        @param argDict A TubeRTDict, filled. If not filled properly, will raise an exception.
        
        @see TubeRTDict
        """

        if not argDict["name"]:
            argDict["name"] = f"Frame"
        _argDict = self.copyObj(argDict) 
        PChecks.check_TubeRTDict(_argDict, self.frames.keys(), self.clog)
        
        self.frames[_argDict["name"]] = BBeam.makeRTframe(_argDict)

        self.frames[_argDict["name"]].setMeta(self.copyObj(world.ORIGIN()), self.copyObj(world.IAX()), self.copyObj(world.INITM()))

        self.clog.info(f"Added tubular frame {_argDict['name']} to system.")
    
    def createGRTFrame(self, argDict : dict): 
        """!
        Create a Gaussian beam distribution of rays from a GRTDict.
        
        The Gaussian frame will be placed in the internal frame dictionary.
        The frame is generated by rejection-sampling a Gaussian distribution in position and direction on an xy-grid.
        Position and orientation trackers for the frame are initialised to the origin and z-axis, respectively.
        
        @ingroup public_api_frames
        
        @param argDict A GRTDict, filled. If not filled properly, will raise an exception.
        
        @see GRTDict
        """

        if not argDict["name"]:
            argDict["name"] = f"Frame"
        
        _argDict = self.copyObj(argDict) 
        PChecks.check_GRTDict(_argDict, self.frames.keys(), self.clog)
        
        self.clog.work(f"Generating Gaussian ray-trace beam.")
        self.clog.work(f"... Sampling ...")
       
        start_time = time.time()
        _argDict["angx0"] = np.degrees(_argDict["lam"] / (np.pi * _argDict["n"] * _argDict["x0"]))
        _argDict["angy0"] = np.degrees(_argDict["lam"] / (np.pi * _argDict["n"] * _argDict["y0"]))

        #check_RTDict(_argDict, self.frames.keys())
        self.frames[_argDict["name"]] = BBeam.makeGRTframe(_argDict)
        self.frames[_argDict["name"]].setMeta(self.copyObj(world.ORIGIN()), self.copyObj(world.IAX()), self.copyObj(world.INITM()))
        
        dtime = time.time() - start_time
        self.clog.work(f"Succesfully sampled {_argDict['nRays']} rays: {dtime} seconds.")
        self.clog.info(f"Added Gaussian frame {_argDict['name']} to system.")

    def calcRayLen(self, *frames : str, start : np.ndarray = None) -> list[np.ndarray]:
        """!
        Calculate total length of a ray-trace beam.
        
        Takes multiple frames and calculates the distance for each ray between frames.
        If the "start" parameter is set to a len-3 Numpy array, the ray length will also include the 
        distance between the position of the ray in the first frame and the given starting point.
        This is useful in case the rays emanate from a single point which is not included as a frame in the internal dictionary.
        
        @ingroup public_api_frames
        
        @param frames Frames between which to calculate total pathlength.
        @param start Point from which to start the calculation, len-3 Numpy array. If given, also calculates distance between point and the first frame. Defaults to None.
        
        @returns out List containing the distances between frames. Can be summed over to obtain total distance.
        """

        for fr in frames:
            PChecks.check_frameSystem(fr, self.frames, self.clog, extern=True)

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

    def createGaussian(self, gaussDict : dict, name_surface : str):
        """!
        Create a vectorial Gaussian beam.
        
        This method creates a general, potentially astigmatic, vectorial Gaussian beam.
        The beam is evaluated with the focus at z = 0. 
        The surface on which the beam is calculated, defined by "name_source", does not have to lie in or be parallel to the xy-plane.
        Instead, the Gaussian beam is evaluated on the surface as-is, evaluating the Gaussian beam at the xyz-points on the surface.
        Still, the focus is at z = 0. If one wishes to displace the focal point, the PO fields and currents need to be translated after generating the Gaussian beam.
        
        @ingroup public_api_po
        
        @param gaussDict A GDict containing parameters for the Gaussian beam.
        @param name_surface Name of plane on which to define Gaussian.
        
        @see GDict
        """

        PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)

        _gaussDict = self.copyObj(gaussDict)
        PChecks.check_GPODict(_gaussDict, self.fields, self.clog)
        
        gauss_in = BBeam.makeGauss(_gaussDict, self.system[name_surface])

        k = 2 * np.pi / _gaussDict["lam"]
        gauss_in[0].setMeta(name_surface, k)
        gauss_in[1].setMeta(name_surface, k)

        self.fields[_gaussDict["name"]] = gauss_in[0]
        self.currents[_gaussDict["name"]] = gauss_in[1]
    
    def createScalarGaussian(self, gaussDict : dict, name_surface : str):
        """!
        Create a scalar Gaussian beam.
        
        This method creates a general, potentially astigmatic, scalar Gaussian beam.
        The beam is evaluated with the focus at z = 0. 
        The surface on which the beam is calculated, defined by "name_source", does not have to lie in or be parallel to the xy-plane.
        Instead, the Gaussian beam is evaluated on the surface as-is, evaluating the Gaussian beam at the xyz-points on the surface.
        Still, the focus is at z = 0. If one wishes to displace the focal point, the PO scalarfield needs to be translated after generating the Gaussian beam.
        
        @ingroup public_api_po
        
        @param gaussDict A GDict containing parameters for the Gaussian beam.
        @param name_surface Name of plane on which to define Gaussian.
        
        @see SGDict
        """

        PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)
        
        _gaussDict = self.copyObj(gaussDict)
        PChecks.check_GPODict(_gaussDict, self.scalarfields, self.clog)
        
        gauss_in = BBeam.makeScalarGauss(_gaussDict, self.system[name_surface])

        k = 2 * np.pi / _gaussDict["lam"]
        gauss_in.setMeta(name_surface, k)

        self.scalarfields[_gaussDict["name"]] = gauss_in

    def runRayTracer(self, runRTDict : dict):
        """!
        Run a ray-trace propagation from a frame to a surface.
        
        The resulting frame is placed in the internal frames dictionary.
        
        @ingroup public_api_frames
        
        @param runRTDict A runRTDict object specifying the ray-trace.
        """

        self.clog.work("*** Starting RT propagation ***")
        
        _runRTDict = self.copyObj(runRTDict)

        PChecks.check_runRTDict(_runRTDict, self.system, self.frames, self.clog)

        _runRTDict["fr_in"] = self.frames[_runRTDict["fr_in"]]
        _runRTDict["t_name"] = self.system[_runRTDict["t_name"]]

        start_time = time.time()
       
        if _runRTDict["device"] == "CPU":
            self.clog.work(f"Hardware: running {_runRTDict['nThreads']} CPU threads.")
            self.clog.work(f"... Calculating ...")
            frameObj = BCPU.RT_CPUd(_runRTDict)

        elif _runRTDict["device"] == "GPU":
            self.clog.work(f"Hardware: running {_runRTDict['nThreads']} CUDA threads per block.")
            self.clog.work(f"... Calculating ...")
            frameObj = BGPU.RT_GPUf(_runRTDict)
        
        dtime = time.time() - start_time
        
        self.clog.work(f"*** Finished: {dtime:.3f} seconds ***")
        self.frames[runRTDict["fr_out"]] = frameObj
        
        self.frames[runRTDict["fr_out"]].setMeta(self.calcRTcenter(runRTDict["fr_out"]), self.calcRTtilt(runRTDict["fr_out"]), self.copyObj(world.INITM()))
    
    def runHybridPropagation(self, hybridDict : dict):
        """!
        Perform a hybrid RT/PO propagation, starting from a reflected field and set of Poynting vectors.
        
        The propagation is done by performing a ray trace from the starting frame into the target surface.
        Then, the starting reflected field is propagated to the target by multiplying each point on the field by the 
        phase factor corresponding to the travel length of the Poynting vector ray associated to the point on the field.
        Stores name of resultant field and frame in the internal association dictionary as two associated objects.
        The name of the association is the surface on which both the target frame and field are defined.
        
        @ingroup public_api_hybrid
        
        @param hybridDict A hybridDict dictionary.
        """

        self.clog.work("*** Starting hybrid propagation ***")
        start_time = time.time()

        PChecks.check_hybridDict(hybridDict, self.system, self.frames, self.fields, self.clog)
        surf = self.fields[hybridDict["field_in"]].surf
        PChecks.check_associations(self.assoc, hybridDict["field_in"], hybridDict["fr_in"], surf, self.clog)

        field = self.copyObj(self.fields[hybridDict["field_in"]])

        verbosity_init = self.verbosity

        self.setLoggingVerbosity(verbose=False)
        self.runRayTracer(hybridDict)
        self.setLoggingVerbosity(verbose=verbosity_init)

        stack = self.calcRayLen(hybridDict["fr_in"], hybridDict["fr_out"], start=hybridDict["start"])
        if hybridDict["start"] is not None:
            expo = np.exp(1j * field.k * stack[1]) * np.sqrt(stack[0] / (2*stack[1] + stack[0])) # Initial curvature

        else:
            expo = np.exp(1j * field.k * stack[0])

        _comps = []
        for i in range(6):
            _comps.append((expo * field[i].ravel()).reshape(field[i].shape))

        field_prop = PTypes.fields(*_comps)
        field_prop.setMeta(hybridDict["t_name"], field.k)

        self.fields[hybridDict["field_out"]] = field_prop

        if hybridDict["interp"]:
            self.interpFrame(hybridDict["fr_out"], hybridDict["field_out"], hybridDict["t_name"], hybridDict["field_out"], comp=hybridDict["comp"])
        
        dtime = time.time() - start_time

        self.assoc[hybridDict["t_name"]] = [hybridDict["field_out"], hybridDict["fr_out"]]
        self.clog.work(f"*** Finished: {dtime:.3f} seconds ***")

    def interpFrame(self, 
                    name_fr_in : str, 
                    name_field : str, 
                    name_target : str, 
                    name_out : str, 
                    comp : FieldComponents = FieldComponents.NONE, 
                    method : str = "nearest"
                    ) -> np.ndarray:
        """!
        Interpolate a frame and an associated field on a regular surface.
        
        The surface should be the target on which the input frame is calculated.
        The "name_field" should be an associated field, calculated in the same hybrid propagation as "name_fr_in".
        
        @ingroup public_api_hybrid
        
        @param name_fr_in Name of input frame.
        @param name_field Name of field object, propagated along with the frame by multiplication.
        @param name_target Name of surface on which to interpolate the field.
        @param name_out Name of output field object in target surface.
        @param comp Component of field to interpolate. Instance of FieldComponents enum object.
        @param method Method for the interpolation.
        
        @returns out Complex numpy array containing interpolated field.
        """

        PChecks.check_frameSystem(name_fr_in, self.frames, self.clog, extern=True)
        PChecks.check_elemSystem(name_target, self.system, self.clog, extern=True)
        
        grids = BRefl.generateGrid(self.system[name_target])

        points = (self.frames[name_fr_in].x, self.frames[name_fr_in].y, self.frames[name_fr_in].z)

        if comp is FieldComponents.NONE:
            _comps = []
            for i in range(6):
                rfield = np.real(self.fields[name_field][i]).ravel()
                ifield = np.imag(self.fields[name_field][i]).ravel()

                grid_interp = (grids.x, grids.y, grids.z)

                rout = griddata(points, rfield, grid_interp, method=method)
                iout = griddata(points, ifield, grid_interp, method=method)

                _comps.append(rout.reshape(self.system[name_target]["gridsize"]) + 1j * iout.reshape(self.system[name_target]["gridsize"]))

            field = PTypes.fields(*_comps)
            field.setMeta(name_target, self.fields[name_field].k)
            self.fields[name_out] = field

            out = field
       
        else:
            rfield = np.real(self.fields[name_field][comp.value]).ravel()
            ifield = np.imag(self.fields[name_field][comp.value]).ravel()

            grid_interp = (grids.x, grids.y, grids.z)

            rout = griddata(points, rfield, grid_interp, method=method)
            iout = griddata(points, ifield, grid_interp, method=method)

            out = rout.reshape(self.system[name_target]["gridsize"]) + 1j * iout.reshape(self.system[name_target]["gridsize"])

            field = self._compToFields(comp, out)
            field.setMeta(name_target, self.fields[name_field].k)

            self.fields[name_out] = field 

        return out

    def calcRTcenter(self, name_frame : str) -> np.ndarray:
        """!
        Calculate the geometric center of a ray-trace frame.
        
        The center is calculated by finding the centroid of the given frame.
        
        @ingroup public_api_frames
        
        @param name_frame Name of frame to calculate center of.
        
        @returns c_f Len-3 Numpy array containing x, y and z co-ordinates of frame center.
        """

        PChecks.check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        frame = self.frames[name_frame]
        c_f = Effs.calcRTcenter(frame)
        return c_f

    def calcRTtilt(self, name_frame : str) -> np.ndarray:
        """!
        Calculate the mean direction normal of a ray-trace frame.
        
        The mean direction is calculated by taking the mean tilt of every ray in the frame.
        
        @ingroup public_api_frames
        
        @param name_frame Name of frame to calculate tilt of.
        
        @returns t_f Len-3 Numpy array containing x, y and z components of frame tilt direction.
        """

        PChecks.check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        frame = self.frames[name_frame]
        t_f = Effs.calcRTtilt(frame)
        return t_f
    
    def calcSpotRMS(self, name_frame : str) -> float:
        """!
        Calculate the RMS spot size of a ray-trace frame.
        
        The RMS spotsize is calculated by taking the root-mean-square of the positions of the rays in the frame.
        
        @ingroup public_api_frames
        
        @param name_frame Name of frame to calculate RMS of.
        
        @returns rms RMS spot size of frame in mm.
        """

        PChecks.check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        frame = self.frames[name_frame]
        rms = Effs.calcRMS(frame)
        return rms

    def calcSpillover(self, name_field : str, comp : FieldComponents, aperDict : dict) -> float:
        """!
        Calculate spillover efficiency of a beam defined on a surface.
        
        The method calculates the spillover using the fraction of the beam that illuminates the region defined in aperDict versus the total beam.
        
        @ingroup public_api_po
        
        @param name_field Name of the PO field.
        @param comp Component of field to calculate spillover of. Instance of FieldComponents enum object.
        @param aperDict An aperDict dictionary containing the parameters for defining the spillover aperture.
        
        @returns spill The spillover efficiency.
        
        @see aperDict
        """

        PChecks.check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        PChecks.check_aperDict(aperDict, self.clog)

        field = self.fields[name_field]
        field_comp = field[comp.value]
        surfaceObj = self.system[field.surf]

        return Effs.calcSpillover(field_comp, surfaceObj, aperDict)

    def calcTaper(self, name_field : str, comp : FieldComponents, aperDict : dict = None) -> float:
        """!
        Calculate taper efficiency of a beam defined on a surface.
        
        The method calculates the taper efficiency using the fraction of the beam that illuminates the region defined in aperDict versus the total beam.
        If aperDict is not given, it will calculate the taper efficiency on the entire beam.
        
        @ingroup public_api_po
        
        @param name_field Name of the PO field.
        @param comp Component of field to calculate taper efficiency of. Instance of FieldComponents enum object.
        @param aperDict An aperDict dictionary containing the parameters for defining the taper aperture. Defaults to None.
        
        @returns taper The taper efficiency.
        
        @see aperDict
        """

        PChecks.check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        aperDict = {} if aperDict is None else aperDict

        if aperDict:
            PChecks.check_aperDict(aperDict, self.clog)

        field = self.fields[name_field]
        field_comp = field[comp.value]
        surfaceObj = self.system[field.surf]

        return Effs.calcTaper(field_comp, surfaceObj, aperDict)

    def calcXpol(self, name_field : str, comp_co : FieldComponents, comp_cr : FieldComponents) -> float:
        """!
        Calculate cross-polar efficiency of a field defined on a surface.
       
        The cross-polar efficiency is described as the ratio of power in the cross-polar component versus the co-polar component.
        The cross-polar efficiency is calculated over the entire field extent.
        
        @ingroup public_api_po
        
        @param name_field Name of the PO field.
        @param comp_co Co-polar component of field. Instance of FieldComponents enum object.
        @param comp_cr Cross-polar component of field. Instance of FieldComponents enum object.
        
        @returns crp The cross-polar efficiency.
        """

        PChecks.check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        field = self.fields[name_field]
        field_co = field[comp_co.value]
        
        field_cr = field[comp_cr.value]
        
        return Effs.calcXpol(field_co, field_cr)

    def fitGaussAbs(self, name_field : str, comp : FieldComponents, thres : float = None, scale : Scales = Scales.LIN, full_output : bool = False, ratio : float = 1) -> np.ndarray:
        """!
        Fit a Gaussian profile to the amplitude of a field component and adds the result to scalar field in system.
        
        The resultant Gaussian fit cannot be propagated using vectorial means, but can be propagated using scalar propagation.
        Note that this method is very sensitive to initial conditions, especially when the beam pattern to which to fit the Gaussian has multiple maxima or is generally
        ill-described by a Gaussian. In the latter case, the method may fail altogether.
        
        @ingroup public_api_po
        
        @param name_field Name of field object.
        @param comp Component of field object. Instance of FieldComponents enum object.
        @param thres Threshold to fit to, in decibels.
        @param scale Fit to amplitude in decibels, linear or logarithmic scale. Instance of Scales enum object.
        @param full_output Return fitted parameters and standard deviations.
        @param ratio Allowed maximal ratio of fit to actual beam. If "None", will just attempt to fit the Gaussian to supplied pattern. 
            If given, will only accept a fit if the ratio of integrated power in the fitted Gaussian to the supplied beam pattern is less 
            than or equal to the given value. Defaults to 1.

        
        @returns popt Fitted beam parameters.
        @returns perr Standard deviation of fitted parameters.
        """

        PChecks.check_fieldSystem(name_field, self.fields, self.clog, extern=True)

        thres = -11 if thres is None else thres

        surfaceObj = self.system[self.fields[name_field].surf]
        field = self.copyObj(np.absolute(self.fields[name_field][comp.value]))

        popt = FGauss.fitGaussAbs(field, surfaceObj, thres, scale, ratio)

        Psi = PTypes.scalarfield(FGauss.generateGauss(popt, surfaceObj, scale=Scales.LIN))
        Psi.setMeta(self.fields[name_field].surf, self.fields[name_field].k)
       
        _name = f"fitGauss_{name_field}"

        num = PChecks.getIndex(_name, self.scalarfields)
        
        if num > 0:
            _name = _name + "_{}".format(num)

        self.scalarfields[_name] = Psi

        if full_output:
            return popt

    def calcMainBeam(self, name_field : str, comp : FieldComponents, thres : float = None, scale : Scales = Scales.LIN) -> float:
        """!
        Calculate main-beam efficiency of a beam pattern.
        
        The main-beam efficiency is calculated by fitting a Gaussian amplitude profile to the central lobe.
        This might reuire fine-tuning the "thres" parameter, or changing the space in whcih to fit by supplying the "mode" parameter.
        Then, the efficiency is defined as the fraction of power in the Gaussian w.r.t. the full pattern.
        Designed for far-field beam patterns, but also applicable to regular fields.
        Note that since this method uses the fitGaussAbs() method, the result is quite sensitive to initial conditions and should therefore be (iteratively) checked for 
        robustness.
        
        @ingroup public_api_po
        
        @param name_field Name of field object.
        @param comp Component of field object. Instance of FieldComponents enum object.
        @param thres Threshold to fit to, in decibels.
        @param scale Fit to amplitude in decibels, linear or logarithmic scale.
        
        @returns eff Main-beam efficiency.
        """

        PChecks.check_fieldSystem(name_field, self.fields, self.clog, extern=True)
        thres = -11 if thres is None else thres

        _thres = self.copyObj(thres)

        self.fitGaussAbs(name_field, comp, thres, scale)
        field = self.copyObj(self.fields[name_field][comp.value])
        surfaceObj = self.system[self.fields[name_field].surf]
        
        eff = Effs.calcMainBeam(field, surfaceObj, self.scalarfields[f"fitGauss_{name_field}"].S)
        return eff
    
    def calcBeamCuts(self, 
                     name_field : str, 
                     comp : FieldComponents, 
                     interp : int = 1001, 
                     phi : float = 0, 
                     center : bool = True, 
                     align : bool = True,
                     norm : FieldComponents = FieldComponents.NONE, 
                     transform : bool = False, 
                     scale : Scales = Scales.dB,
                     full_output : bool = False,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """!
        Calculate cross sections of a beam pattern.
        
        This method calculates cross sections along the cardinal planes of the beam pattern.
        The cardinal planes here are defined to lie along the semi-major and semi-minor axes of the beam pattern.
        It does this by first finding the center and position angle of the beam pattern.
        Then, it creates a snapshot of the current configuration and translates and rotates the beam pattern so that the cardinal planes are 
        oriented along the x- and y-axes.
        It is also possible to not do this and instead directly calculate the cross sections along the x- and y-axes as-is.
        
        @ingroup public_api_po
        
        @param name_field Name of field object.
        @param comp Component of field object. Instance of FieldComponents enum object.
        @param interp Number of points in x and y strip. Defaults to 1001.
        @param phi Manual rotation of cuts w.r.t. to the x-y cardinal planes.
        @param center Whether to center the cardinal planes on the peak of the beam pattern.
        @param align Whether to align the cardinal planes to the beam pattern minor and major axes.
        @param norm Which component to normalise to. Defaults to comp. 
        @param transform Transform surface on which beam is defined. If False, will evaluate beam cuts as if surface is in restframe.
        @param scale Return beamcuts in linear or decibel values. Instance of Scales enum object.
        @param full_output Return x-y coordinates of H and E cuts, instead of just distance to center.
        
        @returns x_cut Beam cross section along the E-plane.
        @returns y_cut Beam cross section along the H-plane.
        @returns x_strip Co-ordinate values for x_cut.
        @returns y_strip Co-ordinate values for y_cut.
        """

        norm = comp if norm is FieldComponents.NONE else norm

        PChecks.check_fieldSystem(name_field, self.fields, self.clog, extern=True)
 
        verbosity_init = self.verbosity

        self.setLoggingVerbosity(verbose=False)
        
        name_surf = self.fields[name_field].surf
        self.snapObj(name_surf, "_")

        self.homeReflector(name_surf)
        grids = self.generateGrids(name_surf, transform=False, spheric=False)

        x_edges = np.array([np.min(grids.x), np.max(grids.x)])
        y_edges = np.array([np.min(grids.y), np.max(grids.y)])

        field = np.absolute(self.fields[name_field][comp.value])
        max_norm = np.nanmax(np.absolute(self.fields[name_field][norm.value]))

        center_use = np.zeros(2)
        rot_use = np.radians(phi)

        if center or align:
            popt = self.fitGaussAbs(name_field, comp, scale=Scales.LIN, full_output=True)
        
            if center:
                center_use = np.array([popt[2], popt[3]])
            
            if align:
                rot_use += popt[4]

        hx_edges = np.cos(rot_use) * x_edges + center_use[0]
        hy_edges = np.sin(rot_use) * x_edges + center_use[1]
        ex_edges = -np.sin(rot_use) * y_edges + center_use[0]
        ey_edges = np.cos(rot_use) * y_edges + center_use[1]
        
        hx_edges_interp = np.linspace(hx_edges[0], hx_edges[1], interp)
        hy_edges_interp = np.linspace(hy_edges[0], hy_edges[1], interp)
        ex_edges_interp = np.linspace(ex_edges[0], ex_edges[1], interp)
        ey_edges_interp = np.linspace(ey_edges[0], ey_edges[1], interp)
        
        h_cut = griddata((grids.x.ravel(), grids.y.ravel()), field.ravel(), (hx_edges_interp, hy_edges_interp), method="cubic") 
        e_cut = griddata((grids.x.ravel(), grids.y.ravel()), field.ravel(), (ex_edges_interp, ey_edges_interp), method="cubic") 
        
        if scale == Scales.dB:
            h_cut = 20 * np.log10(h_cut / max_norm)
            e_cut = 20 * np.log10(e_cut / max_norm)
        
        elif scale == Scales.LIN:
            h_cut = h_cut / max_norm
            e_cut = e_cut / max_norm

        h_strip = np.linspace(x_edges[0], x_edges[1], interp)
        e_strip = np.linspace(y_edges[0], y_edges[1], interp)
        
        self.revertToSnap(name_surf, "_")
        self.deleteSnap(name_surf, "_")

        self.setLoggingVerbosity(verbose=verbosity_init)
        if full_output:
            return h_cut, e_cut, h_strip, e_strip, hx_edges_interp, hy_edges_interp, ex_edges_interp, ey_edges_interp
   
        else:
            return h_cut, e_cut, h_strip, e_strip

    def plotBeamCut(self, 
                    name_field : str, 
                    comp : FieldComponents, 
                    vmin : float = None, 
                    vmax : float = None, 
                    center : bool = True, 
                    align : bool = True, 
                    scale : Scales = Scales.dB, 
                    units : Units = Units.DEG, 
                    name : str = "", 
                    show : bool = True, 
                    save : bool = False, 
                    ret : bool = False
                    ) -> tuple[pt.Figure, pt.Axes]:
        """!
        Plot beam pattern cross sections.
        
        Plot the beam cross sections for a PO field.
        In this case, calcBeamCuts() will try to translate and rotate the supplied beam pattern to lie along the x- and y-axes.
        Note that using the "center" and "align" arguments should not be done when plotting beam cuts of very non-Gaussian beams. For these patterns, it is advised to set the arguments to False and calculate the beam cuts as-is. 
        
        @ingroup public_api_vis
        
        @param name_field Name of field object.
        @param comp Component of field object. Instance of FieldComponents enum object.
        @param comp_cross Cross-polar component. If given, is plotted as well. Defaults to None.
        @param vmin Minimum amplitude value to display. Default is -30.
        @param vmax Maximum amplitude value to display. Default is 0.
        @param center Whether to calculate beam center and center the beam cuts on this point.
        @param align Whether to find position angle of beam cuts and align cut axes to this.
        @param scale Plot in decibels or linear.
        @param units The units of the axes. Instance of Units enum object.
        @param name Name of .png file where plot is saved. Only when save=True. Default is "".
        @param show Show plot. Default is True.
        @param save Save plot to savePath.
        @param ret Return the Figure and Axis object. Default is False.
        
        @returns fig Figure object.
        @returns ax Axes object.
        """

        h_cut, e_cut, h_strip, e_strip = self.calcBeamCuts(name_field, comp, center=center, align=align, scale=scale)

        vmin = np.nanmin([np.nanmin(h_cut), np.nanmin(e_cut)]) if vmin is None else vmin
        vmax = np.nanmax([np.nanmax(h_cut), np.nanmax(e_cut)]) if vmax is None else vmax
        
        fig, ax = PPlot.plotBeamCut(h_cut, 
                                    e_cut, 
                                    h_strip, 
                                    e_strip, 
                                    vmin, 
                                    vmax, 
                                    units)

        if ret:
            return fig, ax

        elif save:
            pt.savefig(fname=self.savePath + '{}_EH_cut.jpg'.format(name),
                        bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

    def calcHPBW(self, 
                 name_field : str, 
                 comp : FieldComponents, 
                 center : bool = False, 
                 align : float = False
                 ) -> tuple [float, float]:
        """!
        Calculate half-power beamwidth.
        
        This is done by directly evaluating the -3 dB points along both cardinal planes of the beam pattern.
        Then, the distance between antipodal half-power points is calculated on an interpolation of the supplied PO field.
        
        @ingroup public_api_po
        
        @param name_field Name of field object.
        @param comp Component of field object. Instance of FieldComponents enum object.
        @param center Whether to center the beam cuts on amplitude center. Use only if beam has well defined amplitude center.
        @param align Whether to take beam cuts along cardinal planes rotated by the position angle.
        
        @returns HPBW_E Half-power beamwidth along E-plane in units of surface of beam.
        @returns HPBW_H Half-power beamwidth along H-plane in units of surface of beam.
        """

        h_cut, e_cut, h_strip, e_strip = self.calcBeamCuts(name_field, comp, center=center, align=align)

        h_masked = h_strip[h_cut > -3]
        e_masked = e_strip[e_cut > -3]

        HPBW_h = np.absolute(np.max(h_masked) - np.min(h_masked))
        HPBW_e = np.absolute(np.max(e_masked) - np.min(e_masked))

        return HPBW_h, HPBW_e

    def createPointSource(self, PSDict : dict, name_surface : str):
        """!
        Generate point-source PO fields and currents.
        
        The point source is generated in the center of the source surface given by "name_surface".
        It is generally a good idea to make this source surface as small as possible, in order to create a "nice" point source.
        If this is too big, the resulting PO field more closely resembles a uniformly illuminated square.
        The H-field is set to 0.
        
        @ingroup public_api_po
        
        @param PSDict A PSDict dictionary, containing parameters for the point source.
        @param name_surface Name of surface on which to define the point-source.
        
        @see PSDict
        """

        PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)
        _PSDict = self.copyObj(PSDict)
        PChecks.check_PSDict(_PSDict, self.fields, self.clog)

        mode = "PMC"
        
        surfaceObj = self.system[name_surface]
        ps = np.zeros(surfaceObj["gridsize"], dtype=complex)

        xs_idx = int((surfaceObj["gridsize"][0] - 1) / 2)
        ys_idx = int((surfaceObj["gridsize"][1] - 1) / 2)

        ps[xs_idx, ys_idx] = _PSDict["E0"] * np.exp(1j * _PSDict["phase"])

        Ex = ps * _PSDict["pol"][0]
        Ey = ps * _PSDict["pol"][1]
        Ez = ps * _PSDict["pol"][2]

        Hx = ps * 0
        Hy = ps * 0
        Hz = ps * 0

        field = PTypes.fields(Ex, Ey, Ez, Hx, Hy, Hz) 
        #current = self.calcCurrents(name_surface, field)
        current = BBeam.calcCurrents(field, self.system[name_surface], mode)
        k =  2 * np.pi / _PSDict["lam"]

        field.setMeta(name_surface, k)
        current.setMeta(name_surface, k)

        self.fields[_PSDict["name"]] = field
        self.currents[_PSDict["name"]] = current

    def createUniformSource(self, UDict : dict, name_surface : str):
        """!
        Generate uniform PO fields and currents.
        
        The uniform field is generated by defining a PO field on the source surface and setting all values to the amplitude specified in 
        the input dictionary. The H-field is set to 0.
        
        @ingroup public_api_po
        
        @param UDict A UDict dictionary, containing parameters for the uniform pattern.
        @param name_surface Name of surface on which to define the uniform pattern.
        
        @see PSDict
        """

        PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)
        _UDict = self.copyObj(UDict)
        PChecks.check_PSDict(_UDict, self.fields, self.clog)
        
        mode = "PMC"

        surfaceObj = self.system[name_surface]
        us = np.ones(surfaceObj["gridsize"], dtype=complex) * _UDict["E0"] * np.exp(1j * _UDict["phase"])

        Ex = us * _UDict["pol"][0]
        Ey = us * _UDict["pol"][1]
        Ez = us * _UDict["pol"][2]

        Hx = us * 0
        Hy = us * 0
        Hz = us * 0

        field = PTypes.fields(Ex, Ey, Ez, Hx, Hy, Hz) 
        current = BBeam.calcCurrents(field, self.system[name_surface], mode)

        k =  2 * np.pi / _UDict["lam"]

        field.setMeta(name_surface, k)
        current.setMeta(name_surface, k)

        self.fields[_UDict["name"]] = field
        self.currents[_UDict["name"]] = current
    
    def createPointSourceScalar(self, PSDict : dict, name_surface : str):
        """!
        Generate point-source scalar PO field.
        
        The point source is generated in the center of the source surface given by "name_surface".
        It is generally a good idea to make this source surface as small as possible, in order to create a "nice" point source.
        If this is too big, the resulting PO field more closely resembles a uniformly illuminated square.
        The H-field is set to 0.
        
        @ingroup public_api_po
        
        @param PSDict A PSDict dictionary, containing parameters for the point source.
        @param name_surface Name of surface on which to define the point-source.
        
        @see PSDict
        """

        PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)
        
        _PSDict = self.copyObj(PSDict)
        PChecks.check_PSDict(_PSDict, self.scalarfields, self.clog)
        
        surfaceObj = self.system[name_surface]
        ps = np.zeros(surfaceObj["gridsize"], dtype=complex)

        xs_idx = int((surfaceObj["gridsize"][0] - 1) / 2)
        ys_idx = int((surfaceObj["gridsize"][1] - 1) / 2)

        ps[xs_idx, ys_idx] = _PSDict["E0"] * np.exp(1j * _PSDict["phase"])
        sfield = PTypes.scalarfield(ps)

        k =  2 * np.pi / _PSDict["lam"]

        sfield.setMeta(name_surface, k)

        self.scalarfields[_PSDict["name"]] = sfield
    
    def createUniformSourceScalar(self, UDict : dict, name_surface : str):
        """!
        Generate scalar uniform PO fields and currents.
        
        The uniform field is generated by defining a PO field on the source surface and setting all values to the amplitude specified in 
        the input dictionary. The H-field is set to 0.
        
        @ingroup public_api_po
        
        @param UDict A UDict dictionary, containing parameters for the uniform pattern.
        @param name_surface Name of surface on which to define the uniform pattern.
        
        @see UDict
        """

        PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)
        _UDict = self.copyObj(UDict)
        PChecks.check_PSDict(_UDict, self.scalarfields, self.clog)

        surfaceObj = self.system[name_surface]
        us = np.ones(surfaceObj["gridsize"], dtype=complex) * _UDict["E0"] * np.exp(1j * _UDict["phase"])

        sfield = PTypes.scalarfield(us)

        k =  2 * np.pi / _UDict["lam"]

        sfield.setMeta(name_surface, k)

        self.scalarfields[_UDict["name"]] = sfield
   
    def interpBeam(self, name : str, gridsize_new : int, obj : Objects = Objects.FIELD):
        """!
        Interpolate a PO beam. Only for beams defined on planar surfaces.
        
        Can interpolate PO fields and currents separately.
        Results are stored in a new PO fields/currents object with the original name appended by 'interp'.
        Also, a new plane will be created with the updated gridsize and name appended by 'interp'.
        
        @ingroup public_api_po
        
        @param name Name of beam to be interpolated.
        @param gridsize_new New gridsizes for interpolation.
        @param obj Whether to interpolate currents or fields.
        """

        if obj == Objects.FIELD:
            PChecks.check_fieldSystem(name, self.fields, self.clog, extern=True)
            obj_out = self.fields[name]

        elif obj == Objects.CURRENT:
            PChecks.check_currentSystem(name, self.currents, self.clog, extern=True)
            obj_out = self.currents[name]

        self.copyElement(obj_out.surf, obj_out.surf + "_interp")
        self.system[obj_out.surf + "_interp"]["gridsize"] = gridsize_new
        self.system[obj_out.surf + "_interp"]["name"] = obj_out.surf + "_interp"

        grids = self.generateGrids(obj_out.surf)
        grids_interp = self.generateGrids(obj_out.surf + "_interp")
        
        points = (grids.x.ravel(), grids.y.ravel())#, grids.z.ravel())
        points_interp = (grids_interp.x.ravel(), grids_interp.y.ravel())#, grids_interp.z.ravel())
        comp_l = []

        for i in range(6):
            _comp = self.copyObj(obj_out[i])
            _cr = np.real(_comp)
            _ci = np.imag(_comp)

            _cr_interp = griddata(points, _cr.ravel(), points_interp)
            _ci_interp = griddata(points, _ci.ravel(), points_interp)
       
            _comp_interp = _cr_interp + 1j * _ci_interp

            comp_l.append(_comp_interp.reshape(gridsize_new))

        if obj == Objects.FIELD:
            obj_interp = PTypes.fields(comp_l[0], comp_l[1], comp_l[2], comp_l[3], comp_l[4], comp_l[5])
            obj_interp.setMeta(obj_out.surf + "_interp", obj_out.k)
            self.fields[name + "_interp"] = obj_interp
        
        elif obj == Objects.CURRENT:
            obj_interp = PTypes.currents(comp_l[0], comp_l[1], comp_l[2], comp_l[3], comp_l[4], comp_l[5])
            obj_interp.setMeta(obj_out.surf + "_interp", obj_out.k)
            self.currents[name + "_interp"] = obj_interp

    def plotBeam2D(self, name_obj : str, comp : FieldComponents = FieldComponents.NONE, contour : str = None, 
                    contour_comp : fieldOrCurrentComponents  = FieldComponents.NONE,
                    vmin : float = None, vmax : float = None, levels : contourLevels = None, 
                    show : bool = True, amp_only : bool = False, save : bool = False, norm : bool = True,
                    aperDict : dict = None, scale : Scales = Scales.dB, project : Projections = Projections.xy,
                    units : Units = Units.MM, name : str = "", titleA : str ="Power", titleP : str = "Phase",
                    unwrap_phase : bool = False, ret : bool = False
                    ) -> tuple[pt.Figure, pt.Axes]:
        """!
        Generate a 2D plot of a PO (scalar)field or current.
        
        Note that matplotlib offers custom control over figures in the matplotlib window.
        This means that most parameters described for this method can be adjusted in the matplotlib plotting window.
        
        @ingroup public_api_vis
        
        @param name_obj Name of field or current to plot.
        @param comp Component of field or current to plot. String of two characters; an uppercase {E, H, J, M} for field followed by a lowercase {x, y, z} for component. (e.g: 'Jz')
        @param contour A PyPO field or current component to plot as contour.
        @param contour_comp Component of contour to plot as contour. If None, assumes the contour is a scalarfield.
        @param vmin Minimum amplitude value to display. Default is -30.
        @param vmax Maximum amplitude value to display. Default is 0.
        @param levels Levels for contourplot.
        @param show Show plot. Default is True.
        @param amp_only Only plot amplitude pattern. Default is False.
        @param save Save plot to /images/ folder.
        @param norm Normalise field (only relevant when plotting linear scale). Default is True.
        @param aperDict Plot an aperture defined in an aperDict object along with the field or current patterns. Default is None.
        @param scale Plot amplitude in linear or decibel values. Instance of Scales enum object.
        @param project Set abscissa and ordinate of plot. Should be given as an instance of the Projection enum. Default is Projection.xy.
        @param units The units of the axes. Instance of Units enum object.
        @param name Name of .png file where plot is saved. Only when save=True. Default is "".
        @param titleA Title of the amplitude plot. Default is "Amp".
        @param titleP Title of the phase plot. Default is "Phase".
        @param unwrap_phase Unwrap the phase patter. Prevents annular structure in phase pattern. Default is False.
        @param ret Return the Figure and Axis object. Default is False.
        
        @returns fig Figure object.
        @returns ax Axes object.

        @see aperDict
        """

        aperDict = {"plot":False} if aperDict is None else aperDict

        if comp == FieldComponents.NONE:
            field_comp = self.scalarfields[name_obj].S
            name_surface = self.scalarfields[name_obj].surf
        
        elif isinstance(comp, FieldComponents):
            PChecks.check_fieldSystem(name_obj, self.fields, self.clog, extern=True)
            field = self.fields[name_obj]
            name_surface = field.surf
            field_comp = field[comp.value]

        elif isinstance(comp, CurrentComponents):
            PChecks.check_currentSystem(name_obj, self.currents, self.clog, extern=True)
            field = self.currents[name_obj] 
            name_surface = field.surf
            field_comp = field[comp.value]

        if contour is not None:
            if contour_comp == FieldComponents.NONE:
                contour_pl = self.scalarfields[contour].S
            
            else:
                if isinstance(contour_comp, FieldComponents):
                    PChecks.check_fieldSystem(contour, self.fields, self.clog, extern=True)
                    contour_pl = self.fields[contour][contour_comp.value]
            
                elif isinstance(contour_comp, CurrentComponents):
                    PChecks.check_currentSystem(contour, self.currents, self.clog, extern=True)
                    contour_pl = self.currents[contour][contour_comp.value]
        else:
            contour_pl = contour

        plotObject = self.system[name_surface]

        fig, ax = PPlot.plotBeam2D(plotObject, field_comp, contour_pl,
                        vmin, vmax, levels, amp_only,
                        norm, aperDict, scale, project,
                        units, titleA, titleP, unwrap_phase)

        if ret:
            return fig, ax

        elif save:
            pt.savefig(fname=os.path.join(self.savePath, f'{plotObject["name"]}_{name}.jpg'), bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

    def plot3D(self, name_surface : str, cmap : cm = cm.cool,
            norm : bool = False, fine : int = 2, show : bool = True, foc1 : bool = False, 
            foc2 : bool = False, save : bool = False, ret : bool = False
            ) -> tuple[pt.Figure, pt.Axes]:
        """!
        Plot a 3D reflector.
        
        Note that matplotlib offers custom control over figures in the matplotlib window.
        This means that most parameters described for this method can be adjusted in the matplotlib plotting window.
        
        @ingroup public_api_vis
        
        @param name_surface Name of reflector to plot.
        @param cmap Colormap of reflector. Default is cool.
        @param norm Plot reflector normals. Default is False.
        @param fine Spacing of normals for plotting. Default is 2.
        @param show Show plot. Default is True.
        @param foc1 Plot focus 1. Default is False.
        @param foc2 Plot focus 2. Default is False.
        @param save Save the plot.
        @param ret Return Figure and Axis. Only used in GUI.

        @returns fig Figure object.
        @returns ax Axes object.
        """

        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        
        if isinstance(name_surface, list) or isinstance(name_surface, np.ndarray):
            for n_s in name_surface:
                PChecks.check_elemSystem(n_s, self.system, self.clog, extern=True)
                plotObject = self.system[n_s]
                PPlot.plot3D(plotObject, ax, fine, cmap, norm, foc1, foc2)
        
        else:
            PChecks.check_elemSystem(name_surface, self.system, self.clog, extern=True)
            plotObject = self.system[name_surface]
            PPlot.plot3D(plotObject, ax, fine, cmap, norm, foc1, foc2)

        if ret:
            return fig, ax
        
        elif save:
            pt.savefig(fname=self.savePath + '{}.jpg'.format(plotObject["name"]),bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()

    def plotSystem(self, cmap : cm = cm.cool, norm : bool = False, fine : int = 2, show : bool = True, foc1 : bool = False, 
                   foc2 : bool = False, save : bool = False, ret : bool = False, select : list[str] = None, RTframes : list[str] = None, 
                   RTcolor : str = "black"
                ) -> tuple[pt.Figure, pt.Axes]:
        """!
        Plot the current system. Plots the reflectors and optionally ray-trace frames in a 3D plot.
        
        The ray-trace frames to plot are supplied as a list to the "RTframes" parameter.
        Note that matplotlib offers custom control over figures in the matplotlib window.
        This means that most parameters described for this method can be adjusted in the matplotlib plotting window.
        
        @ingroup public_api_vis
        
        @param name_surface Name of reflector to plot.
        @param cmap Colormap of reflector. Default is cool.
        @param norm Plot reflector normals. Default is False.
        @param fine Spacing of normals for plotting. Default is 2.
        @param show Show plot. Default is True.
        @param foc1 Plot focus 1. Default is False.
        @param foc2 Plot focus 2. Default is False.
        @param save Save the plot.
        @param ret Return Figure and Axis. Only used in GUI.
        @param select A list of names of reflectors to plot. If not given, plot all reflectors.
        @param RTframes A list of names of frame to plot. If not given, plot no ray-trace frames.

        @returns fig Figure object.
        @returns ax Axes object.
        """


        select = [] if select is None else select
        RTframes = [] if RTframes is None else RTframes
        
        plotDict = {}
        if select:
            for name in select:
                PChecks.check_elemSystem(name, self.system, self.clog, extern=True)
                plotDict[name] = self.system[name]
        else:
            plotDict = self.system
        
        _RTframes = []
        if RTframes:
            for name in RTframes:
                PChecks.check_frameSystem(name, self.frames, self.clog, extern=True)
                _RTframes.append(self.frames[name])


        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        PPlot.plotSystem(plotDict, ax, fine, cmap,norm,
                    foc1, foc2, _RTframes, RTcolor)

        if ret:
            return fig, ax
        
        elif save:
            pt.savefig(fname=self.savePath + 'system.pdf',bbox_inches='tight', dpi=300)
            pt.close()

        elif show:
            pt.show()
    
    def plotGroup(self, name_group : str, show : bool = True, ret : bool = False) -> tuple[pt.Figure, pt.Axes]:
        """!
        Plot a group of reflectors.
        
        Note that matplotlib offers custom control over figures in the matplotlib window.
        This means that most parameters described for this method can be adjusted in the matplotlib plotting window.
        
        @ingroup public_api_vis
        
        @param name_group Name of group to be plotted.
        @param show Show the plot.
        @param ret Whether to return figure and axis.

        @returns fig Figure object.
        @returns ax Axes object.
        """

        select = [x for x in self.groups[name_group]["members"]]

        if ret:
            fig, ax = self.plotSystem(select=select, show=False, ret=True)
            return fig,ax
        else:
            self.plotSystem(select=select, show=show)

    def plotRTframe(self, name_frame : str, project : Projections = Projections.xy, ret : bool = False, aspect : float = 1, units : Units = Units.MM):
        """!
        Create a spot diagram of a ray-trace frame.
        
        Note that matplotlib offers custom control over figures in the matplotlib window.
        This means that most parameters described for this method can be adjusted in the matplotlib plotting window.
        
        @ingroup public_api_vis
        
        @param name_frame Name of frame to plot.
        @param project Set abscissa and ordinate of plot. Should be given as an instance of the Projection enum. Default is Projection.xy.
        @param ret Return Figure and Axis. Default is False.
        @param aspect Aspect ratio of plot. Default is 1.
        @param units Units of the axes for the plot. Instance of Units enum object.

        @returns fig Figure object.
        @returns ax Axes object.
        """

        PChecks.check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        if ret:
            return PPlot.plotRTframe(self.frames[name_frame], project, self.savePath, ret, aspect, units)
        else:
            PPlot.plotRTframe(self.frames[name_frame], project, self.savePath, ret, aspect, units)

    def findRTfocus(self, name_frame : str, f0 : float = None, tol : float = 1e-12) -> np.ndarray:
        """!
        Find the focus of a ray-trace frame.
        
        Adds a new plane to the System, perpendicular to the mean ray-trace tilt of the input frame.
        The new plane is called focal_plane_<name_frame> and stored in the internal system dictionary.
        After completion, the new plane is centered at the ray-trace focus.
        The focus frame is stored as focus_<name_frame> in the internal frames dictionary.
        
        @ingroup public_api_frames
        
        @param name_frame Name of the input frame.
        @param f0 Initial try for focal distance.
        @param verbose Allow verbose System logging.
        
        @returns out The focus co-ordinate.
        """

        PChecks.check_frameSystem(name_frame, self.frames, self.clog, extern=True)
        f0 = 0 if f0 is None else f0
       
        
        tilt = self.calcRTtilt(name_frame)
        center = self.calcRTcenter(name_frame)
        match = self.copyObj(world.IAX())
        
        if np.all(np.isclose(match, -tilt)):
            R = np.eye(4)
        else:
            R = self.findRotation(match, tilt)
        
        t_name = f"focal_plane_{name_frame}"
        fr_out = f"focus_{name_frame}"

        target = {
                "name"      : t_name,
                "gmode"     : "xy",
                "lims_x"    : np.array([-4.2, 4.2]),
                "lims_y"    : np.array([-4.2, 4.2]),
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
                "tol"       : tol,
                "nThreads"  : 1
                }

        self.clog.work(f"Finding focus of {name_frame}...")
        
        verbosity_init = self.verbosity
        self.setLoggingVerbosity(verbose=False)
        
        res = fmin(self._optimiseFocus, f0, args=(runRTDict, tilt), full_output=True, disp=False, ftol=tol)
 
        out = res[0] * tilt + center
        self.translateGrids(t_name, out, mode="absolute")
        
        self.setLoggingVerbosity(verbose=verbosity_init)
        self.clog.result(f"Focus of frame {name_frame}: {*['{:0.3e}'.format(x) for x in out],}, RMS: {res[1]:.3e}")

        return out
        
    def copyObj(self, obj = None): #TODO: check typing
        """!
        Create a deep copy of any object.
        
        @param obj Object do be deepcopied.
        
        @returns copy A deepcopy of obj.
        """

        obj = self if obj is None else obj
        return copy.deepcopy(obj)

    def findRotation(self, v : np.ndarray, u : np.ndarray) -> np.ndarray:
        """!
        Find rotation matrix to rotate v onto u.
        
        @param v Numpy array of length 3. 
        @param u Numpy array of length 3.

        @returns R_transf Rotation martix.
        """

        I = np.eye(3)
        if np.array_equal(v, u):
            return self.copyObj(world.INITM())

        lenv = np.linalg.norm(v)
        lenu = np.linalg.norm(u)

        if lenv == 0 or lenu == 0:
            self.clog.error("Encountered 0-length vector. Cannot proceed.")
            return None

        w = np.cross(v/lenv, u/lenu)

        lenw = np.linalg.norm(w)
        
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        R = I + K + K @ K * (1 - np.dot(v, u)) / lenw**2

        R_transf = self.copyObj(world.INITM())
        R_transf[:-1, :-1] = R
        
        return R_transf
    
    def getAnglesFromMatrix(self, M : np.ndarray) -> np.ndarray:
        """!
        Find x, y and z rotation angles from general rotation matrix.
        Note that the angles are not necessarily the same as the original angles of the matrix.
        However, the matrix constructed by the found angles applies the same 3D rotation as the input matrix.
        
        @param M Numpy array of shape (3,3) containg a general rotation matrix.
        
        @returns r Numpy array of length 3 containing rotation angles around x, y and z.
        """

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

        return r#, testM
    
    def autoConverge(self, source_field : str, name_target : str, tol : float = 1e-2, add : int = 10, patch_size : float = 1/9, max_iter : int = 1000) -> int:
        """!
        Calculate gridsize for which calculation converges.
        
        This function calculates the gridsize for a target surface in order to obtain a convergent solution.
        First, a small patch is selected from the middle of the target surface and given a starting gridsize.
        The brightest component of the source distribution is then selected and copied into a PO scalarfield object.
        The scalarfield is propagated to the target patch and the total incident power is calculated.
        Then, the propagation is performed again but now the gridsize of the patch is smaller.
        The power is calculated again and compared with the previous result.
        If this result is smaller than the given tolerance, the new gridsize is accepted as the converged gridsize.
        If not, another iteration is started.
        If the maximal number of iterations is exceeded, `PyPO` will throw an error and stop.
        
        @ingroup public_api_po
        
        @param source_field Name of field to use for auto convergence. Should be the field that is to be propagated.
        @param name_target Name of target surface.
        @param tol Tolerance for specifying when convergence has been reached.
        @param add Increment in gridsize for each iteration.
        @param patch_size Factor for reducing size of target in order to save time. Should be smaller than 1.
        @param max_iter Maximum number of iterations before auto convergence errors.
        
        @returns gridsize Gridsize, scaled to full target, for which solution converged.
        """

        self.clog.work(f"*** Starting auto-convergence *** ")
        logstate = self.verbosity
        self.setLoggingVerbosity(False)
        gridsize = np.array([1,1])
        diff = 1e99

        P0 = 1e99

        max_E = []
        for i in range(6):
            max_E.append(np.max(np.absolute(self.fields[source_field][i])))

        comp = self.fields[source_field][np.argmax(np.array(max_E))]
  
        self.scalarfields[f"_{source_field}"] = PTypes.scalarfield(comp)
        self.scalarfields[f"_{source_field}"].setMeta(self.fields[source_field].surf, self.fields[source_field].k)

        if self.system[name_target]["gmode"] == 1:
            xu = "lims_u"
            yv = "lims_v"
        
        elif self.system[name_target]["gmode"] == 0:
            xu = "lims_x"
            yv = "lims_y"
        
        self.system[name_target][xu] = patch_size * self.system[name_target][xu]
        self.system[name_target][yv] = patch_size * self.system[name_target][yv]
        
        runPODict = {
                "t_name"    : name_target,
                "s_scalarfield" : f"_{source_field}",
                "epsilon"   : 10,
                "exp"       : "fwd",
                "mode"      : "scalar",
                "name_field"   : "_S_conv"
                }

        n = 0
        norm = 1
        while np.absolute(diff) > tol:
            gridsize += add
            self.system[name_target]["gridsize"] = gridsize
            self.runPO(runPODict)
            
            if n == 0:
                norm = np.max(np.absolute(self.scalarfields["_S_conv"].S))            
            
            _grid = self.generateGrids(name_target)
            P = np.sum(np.absolute(self.scalarfields["_S_conv"].S / norm)**2 * _grid.area)
            diff = np.absolute(P0 - P)
            P0 = P
            self.removeScalarField("_S_conv")
            
            self.setLoggingVerbosity(logstate)
            self.clog.work(f"Difference : {diff} at gridsize {*['{:0.3e}'.format(x) for x in list(gridsize)],}")
            self.setLoggingVerbosity(False)
            
            n += 1
            if n >= max_iter:
                self.setLoggingVerbosity(logstate)
                self.clog.error("Could not find converged solution.")
                self.system[name_target][xu] /= patch_size
                self.system[name_target][yv] /= patch_size
                self.system[name_target]["gridsize"] = (self.system[name_target]["gridsize"] / patch).astype(int)
                return gridsize

        self.setLoggingVerbosity(logstate)
        self.system[name_target][xu] /= patch_size
        self.system[name_target][yv] /= patch_size
        
        gridsize = (gridsize / patch_size).astype(int)
        
        self.system[name_target]["gridsize"] = gridsize
        
        self.clog.result(f"Found converged solution, gridsize: {*['{:0.3e}'.format(x) for x in list(gridsize)],}")
        return gridsize
    
    def runGUIPO(self, runPODict : dict): #TODO: check typing
        """!system
        Instantiate a GUI PO propagation. Stores desired output in the system.fields and/or system.cursystemrents lists.
        If the 'EHP' mode is selected, the reflected Poynting frame is stored in system.frames.
        
        @param PODict Dictionary containing the PO propagation instructions.
        
        @see PODict
        """

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
            out = BCPU.PyPO_CPUd(source, target, _runPODict)

        elif _runPODict["device"] == "GPU":
            out = BGPU.PyPO_GPUf(source, target, _runPODict)
        
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

            frame = self._loadFramePoynt(out[1], _runPODict["t_name"])
            self.frames[_runPODict["name_P"]] = frame
            self.assoc[_runPODict["t_name"]] = [_runPODict["name_EH"], _runPODict["name_P"]]

        elif _runPODict["mode"] == "scalar":
            out.setMeta(_runPODict["t_name"], _runPODict["k"])
            self.scalarfields[_runPODict["name_field"]] = out

        return out
    
    def runGUIRayTracer(self, runRTDict : dict):
        """!
        Run a ray-trace propagation from a frame to a surface in the GUI.
        
        @param runRTDict A runRTDict object specifying the ray-trace.
        """

        _runRTDict = self.copyObj(runRTDict)

        _runRTDict["fr_in"] = self.frames[_runRTDict["fr_in"]]
        _runRTDict["t_name"] = self.system[_runRTDict["t_name"]]
       
        if _runRTDict["device"] == "CPU":
            frameObj = BCPU.RT_CPUd(_runRTDict)

        elif _runRTDict["device"] == "GPU":
            frameObj = BGPU.RT_GPUf(_runRTDict)
        
        self.frames[runRTDict["fr_out"]] = frameObj
        self.frames[runRTDict["fr_out"]].setMeta(self.calcRTcenter(runRTDict["fr_out"]), self.calcRTtilt(runRTDict["fr_out"]), self.copyObj(world.INITM()))
    
    def hybridGUIPropagation(self, hybridDict : dict):
        """!
        Perform a hybrid RT/PO GUI propagation, starting from a reflected field and set of Poynting vectors.
        
        @param hybridDict A hybridDict dictionary.
        """

        field = self.copyObj(self.fields[hybridDict["field_in"]])
        
        self.runGUIRayTracer(hybridDict)

        stack = self.calcRayLen(hybridDict["fr_in"], hybridDict["fr_out"], start=hybridDict["start"])
        if hybridDict["start"] is not None:
            expo = np.exp(1j * field.k * stack[1]) * np.sqrt(stack[0] / (2*stack[1] + stack[0])) # Initial curvature

        else:
            expo = np.exp(1j * field.k * stack[0])

        _comps = []
        for i in range(6):
            _comps.append((expo * field[i].ravel()).reshape(field[i].shape))

        field_prop = PTypes.fields(*_comps)
        field_prop.setMeta(hybridDict["t_name"], field.k)

        self.fields[hybridDict["field_out"]] = field_prop

        if hybridDict["interp"]:
            self.interpFrame(hybridDict["fr_out"], hybridDict["field_out"], hybridDict["t_name"], hybridDict["field_out"], comp=hybridDict["comp"])
    
    def _loadFramePoynt(self, Poynting : PTypes.rfield, name_source : str) -> PTypes.frame:
        """!
        Convert a Poynting vector grid to a frame object.
        
        @param Poynting An rfield object containing reflected Poynting vectors.
        @param name_source Name of reflector on which reflected Poynting vectors are defined
        
        @returns frame_in Frame object containing the Poynting vectors and base points.
        
        @see rfield
        @see frame
        """

        PChecks.check_elemSystem(name_source, self.system, self.clog, extern=True)
        grids = BRefl.generateGrid(self.system[name_source])

        nTot = Poynting.x.shape[0] * Poynting.x.shape[1]
        frame_in = PTypes.frame(nTot, grids.x.ravel(), grids.y.ravel(), grids.z.ravel(),
                        Poynting.x.ravel(), Poynting.y.ravel(), Poynting.z.ravel())

        return frame_in
    
    def _optimiseFocus(self, f0, *args) -> float: #TODO: check typing
        """!
        Cost function for finding a ray-trace frame focus.
        Optimises RMS spot size as function of tilt multiple f0.
        
        @param f0 Tilt multiple for finding focus.
        @param args The runRTDict for propagation and ray-trace tilt of input frame.
        
        @returns RMS The RMS spot size of the frame at f0 times the tilt.
        """

        runRTDict, tilt = args

        trans = f0 * tilt

        self.translateGrids(f"focal_plane_{runRTDict['fr_in']}", trans)
        
        self.runRayTracer(runRTDict)
        RMS = self.calcSpotRMS(f"focus_{runRTDict['fr_in']}")
        self.translateGrids(f"focal_plane_{runRTDict['fr_in']}", -trans)
        #self.removeFrame() 
        return RMS
    
    def _checkBoundPO(self, name : str, transf : np.ndarray):
        """!
        Check if an element to be rotated is bound to a PO field/current.
        If so, rotate vectorial field/current components along.
        
        @param name Name of reflector to be rotated.
        @param transf Array containing the transformation of the reflector.
        """

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
            for field in bound_fields:
                out = BTransf.transformPO(self.fields[field], transf)
                self.fields[field] = self.copyObj(out)

        if bound_currents:
            for current in bound_currents:
                out = BTransf.transformPO(self.currents[current], transf)
                self.currents[current] = self.copyObj(out)
   
    def _compToFields(self, comp : str, field : np.ndarray): #TODO: check typing, Is comp actually a string
        """!
        Transform a single component to a filled fields object by setting all other components to zero.
        
        @param comp Name of component. Instance of FieldComponents enum object.
        @param field Array to be inserted in fields object.
        
        @returns field_c Filled fields object with one component filled.
        """

        null = np.zeros(field.shape, dtype=complex)
        
        clist = []
        for i in range(6):
            if i == comp.value:
                clist.append(field)

            else:
                clist.append(null)

        field_c = PTypes.fields(*clist)

        return field_c
   
    def _absRotationMat(self, rotation : np.ndarray, ori : np.ndarray, pivot): #TODO: check typing pivot
        """!
        Calculate an absolute rotation matrix. Private method.

        The matrix is calculated with respect to the object's orientation, with respect to the z-axis.

        @param rotation Orientation wit respect to the z-axis to rotate to.
        @param ori Orientation of object.
        @param pivot Pivot of object.
        
        @returns Rtot Absolute rotation matrix of dimensions 4 x 4.
        """

        match = world.IAX()
        match_rot = (MatTransf.MatRotate(rotation))[:-1, :-1] @ match
        R = self.findRotation(ori, match_rot)

        Tp = self.copyObj(world.INITM())
        Tpm = self.copyObj(world.INITM())
        Tp[:-1,-1] = pivot
        Tpm[:-1,-1] = -pivot
        
        Rtot = Tp @ R @ Tpm
        return Rtot

    def _fillCoeffs(self, name : str, a : float, b : float, c : float): #TODO: check typing pivot
        """!
        Fill the coeffs values for an internal reflector dictionary.
        
        @param name Name of reflector in system.
        @param a The a coefficient.
        @param b The b coefficient.
        @param c The c coefficient.
        """
        
        self.system[name]["coeffs"][0] = a
        self.system[name]["coeffs"][1] = b
        self.system[name]["coeffs"][2] = c

