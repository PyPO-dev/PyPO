# Standard Python imports
import numbers
import scipy.optimize as opt
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
from src.PyPO.CustomLogger import CustomLogger
import src.PyPO.Plotter as plt
import src.PyPO.Efficiencies as effs
import src.PyPO.FitGauss as fgs

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
    def __init__(self, redirect=None, context=None, verbose=True):
        self.num_ref = 0
        self.num_cam = 0
        self.nThreads_cpu = os.cpu_count()
        
        Config.initPrint(redirect)
        Config.setContext(context)
        # Internal dictionaries
        self.system = {}
        self.frames = {}
        self.fields = {}
        self.currents = {}
        self.scalarfields = {}

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
        
        self.clog_mgr = CustomLogger(os.path.basename(__file__))
        self.clog = self.clog_mgr.getCustomLogger() if verbose else self.clog_mgr.getCustomLogger(open(os.devnull, "w"))

        self.clog.info("INITIALIZED EMPTY SYSTEM.")
    ##
    # Destructor. Deletes any reference to the logger assigned to current system.
    def __del__(self):
        self.clog.info("EXITING SYSTEM.")
        del self.clog_mgr
        del self.clog

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
            
            self.system.update(sys_copy)
            self.fields.update(sys_copy)
            self.currents.update(sys_copy)
            self.frames.update(sys_copy)

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

        check_ElemDict(reflDict, self.system.keys(), self.num_ref) 

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
            pivot = offTrans

            self.system[reflDict["name"]]["transf"] = MatRotate(offRot, reflDict["transf"], pivot)
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, reflDict["transf"])

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = -1

        elif reflDict["pmode"] == "manual":
            self.system[reflDict["name"]]["coeffs"] = np.array([reflDict["coeffs"][0], reflDict["coeffs"][1], -1])

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gcenter"] = np.zeros(2)
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            # Convert v in degrees to radians
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1
        
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
        check_ElemDict(reflDict, self.system.keys(), self.num_ref) 
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
            self.system[reflDict["name"]]["gcenter"] = np.zeros(2)
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

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
        check_ElemDict(reflDict, self.system.keys(), self.num_ref) 
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
            rotation = np.array([0, rot_y, rot_z])

            a = np.sqrt(np.dot(diff, diff)) / (2 * ecc)
            b = a * np.sqrt(1 - ecc**2)
            
            _transf = MatRotate(rotation, reflDict["transf"])
            self.system[reflDict["name"]]["transf"] = MatTranslate(trans, _transf)

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = b


        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gcenter"] = np.zeros(2)
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

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
        check_ElemDict(reflDict, self.system.keys(), self.num_ref) 

        self.system[reflDict["name"]] = self.copyObj(reflDict)
        self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

        self.system[reflDict["name"]]["coeffs"][0] = -1
        self.system[reflDict["name"]]["coeffs"][1] = -1
        self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gcenter"] = np.zeros(2)
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["gmode"] = 1

        elif reflDict["gmode"] == "AoE":
            # Assume is given in degrees
            # Convert Az and El to radians
            self.system[reflDict["name"]]["gcenter"] = np.zeros(2)
            self.system[reflDict["name"]]["ecc_uv"] = 0
            self.system[reflDict["name"]]["rot_uv"] = 0
            self.system[reflDict["name"]]["gmode"] = 0

            self.system[reflDict["name"]]["lims_Az"] = [self.system[reflDict["name"]]["lims_Az"][0],
                                                        self.system[reflDict["name"]]["lims_Az"][1]]

            self.system[reflDict["name"]]["lims_El"] = [self.system[reflDict["name"]]["lims_El"][0],
                                                        self.system[reflDict["name"]]["lims_El"][1]]

            self.system[reflDict["name"]]["gmode"] = 2

        self.clog.info(f"Added plane {reflDict['name']} to system.")
        self.num_ref += 1
    
    ##
    # Rotate reflector grids.
    #
    # Apply a rotation, around a center of rotation, to a (selection of) reflector(s).
    #
    # @param name Reflector name or list of reflector names.
    # @param rotation Numpy ndarray of length 3, containing rotation angles around x, y and z axes, in degrees.
    # @param pivot Numpy ndarray of length 3, containing pivot x, y and z co-ordinates, in mm. Defaults to origin. 
    def rotateGrids(self, name, rotation, pivot=None):
        pivot = np.zeros(3) if pivot is None else pivot

        if isinstance(name, list):
            for _name in name:
                self.system[_name]["transf"] = MatRotate(rotation, self.system[_name]["transf"], pivot)
            
        else:
            self.system[name]["transf"] = MatRotate(rotation, self.system[name]["transf"], pivot)
        self.clog.info(f"Rotated {name} by {*['{:0.3e}'.format(x) for x in list(rotation)],} degrees around {*['{:0.3e}'.format(x) for x in list(pivot)],}.")

    ##
    # Translate reflector grids.
    #
    # Apply a translation to a (selection of) reflector(s).
    #
    # @param name Reflector name or list of reflector names.
    # @param translation Numpy ndarray of length 3, containing translation x, y and z co-ordinates, in mm.
    def translateGrids(self, name, translation):
        if isinstance(name, list):
            for _name in name:
                self.system[_name]["transf"] = MatTranslate(translation, self.system[_name]["transf"])
        else:
            self.system[name]["transf"] = MatTranslate(translation, self.system[name]["transf"])
        
        self.clog.info(f"Translated {name} by {*['{:0.3e}'.format(x) for x in list(translation)],} millimeters.")

    ##
    # Home a reflector back into default configuration.
    #
    # Set internal transformation matrix of a (selection of) reflector(s) to identity.
    #
    # @param name Reflector name or list of reflector names to be homed.
    def homeReflector(self, name, trans=True, rot=True):
        if isinstance(name, list):
            if trans:
                for _name in name:
                    _transf = np.eye(4)
                    _transf[:-1, :-1] = self.system[_name]["transf"][:-1, :-1]
                    self.system[_name]["transf"] = _transf
            
            if rot:
                for _name in name:
                    _transf = self.system[_name]["transf"]
                    _transf[:-1, :-1] = np.zeros(3)
                    self.system[_name]["transf"] = _transf

                    
        else:
            if trans:
                _transf = np.eye(4)
                _transf[:-1, :-1] = self.system[name]["transf"][:-1, :-1]
                self.system[name]["transf"] = _transf
            
            if rot:
                _transf = self.system[name]["transf"]
                _transf[:-1, :-1] = np.zeros(3)
                self.system[name]["transf"] = _transf

        self.clog.info(f"Transforming {name} to home position.")
    
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
        
        with open(os.path.join(path, "frames.pys"), 'wb') as file: 
            pickle.dump(self.frames, file)
        
        with open(os.path.join(path, "fields.pys"), 'wb') as file: 
            pickle.dump(self.fields, file)
        
        with open(os.path.join(path, "currents.pys"), 'wb') as file: 
            pickle.dump(self.currents, file)
        
        with open(os.path.join(path, "scalarfields.pys"), 'wb') as file: 
            pickle.dump(self.scalarfields, file)

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
        self.clog.info(f"Removed element {name} from system.")
        for n in name:
            del self.system[n]
    
    ##
    # Remove a ray-trace frame from system
    #
    # @param frameName Name of frame to be removed.
    def removeFrame(self, *frameName):
        self.clog.info(f"Removed frame {frameName} from system.")
        for fn in frameName:
            del self.frames[fn]
    
    ##
    # Remove a PO field from system
    #
    # @param fieldName Name of field to be removed.
    def removeField(self, fieldName):
        self.clog.info(f"Removed PO field {fieldName} from system.")
        del self.fields[fieldName]
    
    ##
    # Remove a PO current from system
    #
    # @param curentName Name of current to be removed.
    def removeCurrent(self, currentName):
        self.clog.info(f"Removed PO current {currentName} from system.")
        del self.currents[currentName]

    ##
    # Remove a scalar PO field from system
    #
    # @param fieldName Name of scalar field to be removed.
    def removeScalarField(self, fieldName):
        self.clog.info(f"Removed scalar PO field {fieldName} from system.")
        del self.scalarfields[fieldName]
    
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
        currents = calcCurrents(fields, self.system[name_source], mode)
        return currents

    ##
    # Instantiate a PO propagation. Stores desired output in the system.fields and/or system.currents lists.
    # If the 'EHP' mode is selected, the reflected Poynting frame is stored in system.frames.
    #
    # @param PODict Dictionary containing the PO propagation instructions.
    #
    # @see PODict
    def runPO(self, PODict):
        self.clog.info("*** Starting PO propagation ***")
       
        check_runPODict(PODict, self.system, self.currents, self.scalarfields)

        if PODict["mode"] != "scalar":
            sc_name = PODict["s_current"]
            PODict["s_current"] = self.currents[PODict["s_current"]]
            self.clog.info(f"Propagating {sc_name} on {PODict['s_current'].surf} to {PODict['t_name']}, propagation mode: {PODict['mode']}.")
            source = self.system[PODict["s_current"].surf]
            PODict["k"] = PODict["s_current"].k

        else:
            sc_name = PODict["s_scalarfield"]
            PODict["s_scalarfield"] = self.scalarfields[PODict["s_scalarfield"]]
            self.clog.info(f"Propagating {sc_name} on {PODict['s_scalarfield'].surf} to {PODict['t_name']}, propagation mode: {PODict['mode']}.")
            source = self.system[PODict["s_scalarfield"].surf]
            PODict["k"] = PODict["s_scalarfield"].k
       
        target = self.system[PODict["t_name"]]
        
        start_time = time.time()
        
        if PODict["device"] == "CPU":
            self.clog.info(f"Hardware: running {PODict['nThreads']} CPU threads.")
            self.clog.info(f"... Calculating ...")
            out = PyPO_CPUd(source, target, PODict)

        elif PODict["device"] == "GPU":
            self.clog.info(f"Hardware: running {PODict['nThreads']} CUDA threads per block.")
            self.clog.info(f"... Calculating ...")
            out = PyPO_GPUf(source, target, PODict)

        dtime = time.time() - start_time
        
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

        elif PODict["mode"] == "scalar":
            out.setMeta(PODict["t_name"], PODict["k"])
            self.scalarfields[PODict["name_field"]] = out

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
        
        check_TubeRTDict(argDict, self.frames.keys())
        self.frames[argDict["name"]] = makeRTframe(argDict)
    
    ##
    # Create a Gaussian beam distribution of rays from a GRTDict.
    #
    # @param argDict A GRTDict, filled. If not filled properly, will raise an exception.
    #
    # @see GRTDict
    def createGRTFrame(self, argDict):
        self.clog.info(f"Generating Gaussian ray-trace beam.")
        self.clog.info(f"... Sampling ...")
        if not argDict["name"]:
            argDict["name"] = f"Frame_{len(self.frames)}"
       
        start_time = time.time()
        argDict["angx0"] = np.degrees(argDict["lam"] / (np.pi * argDict["n"] * argDict["x0"]))
        argDict["angy0"] = np.degrees(argDict["lam"] / (np.pi * argDict["n"] * argDict["y0"]))

        #check_RTDict(argDict, self.frames.keys())
        self.frames[argDict["name"]] = makeGRTframe(argDict)
        dtime = time.time() - start_time
        self.clog.info(f"Succesfully sampled {argDict['nRays']} rays: {dtime} seconds.")

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

        check_runRTDict(_runRTDict, self.system, self.frames)

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
        self.frames[_runRTDict["fr_out"]] = frameObj

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

    ##
    # Calculate the geometric center of a ray-trace frame.
    #
    # @param name_frame Name of frame to calculate center of.
    #
    # @returns c_f Len-3 Numpy array containing x, y and z co-ordinates of frame center.
    def calcRTcenter(self, name_frame):
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
        aperDict = {} if aperDict is None else aperDict

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
        thres = -11 if thres is None else thres
        mode = "dB" if mode is None else mode
        
        self.fitGaussAbs(name_field, comp, thres, mode)
        field = getattr(self.fields[name_field], comp)
        surfaceObj = self.system[self.fields[name_field].surf]
        
        eff = effs.calcMainBeam(field, surfaceObj, self.scalarfields[f"fitGauss_{name_field}"].S)

        return eff
    
    ##
    # Generate point-source PO fields and currents.
    #
    # @param PSDict A PSDict dictionary, containing parameters for the point source.
    # @param name_surface Name of surface on which to define the point-source.
    #
    # @see PSDict
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

    ##
    # Generate point-source scalar PO field.
    #
    # @param PSDict A PSDict dictionary, containing parameters for the point source.
    # @param name_surface Name of surface on which to define the point-source.
    #
    # @see PSDict
    def generatePointSourceScalar(self, PSDict, name_surface):
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
                    vmin=-30, vmax=0, show=True, amp_only=False,
                    save=False, interpolation=None,
                    aperDict=None, mode='dB', project='xy',
                    units="", name="", titleA="Amp", titleP="Phase",
                    unwrap_phase=False, ret=False):

        aperDict = {"plot":False} if aperDict is None else aperDict
        
        if comp is None:
            field_comp = self.scalarfields[name_obj].S
            name_surface = self.scalarfields[name_obj].surf
        
        elif comp[0] == "E" or comp[0] == "H":
            field = self.fields[name_obj]
            name_surface = field.surf
        
            if comp in self.EHcomplist:
                field_comp = getattr(field, comp)

        elif comp[0] == "J" or comp[0] == "M":
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
                        save, interpolation,
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
                plotObject = self.system[n_s]
                plt.plot3D(plotObject, ax, fine, cmap, norm, foc1, foc2)
        
        else:
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
                norm=False, fine=2, show=True, foc1=False, foc2=False, save=False, ret=False, select=None, RTframes=None):

        select = [] if select is None else select
        RTframes = [] if RTframes is None else RTframes
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
        
    ##
    # Create a spot diagram of a ray-trace frame.
    #
    # @param frame_name Name of frame to plot.
    # @param project Set abscissa and ordinate of plot. Should be given as a string. Default is "xy".
    # @param ret Return Figure and Axis. Default is False.
    # @param aspect Aspect ratio of plot. Default is 1.
    def plotRTframe(self, frame_name, project="xy", ret=False, aspect=1):
        if ret:
            return plt.plotRTframe(self.frames[frame_name], project, self.savePath, ret, aspect)
        else:
            plt.plotRTframe(self.frames[frame_name], project, self.savePath, ret, aspect)

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
            return I

        lenv = np.linalg.norm(v)
        lenu = np.linalg.norm(u)
        if lenv == 0 or lenu == 0:
            self.clog.error("Encountered 0-length vector. Cannot proceed.")
            exit(0)

        w = np.cross(v, u)

        lenw = np.linalg.norm(w)
        
        w = w / lenw
        
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        #print(K)
        theta = np.arcsin(lenw / (lenv * lenu))
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        R_transf = np.eye(4)
        R_transf[:-1, :-1] = R
        return R_transf

    def findRTfocus(self, name_frame, f0=None, verbose=False):
        f0 = 0 if f0 is None else f0
        
        tilt = self.calcRTtilt(name_frame)
        center = self.calcRTcenter(name_frame)
        match = np.array([0, 0, 1])

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
        self.setLoggingVerbosity(verbose=False)
        res = opt.fmin(self._optimiseFocus, f0, args=(runRTDict, tilt), full_output=True, disp=False)
        if verbose:
            self.setLoggingVerbosity(verbose=True)

        out = res[0] * tilt + center
        self.clog.info(f"Focus: {*['{:0.3e}'.format(x) for x in out],}, RMS: {res[1]:.3e}")

        return out

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
        #self.clog.debug(M[1,2])

        r = np.degrees(np.array([rx, ry, rz]))

        testM = MatRotate(r, np.eye(4), pivot=None, radians=False)

        return r, testM

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
