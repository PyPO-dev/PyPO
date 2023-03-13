import numpy as np
import os
import ctypes
import pathlib

nThreads_cpu = os.cpu_count()

from src.PyPO.PyPOTypes import *
from src.PyPO.CustomLogger import CustomLogger

PO_modelist = ["JM", "EH", "JMEH", "EHP", "FF", "scalar"]

clog_mgr = CustomLogger(os.path.basename(__file__))
clog = clog_mgr.getCustomLogger()

def has_CUDA():
    has = False

    win_cuda = os.path.exists(pathlib.Path(__file__).parents[2]/"out/build/Debug/pypogpu.dll")    
    nix_cuda = os.path.exists(pathlib.Path(__file__).parents[2]/"out/build/libpypogpu.so")
    mac_cuda = os.path.exists(pathlib.Path(__file__).parents[2]/"out/build/libpypogpu.dylib")

    has = win_cuda or nix_cuda or mac_cuda

    return has

def check_elemSystem(name, elements, errStr="", extern=False):
    if name not in elements:
        errStr += errMsg_noelem(name)

    if extern:
        if errStr:
            errList = errStr.split("\n")[:-1]

            for err in errList:
                clog.error(err)
            raise ElemNameError()
    
    else:
        return errStr

def check_fieldSystem(name, fields, errStr="", extern=False):
    if name not in fields:
        errStr += errMsg_nofield(name)

    if extern:
        if errStr:
            errList = errStr.split("\n")[:-1]

            for err in errList:
                clog.error(err)
            raise FieldNameError()
    
    else:
        return errStr

def check_currentSystem(name, currents, errStr="", extern=False):
    if name not in currents:
        errStr += errMsg_nocurrent(name)

    if extern:
        if errStr:
            errList = errStr.split("\n")[:-1]

            for err in errList:
                clog.error(err)
            raise CurrentNameError()
    
    else:
        return errStr

def check_scalarfieldSystem(name, scalarfields, errStr="", extern=False):
    if name not in scalarfields:
        errStr += errMsg_noscalarfield(name)

    if extern:
        if errStr:
            errList = errStr.split("\n")[:-1]

            for err in errList:
                clog.error(err)
            raise ScalarFieldNameError()
    
    else:
        return errStr

def check_frameSystem(name, frames, errStr="", extern=False):
    if name not in frames:
        errStr += errMsg_noframe(name)

    if extern:
        if errStr:
            errList = errStr.split("\n")[:-1]

            for err in errList:
                clog.error(err)
            raise FrameNameError()
    
    else:
        return errStr

def check_groupSystem(name, groups, errStr="", extern=False):
    if name not in groups:
        errStr += errMsg_nogroup(name)

    if extern:
        if errStr:
            errList = errStr.split("\n")[:-1]

            for err in errList:
                clog.error(err)
            raise GroupNameError()
    
    else:
        return errStr

# Error classes to be used
class InputReflError(Exception):
    pass

class InputRTError(Exception):
    pass

class RunRTError(Exception):
    pass

class RunPOError(Exception):
    pass

class ElemNameError(Exception):
    pass

class FieldNameError(Exception):
    pass

class CurrentNameError(Exception):
    pass

class FrameNameError(Exception):
    pass

class ScalarFieldNameError(Exception):
    pass

class GroupNameError(Exception):
    pass

# Error message definitions
def errMsg_name(elemName):
    return f"Name \"{elemName}\" already in use. Choose different name.\n"

def errMsg_field(fieldName, elemName):
    return f"Missing field \"{fieldName}\", element {elemName}.\n"

def errMsg_type(fieldName, inpType, elemName, fieldType):
    return f"Wrong type {inpType} in field \"{fieldName}\", element {elemName}. Expected {fieldType}.\n"

def errMsg_option(fieldName, option, elemName, args):
    if len(args) == 2:
        return f"Unknown option \"{option}\" in field \"{fieldName}\", element {elemName}. Expected \"{args[0]}\" or \"{args[1]}\".\n"

    elif len(args) == 3:
        return f"Unknown option \"{option}\" in field \"{fieldName}\", element {elemName}. Expected \"{args[0]}\", \"{args[1]}\" or \"{args[2]}\".\n"

def errMsg_shape(fieldName, shape, elemName, shapeExpect):
    return f"Incorrect input shape of {shape} for field \"{fieldName}\", element {elemName}. Expected {shapeExpect}.\n"

def errMsg_value(fieldName, value, elemName):
    return f"Incorrect value {value} encountered in field \"{fieldName}\", element {elemName}.\n"

def errMsg_noelem(elemName):
    return f"Element {elemName} not in system.\n"

def errMsg_noframe(frameName):
    return f"Frame {frameName} not in system.\n"

def errMsg_nofield(fieldName):
    return f"Field {fieldName} not in system.\n"

def errMsg_nocurrent(Name):
    return f"Current {Name} not in system.\n"

def errMsg_noscalarfield(scalarfieldName):
    return f"Scalar field {scalarfieldName} not in system.\n"

def errMsg_nogroup(groupName):
    return f"Group {groupName} not in system.\n"

# Check blocks for different datatypes
def block_ndarray(fieldName, elemDict, shape):
    _errStr = ""
    if not isinstance(elemDict[fieldName], np.ndarray):
        _errStr += errMsg_type(fieldName, type(elemDict[fieldName]), elemDict["name"], np.ndarray)

    elif not elemDict[fieldName].shape == shape:
        _errStr += errMsg_shape(fieldName, elemDict[fieldName].shape, elemDict["name"], f"{shape}")
    
    return _errStr

##
# Check element input dictionary.
#
# Checks the input dictionary for errors. Raises exceptions when encountered.
#
# @param elemName Name of element, string.
# @param nameList List of names in system dictionary.
def check_ElemDict(elemDict, nameList, num_ref):
    
    errStr = ""
   
    elemDict["transf"] = np.eye(4)
   
    if not "pos" in elemDict:
        elemDict["pos"] = np.zeros(3)

    else:
        errStr += block_ndarray("pos", elemDict, (3,))

    if not "ori" in elemDict:
        elemDict["ori"] = np.array([0,0,1])
    
    else:
        errStr += block_ndarray("ori", elemDict, (3,))

    if not "flip" in elemDict:
        elemDict["flip"] = False

    else:
        if not isinstance(elemDict["flip"], bool):
            clog.warning("Invalid option {elemDict['flip']} for flip. Defaulting to False.")

    if elemDict["type"] == 0:
        if not "name" in elemDict:
            elemDict["name"] = "Parabola"
        
        if "pmode" in elemDict:
            if elemDict["pmode"] == "focus":
                if "vertex" in elemDict:
                    errStr += block_ndarray("vertex", elemDict, (3,))
                else:
                    errStr += errMsg_field("vertex", elemDict["name"])

                if "focus_1" in elemDict:
                    errStr += block_ndarray("focus_1", elemDict, (3,))
                else:
                    errStr += errMsg_field("focus_1", elemDict["name"])

            elif elemDict["pmode"] == "manual":
                if "coeffs" in elemDict:
                    errStr += block_ndarray("coeffs", elemDict, (2,))

                else:
                    errStr += errMsg_field("coeffs", elemDict["name"])

            else:
                args = ["focus", "manual"]
                errStr += errMsg_option("pmode", elemDict["pmode"], elemDict["name"], args=args)

        else:
            errStr += errMsg_field("pmode", elemDict["name"])

    elif elemDict["type"] == 1 or elemDict["type"] == 2:
        if elemDict["type"] == 1:
            
            if not "name" in elemDict:
                elemDict["name"] = "Hyperbola"
        
        else:
            if not "name" in elemDict:
                elemDict["name"] = "Ellipse"

        if "pmode" in elemDict:
            if elemDict["pmode"] == "focus":
                if "focus_1" in elemDict:
                    errStr += block_ndarray("focus_1", elemDict, (3,))
                else:
                    errStr += errMsg_field("focus_1", elemDict["name"])

                if "focus_2" in elemDict:
                    errStr += block_ndarray("focus_2", elemDict, (3,))
                else:
                    errStr += errMsg_field("focus_2", elemDict["name"])
                
                if "ecc" in elemDict:
                    if not ((isinstance(elemDict["ecc"], float) or isinstance(elemDict["ecc"], int))):
                        errStr += errMsg_type("ecc", type(elemDict["ecc"]), elemDict["name"], [float, int])

                    elif elemDict["type"] == 1:
                        if elemDict["ecc"] <= 1:
                            errStr += errMsg_value("ecc", elemDict["ecc"], elemDict["name"])

                    elif elemDict["type"] == 2:
                        if elemDict["ecc"] < 0 or elemDict["ecc"] >= 1:
                            errStr += errMsg_value("ecc", elemDict["ecc"], elemDict["name"])
                
                else:
                    errStr += errMsg_field("ecc", elemDict["name"])
            
            elif elemDict["pmode"] == "manual":
                if "coeffs" in elemDict:
                    errStr += block_ndarray("coeffs", elemDict, (3,))
                else:
                    errStr += errMsg_field("coeffs", elemDict["name"])

            else:
                args = ["focus", "manual"]
                errStr += errMsg_option("pmode", elemDict["pmode"], elemDict["name"], args=args)

    elif elemDict["type"] == 3:
        if not "name" in elemDict:
            elemDict["name"] = "plane"

    if "gmode" in elemDict:
        if elemDict["gmode"] == "xy":
            if "lims_x" in elemDict:
                errStr += block_ndarray("lims_x", elemDict, (2,))
            else:
                errStr += errMsg_field("lims_x", elemDict["name"])

            if "lims_y" in elemDict:
                errStr += block_ndarray("lims_y", elemDict, (2,))
            else:
                errStr += errMsg_field("lims_y", elemDict["name"])

        elif elemDict["gmode"] == "uv":
            if not "gcenter" in elemDict:
                elemDict["gcenter"] = np.zeros(2)
           
            if not "ecc_uv" in elemDict:
                elemDict["ecc_uv"] = 0

            if not "rot_uv" in elemDict:
                elemDict["rot_uv"] = 0

            if "lims_u" in elemDict:
                errStr += block_ndarray("lims_u", elemDict, (2,))

                if elemDict["lims_u"][0] < 0:
                    errStr += errMsg_value("lims_u", elemDict["lims_u"][0], elemDict["name"])

                if elemDict["lims_u"][1] < 0:
                    errStr += errMsg_value("lims_u", elemDict["lims_u"][1], elemDict["name"])
            else:
                errStr += errMsg_field("lims_u", elemDict["name"])

            if "lims_v" in elemDict:
                errStr += block_ndarray("lims_v", elemDict, (2,))

                if elemDict["lims_v"][0] < 0:
                    errStr += errMsg_value("lims_v", elemDict["lims_v"][0], elemDict["name"])
 
                if elemDict["lims_v"][1] > 360:
                    errStr += errMsg_value("lims_v", elemDict["lims_v"][1], elemDict["name"])
            else:
                errStr += errMsg_field("lims_v", elemDict["name"])

            if "ecc_uv" in elemDict:
                if not ((isinstance(elemDict["ecc_uv"], float) or isinstance(elemDict["ecc_uv"], int))):
                    errStr += errMsg_type("ecc_uv", type(elemDict["ecc_uv"]), elemDict["name"], [float, int])

                if elemDict["ecc_uv"] < 0 or elemDict["ecc_uv"] > 1:
                    errStr += errMsg_value("ecc_uv", elemDict["ecc_uv"], elemDict["name"])

            if "rot_uv" in elemDict:
                if not ((isinstance(elemDict["rot_uv"], float) or isinstance(elemDict["rot_uv"], int))):
                    errStr += errMsg_type("rot_uv", type(elemDict["rot_uv"]), elemDict["name"], [float, int])
        
            if "gcenter" in elemDict:
                errStr += block_ndarray("gcenter", elemDict, (2,))

        elif elemDict["gmode"] == "AoE":
            if "lims_Az" in elemDict:
                errStr += block_ndarray("lims_Az", elemDict, (2,))
            else:
                errStr += errMsg_field("lims_Az", elemDict["name"])

            if "lims_El" in elemDict:
                errStr += block_ndarray("lims_El", elemDict, (2,))
            else:
                errStr += errMsg_field("lims_El", elemDict["name"])
    
        else:
            args = ["xy", "uv", "AoE (plane only)"]
            errStr += errMsg_option("gmode", elemDict["gmode"], elemDict["name"], args=args)

    else:
        errStr += errMsg_field("gmode", elemDict["name"])

    if "gridsize" in elemDict:
        errStr += block_ndarray("gridsize", elemDict, (2,))

        if not (isinstance(elemDict["gridsize"][0], np.int64) or isinstance(elemDict["gridsize"][0], np.int32)):
            errStr += errMsg_type("gridsize[0]", type(elemDict["gridsize"][0]), elemDict["name"], [np.int64, np.int32])

        if not (isinstance(elemDict["gridsize"][1], np.int64) or isinstance(elemDict["gridsize"][1], np.int32)):
            errStr += errMsg_type("gridsize[1]", type(elemDict["gridsize"][1]), elemDict["name"], [np.int64, np.int32])
    
    if elemDict["name"] in nameList:
        elemDict["name"] = elemDict["name"] + "_{}".format(num_ref)

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputReflError()
    
    else:
        return 0

def check_TubeRTDict(TubeRTDict, nameList):
    errStr = ""
    
    if TubeRTDict["name"] in nameList:
        errStr += errMsg_name(TubeRTDict["name"])

    if "nRays" in TubeRTDict:
        if not isinstance(TubeRTDict["nRays"], int):
            errStr += errMsg_type("nRays", type(TubeRTDict["nRays"]), "TubeRTDict", int)

    else:
        errStr += errMsg_field("nRays", "TubeRTDict")

    if "nRing" in TubeRTDict:
        if not isinstance(TubeRTDict["nRing"], int):
            errStr += errMsg_type("nRing", type(TubeRTDict["nRays"]), "TubeRTDict", int)

    else:
        errStr += errMsg_field("nRays", "TubeRTDict")


    if "angx" in TubeRTDict:
        if not ((isinstance(TubeRTDict["angx"], float) or isinstance(TubeRTDict["angx"], int))):
            errStr += errMsg_type("angx", type(TubeRTDict["angx"]), "TubeRTDict", [float, int])

    else:
        errStr += errMsg_field("angx", "TubeRTDict")


    if "angy" in TubeRTDict:
        if not ((isinstance(TubeRTDict["angy"], float) or isinstance(TubeRTDict["angy"], int))):
            errStr += errMsg_type("angy", type(TubeRTDict["angy"]), "TubeRTDict", [float, int])

    else:
        errStr += errMsg_field("angy", "TubeRTDict")


    if "a" in TubeRTDict:
        if not ((isinstance(TubeRTDict["a"], float) or isinstance(TubeRTDict["a"], int))):
            errStr += errMsg_type("a", type(TubeRTDict["a"]), "TubeRTDict", [float, int])

    else:
        errStr += errMsg_field("a", "TubeRTDict")


    if "b" in TubeRTDict:
        if not ((isinstance(TubeRTDict["b"], float) or isinstance(TubeRTDict["b"], int))):
            errStr += errMsg_type("b", type(TubeRTDict["b"]), "TubeRTDict", [float, int])

    else:
        errStr += errMsg_field("b", "TubeRTDict")

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputRTError()

def check_runRTDict(runRTDict, elements, frames):
    errStr = ""
   
    cuda = has_CUDA()
    errStr = check_frameSystem(runRTDict["fr_in"], frames, errStr)
    errStr = check_elemSystem(runRTDict["t_name"], elements, errStr)

    if "tol" not in runRTDict:
        runRTDict["tol"] = 1e-3

    elif "tol" in runRTDict:
        if runRTDict["tol"] < 0:
            clog.warning("Negative tolerances are not allowed. Changing sign.")
            runRTDict["tol"] *= -1


    if "t0" not in runRTDict:
        runRTDict["t0"] = 1

    if "device" not in runRTDict:
        runRTDict["device"] = "CPU"
    
    if "device" in runRTDict:
        if runRTDict["device"] != "CPU" and runRTDict["device"] != "GPU":
            clog.warning(f"Device {runRTDict['device']} unknown. Defaulting to CPU.")
            runRTDict["device"] = "CPU"

        if runRTDict["device"] == "GPU" and not cuda:
            clog.warning(f"No PyPO CUDA libraries found. Defaulting to CPU.")
            runRTDict["device"] = "CPU"

        if runRTDict["device"] == "CPU":
            if "nThreads" in runRTDict:
                if runRTDict["nThreads"] > nThreads_cpu:
                    clog.warning(f"Insufficient CPU threads available, automatically reducing threadcount.")
                    runRTDict["nThreads"] = nThreads_cpu

            else:
                runRTDict["nThreads"] = nThreads_cpu

        elif runRTDict["device"] == "GPU":
            if "nThreads" not in runRTDict:
                runRTDict["nThreads"] = 256


    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise RunRTError()

def check_GBDict(GBDict):
    errStr = ""

def check_runPODict(runPODict, elements, currents, scalarfields):
    errStr = ""

    cuda = has_CUDA()
    
    if not "exp" in runPODict:
        runPODict["exp"] = "fwd"

    if "mode" not in runPODict:
        errStr += f"Please provide propagation mode.\n"
    
    else:
        if runPODict["mode"] not in PO_modelist:
            errStr += f"{runPODict['mode']} is not a valid propagation mode.\n"

        if "s_current" in runPODict:
            errStr = check_currentSystem(runPODict["s_current"], currents, errStr)
        
        if "s_scalarfield" in runPODict:
            errStr = check_frameSystem(runPODict["s_scalarfield"], scalarfields, errStr)
    
    errStr = check_elemSystem(runPODict["t_name"], elements, errStr)
   
    if "epsilon" not in runPODict:
        runPODict["epsilon"] = 1

    if "device" not in runPODict:
        runPODict["device"] = "CPU"

    if "device" in runPODict:
        if runPODict["device"] != "CPU" and runPODict["device"] != "GPU":
            clog.warning(f"Device {runPODict['device']} unknown. Defaulting to CPU.")
            runPODict["device"] = "CPU"

        if runPODict["device"] == "GPU" and not cuda:
            clog.warning(f"No PyPO CUDA libraries found. Defaulting to CPU.")
            runPODict["device"] = "CPU"

        if runPODict["device"] == "CPU":
            
            if "nThreads" in runPODict:
                if runPODict["nThreads"] > nThreads_cpu:
                    clog.warning(f"Insufficient CPU threads available, automatically reducing threadcount.")
                    runPODict["nThreads"] = nThreads_cpu

            else:
                runPODict["nThreads"] = nThreads_cpu

        elif runPODict["device"] == "GPU":
            if "nThreads" not in runPODict:
                runPODict["nThreads"] = 256


    if errStr:
        errList = errStr.split("\n")[:-1]
        for err in errList:
            clog.error(err)
        raise RunPOError()

def check_ellipseLimits(ellipsoid):
    buff = 1000
    idx_lim = 0

    if ellipsoid["coeffs"][1] < ellipsoid["coeffs"][0]:
        idx_lim = 1

    if ellipsoid["gmode"] == 0:
        if np.absolute(ellipsoid["lims_x"][0]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Lower x-limit of {ellipsoid['lims_x'][0]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_x"][0] = ellipsoid["coeffs"][idx_lim] + ellipsoid["coeffs"][0] / buff
        
        if np.absolute(ellipsoid["lims_x"][1]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Upper x-limit of {ellipsoid['lims_x'][1]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_x"][1] = ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_x"][1] / buff
        
        if np.absolute(ellipsoid["lims_y"][0]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Lower y-limit of {ellipsoid['lims_y'][0]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_y"][0] = ellipsoid["coeffs"][idx_lim] + ellipsoid["lims_y"][0] / buff
        
        if np.absolute(ellipsoid["lims_y"][1]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Upper y-limit of {ellipsoid['lims_y'][1]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_y"][1] = ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_y"][1] / buff

    elif ellipsoid["gmode"] == 1:
        if np.absolute(ellipsoid["lims_u"][0]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Lower u-limit of {ellipsoid['lims_u'][0]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_u"][0] = ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_u"][0] / buff
 
        if np.absolute(ellipsoid["lims_u"][1]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Upper u-limit of {ellipsoid['lims_u'][1]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_u"][1] = ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_u"][1] / buff

