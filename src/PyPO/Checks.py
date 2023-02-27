import numpy as np
import os
import ctypes
import pathlib

nThreads_cpu = os.cpu_count()

from src.PyPO.PyPOTypes import *
from src.PyPO.CustomLogger import CustomLogger

PO_modelist = ["JM", "EH", "JMEH", "EHP", "FF", "scalar"]

def has_CUDA():
    has = False

    win_cuda = os.path.exists(pathlib.Path(__file__).parents[2]/"out/build/Debug/pypogpu.dll")    
    nix_cuda = os.path.exists(pathlib.Path(__file__).parents[2]/"out/build/libpypogpu.so")
    mac_cuda = os.path.exists(pathlib.Path(__file__).parents[2]/"out/build/libpypogpu.dylib")

    has = win_cuda or nix_cuda or mac_cuda

    return has

# Error classes to be used
class InputReflError(Exception):
    pass

class InputRTError(Exception):
    pass

class RunRTError(Exception):
    pass

class RunPOError(Exception):
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
    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
    
    errStr = ""
   
    elemDict["transf"] = np.eye(4)
    
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
    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
    
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

    if "tChief" in TubeRTDict:
        errStr += block_ndarray("tChief", TubeRTDict, (3,))
    else:
        errStr += errMsg_field("tChief", "TubeRTDict")

    if "oChief" in TubeRTDict:
        errStr += block_ndarray("oChief", TubeRTDict, (3,))
    else:
        errStr += errMsg_field("oChief", "TubeRTDict")

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputRTError()

def check_runRTDict(runRTDict, elements, frames):
    errStr = ""

    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
   
    cuda = has_CUDA()
    if runRTDict["fr_in"] not in frames:
        errStr += errMsg_noframe(runRTDict["fr_in"])
    if runRTDict["t_name"] not in elements:
        errStr += errMsg_noelem(runRTDict["t_name"])
   
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
    

    elif "device" in runRTDict:
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

    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
   
    cuda = has_CUDA()
    
    if not "exp" in runPODict:
        runPODict["exp"] = "fwd"

    if "mode" not in runPODict:
        errStr += f"Please provide propagation mode.\n"
    
    else:
        if runPODict["mode"] not in PO_modelist:
            errStr += f"{runPODict['mode']} is not a valid propagation mode.\n"

        if "s_current" in runPODict:
            if runPODict["s_current"] not in currents:
                errStr += errMsg_nocurrent(runPODict["s_current"])
        
        if "s_scalarfield" in runPODict:
            if runPODict["s_scalarfield"] not in scalarfields:
                errStr += errMsg_noscalarfield(runPODict["scalarfield"])
    
    if runPODict["t_name"] not in elements:
        errStr += errMsg_noelem(runPODict["t_name"])
   
    if "epsilon" not in runPODict:
        runPODict["epsilon"] = 1

    if "device" not in runPODict:
        runPODict["device"] = "CPU"

    elif "device" in runPODict:
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
    
