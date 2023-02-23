import numpy as np
import os

from src.PyPO.PyPOTypes import *
from src.PyPO.CustomLogger import CustomLogger

# Error classes to be used
class InputReflError(Exception):
    pass

class InputRTError(Exception):
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
def check_ElemDict(elemDict, nameList):
    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
    
    errStr = ""
   
    if elemDict["name"] in nameList:
        errStr += errMsg_name(elemDict["name"])

    if elemDict["type"] == 0:

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
    
    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputReflError()
    
    else:
        return 0

def check_RTDict(RTDict, nameList):
    errStr = ""
    if RTDict["name"] in nameList:
        errStr += errMsg_name(RTDict["name"])

    if "nRays" in RTDict:
        if not isinstance(RTDict["nRays"], int):
            errStr += errMsg_type("nRays", type(RTDict["nRays"]), "RTDict", int)

    else:
        errStr += errMsg_field("nRays", "RTDict")

    if "nRing" in RTDict:
        if not isinstance(RTDict["nRing"], int):
            errStr += errMsg_type("nRing", type(RTDict["nRays"]), "RTDict", int)

    else:
        errStr += errMsg_field("nRays", "RTDict")


    if "angx" in RTDict:
        if not ((isinstance(RTDict["angx"], float) or isinstance(RTDict["angx"], int))):
            errStr += errMsg_type("angx", type(RTDict["angx"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("angx", "RTDict")


    if "angy" in RTDict:
        if not ((isinstance(RTDict["angy"], float) or isinstance(RTDict["angy"], int))):
            errStr += errMsg_type("angy", type(RTDict["angy"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("angy", "RTDict")


    if "a" in RTDict:
        if not ((isinstance(RTDict["a"], float) or isinstance(RTDict["a"], int))):
            errStr += errMsg_type("a", type(RTDict["a"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("a", "RTDict")


    if "b" in RTDict:
        if not ((isinstance(RTDict["b"], float) or isinstance(RTDict["b"], int))):
            errStr += errMsg_type("b", type(RTDict["b"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("b", "RTDict")

    if "tChief" in RTDict:
        errStr += block_ndarray("tChief", RTDict, (3,))
    else:
        errStr += errMsg_field("tChief", "RTDict")

    if "oChief" in RTDict:
        errStr += block_ndarray("oChief", RTDict, (3,))
    else:
        errStr += errMsg_field("oChief", "RTDict")

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputRTError()

def check_GBDict(GBDict):
    errStr = ""
