import numpy as np
import os
import pathlib
import re

nThreads_cpu = os.cpu_count()

from PyPO.PyPOTypes import *
import PyPO.Config as Config
import PyPO.WorldParam as world

PO_modelist = ["JM", "EH", "JMEH", "EHP", "FF", "scalar"]

##
# @file
# File containing all commonly used checks for PyPO user input.

##
# Get the regular expression for checking if an object already exists.
# Counts the amount of occurrences in order to avoid conflicting names.
#
# @param name Name of object.
# @param nameList List of names to check.
#
# @returns num Increment of highest occurrence of number.
def getIndex(name, nameList):
    regex = f"(?<!.){name}(_(\d*(?![ -~])))*(?![ -~])"
    l = re.compile(regex)
    match = list(filter(l.match, nameList))
    match_spl = [int(x.replace(name + "_", "")) if x != name else 0 for x in match]
    num = 0
    if match_spl and not Config.override:
        num = max(match_spl) + 1

    return num

##
# Check if the CUDA dynamically linked libraries exist.
# Checks the paths for Windows, Linux and Mac OS.
def has_CUDA():
    has = False
    path_cur = pathlib.Path(__file__).parent.resolve()

    win_cuda = os.path.exists(os.path.join(path_cur, "pypogpu.dll"))
    nix_cuda = os.path.exists(os.path.join(path_cur, "libpypogpu.so"))
    mac_cuda = os.path.exists(os.path.join(path_cur, "libpypogpu.dylib"))

    has = win_cuda or nix_cuda or mac_cuda

    return has

##
# Check if a specified element is in the system dictionary.
#
# @param name Name of element.
# @param elements The system dictionary containing all elements.
# @param clog CustomLogger object.
# @param errStr Error string for appending error messages.
# @param extern Whether this function is called from System or from here.
#
# @returns errStr The error string with any new entries appended.
def check_elemSystem(name, elements, clog, errStr="", extern=False):
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

##
# Check if a specified field is in the fields dictionary.
#
# @param name Name of field.
# @param fields The fields dictionary containing all fields.
# @param clog CustomLogger object.
# @param errStr Error string for appending error messages.
# @param extern Whether this function is called from System or from here.
#
# @returns errStr The error string with any new entries appended.
def check_fieldSystem(name, fields, clog, errStr="", extern=False):
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

##
# Check if a specified current is in the currents dictionary.
#
# @param name Name of current.
# @param currents The currents dictionary containing all currents.
# @param clog CustomLogger object.
# @param errStr Error string for appending error messages.
# @param extern Whether this function is called from System or from here.
#
# @returns errStr The error string with any new entries appended.
def check_currentSystem(name, currents, clog, errStr="", extern=False):
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

##
# Check if a specified scalarfield is in the scalarfields dictionary.
#
# @param name Name of scalarfield.
# @param scalarfields The scalarfields dictionary containing all scalarfields.
# @param clog CustomLogger object.
# @param errStr Error string for appending error messages.
# @param extern Whether this function is called from System or from here.
#
# @returns errStr The error string with any new entries appended.
def check_scalarfieldSystem(name, scalarfields, clog, errStr="", extern=False):
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

##
# Check if a specified frame is in the frames dictionary.
#
# @param name Name of frame.
# @param frames The frames dictionary containing all frames.
# @param clog CustomLogger object.
# @param errStr Error string for appending error messages.
# @param extern Whether this function is called from System or from here.
#
# @returns errStr The error string with any new entries appended.
def check_frameSystem(name, frames, clog, errStr="", extern=False):
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

##
# Check if a specified group is in the groups dictionary.
#
# @param name Name of group.
# @param groups The groups dictionary containing all groups.
# @param clog CustomLogger object.
# @param errStr Error string for appending error messages.
# @param extern Whether this function is called from System or from here.
#
# @returns errStr The error string with any new entries appended.
def check_groupSystem(name, groups, clog, errStr="", extern=False):
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

##
# Input reflector error. Raised when an error is encountered in an input reflector dictionary.
class InputReflError(Exception):
    pass

##
# Input ray-trace error. Raised when an error is encountered in an input ray-trace dictionary.
class InputRTError(Exception):
    pass

##
# Propagate ray-trace error. Raised when an error is encountered in a ray-trace propagation dictionary.
class RunRTError(Exception):
    pass

##
# Input physical optics error. Raised when an error is encountered in an input PO beam dictionary.
class InputPOError(Exception):
    pass

##
# Propagate physical optics error. Raised when an error is encountered in a physical optics propagation dictionary.
class RunPOError(Exception):
    pass

##
# Hybrid propagation error. Raised when an error is encountered in a hybrid propagation dictionary. 
class HybridPropError(Exception):
    pass

##
# Element name error. Raised when specified element cannot be found in the system dictionary. 
class ElemNameError(Exception):
    pass

##
# Field name error. Raised when specified field cannot be found in the fields dictionary. 
class FieldNameError(Exception):
    pass

##
# Current name error. Raised when specified current cannot be found in the currents dictionary. 
class CurrentNameError(Exception):
    pass

##
# Frame name error. Raised when specified frame cannot be found in the frames dictionary. 
class FrameNameError(Exception):
    pass

##
# Scalarfield name error. Raised when specified scalarfield cannot be found in the scalarfields dictionary. 
class ScalarFieldNameError(Exception):
    pass

##
# Group name error. Raised when specified group cannot be found in the groups dictionary. 
class GroupNameError(Exception):
    pass

##
# Merge beamerror. Raised when beams are to be merged but are not on same surface. 
class MergeBeamError(Exception):
    pass

##
# ApertureError. Raised when aperDict is filled incorrectly.
class ApertureError(Exception):
    pass

##
# Error message when a mandatory field has not been filled in a dictionary.
#
# @param fieldName Name of field in dictionary that is not filled.
# @param elemName Name of dictionary where error occurred. 
#
# @returns errStr The errorstring.
def errMsg_field(fieldName, elemName):
    return f"Missing field \"{fieldName}\", element {elemName}.\n"

##
# Error message when a field has not been filled has been filled with an incorrect type.
#
# @param fieldName Name of field in dictionary that is incorrectly filled.
# @param inpType Type of given input.
# @param elemName Name of dictionary where error occurred. 
# @param fieldType Expected type of input.
#
# @returns errStr The errorstring.
def errMsg_type(fieldName, inpType, elemName, fieldType):
    return f"Wrong type {inpType} in field \"{fieldName}\", element {elemName}. Expected {fieldType}.\n"

##
# Error message when a field has an unknown option.
#
# @param fieldName Name of field in dictionary.
# @param option Given option.
# @param elemName Name of dictionary where error occurred. 
# @param args Expected options.
#
# @returns errStr The errorstring.
def errMsg_option(fieldName, option, elemName, args):
    if len(args) == 2:
        return f"Unknown option \"{option}\" in field \"{fieldName}\", element {elemName}. Expected \"{args[0]}\" or \"{args[1]}\".\n"

    elif len(args) == 3:
        return f"Unknown option \"{option}\" in field \"{fieldName}\", element {elemName}. Expected \"{args[0]}\", \"{args[1]}\" or \"{args[2]}\".\n"

##
# Error message when a field has an incorrect shape.
#
# @param fieldName Name of field in dictionary.
# @param shape Shape of input.
# @param elemName Name of dictionary where error occurred.
# @param shapeExpect Expected input shape for field.
#
# @returns errStr The errorstring.
def errMsg_shape(fieldName, shape, elemName, shapeExpect):
    return f"Incorrect input shape of {shape} for field \"{fieldName}\", element {elemName}. Expected {shapeExpect}.\n"

##
# Error message when a wrong input value is encountered.
#
# @param fieldName Name of field where incorrect value is encountered.
# @param value Input value.
# @param Name of dictionary where error occurred.
#
# @returns errStr The errorstring.
def errMsg_value(fieldName, value, elemName):
    return f"Incorrect value {value} encountered in field \"{fieldName}\", element {elemName}.\n"

##
# Error message when a reflector element is not present in System.
#
# @param elemName Name of element.
#
# @returns errStr The errorstring.
def errMsg_noelem(elemName):
    return f"Element {elemName} not in system.\n"

##
# Error message when a frame object is not present in System.
#
# @param frameName Name of frame.
#
# @returns errStr The errorstring.
def errMsg_noframe(frameName):
    return f"Frame {frameName} not in system.\n"

##
# Error message when a field object is not present in System.
#
# @param fieldName Name of field.
#
# @returns errStr The errorstring.
def errMsg_nofield(fieldName):
    return f"Field {fieldName} not in system.\n"

##
# Error message when a current object is not present in System.
#
# @param currentName Name of current.
#
# @returns errStr The errorstring.
def errMsg_nocurrent(currentName):
    return f"Current {currentName} not in system.\n"

##
# Error message when a scalarfield object is not present in System.
#
# @param scalarfieldName Name of scalarfield.
#
# @returns errStr The errorstring.
def errMsg_noscalarfield(scalarfieldName):
    return f"Scalar field {scalarfieldName} not in system.\n"

##
# Error message when a group is not present in System.
#
# @param groupName Name of group.
#
# @returns errStr The errorstring.
def errMsg_nogroup(groupName):
    return f"Group {groupName} not in system.\n"

##
# Error message when beams are to be merged but are not on the same surface.
#
# @param beamName Name of field/current that is not on surface.
# @param surf0 Zeroth surface, taken as the merging surface.
# @param surfd Surface on which current beam is defined.
#
# @returns errStr The errorstring.
def errMsg_mergebeam(beamName, surf0, surfd):
    return f"Cannot merge {beamName}, defined on {surfd}, on merging surface {surf0}.\n"

## 
# Check if an input array has correct shape.
#
# @param fieldName Name of field containing array.
# @param elemDict Dictionary containing field.
# @param shape Expected shape of input array.
#
# @returns errStr The errorstring.
def block_ndarray(fieldName, elemDict, shape, cust_name=False):
    _errStr = ""

    if not cust_name:
        nameObj = elemDict["name"]
    else:
        nameObj = cust_name

    if not isinstance(elemDict[fieldName], np.ndarray):
        _errStr += errMsg_type(fieldName, type(elemDict[fieldName]), nameObj, np.ndarray)

    elif not elemDict[fieldName].shape == shape:
        _errStr += errMsg_shape(fieldName, elemDict[fieldName].shape, nameObj, f"{shape}")
    
    return _errStr

##
# Check element input dictionary.
#
# Checks the input dictionary for errors. Raises exceptions when encountered.
#
# @param elemName Name of element, string.
# @param nameList List of names in system dictionary.
# @param clog CustomLogger object.
def check_ElemDict(elemDict, nameList, clog): 
    errStr = ""
   
    elemDict["transf"] = world.INITM() 
   
    if not "pos" in elemDict:
        elemDict["pos"] = world.ORIGIN()

    else:
        errStr += block_ndarray("pos", elemDict, (3,))

    if not "ori" in elemDict:
        elemDict["ori"] = world.IAX()
    
    else:
        errStr += block_ndarray("ori", elemDict, (3,))

    if not "flip" in elemDict:
        elemDict["flip"] = False
    
    if not "gcenter" in elemDict:
        elemDict["gcenter"] = np.zeros(2)
   
    if not "ecc_uv" in elemDict:
        elemDict["ecc_uv"] = 0

    if not "rot_uv" in elemDict:
        elemDict["rot_uv"] = 0

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

        if "orient" not in elemDict:
            elemDict["orient"] = "x"

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
            elemDict["name"] = "Plane"
        
    num = getIndex(elemDict["name"], nameList)
    if num > 0:
        elemDict["name"] = elemDict["name"] + "_{}".format(num)
    
    if "gmode" in elemDict:
        if elemDict["gmode"] == "xy" or elemDict["gmode"] == 0:
            if "lims_x" in elemDict:
                errStr += block_ndarray("lims_x", elemDict, (2,))
            else:
                errStr += errMsg_field("lims_x", elemDict["name"])

            if "lims_y" in elemDict:
                errStr += block_ndarray("lims_y", elemDict, (2,))
            else:
                errStr += errMsg_field("lims_y", elemDict["name"])

        elif elemDict["gmode"] == "uv" or elemDict["gmode"] == 1:
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

        elif elemDict["gmode"] == "AoE" or elemDict["gmode"] == 2:
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
   
        if elemDict["gridsize"][0] < 0 or elemDict["gridsize"][1] < 0:
            clog.warning(f"Negative gridsize encountered in {elemDict['name']}. Changing sign.")
            elemDict["gridsize"] = np.absolute(elemDict["gridsize"])

    else:
        errStr += errMsg_field("gridsize", elemDict["name"])


    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputReflError()
    
    else:
        return 0

##
# Check a tubular input frame dictionary.
#
# @param TubeRTDict A TubeRTDict object.
# @param namelist List containing names of frames in System.
# @param clog CustomLogger object.
#
# @see TubeRTDict
def check_TubeRTDict(TubeRTDict, nameList, clog):
    errStr = ""
    
    if "name" not in TubeRTDict:
        TubeRTDict["name"] = "TubeFrame"
    
    num = getIndex(TubeRTDict["name"], nameList)

    if num > 0:
        TubeRTDict["name"] = TubeRTDict["name"] + "_{}".format(num)
    
    if "nRays" in TubeRTDict:
        if not isinstance(TubeRTDict["nRays"], int):
            errStr += errMsg_type("nRays", type(TubeRTDict["nRays"]), "TubeRTDict", int)
        
        elif TubeRTDict["nRays"] < 0:
            clog.warning(f"Negative value {TubeRTDict['nRays']} encountered in TubeRTDict. Changing sign")
            TubeRTDict["nRays"] *= -1

    else:
        errStr += errMsg_field("nRays", "TubeRTDict")

    if "nRing" in TubeRTDict:
        if not isinstance(TubeRTDict["nRing"], int):
            errStr += errMsg_type("nRing", type(TubeRTDict["nRays"]), "TubeRTDict", int)
        
        elif TubeRTDict["nRing"] < 0:
            clog.warning(f"Negative value {TubeRTDict['nRing']} encountered in TubeRTDict. Changing sign")
            TubeRTDict["nRing"] *= -1

    else:
        errStr += errMsg_field("nRing", "TubeRTDict")


    if "angx0" in TubeRTDict:
        if not ((isinstance(TubeRTDict["angx0"], float) or isinstance(TubeRTDict["angx0"], int))):
            errStr += errMsg_type("angx0", type(TubeRTDict["angx0"]), "TubeRTDict", [float, int])

    else:
        errStr += errMsg_field("angx0", "TubeRTDict")


    if "angy0" in TubeRTDict:
        if not ((isinstance(TubeRTDict["angy0"], float) or isinstance(TubeRTDict["angy0"], int))):
            errStr += errMsg_type("angy0", type(TubeRTDict["angy0"]), "TubeRTDict", [float, int])

    else:
        errStr += errMsg_field("angy0", "TubeRTDict")


    if "x0" in TubeRTDict:
        if not ((isinstance(TubeRTDict["x0"], float) or isinstance(TubeRTDict["x0"], int))):
            errStr += errMsg_type("x0", type(TubeRTDict["x0"]), "TubeRTDict", [float, int])
        
        elif TubeRTDict["x0"] < 0:
            clog.warning(f"Encountered negative value {TubeRTDict['x0']} in field 'x0' in TubeRTDict {TubeRTDict['name']}. Changing sign.")
            TubeRTDict["x0"] *= -1

    else:
        errStr += errMsg_field("x0", "TubeRTDict")


    if "y0" in TubeRTDict:
        if not ((isinstance(TubeRTDict["y0"], float) or isinstance(TubeRTDict["y0"], int))):
            errStr += errMsg_type("y0", type(TubeRTDict["y0"]), "TubeRTDict", [float, int])
        
        elif TubeRTDict["y0"] < 0:
            clog.warning(f"Encountered negative value {TubeRTDict['y0']} in field 'y0' in TubeRTDict {TubeRTDict['name']}. Changing sign.")
            TubeRTDict["y0"] *= -1

    else:
        errStr += errMsg_field("y0", "TubeRTDict")

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputRTError()

##
# Check a Gaussian input frame dictionary.
#
# @param GRTDict A GRTDict object.
# @param namelist List containing names of frames in System.
# @param clog CustomLogger object.
#
# @see GRTDict
def check_GRTDict(GRTDict, nameList, clog):
    errStr = ""
    
    if "name" not in GRTDict:
        GRTDict["name"] = "GaussFrame"
    
    num = getIndex(GRTDict["name"], nameList)

    if num > 0:
        GRTDict["name"] = GRTDict["name"] + "_{}".format(num)

    if "nRays" in GRTDict:
        if not isinstance(GRTDict["nRays"], int):
            errStr += errMsg_type("nRays", type(GRTDict["nRays"]), "GRTDict", int)

        elif GRTDict["nRays"] < 0:
            clog.warning(f"Negative value {GRTDict['nRays']} encountered in GRTDict. Changing sign")
            GRTDict["nRays"] *= -1

    else:
        errStr += errMsg_field("nRays", "GRTDict")

    if "lam" in GRTDict:
        if GRTDict["lam"] == 0 + 0j:
            clog.info(f"Never heard of a complex-valued wavelength of zero, but good try.. Therefore changing wavelength now to 'lam' equals {np.pi:.42f}!")
            GRTDict["lam"] = np.pi

        if not (isinstance(GRTDict["lam"], float) or isinstance(GRTDict["lam"], int)):
            errStr += errMsg_type("lam", type(GRTDict["lam"]), "GRTDict", [float, int])
        
        elif GRTDict["lam"] < 0:
            clog.warning(f"Encountered negative value {GRTDict['lam']} in field 'lam' in GRTDict {GRTDict['name']}. Changing sign.")
            GRTDict["lam"] *= -1

    else:
        errStr += errMsg_field("lam", "GRTDict")

    if "x0" in GRTDict:
        if not ((isinstance(GRTDict["x0"], float) or isinstance(GRTDict["x0"], int))):
            errStr += errMsg_type("x0", type(GRTDict["x0"]), "GRTDict", [float, int])

        elif GRTDict["x0"] < 0:
            clog.warning(f"Encountered negative value {GRTDict['x0']} in field 'x0' in GRTDict {GRTDict['name']}. Changing sign.")
            GRTDict["x0"] *= -1

    else:
        errStr += errMsg_field("x0", "GRTDict")


    if "y0" in GRTDict:
        if not ((isinstance(GRTDict["y0"], float) or isinstance(GRTDict["y0"], int))):
            errStr += errMsg_type("y0", type(GRTDict["y0"]), "GRTDict", [float, int])
        
        elif GRTDict["y0"] < 0:
            clog.warning(f"Encountered negative value {GRTDict['y0']} in field 'y0' in GRTDict {GRTDict['name']}. Changing sign.")
            GRTDict["y0"] *= -1

    else:
        errStr += errMsg_field("y0", "GRTDict")

    if "n" in GRTDict:
        if not ((isinstance(GRTDict["n"], float) or isinstance(GRTDict["n"], int))):
            errStr += errMsg_type("n", type(GRTDict["n"]), "GRTDict", [float, int])

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputRTError()

##
# Check a ray-trace propagation input dictionary.
#
# @param runRTDict A runRTDict.
# @param elements List containing names of surfaces in System.
# @param frames List containing names of frames in System.
# @param clog CustomLogger object.
def check_runRTDict(runRTDict, elements, frames, clog):
    errStr = ""
   
    cuda = has_CUDA()
    errStr = check_frameSystem(runRTDict["fr_in"], frames, clog, errStr)
    errStr = check_elemSystem(runRTDict["t_name"], elements, clog, errStr)

    num = getIndex(runRTDict["fr_out"], frames)

    if num > 0:
        runRTDict["fr_out"] = runRTDict["fr_out"] + "_{}".format(num)

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

##
# Check a point source input beam dictionary.
#
# @param PSDict A PSDict object.
# @param namelist List containing names of fields in System.
# @param clog CustomLogger object.
#
# @see PSDict
def check_PSDict(PSDict, nameList, clog):
    errStr = ""
    
    if "name" not in PSDict:
        PSDict["name"] = "PointSourcePO"
    
    num = getIndex(PSDict["name"], nameList)

    if num > 0:
        PSDict["name"] = PSDict["name"] + "_{}".format(num)

    if "lam" in PSDict:
        if PSDict["lam"] == 0 + 0j:
            clog.info(f"Never heard of a complex-valued wavelength of zero, but good try.. Therefore changing wavelength now to 'lam' equals {np.pi:.42f}!")
            PSDict["lam"] = np.pi

        if not ((isinstance(PSDict["lam"], float) or isinstance(PSDict["lam"], int))):
            errStr += errMsg_type("lam", type(PSDict["lam"]), "PSDict", [float, int])
        
        elif PSDict["lam"] < 0:
            clog.warning(f"Encountered negative value {PSDict['lam']} in field 'lam' in PSDict {PSDict['name']}. Changing sign.")
            PSDict["lam"] *= -1

    else:
        errStr += errMsg_field("lam", "PSDict")

    if "phase" in PSDict:
        if not ((isinstance(PSDict["phase"], float) or isinstance(PSDict["phase"], int))):
            errStr += errMsg_type("phase", type(PSDict["phase"]), "PSDict", [float, int])

    else:
        PSDict["phase"] = 0

    if "pol" in PSDict:
        errStr += block_ndarray("pol", PSDict, (3,))

    else:
        PSDict["pol"] = np.array([1, 0, 0])

    if "E0" in PSDict:
        if not ((isinstance(PSDict["E0"], float) or isinstance(PSDict["E0"], int))):
            errStr += errMsg_type("E0", type(PSDict["E0"]), "PSDict", [float, int])

    else:
        PSDict["E0"] = 1

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputPOError()

##
# Check a Gaussian input beam dictionary.
#
# @param GPODict A GPODict object.
# @param namelist List containing names of fields in System.
# @param clog CustomLogger object.
#
# @see GPODict
def check_GPODict(GPODict, nameList, clog):
    errStr = ""
    
    if "name" not in GPODict:
        GPODict["name"] = "GaussianBeamPO"
    
    num = getIndex(GPODict["name"], nameList)

    if num > 0:
        GPODict["name"] = GPODict["name"] + "_{}".format(num)

    if "lam" in GPODict:
        if GPODict["lam"] == 0 + 0j:
            clog.info(f"Never heard of a complex-valued wavelength of zero, but good try.. Therefore changing wavelength now to 'lam' equals {np.pi:.42f}!")
            GPODict["lam"] = np.pi

        if not ((isinstance(GPODict["lam"], float) or isinstance(GPODict["lam"], int))):
            errStr += errMsg_type("lam", type(GPODict["lam"]), "GPODict", [float, int])
        
        elif GPODict["lam"] < 0:
            clog.warning(f"Encountered negative value {GPODict['lam']} in field 'lam' in GPODict {GPODict['name']}. Changing sign.")
            GPODict["lam"] *= -1

    else:
        errStr += errMsg_field("lam", "GPODict")

    if "w0x" in GPODict:
        if not ((isinstance(GPODict["w0x"], float) or isinstance(GPODict["w0x"], int))):
            errStr += errMsg_type("w0x", type(GPODict["w0x"]), "GPODict", [float, int])

        elif GPODict["w0x"] < 0:
            clog.warning(f"Encountered negative value {GPODict['w0x']} in field 'w0x' in GPODict {GPODict['name']}. Changing sign.")
            GPODict["w0x"] *= -1

    else:
        errStr += errMsg_field("w0x", "GPODict")


    if "w0y" in GPODict:
        if not ((isinstance(GPODict["w0y"], float) or isinstance(GPODict["w0y"], int))):
            errStr += errMsg_type("w0y", type(GPODict["w0y"]), "GPODict", [float, int])
        
        elif GPODict["w0y"] < 0:
            clog.warning(f"Encountered negative value {GPODict['w0y']} in field 'w0y' in GPODict {GPODict['name']}. Changing sign.")
            GPODict["w0y"] *= -1

    else:
        errStr += errMsg_field("w0y", "GPODict")

    if "n" in GPODict:
        if not ((isinstance(GPODict["n"], float) or isinstance(GPODict["n"], int))):
            errStr += errMsg_type("n", type(GPODict["n"]), "GPODict", [float, int])

        elif GPODict["n"] < 1 and GPODict >= 0:
            clog.warning("Refractive indices smaller than unity are not allowed. Changing to 1.")

    else:
        GPODict["n"] = 1

    if "dxyz" in GPODict:
        if not ((isinstance(GPODict["dxyz"], float) or isinstance(GPODict["dxyz"], int))):
            errStr += errMsg_type("dxyz", type(GPODict["dxyz"]), "GPODict", [float, int])

    else:
        GPODict["dxyz"] = 0

    if "pol" in GPODict:
        errStr += block_ndarray("pol", GPODict, (3,))

    else:
        GPODict["pol"] = np.array([1, 0, 0])

    if "E0" in GPODict:
        if not ((isinstance(GPODict["E0"], float) or isinstance(GPODict["E0"], int))):
            errStr += errMsg_type("E0", type(GPODict["E0"]), "GPODict", [float, int])

    else:
        GPODict["E0"] = 1

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise InputPOError()

##
# Check a physical optics propagation input dictionary.
#
# @param runPODict A runPODict.
# @param elements List containing names of surfaces in System.
# @param currents List containing names of currents in System.
# @param scalarfields List containing names of scalarfields in System.
# @param clog CustomLogger object.
def check_runPODict(runPODict, elements, fields, currents, scalarfields, frames, clog):
    errStr = ""

    cuda = has_CUDA()
    
    if not "exp" in runPODict:
        runPODict["exp"] = "fwd"

    if "t_name" in runPODict:
        check_elemSystem(runPODict["t_name"], elements, clog)

    else:
        errStr += errMsg_field("t_name", "runPODict")
    
    if "mode" not in runPODict:
        errStr += f"Please provide propagation mode.\n"
   
    else:
        if runPODict["mode"] not in PO_modelist:
            errStr += f"{runPODict['mode']} is not a valid propagation mode.\n"

        if "s_current" in runPODict:
            errStr = check_currentSystem(runPODict["s_current"], currents, clog, errStr)
        
        if "s_scalarfield" in runPODict:
            errStr = check_scalarfieldSystem(runPODict["s_scalarfield"], scalarfields, clog, errStr)
    
    errStr = check_elemSystem(runPODict["t_name"], elements, clog, errStr)
  
    if runPODict["mode"] == "JM":
        if "name_JM" not in runPODict:
            errStr += errMsg_field("name_JM", "runPODict")
        
        num = getIndex(runPODict["name_JM"], currents)

        if num > 0:
            runPODict["name_JM"] = runPODict["name_JM"] + "_{}".format(num)
    
    if runPODict["mode"] == "EH":
        if "name_EH" not in runPODict:
            errStr += errMsg_field("name_EH", "runPODict")
        
        num = getIndex(runPODict["name_EH"], fields)

        if num > 0:
            runPODict["name_EH"] = runPODict["name_EH"] + "_{}".format(num)
    
    if runPODict["mode"] == "JMEH":
        if "name_EH" not in runPODict:
            errStr += errMsg_field("name_EH", "runPODict")
        
        num = getIndex(runPODict["name_EH"], fields)

        if num > 0:
            runPODict["name_EH"] = runPODict["name_EH"] + "_{}".format(num)
        
        if "name_JM" not in runPODict:
            errStr += errMsg_field("name_JM", "runPODict")
        
        num = getIndex(runPODict["name_JM"], currents)

        if num > 0:
            runPODict["name_JM"] = runPODict["name_JM"] + "_{}".format(num)
    
    if runPODict["mode"] == "EHP":
        if "name_EH" not in runPODict:
            errStr += errMsg_field("name_EH", "runPODict")
        
        num = getIndex(runPODict["name_EH"], fields)

        if num > 0:
            runPODict["name_EH"] = runPODict["name_EH"] + "_{}".format(num)
        
        if "name_P" not in runPODict:
            errStr += errMsg_field("name_P", "runPODict")
        
        num = getIndex(runPODict["name_P"], frames)

        if num > 0:
            runPODict["name_P"] = runPODict["name_P"] + "_{}".format(num)

    if runPODict["mode"] == "FF":
        if "name_EH" not in runPODict:
            errStr += errMsg_field("name_EH", "runPODict")
        
        num = getIndex(runPODict["name_EH"], fields)

        if num > 0:
            runPODict["name_EH"] = runPODict["name_EH"] + "_{}".format(num)
    
    if runPODict["mode"] == "scalar":
        if "name_field" not in runPODict:
            errStr += errMsg_field("name_field", "runPODict")
        
        num = getIndex(runPODict["name_field"], scalarfields)

        if num > 0:
            runPODict["name_field"] = runPODict["name_field"] + "_{}".format(num)

    if "epsilon" not in runPODict:
        runPODict["epsilon"] = 1

    if "device" not in runPODict:
        if cuda:
            runPODict["device"] = "GPU"

        else:
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

##
# Check a hybrid propagation input dictionary.
#
# @param hybridDict A hybridDict.
# @param elements List containing names of surfaces in System.
# @param frames List containing names of frames in System.
# @param fields List containing names of frames in System.
# @param clog CustomLogger object.
def check_hybridDict(hybridDict, elements, frames, fields, clog):
    errStr = ""
   
    errStr = check_frameSystem(hybridDict["fr_in"], frames, clog, errStr)
    errStr = check_elemSystem(hybridDict["t_name"], elements, clog, errStr)
    errStr = check_fieldSystem(hybridDict["field_in"], fields, clog, errStr)

    if "start" not in hybridDict:
        hybridDict["start"] = None

    elif "start" in hybridDict:
        if hybridDict["start"] is not None:
            errStr += block_ndarray("start", hybridDict, (3,), cust_name="hybridDict")
    
    if "interp" not in hybridDict:
        hybridDict["interp"] = True
        
    elif "interp" in hybridDict:
        if not isinstance(hybridDict["interp"], bool):
            errStr += errMsg_type("interp", type(hybridDict["interp"]), "hybridDict", bool)
    
    if "comp" not in hybridDict:
        hybridDict["comp"] = True
        
    elif "comp" in hybridDict:
        if not isinstance(hybridDict["comp"], str) and hybridDict["comp"] != True:
            errStr += errMsg_type("comp", type(hybridDict["comp"]), "hybridDict", str)

    num = getIndex(hybridDict["fr_out"], frames)

    if num > 0:
        hybridDict["fr_out"] = hybridDict["fr_out"] + "_{}".format(num)
    
    num = getIndex(hybridDict["field_out"], fields)

    if num > 0:
        hybridDict["field_out"] = hybridDict["field_out"] + "_{}".format(num)
    
    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise HybridPropError()

##
# CHeck if aperture dictionary is valid.
#
# @param aperDict An aperture dictionary.
# @param clog CustomLogger object.
#
# @see aperDict
def check_aperDict(aperDict, clog):
    errStr = ""

    if "plot" in aperDict:
        if not isinstance(aperDict["plot"], bool):
            errStr += errMsg_type("plot", type(aperDict["plot"]), "aperDict", bool)

    else:
        aperDict["plot"] = True

    if "center" in aperDict:
        errStr += block_ndarray("center", aperDict, (2,), cust_name="aperDict")

    else:
        aperDict["center"] = np.zeros(2)

    if not "outer" in aperDict:
        errStr += errMsg_field("outer", "aperDict")
    
    if not "inner" in aperDict:
        errStr += errMsg_field("inner", "aperDict")

    if errStr:
        errList = errStr.split("\n")[:-1]

        for err in errList:
            clog.error(err)
        raise ApertureError()
##
# Check if ellipsoid limits are valid points.
# If not, reduces limits to acceptable values.
#
# @param ellipsoid A reflDict containing description of ellipsoid surface.
# @param clog CustomLogger object.
def check_ellipseLimits(ellipsoid, clog):
    buff = 1000
    idx_lim = 0
    if ellipsoid["coeffs"][1] < ellipsoid["coeffs"][0]:
        idx_lim = 1

    if ellipsoid["gmode"] == 0:
        if np.absolute(ellipsoid["lims_x"][0]) > ellipsoid["coeffs"][idx_lim]:
            sgn = np.sign((ellipsoid["lims_x"][0]))
            clog.warning(f"Lower x-limit of {ellipsoid['lims_x'][0]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {sgn*ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_x"][0] = sgn * (ellipsoid["coeffs"][idx_lim] + ellipsoid["coeffs"][0] / buff)
        
        if np.absolute(ellipsoid["lims_x"][1]) > ellipsoid["coeffs"][idx_lim]:
            sgn = np.sign((ellipsoid["lims_x"][1]))
            clog.warning(f"Upper x-limit of {ellipsoid['lims_x'][1]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {sgn*ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_x"][1] = sgn * (ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_x"][1] / buff)
        
        if np.absolute(ellipsoid["lims_y"][0]) > ellipsoid["coeffs"][idx_lim]:
            sgn = np.sign((ellipsoid["lims_y"][0]))
            clog.warning(f"Lower y-limit of {ellipsoid['lims_y'][0]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {sgn*ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_y"][0] = sgn * (ellipsoid["coeffs"][idx_lim] + ellipsoid["lims_y"][0] / buff)
        
        if np.absolute(ellipsoid["lims_y"][1]) > ellipsoid["coeffs"][idx_lim]:
            sgn = np.sign((ellipsoid["lims_y"][1]))
            clog.warning(f"Upper y-limit of {ellipsoid['lims_y'][1]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {sgn*ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_y"][1] = sgn * (ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_y"][1] / buff)

    elif ellipsoid["gmode"] == 1:
        if np.absolute(ellipsoid["lims_u"][0]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Lower u-limit of {ellipsoid['lims_u'][0]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_u"][0] = ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_u"][0] / buff
 
        if np.absolute(ellipsoid["lims_u"][1]) > ellipsoid["coeffs"][idx_lim]:
            clog.warning(f"Upper u-limit of {ellipsoid['lims_u'][1]:.3f} incompatible with ellipsoid {ellipsoid['name']}. Changing to {ellipsoid['coeffs'][idx_lim]}.")
            ellipsoid["lims_u"][1] = ellipsoid["coeffs"][idx_lim] - ellipsoid["lims_u"][1] / buff

##
# Check if beams to be merged are defined on same surface.
# If not, raise MergeBeam Error.
#
# @param beams Fields/currents to be merged.
# @param checkDict System c=dictionary containing fields/currents.
# @param clog CustomLogger object.
def check_sameBound(beams, checkDict, clog):
    errStr = ""
    surf0 = checkDict[beams[0]].surf
    for i in range(len(beams) - 1):
        if checkDict[beams[i+1]].surf != surf0:
            errStr += errMsg_mergebeam(beams[i+1], surf0, checkDict[beams[i+1]].surf)
    
    if errStr:
        errList = errStr.split("\n")[:-1]
        for err in errList:
            clog.error(err)
        
        raise MergeBeamError()

##
# Check if field and frame are associated on the same surface.
# Used for hybrid propagations.
# Currently, can only have one single association per surface!
#
# @param associations All present associations in system.
# @param fieldName Name of field to be propagated.
# @param frameName Name of frame to be propagated.
# @param surf Name of surface from which a hybrid propagation is performed.
# @param clog CustomLogger object.
def check_associations(associations, fieldName, frameName, surf, clog):
    if surf not in associations.keys():
        clog.error(f"Surface {surf} not found in associations.")
        
        raise HybridPropError
    
    else:
        if (fieldName not in associations[surf]) or (frameName not in associations[surf]):
            clog.error(f"Field {fieldName} and frame {frameName} are not associated.")
            
            raise HybridPropError





