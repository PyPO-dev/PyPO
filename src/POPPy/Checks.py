import numpy as np
from src.POPPy.POPPyTypes import *

class InputReflError(Exception):
    pass

class InputRTError(Exception):
    pass

def errMsg_field(fieldName, elemName):
    return "\nMissing field \"{}\", element {}.".format(fieldName, elemName)

def errMsg_type(fieldName, inpType, elemName, fieldType):
    return "\nWrong type {} in field \"{}\", element {}. Expected {}.".format(inpType, fieldName, elemName, fieldType)

def errMsg_option(fieldName, option, elemName, args):
    if len(args) == 2:
        return "\nUnknown option \"{}\" in field \"{}\", element {}. Expected \"{}\" or \"{}\".".format(option, fieldName, elemName, args[0], args[1])

    elif len(args) == 3:
        return "\nUnknown option \"{}\" in field \"{}\", element {}. Expected \"{}\", \"{}\" or \"{}\".".format(option, fieldName, elemName, args[0], args[1], args[2])

def errMsg_shape(fieldName, shape, elemName, shapeExpect):
    return "\nIncorrect input shape of {} for field \"{}\", element {}. Expected {}.".format(shape, fieldName, elemName, shapeExpect)

def errMsg_value(fieldName, value, elemName):
    return "\nIncorrect value {} encountered in field \"{}\", element {}.".format(value, fieldName, elemName)

def check_ElemDict(elemDict):
    """
    Check if the element dictionary has been properly filled.
    """

    errStr = ""
    
    if elemDict["type"] == 0:

        if "pmode" in elemDict:
            if elemDict["pmode"] == "focus":
                if "vertex" in elemDict:
                    if not isinstance(elemDict["vertex"], np.ndarray):
                        errStr += errMsg_type("vertex", type(elemDict["vertex"]), elemDict["name"], np.ndarray)

                    elif not elemDict["vertex"].shape == (3,):
                        errStr += errMsg_shape("vertex", elemDict["vertex"].shape, elemDict["name"], "(3,)")
                
                else:
                    errStr += errMsg_field("vertex", elemDict["name"])

                if "focus_1" in elemDict:
                    if not isinstance(elemDict["focus_1"], np.ndarray):
                         errStr += errMsg_type("focus_1", type(elemDict["focus_1"]), elemDict["name"], np.ndarray)
                     
                    elif not elemDict["focus_1"].shape == (3,):
                        errStr += errMsg_shape("focus_1", elemDict["focus_1"].shape, elemDict["name"], "(3,)")

                else:
                    errStr += errMsg_field("focus_1", elemDict["name"])

            elif elemDict["pmode"] == "manual":
                if "coeffs" in elemDict:
                    if not isinstance(elemDict["coeffs"], np.ndarray):
                        errStr += errMsg_type("coeffs", type(elemDict["coeffs"]), elemDict["name"], np.ndarray)

                    elif not elemDict["coeffs"].shape == (2,):
                        errStr += errMsg_shape("coeffs", elemDict["coeffs"].shape, elemDict["name"], "(2,)")

            else:
                args = ["focus", "manual"]
                errStr += errMsg_option("pmode", elemDict["pmode"], elemDict["name"], args=args)

        else:
            errStr += errMsg_field("pmode", elemDict["name"])

    elif elemDict["type"] == 1 or elemDict["type"] == 2:

        if "pmode" in elemDict:
            if elemDict["pmode"] == "focus":
                if "focus_1" in elemDict:
                    if not isinstance(elemDict["focus_1"], np.ndarray):
                        errStr += errMsg_type("focus_1", type(elemDict["focus_1"]), elemDict["name"], np.ndarray)

                    elif not elemDict["focus_1"].shape == (3,):
                        errStr += errMsg_shape("focus_1", elemDict["focus_1"].shape, elemDict["name"], "(3,)")

                else:
                    errStr += errMsg_field("focus_1", elemDict["name"])

                if "focus_2" in elemDict:
                    if not isinstance(elemDict["focus_2"], np.ndarray):
                         errStr += errMsg_type("focus_2", type(elemDict["focus_2"]), elemDict["name"], np.ndarray)
                     
                    elif not elemDict["focus_2"].shape == (3,):
                        errStr += errMsg_shape("focus_2", elemDict["focus_2"].shape, elemDict["name"], "(3,)")

                else:
                    errStr += errMsg_field("focus_2", elemDict["name"])
                
                if "ecc" in elemDict:
                    if not (isinstance(elemDict["ecc"], float) or isinstance(elemDict["ecc"], int)):
                        errStr += errMsg_type("ecc", type(elemDict["ecc"]), elemDict["name"], [float, int])

                else:
                    errStr += errMsg_field("ecc", elemDict["name"])


            elif elemDict["pmode"] == "manual":
                if "coeffs" in elemDict:
                    if not isinstance(elemDict["coeffs"], np.ndarray):
                        errStr += errMsg_type("coeffs", type(elemDict["coeffs"]), elemDict["name"], np.ndarray)

                    elif not elemDict["coeffs"].shape == (3,):
                        errStr += errMsg_shape("coeffs", elemDict["coeffs"].shape, elemDict["name"], "(3,)")

                else:
                    errStr += errMsg_field("coeffs", elemDict["name"])


            else:
                args = ["focus", "manual"]
                errStr += errMsg_option("pmode", elemDict["pmode"], elemDict["name"], args=args)

    if "gmode" in elemDict:
        if elemDict["gmode"] == "xy":
            if "lims_x" in elemDict:
                if not isinstance(elemDict["lims_x"], np.ndarray):
                    errStr += errMsg_type("lims_x", type(elemDict["lims_x"]), elemDict["name"], np.ndarray)

                elif not elemDict["lims_x"].shape == (2,):
                    errStr += errMsg_shape("lims_x", elemDict["lims_x"].shape, elemDict["name"], "(2,)")

            else:
                errStr += errMsg_field("lims_x", elemDict["name"])

            if "lims_y" in elemDict:
                if not isinstance(elemDict["lims_y"], np.ndarray):
                    errStr += errMsg_type("lims_y", type(elemDict["lims_y"]), elemDict["name"], np.ndarray)

                elif not elemDict["lims_y"].shape == (2,):
                    errStr += errMsg_shape("lims_y", elemDict["lims_y"].shape, elemDict["name"], "(2,)")

            else:
                errStr += errMsg_field("lims_y", elemDict["name"])

        elif elemDict["gmode"] == "uv":
            if "lims_u" in elemDict:
                if not isinstance(elemDict["lims_u"], np.ndarray):
                    errStr += errMsg_type("lims_u", type(elemDict["lims_u"]), elemDict["name"], np.ndarray)

                elif not elemDict["lims_u"].shape == (2,):
                    errStr += errMsg_shape("lims_u", elemDict["lims_u"].shape, elemDict["name"], "(2,)")

                if elemDict["lims_u"][0] < 0:
                    errStr += errMsg_value("lims_u", elemDict["lims_u"], elemDict["name"])

            else:
                errStr += errMsg_field("lims_u", elemDict["name"])

            if "lims_v" in elemDict:
                if not isinstance(elemDict["lims_v"], np.ndarray):
                    errStr += errMsg_type("lims_v", type(elemDict["lims_v"]), elemDict["name"], np.ndarray)

                elif not elemDict["lims_v"].shape == (2,):
                    errStr += errMsg_shape("lims_v", elemDict["lims_v"].shape, elemDict["name"], "(2,)")

                if elemDict["lims_v"][0] < 0:
                    errStr += errMsg_value("lims_v", elemDict["lims_v"][0], elemDict["name"])
 
                if elemDict["lims_v"][1] > 360:
                    errStr += errMsg_value("lims_v", elemDict["lims_v"][1], elemDict["name"])
            else:
                errStr += errMsg_field("lims_v", elemDict["name"])
    

        elif elemDict["gmode"] == "AoE":
            if "lims_Az" in elemDict:
                if not isinstance(elemDict["lims_Az"], np.ndarray):
                    errStr += errMsg_type("lims_Az", type(elemDict["lims_Az"]), elemDict["name"], np.ndarray)

                elif not elemDict["lims_Az"].shape == (2,):
                    errStr += errMsg_shape("lims_Az", elemDict["lims_Az"].shape, elemDict["name"], "(2,)")

            else:
                errStr += errMsg_field("lims_Az", elemDict["name"])

            if "lims_El" in elemDict:
                if not isinstance(elemDict["lims_El"], np.ndarray):
                    errStr += errMsg_type("lims_El", type(elemDict["lims_El"]), elemDict["name"], np.ndarray)

                elif not elemDict["lims_El"].shape == (2,):
                    errStr += errMsg_shape("lims_El", elemDict["lims_El"].shape, elemDict["name"], "(2,)")

            else:
                errStr += errMsg_field("lims_El", elemDict["name"])
    
        else:
            args = ["xy", "uv", "AoE (plane only)"]
            errStr += errMsg_option("gmode", elemDict["gmode"], elemDict["name"], args=args)

    else:
        errStr += errMsg_field("gmode", elemDict["name"])

    if "gridsize" in elemDict:
        if not isinstance(elemDict["gridsize"], np.ndarray):
            errStr += errMsg_type("gridsize", type(elemDict["gridsize"]), elemDict["name"], np.ndarray)

        elif not elemDict["gridsize"].shape == (2,):
            errStr += errMsg_shape("gridsize", elemDict["gridsize"], elemDict["name"], "(2,)")

        if not isinstance(elemDict["gridsize"][0], np.int64):
            errStr += errMsg_type("gridsize[0]", type(elemDict["gridsize"][0]), elemDict["name"], np.int64)

        if not isinstance(elemDict["gridsize"][1], np.int64):
            errStr += errMsg_type("gridsize[1]", type(elemDict["gridsize"][1]), elemDict["name"], np.int64)
    
    if errStr:
        raise InputReflError(errStr)

def check_RTDict(RTDict):
    errStr = ""

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
        if not (isinstance(RTDict["angx"], float) or isinstance(RTDict["angx"], int)):
            errStr += errMsg_type("angx", type(RTDict["angx"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("angx", "RTDict")


    if "angy" in RTDict:
        if not (isinstance(RTDict["angy"], float) or isinstance(RTDict["angy"], int)):
            errStr += errMsg_type("angy", type(RTDict["angy"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("angy", "RTDict")


    if "a" in RTDict:
        if not (isinstance(RTDict["a"], float) or isinstance(RTDict["a"], int)):
            errStr += errMsg_type("a", type(RTDict["a"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("a", "RTDict")


    if "b" in RTDict:
        if not (isinstance(RTDict["b"], float) or isinstance(RTDict["b"], int)):
            errStr += errMsg_type("b", type(RTDict["b"]), "RTDict", [float, int])

    else:
        errStr += errMsg_field("b", "RTDict")

    if "tChief" in RTDict:
        if not isinstance(RTDict["tChief"], np.ndarray):
            errStr += errMsg_type("tChief", type(RTDict["tChief"]), "RTDict", np.ndarray)

        elif not RTDict["tChief"].shape == (3,):
            errStr += errMsg_shape("tChief", RTDict["tChief"].shape, "RTDict", "(3,)")

    else:
        errStr += errMsg_field("tChief", "RTDict")

    if "oChief" in RTDict:
        if not isinstance(RTDict["oChief"], np.ndarray):
            errStr += errMsg_type("oChief", type(RTDict["oChief"]), "RTDict", np.ndarray)
                     
        elif not RTDict["oChief"].shape == (3,):
            errStr += errMsg_shape("oChief", RTDict["oChief"].shape, "RTDict", "(3,)")

    else:
        errStr += errMsg_field("oChief", "RTDict")

    if errStr:
        raise InputRTError(errStr)
