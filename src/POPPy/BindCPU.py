import ctypes
import numpy as np

from src.POPPy.BindUtils import *
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

import sys

#############################################################################
#                                                                           #
#              List of bindings for the CPU interface of POPPy.             #
#                                                                           #
#############################################################################

def loadCPUlib():
    lib = ctypes.CDLL('./src/C++/libpoppycpu.so')

    lib.propagateToGrid_JM.argtypes = [ctypes.POINTER(c2Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_JM.restype = None

    lib.propagateToGrid_EH.argtypes = [ctypes.POINTER(c2Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_EH.restype = None

    lib.propagateToGrid_JMEH.argtypes = [ctypes.POINTER(c4Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_JMEH.restype = None

    lib.propagateToGrid_EHP.argtypes = [ctypes.POINTER(c2rBundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_EHP.restype = None

    lib.propagateToGrid_scalar.argtypes = [ctypes.POINTER(arrC1), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(arrC1),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_scalar.restype = None

    lib.propagateToFarField.argtypes = [ctypes.POINTER(c2Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToFarField.restype = None

    return lib

# WRAPPER FUNCTIONS DOUBLE PREC
def POPPy_CPUd(source, target, currents, k, epsilon, t_direction, nThreads, mode):
    lib = loadCPUlib()

    # Create structs with pointers for source and target
    csp = reflparams()
    ctp = reflparams()

    cs = reflcontainer()
    ct = reflcontainer()

    gs = source["gridsize"][0] * source["gridsize"][1]
    gt = target["gridsize"][0] * target["gridsize"][1]

    allfill_reflparams(csp, source, ctypes.c_double)
    allfill_reflparams(ctp, target, ctypes.c_double)

    allocate_reflcontainer(cs, gs, ctypes.c_double)
    allocate_reflcontainer(ct, gt, ctypes.c_double)

    if mode == "Scalar":
        c_cfield = arrC1()
        fieldConv(field, c_field, gs, ctypes.c_double)

    else:
        c_currents = c2Bundle()
        currentConv(currents, c_currents, gs, ctypes.c_double)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    k           = ctypes.c_double(k)
    nThreads    = ctypes.c_int(nThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    if mode == "JM":
        res = c2Bundle()
        allocate_c2Bundle(res, gt, ctypes.c_double)

        lib.propagateToGrid_JM(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, nThreads,
                                epsilon, t_direction)

        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents')

        return JM

    elif mode == "EH":
        res = c2Bundle()
        allocate_c2Bundle(res, gt, ctypes.c_double)

        lib.propagateToGrid_EH(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, nThreads,
                                epsilon, t_direction)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

        return EH

    elif mode == "JMEH":
        res = c4Bundle()
        allocate_c4Bundle(res, gt, ctypes.c_double)

        lib.propagateToGrid_JMEH(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, nThreads,
                                epsilon, t_direction)

        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape)

        return [JM, EH]

    elif mode == "EHP":
        res = c2rBundle()
        allocate_c2rBundle(res, gt, ctypes.c_double)

        lib.propagateToGrid_EHP(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, nThreads,
                                epsilon, t_direction)

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape)

        return [EH, Pr]

    elif mode == "Scalar":
        res = arrC1()
        allocate_arrC1(res, gt, ctypes.c_double)

        lib.propagateToGrid_scalar(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, nThreads,
                                epsilon, t_direction)

        # Unpack filled struct
        E = arrC1ToObj(res, shape=target_shape)

        return E

    elif mode == "FF":
        res = c2Bundle()
        allocate_c2Bundle(res, gt, ctypes.c_double)

        lib.propagateToFarField(ctypes.byref(res), csp, ctp,
                                ctypes.byref(cs), ctypes.byref(ct),
                                ctypes.byref(c_currents), k, nThreads,
                                epsilon, t_direction)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

        return EH

if __name__ == "__main__":
    rint("Bindings for POPPy CPU.")
