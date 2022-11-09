import ctypes
import numpy as np

from src.POPPy.BindUtils import *
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

import threading

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

    lib.propagateRays.argtypes = [reflparams, ctypes.POINTER(cframe),
                                ctypes.POINTER(cframe), ctypes.c_int, ctypes.c_double, ctypes.c_double]

    lib.propagateRays.restype = None

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

    args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
            ctypes.byref(c_currents), k, nThreads, epsilon,
            t_direction]

    if mode == "JM":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_JM, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents', np_t=np.float64)

        return JM

    elif mode == "EH":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_EH, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

    elif mode == "JMEH":
        res = c4Bundle()
        args.insert(0, res)

        allocate_c4Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_JMEH, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape, np_t=np.float64)

        return [JM, EH]

    elif mode == "EHP":
        res = c2rBundle()
        args.insert(0, res)

        allocate_c2rBundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_EHP, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape, np_t=np.float64)

        return [EH, Pr]

    elif mode == "Scalar":
        res = arrC1()
        args.insert(0, res)

        allocate_arrC1(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_scalar, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        E = arrC1ToObj(res, shape=target_shape)

        return E

    elif mode == "FF":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToFarField, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

def RT_CPUd(target, fr_in, epsilon, t0, nThreads):
    lib = loadCPUlib()

    inp = cframe()
    res = cframe()

    allocate_cframe(res, fr_in.size, ctypes.c_double)
    allfill_cframe(inp, fr_in, fr_in.size, ctypes.c_double)

    ctp = reflparams()
    allfill_reflparams(ctp, target, ctypes.c_double)

    nThreads    = ctypes.c_int(nThreads)
    epsilon     = ctypes.c_double(epsilon)
    t0          = ctypes.c_double(t0)

    args = [ctp, ctypes.byref(inp), ctypes.byref(res),
                        nThreads, epsilon, t0]

    t = threading.Thread(target=lib.propagateRays, args=args)
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exit
        t.join(.1)

    shape = (fr_in.size,)
    fr_out = frameToObj(res, np_t=np.float64, shape=shape)

    return fr_out

if __name__ == "__main__":
    rint("Bindings for POPPy CPU.")
