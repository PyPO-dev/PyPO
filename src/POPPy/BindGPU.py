import ctypes
import math
import numpy as np
import os
import sys
import pathlib
from src.POPPy.BindUtils import *
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

import threading

#############################################################################
#                                                                           #
#              List of bindings for the GPU interface of POPPy.             #
#                                                                           #
#############################################################################

def loadGPUlib():
    try:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build/Debug"
        lib = ctypes.CDLL(str(LD_PATH/"poppygpu.dll"))
    except:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build"
        lib = ctypes.CDLL(LD_PATH/"libpoppygpu.so")

    lib.callKernelf_JM.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_JM.restype = None

    lib.callKernelf_EH.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_EH.restype = None

    lib.callKernelf_JMEH.argtypes = [ctypes.POINTER(c4Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_JMEH.restype = None

    lib.callKernelf_EHP.argtypes = [ctypes.POINTER(c2rBundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_EHP.restype = None

    lib.callKernelf_FF.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_FF.restype = None

    lib.callRTKernel.argtypes = [reflparamsf, ctypes.POINTER(cframef), ctypes.POINTER(cframef),
                                ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callRTKernel.restype = None

    return lib

# WRAPPER FUNCTIONS DOUBLE PREC

#### SINGLE PRECISION
def POPPy_GPUf(source, target, PODict):
    lib = loadGPUlib()

    # Create structs with pointers for source and target
    csp = reflparamsf()
    ctp = reflparamsf()

    cs = reflcontainerf()
    ct = reflcontainerf()

    gs = source["gridsize"][0] * source["gridsize"][1]
    gt = target["gridsize"][0] * target["gridsize"][1]

    allfill_reflparams(csp, source, ctypes.c_float)
    allfill_reflparams(ctp, target, ctypes.c_float)

    allocate_reflcontainer(cs, gs, ctypes.c_float)
    allocate_reflcontainer(ct, gt, ctypes.c_float)

    if PODict["mode"] == "Scalar":
        c_cfield = arrC1f()
        fieldConv(PODict["s_field"], c_field, gs, ctypes.c_float)

    else:
        c_currents = c2Bundlef()
        currentConv(PODict["s_current"], c_currents, gs, ctypes.c_float)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    nBlocks = math.ceil(gt / PODict["nThreads"])

    if PODict["exp"] == "fwd":
        exp_prop = -1

    elif PODict["exp"] == "bwd":
        exp_prop = 1

    k           = ctypes.c_float(PODict["k"])
    nThreads    = ctypes.c_int(PODict["nThreads"])
    nBlocks     = ctypes.c_int(nBlocks)
    epsilon     = ctypes.c_float(PODict["epsilon"])
    t_direction = ctypes.c_float(exp_prop)

    args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
            ctypes.byref(c_currents), k, epsilon,
            t_direction, nBlocks, nThreads]

    if PODict["mode"] == "JM":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        t = threading.Thread(target=lib.callKernelf_JM, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents', np_t=np.float64)

        return JM

    elif PODict["mode"] == "EH":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        t = threading.Thread(target=lib.callKernelf_EH, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

    elif PODict["mode"] == "JMEH":
        res = c4Bundlef()

        args.insert(0, res)

        allocate_c4Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        t = threading.Thread(target=lib.callKernelf_JMEH, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape, np_t=np.float64)

        return [JM, EH]

    elif PODict["mode"] == "EHP":
        res = c2rBundlef()

        args.insert(0, res)

        allocate_c2rBundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        t = threading.Thread(target=lib.callKernelf_EHP, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape, np_t=np.float64)

        return [EH, Pr]

    elif PODict["mode"] == "FF":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        t = threading.Thread(target=lib.callKernelf_FF, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            t.join(.1)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

def RT_GPUf(target, fr_in, epsilon, t0, nThreads):
    lib = loadGPUlib()

    ctp = reflparamsf()
    allfill_reflparams(ctp, target, ctypes.c_float)

    inp = cframef()
    res = cframef()

    allocate_cframe(res, fr_in.size, ctypes.c_float)
    allfill_cframe(inp, fr_in, fr_in.size, ctypes.c_float)

    nBlocks = math.ceil(fr_in.size / nThreads)

    nBocks      = ctypes.c_int(nBlocks)
    nThreads    = ctypes.c_int(nThreads)
    epsilon     = ctypes.c_float(epsilon)
    t0          = ctypes.c_float(t0)

    args = [ctp, ctypes.byref(inp), ctypes.byref(res),
            epsilon, t0, nBlocks, nThreads]

    t = threading.Thread(target=lib.callRTKernel, args=args)
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exit
        t.join(.1)

    shape = (fr_in.size,)
    fr_out = frameToObj(res, np_t=np.float32, shape=shape)

    return fr_out

if __name__ == "__main__":
    print("Bindings for POPPy GPU.")
