import ctypes
import math
import numpy as np
import os
import pathlib
from PyPO.BindUtils import *
from PyPO.Structs import *
from PyPO.PyPOTypes import *
import PyPO.Config as Config
import PyPO.Threadmgr as TManager

##
# @file
# Bindings for the ctypes interface for PyPO. 
# These bindings are concerned with propagations for the ray-tracer and the physical optics on the GPU.

##
# Load the pypogpu shared library. Will detect the operating system and link the library accordingly.
#
# @returns lib The ctypes library containing the C/C++ functions.
def loadGPUlib():
    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpypogpu.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypogpu.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypogpu.dylib"))
    
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
    
    lib.callKernelf_scalar.argtypes = [ctypes.POINTER(arrC1f), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(arrC1f), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_scalar.restype = None

    lib.callRTKernel.argtypes = [reflparamsf, ctypes.POINTER(cframef), ctypes.POINTER(cframef),
                                ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callRTKernel.restype = None

    return lib

##
# Perform a PO propagation on the GPU.
# Note that the calculations are always done in single precision for the GPU.
# Depending on the 'mode' parameter in the runPODict, this function returns different objects.
# Please see the dictionary templates for an overview.
#
# @param source A reflDict dictionary of the surface on which the source currents/scalarfields are defined.
# @param target A reflDict dictionary of the target surface on which the results are calculated.
# @param runPODict A runPODict dictionary containing the relevant propagation parameters.
#
# @see runPODict
def PyPO_GPUf(source, target, runPODict):
    lib = loadGPUlib()
    mgr = TManager.Manager(Config.context)

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

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    nBlocks = math.ceil(gt / runPODict["nThreads"])

    if runPODict["exp"] == "fwd":
        exp_prop = -1

    elif runPODict["exp"] == "bwd":
        exp_prop = 1

    k           = ctypes.c_float(runPODict["k"])
    nThreads    = ctypes.c_int(runPODict["nThreads"])
    nBlocks     = ctypes.c_int(nBlocks)
    epsilon     = ctypes.c_float(runPODict["epsilon"])
    t_direction = ctypes.c_float(exp_prop)
    
    if runPODict["mode"] == "scalar":
        c_field = arrC1f()
        sfieldConv(runPODict["s_scalarfield"], c_field, gs, ctypes.c_float)
        args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
                ctypes.byref(c_field), k, epsilon,
                t_direction, nBlocks, nThreads]

    else:
        c_currents = c2Bundlef()
        allfill_c2Bundle(c_currents, runPODict["s_current"], gs, ctypes.c_float)
        args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
                ctypes.byref(c_currents), k, epsilon,
                t_direction, nBlocks, nThreads]

    if runPODict["mode"] == "JM":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        #t = mgr.new_sthread(target=lib.callKernelf_JM, args=args)
        mgr.new_sthread(target=lib.callKernelf_JM, args=args)
        
        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents', np_t=np.float64)

        return JM

    elif runPODict["mode"] == "EH":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_EH, args=args)
        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

    elif runPODict["mode"] == "JMEH":
        res = c4Bundlef()

        args.insert(0, res)

        allocate_c4Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_JMEH, args=args)
        
        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape, np_t=np.float64)

        return [JM, EH]

    elif runPODict["mode"] == "EHP":
        res = c2rBundlef()

        args.insert(0, res)

        allocate_c2rBundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_EHP, args=args)

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape, np_t=np.float64)

        return [EH, Pr]

    elif runPODict["mode"] == "FF":
        res = c2Bundlef()
        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_FF, args=args)
        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH
    
    elif runPODict["mode"] == "scalar":
        res = arrC1f()
        args.insert(0, res)

        allocate_arrC1(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_scalar, args=args)
        # Unpack filled struct
        S = arrC1ToObj(res, shape=target_shape, np_t=np.float64)

        return S

##
# Perform an RT propagation on the GPU.
# Note that the calculations are always done in single precision for the GPU.
#
# @param runRTDict A runRTDict dictionary containing the relevant propagation parameters.
#
# @see runRTDict
def RT_GPUf(runRTDict):
    lib = loadGPUlib()
    mgr = TManager.Manager(Config.context)

    ctp = reflparamsf()
    allfill_reflparams(ctp, runRTDict["t_name"], ctypes.c_float)

    inp = cframef()
    res = cframef()

    allocate_cframe(res, runRTDict["fr_in"].size, ctypes.c_float)
    allfill_cframe(inp, runRTDict["fr_in"], runRTDict["fr_in"].size, ctypes.c_float)

    nBlocks = math.ceil(runRTDict["fr_in"].size / runRTDict["nThreads"])

    nBlocks      = ctypes.c_int(nBlocks)
    nThreads    = ctypes.c_int(runRTDict["nThreads"])
    tol         = ctypes.c_float(runRTDict["tol"])
    t0          = ctypes.c_float(runRTDict["t0"])

    args = [ctp, ctypes.byref(inp), ctypes.byref(res),
            tol, t0, nBlocks, nThreads]

    mgr.new_sthread(target=lib.callRTKernel, args=args)
    
    shape = (runRTDict["fr_in"].size,)
    fr_out = frameToObj(res, np_t=np.float32, shape=shape)

    return fr_out

if __name__ == "__main__":
    print("Bindings for PyPO GPU.")
